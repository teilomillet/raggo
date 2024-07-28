package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/teilomillet/goal"
	"github.com/teilomillet/raggo"
)

type ResumeCache struct {
	Hash          string
	Summary       string
	Info          ResumeInfo
	LastProcessed time.Time
}

// ResumeInfo represents structured information extracted from a resume
type ResumeInfo struct {
	Name                string   `json:"name"`
	ContactDetails      string   `json:"contact_details"`
	ProfessionalSummary string   `json:"professional_summary"`
	Skills              []string `json:"skills"`
	WorkExperience      string   `json:"work_experience"`
	Education           string   `json:"education"`
	AdditionalInfo      string   `json:"additional_info"`
}

func main() {

	// Set log level to Debug for more detailed output
	raggo.SetLogLevel(raggo.LogLevelWarn)

	// Initialize the LLM
	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-4o-mini"),
		goal.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		goal.SetMaxTokens(2048),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Initialize raggo components
	parser := raggo.NewParser()
	chunker, err := raggo.NewChunker(
		raggo.ChunkSize(512),
		raggo.ChunkOverlap(64),
	)
	if err != nil {
		log.Fatalf("Failed to create chunker: %v", err)
	}

	embedder, err := raggo.NewEmbedder(
		raggo.SetProvider("openai"),
		raggo.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.SetModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	embeddingService := raggo.NewEmbeddingService(embedder)

	vectorDB, err := raggo.NewVectorDB(
		raggo.SetVectorDBType("memory"),
		raggo.SetVectorDBDimension(1536),
	)
	if err != nil {
		log.Fatalf("Failed to create vector database: %v", err)
	}
	defer vectorDB.Close()

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	// Construct the path to the testdata directory
	resumeDir := filepath.Join(wd, "testdata")
	log.Printf("Processing resumes from directory: %s", resumeDir)

	// Process resumes
	err = processResumes(llm, parser, chunker, embeddingService, vectorDB, resumeDir)
	if err != nil {
		log.Fatalf("Failed to process resumes: %v", err)
	}

	// Job offer
	jobOffer := "We are looking for a software engineer with 5+ years of experience in Go, distributed systems, and cloud technologies. The ideal candidate should have strong problem-solving skills and experience with Docker and Kubernetes."

	// Perform normal vector search
	fmt.Println("\nPerforming normal vector search:")
	candidates, err := searchCandidates(context.Background(), llm, embeddingService, vectorDB, jobOffer)
	if err != nil {
		log.Fatalf("Failed to search candidates: %v", err)
	}

	printCandidates(candidates)

	// Perform hybrid search
	fmt.Println("\nPerforming hybrid search:")
	hybridCandidates, err := hybridSearchCandidates(context.Background(), llm, embeddingService, vectorDB, jobOffer)
	if err != nil {
		log.Fatalf("Failed to perform hybrid search for candidates: %v", err)
	}

	printCandidates(hybridCandidates)
}

func processResumes(llm goal.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDB raggo.VectorDB, resumeDir string) error {
	files, err := filepath.Glob(filepath.Join(resumeDir, "*.pdf"))
	if err != nil {
		return fmt.Errorf("failed to list resume files: %w", err)
	}
	if len(files) == 0 {
		return fmt.Errorf("no PDF files found in directory: %s", resumeDir)
	}
	log.Printf("Found %d resume files to process", len(files))

	cacheDir := filepath.Join(resumeDir, ".cache")
	err = os.MkdirAll(cacheDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 5) // Limit concurrency to 5 goroutines

	for _, file := range files {
		wg.Add(1)
		go func(file string) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore

			err := processResume(file, llm, parser, chunker, embeddingService, vectorDB, cacheDir)
			if err != nil {
				log.Printf("Error processing resume %s: %v", file, err)
			}
		}(file)
	}

	wg.Wait()

	// Count total items in the collection
	if counter, ok := vectorDB.(interface{ Count(string) (int, error) }); ok {
		count, err := counter.Count("resumes")
		if err != nil {
			log.Printf("Failed to get collection item count: %v", err)
		} else {
			log.Printf("Total items in 'resumes' collection: %d", count)
		}
	}

	return nil
}

func processResume(file string, llm goal.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDB raggo.VectorDB, cacheDir string) error {
	log.Printf("Processing resume: %s", file)

	cacheFile := filepath.Join(cacheDir, filepath.Base(file)+".cache")
	var cache ResumeCache

	cacheData, err := os.ReadFile(cacheFile)
	if err == nil {
		err = json.Unmarshal(cacheData, &cache)
		if err != nil {
			log.Printf("Failed to unmarshal cache data: %v", err)
		}
	}

	var standardizedSummary string
	var resumeInfo *ResumeInfo

	if cache.Summary != "" && cache.Info.Name != "" {
		log.Printf("Using cached summary and info for resume: %s", file)
		standardizedSummary = cache.Summary
		resumeInfo = &cache.Info
	} else {
		// Parse the resume and create summary if not in cache
		doc, err := parser.Parse(file)
		if err != nil {
			return fmt.Errorf("failed to parse resume: %w", err)
		}

		standardizedSummary, err = createStandardizedSummary(llm, doc.Content)
		if err != nil {
			return fmt.Errorf("failed to create standardized summary: %w", err)
		}

		resumeInfo, err = extractStructuredData(llm, standardizedSummary)
		if err != nil {
			return fmt.Errorf("failed to extract structured data: %w", err)
		}

		// Update cache
		cache.Summary = standardizedSummary
		cache.Info = *resumeInfo
		cache.LastProcessed = time.Now()

		cacheData, err = json.Marshal(cache)
		if err != nil {
			log.Printf("Failed to marshal cache data: %v", err)
		} else {
			err = os.WriteFile(cacheFile, cacheData, 0644)
			if err != nil {
				log.Printf("Failed to write cache file: %v", err)
			}
		}
	}

	// Always generate and save embeddings
	fullContent := combineResumeInfo(resumeInfo, standardizedSummary)
	chunks := chunker.Chunk(fullContent)
	embeddedChunks, err := embeddingService.EmbedChunks(context.Background(), chunks)
	if err != nil {
		return fmt.Errorf("failed to embed chunks: %w", err)
	}

	// Add structured data to each chunk's metadata
	for i := range embeddedChunks {
		embeddedChunks[i].Metadata["name"] = resumeInfo.Name
		embeddedChunks[i].Metadata["skills"] = strings.Join(resumeInfo.Skills, ", ")
		embeddedChunks[i].Metadata["professional_summary"] = resumeInfo.ProfessionalSummary
		embeddedChunks[i].Metadata["work_experience"] = resumeInfo.WorkExperience
	}

	log.Printf("Embedded chunks structure for %s:", file)
	for i, chunk := range embeddedChunks {
		log.Printf("  Chunk %d:", i)
		log.Printf("    Text: %s", truncateString(chunk.Text, 50))
		log.Printf("    Metadata: %+v", chunk.Metadata)
		log.Printf("    Embedding Fields: %v", reflect.ValueOf(chunk.Embeddings).MapKeys())
	}

	log.Printf("Saving embeddings for resume: %s", file)
	err = vectorDB.SaveEmbeddings(context.Background(), "resumes", embeddedChunks)
	if err != nil {
		return fmt.Errorf("failed to save embeddings: %w", err)
	}
	log.Printf("Embeddings saved successfully for resume: %s", file)

	// Add debug logging for database schema
	if debugDB, ok := vectorDB.(interface {
		DescribeCollection(string) (map[string]string, error)
	}); ok {
		if schema, err := debugDB.DescribeCollection("resumes"); err == nil {
			log.Printf("Collection 'resumes' schema after saving %s:", file)
			for field, fieldType := range schema {
				log.Printf("  %s: %s", field, fieldType)
			}
		} else {
			log.Printf("Failed to describe collection: %v", err)
		}
	} else {
		log.Printf("VectorDB does not support schema description")
	}

	return nil
}

func deduplicateResults(results []raggo.SearchResult) []raggo.SearchResult {
	seen := make(map[string]bool)
	deduplicated := []raggo.SearchResult{}

	for _, result := range results {
		name, ok := result.Fields["name"].(string)
		if !ok {
			continue
		}
		if !seen[name] {
			seen[name] = true
			deduplicated = append(deduplicated, result)
		}
	}

	return deduplicated
}

func createStandardizedSummary(llm goal.LLM, content string) (string, error) {
	summarizePrompt := goal.NewPrompt(
		"Summarize the given resume in the following standardized format:\n" +
			"Personal Information: [Name, contact details]\n" +
			"Professional Summary: [2-3 sentences summarizing key qualifications and career objectives]\n" +
			"Skills: [List of key skills, separated by commas]\n" +
			"Work Experience: [For each position: Job Title, Company, Dates, 1-2 key responsibilities or achievements]\n" +
			"Education: [Degree, Institution, Year]\n" +
			"Additional Information: [Any other relevant details like certifications, languages, etc.]\n" +
			"Ensure that the summary is concise and captures the most important information from the resume.\n\n" +
			"Resume Content:\n" + content,
	)

	standardizedSummary, err := llm.Generate(context.Background(), summarizePrompt)
	if err != nil {
		return "", fmt.Errorf("failed to create standardized summary: %w", err)
	}

	return standardizedSummary, nil
}

func extractStructuredData(llm goal.LLM, standardizedSummary string) (*ResumeInfo, error) {
	extractPrompt := goal.NewPrompt(fmt.Sprintf(`
Extract the following information from the given resume summary:
- Name
- Contact Details
- Professional Summary
- Skills (as a list)
- Work Experience
- Education
- Additional Information

Resume Summary:
%s

Respond with a JSON object containing these fields.
`, standardizedSummary))

	resumeInfo, err := goal.ExtractStructuredData[ResumeInfo](context.Background(), llm, extractPrompt.String())
	if err != nil {
		return nil, fmt.Errorf("failed to extract structured data: %w", err)
	}

	return resumeInfo, nil
}

func combineResumeInfo(resumeInfo *ResumeInfo, standardizedSummary string) string {
	return fmt.Sprintf(`Name: %s
Contact Details: %s
Professional Summary: %s
Skills: %s
Work Experience: %s
Education: %s
Additional Information: %s

Detailed Summary:
%s`,
		resumeInfo.Name,
		resumeInfo.ContactDetails,
		resumeInfo.ProfessionalSummary,
		strings.Join(resumeInfo.Skills, ", "),
		resumeInfo.WorkExperience,
		resumeInfo.Education,
		resumeInfo.AdditionalInfo,
		standardizedSummary)
}

func getCollectionItemCount(vectorDB raggo.VectorDB, collectionName string) (int, error) {
	// This is a placeholder. You'll need to implement this based on your VectorDB interface
	// It should return the number of items in the specified collection
	return 0, fmt.Errorf("getCollectionItemCount not implemented")
}

func searchCandidates(ctx context.Context, llm goal.LLM, embeddingService *raggo.EmbeddingService, vectorDB raggo.VectorDB, jobOffer string) ([]raggo.SearchResult, error) {
	// Summarize the job offer
	jobSummary, err := goal.Summarize(ctx, llm, jobOffer)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize job offer: %w", err)
	}

	// Create a single chunk for the job summary
	jobChunk := raggo.Chunk{
		Text:          jobSummary,
		TokenSize:     len(strings.Split(jobSummary, " ")), // Simple word count as a proxy for token size
		StartSentence: 0,
		EndSentence:   1,
	}

	// Embed the job summary
	jobEmbeddings, err := embeddingService.EmbedChunks(ctx, []raggo.Chunk{jobChunk})
	if err != nil {
		return nil, fmt.Errorf("failed to embed job summary: %w", err)
	}

	if len(jobEmbeddings) == 0 {
		return nil, fmt.Errorf("no embeddings generated for job summary")
	}

	// Search for matching candidates
	results, err := vectorDB.Search(ctx, "resumes", jobEmbeddings[0].Embeddings["default"], 5, raggo.NewDefaultSearchParam())
	if err != nil {
		return nil, fmt.Errorf("failed to search for candidates: %w", err)
	}

	return results, nil
}

func hybridSearchCandidates(ctx context.Context, llm goal.LLM, embeddingService *raggo.EmbeddingService, vectorDB raggo.VectorDB, jobOffer string) ([]raggo.SearchResult, error) {
	// Summarize the job offer
	jobSummary, err := goal.Summarize(ctx, llm, jobOffer)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize job offer: %w", err)
	}

	// Create a single chunk for the job summary
	jobChunk := raggo.Chunk{
		Text:          jobSummary,
		TokenSize:     len(strings.Split(jobSummary, " ")), // Simple word count as a proxy for token size
		StartSentence: 0,
		EndSentence:   1,
	}

	// Embed the job summary
	jobEmbeddings, err := embeddingService.EmbedChunks(ctx, []raggo.Chunk{jobChunk})
	if err != nil {
		return nil, fmt.Errorf("failed to embed job summary: %w", err)
	}

	if len(jobEmbeddings) == 0 {
		return nil, fmt.Errorf("no embeddings generated for job summary")
	}

	// Prepare queries for hybrid search
	queries := make(map[string][]float64)
	for field, embedding := range jobEmbeddings[0].Embeddings {
		queries[field] = embedding
	}

	// Define the fields we want to retrieve
	fields := []string{"name", "skills", "professional_summary", "work_experience"}

	// Perform hybrid search with an increased limit
	results, err := vectorDB.HybridSearch(ctx, "resumes", queries, fields, 10, raggo.NewDefaultSearchParam()) // Increased limit to 10
	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid search for candidates: %w", err)
	}

	// Deduplicate results
	dedupedResults := deduplicateResults(results)

	// Limit to top 5 after deduplication
	if len(dedupedResults) > 5 {
		dedupedResults = dedupedResults[:5]
	}

	return dedupedResults, nil
}

func printCandidates(candidates []raggo.SearchResult) {
	fmt.Println("Top candidates for the job offer:")
	for i, candidate := range candidates {
		fmt.Printf("%d. Score: %.4f\n", i+1, candidate.Score)

		if name, ok := candidate.Fields["name"].(string); ok {
			fmt.Printf("   Name: %s\n", name)
		}
		if skills, ok := candidate.Fields["skills"].([]string); ok {
			fmt.Printf("   Skills: %s\n", strings.Join(skills, ", "))
		}
		if summary, ok := candidate.Fields["professional_summary"].(string); ok {
			fmt.Printf("   Professional Summary: %s\n", truncateString(summary, 100))
		}
		if experience, ok := candidate.Fields["work_experience"].(string); ok {
			fmt.Printf("   Work Experience: %s\n", truncateString(experience, 100))
		}

		fmt.Println()
	}
}

func truncateString(s string, maxLength int) string {
	if len(s) <= maxLength {
		return s
	}
	return s[:maxLength] + "..."
}

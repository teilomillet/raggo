package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
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

	// Initialize InMemory VectorDBManager
	inMemoryManager, err := raggo.NewVectorDBManager("memory",
		raggo.WithDimension(1536),
		raggo.WithTopK(5),
	)
	if err != nil {
		log.Fatalf("Failed to create InMemory VectorDBManager: %v", err)
	}
	defer inMemoryManager.Close()

	// Initialize Milvus VectorDBManager
	milvusManager, err := raggo.NewVectorDBManager("milvus",
		raggo.WithAddress("localhost:19530"),
		raggo.WithDimension(1536),
		raggo.WithTopK(5),
	)
	if err != nil {
		log.Fatalf("Failed to create Milvus VectorDBManager: %v", err)
	}
	defer milvusManager.Close()

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	// Construct the path to the testdata directory
	resumeDir := filepath.Join(wd, "testdata")
	log.Printf("Processing resumes from directory: %s", resumeDir)

	// Process resumes for both databases
	err = processResumes(llm, parser, chunker, embeddingService, inMemoryManager, resumeDir)
	if err != nil {
		log.Fatalf("Failed to process resumes for InMemory DB: %v", err)
	}

	err = processResumes(llm, parser, chunker, embeddingService, milvusManager, resumeDir)
	if err != nil {
		log.Fatalf("Failed to process resumes for Milvus DB: %v", err)
	}

	// Job offer
	jobOffer := "We are looking for a software engineer with 5+ years of experience in Go, distributed systems, and cloud technologies. The ideal candidate should have strong problem-solving skills and experience with Docker and Kubernetes."

	// Perform searches for both databases
	fmt.Println("\nPerforming searches on InMemory DB:")
	performSearches(llm, embeddingService, inMemoryManager, jobOffer)

	fmt.Println("\nPerforming searches on Milvus DB:")
	performSearches(llm, embeddingService, milvusManager, jobOffer)
}

func performSearches(llm goal.LLM, embeddingService *raggo.EmbeddingService, vectorDBManager *raggo.VectorDBManager, jobOffer string) {
	// Perform normal vector search
	fmt.Println("Normal vector search:")
	candidates, err := searchCandidates(context.Background(), llm, embeddingService, vectorDBManager, jobOffer)
	if err != nil {
		log.Printf("Failed to search candidates: %v", err)
	} else {
		printCandidates(candidates)
	}

	// Perform hybrid search
	fmt.Println("\nHybrid search:")
	hybridCandidates, err := hybridSearchCandidates(context.Background(), llm, embeddingService, vectorDBManager, jobOffer)
	if err != nil {
		log.Printf("Failed to perform hybrid search for candidates: %v", err)
	} else {
		printCandidates(hybridCandidates)
	}
}

func processResumes(llm goal.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDBManager *raggo.VectorDBManager, resumeDir string) error {
	// Ensure the 'resumes' collection exists
	err := vectorDBManager.EnsureCollection("resumes")
	if err != nil {
		return fmt.Errorf("failed to ensure 'resumes' collection: %w", err)
	}

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
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			err := processResume(file, llm, parser, chunker, embeddingService, vectorDBManager, cacheDir)
			if err != nil {
				log.Printf("Error processing resume %s: %v", file, err)
			}
		}(file)
	}

	wg.Wait()

	return nil
}

func processResume(file string, llm goal.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDBManager *raggo.VectorDBManager, cacheDir string) error {
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

	// Convert embeddings to float32 and prepare metadata
	vectors := make([][]float32, len(embeddedChunks))
	metadata := make([]map[string]interface{}, len(embeddedChunks))
	for i, chunk := range embeddedChunks {
		vectors[i] = convertToFloat32(chunk.Embeddings["default"])
		metadata[i] = chunk.Metadata
	}

	log.Printf("Saving embeddings for resume: %s", file)
	err = vectorDBManager.InsertVectors("resumes", vectors, metadata)
	if err != nil {
		return fmt.Errorf("failed to save embeddings: %w", err)
	}
	log.Printf("Embeddings saved successfully for resume: %s", file)

	return nil
}

func deduplicateResults(results []raggo.SearchResult) []raggo.SearchResult {
	seen := make(map[string]bool)
	deduplicated := []raggo.SearchResult{}

	for _, result := range results {
		name, ok := result.Metadata["name"].(string)
		if !ok {
			// If we can't get the name, just include the result
			deduplicated = append(deduplicated, result)
			continue
		}
		if !seen[name] {
			seen[name] = true
			deduplicated = append(deduplicated, result)
		}
	}

	fmt.Printf("Deduplication: input %d, output %d\n", len(results), len(deduplicated))
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

func searchCandidates(ctx context.Context, llm goal.LLM, embeddingService *raggo.EmbeddingService, vectorDBManager *raggo.VectorDBManager, jobOffer string) ([]raggo.SearchResult, error) {

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
	results, err := vectorDBManager.Search("resumes", convertToFloat32(jobEmbeddings[0].Embeddings["default"]))
	if err != nil {
		return nil, fmt.Errorf("failed to search for candidates: %w", err)
	}

	return results, nil
}

func hybridSearchCandidates(ctx context.Context, llm goal.LLM, embeddingService *raggo.EmbeddingService, vectorDBManager *raggo.VectorDBManager, jobOffer string) ([]raggo.SearchResult, error) {
	// Summarize the job offer
	jobSummary, err := goal.Summarize(ctx, llm, jobOffer)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize job offer: %w", err)
	}
	fmt.Printf("Job Summary: %s\n", jobSummary)

	// Create a single chunk for the job summary
	jobChunk := raggo.Chunk{
		Text:          jobSummary,
		TokenSize:     len(strings.Split(jobSummary, " ")),
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
	fmt.Printf("Number of job embeddings: %d\n", len(jobEmbeddings))

	// Convert query vectors to float32
	queryVectors := make([][]float32, 0)
	for _, embedding := range jobEmbeddings[0].Embeddings {
		queryVectors = append(queryVectors, convertToFloat32(embedding))
	}
	fmt.Printf("Number of query vectors: %d\n", len(queryVectors))

	// Ensure we have at least two query vectors for hybrid search
	if len(queryVectors) < 2 {
		queryVectors = append(queryVectors, queryVectors[0])
	}

	results, err := vectorDBManager.HybridSearch("resumes", queryVectors)
	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid search for candidates: %w", err)
	}
	fmt.Printf("Number of hybrid search results: %d\n", len(results))
	for i, result := range results {
		fmt.Printf("Result %d: ID=%d, Score=%f, Name=%v\n", i, result.ID, result.Score, result.Metadata["name"])
	}

	// Deduplicate results
	dedupedResults := deduplicateResults(results)
	fmt.Printf("Number of deduplicated results: %d\n", len(dedupedResults))

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

		if name, ok := candidate.Metadata["name"].(string); ok {
			fmt.Printf("   Name: %s\n", name)
		}
		if skills, ok := candidate.Metadata["skills"].(string); ok {
			fmt.Printf("   Skills: %s\n", skills)
		}
		if summary, ok := candidate.Metadata["professional_summary"].(string); ok {
			fmt.Printf("   Professional Summary: %s\n", truncateString(summary, 100))
		}
		if experience, ok := candidate.Metadata["work_experience"].(string); ok {
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

// Add helper function for float64 to float32 conversion
func convertToFloat32(input []float64) []float32 {
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(v)
	}
	return output
}

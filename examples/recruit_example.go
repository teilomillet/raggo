package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo"
)

var debug = raggo.Debug
var info = raggo.Info
var warn = raggo.Warn
var errorLog = raggo.Error

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
	fmt.Println("Starting recruit_example.go")

	// Set log level to Debug for more detailed output
	raggo.SetLogLevel(raggo.LogLevelDebug)
	info("Log level set")

	// Initialize the LLM
	debug("Initializing LLM...")
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(2048),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}
	info("LLM initialized")

	// Initialize raggo components
	debug("Initializing parser...")
	parser := raggo.NewParser()
	debug("Parser initialized")

	debug("Initializing chunker...")
	chunker, err := raggo.NewChunker(
		raggo.ChunkSize(512),
		raggo.ChunkOverlap(64),
	)
	if err != nil {
		log.Fatalf("Failed to create chunker: %v", err)
	}
	debug("Chunker initialized")

	debug("Initializing embedder...")
	embedder, err := raggo.NewEmbedder(
		raggo.WithEmbedderProvider("openai"),
		raggo.WithEmbedderAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.WithEmbedderModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}
	debug("Embedder initialized")

	embeddingService := raggo.NewEmbeddingService(embedder)
	debug("Embedding service created")

	debug("Creating VectorDB...")
	vectorDB, err := raggo.NewVectorDB(
		raggo.WithType("milvus"),
		raggo.WithAddress("localhost:19530"),
		raggo.WithMaxPoolSize(10),
		raggo.WithTimeout(30*time.Second),
	)
	if err != nil {
		log.Fatalf("Failed to create vector database: %v", err)
	}
	debug("VectorDB created")

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	debug("Connecting to VectorDB...")
	err = vectorDB.Connect(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Milvus database: %v", err)
	}
	debug("Connected to VectorDB")

	// Define the schema and create the collection
	collectionName := "resumes"
	schema := raggo.Schema{
		Name: collectionName,
		Fields: []raggo.Field{
			{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
			{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
			{Name: "name", DataType: "varchar", MaxLength: 255},
			{Name: "skills", DataType: "varchar", MaxLength: 1000},
			{Name: "professional_summary", DataType: "varchar", MaxLength: 2000},
			{Name: "work_experience", DataType: "varchar", MaxLength: 5000},
		},
	}

	exists, err := vectorDB.HasCollection(context.Background(), collectionName)
	if err != nil {
		log.Fatalf("Failed to check if collection exists: %v", err)
	}
	if exists {
		debug("Dropping existing collection: %s", collectionName)
		err = vectorDB.DropCollection(context.Background(), collectionName)
		if err != nil {
			log.Fatalf("Failed to drop existing collection: %v", err)
		}
	}
	debug("Creating collection with schema:")
	for _, field := range schema.Fields {
		debug("Field: %s, Type: %s, PrimaryKey: %v, AutoID: %v, Dimension: %d, MaxLength: %d",
			field.Name, field.DataType, field.PrimaryKey, field.AutoID, field.Dimension, field.MaxLength)
	}
	err = vectorDB.CreateCollection(context.Background(), collectionName, schema)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	debug("Collection created successfully")

	// Add this block to create the index
	debug("Creating index on Embedding field...")
	err = vectorDB.CreateIndex(context.Background(), collectionName, "Embedding", raggo.Index{
		Type:   "HNSW",
		Metric: "L2",
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 64,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create index: %v", err)
	}
	debug("Index created successfully")

	debug("Loading collection: %s", collectionName)
	err = vectorDB.LoadCollection(context.Background(), collectionName)
	if err != nil {
		log.Fatalf("Failed to load collection: %v", err)
	}
	debug("Collection loaded successfully")

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	// Construct the path to the testdata directory
	resumeDir := filepath.Join(wd, "testdata")
	log.Printf("Processing resumes from directory: %s", resumeDir)

	// Process resumes
	debug("Starting resume processing...")
	err = processResumes(llm, parser, chunker, embeddingService, vectorDB, resumeDir)
	if err != nil {
		log.Fatalf("Failed to process resumes: %v", err)
	}
	debug("Resume processing completed.")

	debug("\nPreparing job offer for search...")

	// Job offer
	jobOffer := "We are looking for a software engineer with 5+ years of experience in Go, distributed systems, and cloud technologies. The ideal candidate should have strong problem-solving skills and experience with Docker and Kubernetes."

	vectorDB.SetColumnNames([]string{"name", "skills", "professional_summary", "work_experience"})

	// Perform normal vector search
	info("\nPerforming normal vector search:")
	candidates, err := searchCandidates(context.Background(), llm, embeddingService, vectorDB, jobOffer)
	if err != nil {
		log.Fatalf("Failed to search candidates: %v", err)
	}

	printCandidates(candidates)

	// Perform hybrid search
	info("\nStarting hybrid search process...")
	hybridCandidates, err := hybridSearchCandidates(context.Background(), llm, embeddingService, vectorDB, jobOffer)
	if err != nil {
		log.Fatalf("Failed to perform hybrid search for candidates: %v", err)
	}
	fmt.Printf("Hybrid search completed. Found %d candidates.\n", len(hybridCandidates))

	printCandidates(hybridCandidates)
}

func processResumes(llm gollm.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDB *raggo.VectorDB, resumeDir string) error {
	fmt.Printf("Scanning directory: %s\n", resumeDir)
	files, err := filepath.Glob(filepath.Join(resumeDir, "*.pdf"))
	if err != nil {
		return fmt.Errorf("failed to list resume files: %w", err)
	}
	if len(files) == 0 {
		return fmt.Errorf("no PDF files found in directory: %s", resumeDir)
	}
	fmt.Printf("Found %d resume files to process\n", len(files))

	cacheDir := filepath.Join(resumeDir, ".cache")
	err = os.MkdirAll(cacheDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	var wg sync.WaitGroup
	semaphore := make(chan struct{}, 5) // Limit concurrency to 5 goroutines

	fmt.Println("Starting to process resumes...")
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

	return nil
}

func processResume(file string, llm gollm.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDB *raggo.VectorDB, cacheDir string) error {
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

	records := make([]raggo.Record, len(embeddedChunks))
	for i, chunk := range embeddedChunks {
		records[i] = raggo.Record{
			Fields: map[string]interface{}{
				"Embedding":            chunk.Embeddings["default"],
				"name":                 truncateString(resumeInfo.Name, 255),
				"skills":               truncateString(strings.Join(resumeInfo.Skills, ", "), 1000),
				"professional_summary": truncateString(resumeInfo.ProfessionalSummary, 2000),
				"work_experience":      truncateString(resumeInfo.WorkExperience, 5000),
			},
		}
	}

	debug("Inserting %d records for resume: %s", len(records), file)
	err = vectorDB.Insert(context.Background(), "resumes", records)
	if err != nil {
		return fmt.Errorf("failed to insert records: %w", err)
	}
	debug("Successfully inserted %d records for resume: %s", len(records), file)

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

func createStandardizedSummary(llm gollm.LLM, content string) (string, error) {
	summarizePrompt := gollm.NewPrompt(
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

func extractStructuredData(llm gollm.LLM, standardizedSummary string) (*ResumeInfo, error) {
	extractPrompt := gollm.NewPrompt(fmt.Sprintf(`
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

	resumeInfo := &ResumeInfo{}
	_, err := llm.GenerateWithSchema(context.Background(), extractPrompt, resumeInfo)
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

func searchCandidates(ctx context.Context, llm gollm.LLM, embeddingService *raggo.EmbeddingService, vectorDB *raggo.VectorDB, jobOffer string) ([]raggo.SearchResult, error) {

	fmt.Println("Starting candidate search...")

	// Summarize the job offer
	jobSummary, err := llm.Generate(ctx, llm.NewPrompt(jobOffer))
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

	debug("Performing search on collection: %s", "resumes")
	debug("Search vector length: %d", len(jobEmbeddings[0].Embeddings["default"]))

	// Search for matching candidates
	debug("Performing search on collection: %s with vector of length %d", "resumes", len(jobEmbeddings[0].Embeddings["default"]))
	results, err := vectorDB.Search(ctx, "resumes", map[string]raggo.Vector{"Embedding": jobEmbeddings[0].Embeddings["default"]}, 5, "L2", map[string]interface{}{
		"type": "HNSW",
		"ef":   100,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to search for candidates: %w", err)
	}
	debug("Search completed. Found %d results", len(results))

	return results, nil
}

func hybridSearchCandidates(ctx context.Context, llm gollm.LLM, embeddingService *raggo.EmbeddingService, vectorDB *raggo.VectorDB, jobOffer string) ([]raggo.SearchResult, error) {
	fmt.Println("Starting optimized hybrid search for candidates...")

	// Extract different aspects of the job offer
	jobAspects, err := extractJobAspects(ctx, llm, jobOffer)
	if err != nil {
		return nil, fmt.Errorf("failed to extract job aspects: %w", err)
	}

	// Create chunks for each job aspect
	var jobChunks []raggo.Chunk
	for _, text := range jobAspects {
		jobChunks = append(jobChunks, raggo.Chunk{
			Text:          text,
			TokenSize:     len(strings.Split(text, " ")),
			StartSentence: 0,
			EndSentence:   1,
		})
	}

	// Embed all job aspects
	jobEmbeddings, err := embeddingService.EmbedChunks(ctx, jobChunks)
	if err != nil {
		return nil, fmt.Errorf("failed to embed job aspects: %w", err)
	}

	if len(jobEmbeddings) == 0 {
		return nil, fmt.Errorf("no embeddings generated for job aspects")
	}
	fmt.Printf("Generated %d embeddings for job aspects\n", len(jobEmbeddings))

	// Combine all aspect embeddings into a single vector
	combinedVector := make(raggo.Vector, len(jobEmbeddings[0].Embeddings["default"]))
	for _, embedding := range jobEmbeddings {
		for i, v := range embedding.Embeddings["default"] {
			combinedVector[i] += v
		}
	}
	// Normalize the combined vector
	magnitude := 0.0
	for _, v := range combinedVector {
		magnitude += v * v
	}
	magnitude = math.Sqrt(magnitude)
	for i := range combinedVector {
		combinedVector[i] /= magnitude
	}

	// Prepare search vector
	searchVectors := map[string]raggo.Vector{
		"Embedding": combinedVector,
	}

	// Perform hybrid search
	fmt.Println("Performing optimized hybrid search...")
	results, err := vectorDB.HybridSearch(ctx, "resumes", searchVectors, 10, "L2", map[string]interface{}{
		"type": "HNSW",
		"ef":   100,
	}, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid search for candidates: %w", err)
	}
	// Deduplicate results
	dedupedResults := deduplicateResults(results)
	fmt.Printf("After deduplication: %d results\n", len(dedupedResults))

	// Limit to top 5 after deduplication
	if len(dedupedResults) > 5 {
		dedupedResults = dedupedResults[:5]
		fmt.Println("Limited to top 5 results")
	}

	return dedupedResults, nil
}

func extractJobAspects(ctx context.Context, llm gollm.LLM, jobOffer string) (map[string]string, error) {
	prompt := gollm.NewPrompt(`
	Extract and summarize the following aspects from the job offer:
	1. Required Skills
	2. Experience Level
	3. Job Responsibilities
	4. Company Culture

	For each aspect, provide a brief summary (1-2 sentences).
	Respond with a JSON object where the keys are the aspect names and the values are the summaries.

	Job Offer:
	` + jobOffer)

	aspectsJSON, err := llm.Generate(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to generate job aspects: %w", err)
	}

	var aspects map[string]string
	err = json.Unmarshal([]byte(aspectsJSON), &aspects)
	if err != nil {
		return nil, fmt.Errorf("failed to parse job aspects: %w", err)
	}

	return aspects, nil
}

func printCandidates(candidates []raggo.SearchResult) {
	fmt.Println("Top candidates for the job offer:")
	for i, candidate := range candidates {
		fmt.Printf("%d. Score: %.4f\n", i+1, candidate.Score)
		if name, ok := candidate.Fields["name"].(string); ok {
			fmt.Printf("   Name: %s\n", name)
		}
		if skills, ok := candidate.Fields["skills"].(string); ok {
			fmt.Printf("   Skills: %s\n", skills)
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
	return s[:maxLength]
}

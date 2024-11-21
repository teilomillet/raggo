package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo"
	"golang.org/x/time/rate"
)

const (
	EXPECTED_FILE_COUNT = 5000
	GPT_RPM_LIMIT       = 5000    // Requests per minute for GPT models
	GPT_TPM_LIMIT       = 4000000 // Tokens per minute for gpt-4o-mini
	EMBED_RPM_LIMIT     = 5000    // Requests per minute for embedding
	EMBED_TPM_LIMIT     = 5000000 // Tokens per minute for text-embedding-3-small
	MAX_CONCURRENT      = 10      // Maximum number of concurrent goroutines
)

type RateLimiter struct {
	requestLimiter *rate.Limiter
	tokenLimiter   *rate.Limiter
	mu             sync.Mutex
}

func NewRateLimiter(rpm, tpm int) *RateLimiter {
	return &RateLimiter{
		requestLimiter: rate.NewLimiter(rate.Limit(rpm/60), rpm),
		tokenLimiter:   rate.NewLimiter(rate.Limit(tpm/60), tpm),
	}
}

func (rl *RateLimiter) Wait(ctx context.Context, tokens int) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if err := rl.requestLimiter.Wait(ctx); err != nil {
		return err
	}
	return rl.tokenLimiter.WaitN(ctx, tokens)
}

func (rl *RateLimiter) UpdateLimits(remainingRequests, remainingTokens int, resetRequests, resetTokens time.Duration) {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	rl.requestLimiter.SetBurst(remainingRequests)
	rl.tokenLimiter.SetBurst(remainingTokens)

	rl.requestLimiter.SetLimit(rate.Limit(float64(remainingRequests) / resetRequests.Seconds()))
	rl.tokenLimiter.SetLimit(rate.Limit(float64(remainingTokens) / resetTokens.Seconds()))
}

var (
	gptLimiter   *RateLimiter
	embedLimiter *RateLimiter
)

func init() {
	gptLimiter = NewRateLimiter(GPT_RPM_LIMIT, GPT_TPM_LIMIT)
	embedLimiter = NewRateLimiter(EMBED_RPM_LIMIT, EMBED_TPM_LIMIT)
}

func main() {
	raggo.SetLogLevel(raggo.LogLevelDebug)

	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	targetDir := filepath.Join(wd, "benchmark_data")

	parser := raggo.NewParser()
	chunker, err := raggo.NewChunker(
		raggo.ChunkSize(512),
		raggo.ChunkOverlap(64),
	)
	if err != nil {
		log.Fatalf("Failed to create chunker: %v", err)
	}

	embedder, err := raggo.NewEmbedder(
		raggo.WithEmbedderProvider("openai"),
		raggo.WithEmbedderAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.WithEmbedderModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(2048),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	milvusDB, err := raggo.NewVectorDB(
		raggo.WithType("milvus"),
		raggo.WithAddress("localhost:19530"),
		raggo.WithMaxPoolSize(10),
		raggo.WithTimeout(30*time.Second),
	)
	if err != nil {
		log.Fatalf("Failed to create Milvus vector database: %v", err)
	}
	defer milvusDB.Close()

	err = milvusDB.Connect(context.Background())
	if err != nil {
		log.Fatalf("Failed to connect to Milvus database: %v", err)
	}

	collectionName := "pdf_embeddings"
	createMilvusCollection(milvusDB, collectionName)

	benchmarkPDFProcessing(parser, chunker, embedder, llm, milvusDB, collectionName, targetDir)
}

func createMilvusCollection(milvusDB *raggo.VectorDB, collectionName string) {
	ctx := context.Background()

	exists, err := milvusDB.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("Failed to check if collection exists: %v", err)
	}

	if exists {
		err = milvusDB.DropCollection(ctx, collectionName)
		if err != nil {
			log.Fatalf("Failed to drop existing collection: %v", err)
		}
		log.Printf("Dropped existing collection %s\n", collectionName)
	}

	schema := raggo.Schema{
		Name: collectionName,
		Fields: []raggo.Field{
			{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
			{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
			{Name: "Text", DataType: "varchar", MaxLength: 65535},
		},
	}

	err = milvusDB.CreateCollection(ctx, collectionName, schema)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	log.Printf("Successfully created collection %s", collectionName)

	index := raggo.Index{
		Type:   "HNSW",
		Metric: "L2",
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 256,
		},
	}
	err = milvusDB.CreateIndex(ctx, collectionName, "Embedding", index)
	if err != nil {
		log.Fatalf("Failed to create index on Embedding: %v", err)
	}

	err = milvusDB.LoadCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("Failed to load collection: %v", err)
	}
}

func benchmarkPDFProcessing(parser raggo.Parser, chunker raggo.Chunker, embedder raggo.Embedder, llm gollm.LLM, milvusDB *raggo.VectorDB, collectionName, targetDir string) {
	files, err := filepath.Glob(filepath.Join(targetDir, "*.pdf"))
	if err != nil {
		log.Fatalf("Failed to list PDF files: %v", err)
	}

	if len(files) != EXPECTED_FILE_COUNT {
		log.Fatalf("Expected %d files, but found %d. Please run the bash script to prepare the correct number of files.", EXPECTED_FILE_COUNT, len(files))
	}

	fmt.Printf("Starting PDF processing benchmark with %d files...\n", len(files))

	start := time.Now()

	var wg sync.WaitGroup
	var mu sync.Mutex
	errorCount := 0
	successCount := 0
	totalTokens := 0
	embedCount := 0
	summaryCount := 0

	sem := make(chan struct{}, MAX_CONCURRENT)

	for _, file := range files {
		sem <- struct{}{}
		wg.Add(1)
		go func(filePath string) {
			defer func() {
				<-sem
				wg.Done()
			}()
			tokens, embeds, summaries, err := processAndEmbedPDF(parser, chunker, embedder, llm, milvusDB, collectionName, filePath)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				log.Printf("Error processing %s: %v", filePath, err)
				errorCount++
			} else {
				successCount++
				totalTokens += tokens
				embedCount += embeds
				summaryCount += summaries
			}
		}(file)
	}

	wg.Wait()
	duration := time.Since(start)

	fmt.Printf("\nBenchmark Results:\n")
	fmt.Printf("Total files: %d\n", len(files))
	fmt.Printf("Files successfully processed: %d\n", successCount)
	fmt.Printf("Files with processing errors: %d\n", errorCount)
	fmt.Printf("Total processing time: %v\n", duration)
	fmt.Printf("Total tokens processed: %d\n", totalTokens)
	fmt.Printf("Total embeddings created: %d\n", embedCount)
	fmt.Printf("Total summaries generated: %d\n", summaryCount)
	if successCount > 0 {
		fmt.Printf("Average time per successfully processed file: %v\n", duration/time.Duration(successCount))
	}
	fmt.Printf("Average tokens per second: %.2f\n", float64(totalTokens)/duration.Seconds())
	fmt.Printf("Average embeddings per second: %.2f\n", float64(embedCount)/duration.Seconds())
	fmt.Printf("Average summaries per second: %.2f\n", float64(summaryCount)/duration.Seconds())
}

func processAndEmbedPDF(parser raggo.Parser, chunker raggo.Chunker, embedder raggo.Embedder, llm gollm.LLM, milvusDB *raggo.VectorDB, collectionName, filePath string) (int, int, int, error) {
	log.Printf("Processing file: %s", filePath)

	doc, err := parser.Parse(filePath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("error parsing PDF: %w", err)
	}

	if len(doc.Content) == 0 {
		return 0, 0, 0, fmt.Errorf("parsed PDF content is empty")
	}

	// Wait for rate limit before generating summary
	if err := gptLimiter.Wait(context.Background(), len(doc.Content)); err != nil {
		return 0, 0, 0, fmt.Errorf("rate limit wait error: %w", err)
	}

	summaryPrompt := gollm.NewPrompt(fmt.Sprintf("Summarize the following text in 2-3 sentences:\n\n%s", doc.Content))
	summary, err := llm.Generate(context.Background(), summaryPrompt)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("error generating summary: %w", err)
	}

	// Update rate limiter based on response headers
	// Note: You'll need to modify your gollm.LLM interface to return these headers
	// gptLimiter.UpdateLimits(remainingRequests, remainingTokens, resetRequests, resetTokens)

	chunks := chunker.Chunk(summary)

	totalTokens := 0
	embedCount := 0

	var records []raggo.Record

	for _, chunk := range chunks {
		chunkTokens := len(chunk.Text) // Simple approximation
		totalTokens += chunkTokens

		if err := embedLimiter.Wait(context.Background(), chunkTokens); err != nil {
			return totalTokens, embedCount, 1, fmt.Errorf("embed rate limit wait error: %w", err)
		}

		embedding, err := embedder.Embed(context.Background(), chunk.Text)
		if err != nil {
			return totalTokens, embedCount, 1, fmt.Errorf("error embedding chunk: %w", err)
		}
		embedCount++

		// Update embed rate limiter based on response headers
		// embedLimiter.UpdateLimits(remainingRequests, remainingTokens, resetRequests, resetTokens)

		records = append(records, raggo.Record{
			Fields: map[string]interface{}{
				"Embedding": embedding,
				"Text":      chunk.Text,
			},
		})
	}

	// Insert records into Milvus
	err = milvusDB.Insert(context.Background(), collectionName, records)
	if err != nil {
		return totalTokens, embedCount, 1, fmt.Errorf("error inserting records into Milvus: %w", err)
	}

	log.Printf("Successfully processed and inserted %s: %d tokens, %d embeddings", filePath, totalTokens, embedCount)

	return totalTokens, embedCount, 1, nil
}

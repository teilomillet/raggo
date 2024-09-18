package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/pkoukk/tiktoken-go"
	"github.com/teilomillet/raggo"
	"golang.org/x/time/rate"
)

const (
	embeddingTPM = 5_000_000 // Tokens per minute for text-embedding-3-small
	embeddingRPM = 5_000     // Requests per minute for text-embedding-3-small
)

type RateLimitedEmbedder struct {
	embedder       raggo.Embedder
	tokenLimiter   *rate.Limiter
	requestLimiter *rate.Limiter
	tokenCounter   *TikTokenCounter
}

func NewRateLimitedEmbedder(embedder raggo.Embedder) (*RateLimitedEmbedder, error) {
	tokenCounter, err := NewTikTokenCounter("cl100k_base") // Use the appropriate encoding for your model
	if err != nil {
		return nil, fmt.Errorf("failed to create TikTokenCounter: %w", err)
	}

	return &RateLimitedEmbedder{
		embedder:       embedder,
		tokenLimiter:   rate.NewLimiter(rate.Limit(embeddingTPM/60), embeddingTPM), // Convert TPM to tokens per second
		requestLimiter: rate.NewLimiter(rate.Limit(embeddingRPM/60), embeddingRPM), // Convert RPM to requests per second
		tokenCounter:   tokenCounter,
	}, nil
}

func (rle *RateLimitedEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	tokens := rle.tokenCounter.Count(text)

	if err := rle.tokenLimiter.WaitN(ctx, tokens); err != nil {
		return nil, fmt.Errorf("rate limit exceeded for tokens: %w", err)
	}

	if err := rle.requestLimiter.Wait(ctx); err != nil {
		return nil, fmt.Errorf("rate limit exceeded for requests: %w", err)
	}

	return rle.embedder.Embed(ctx, text)
}

type TikTokenCounter struct {
	tke *tiktoken.Tiktoken
}

func NewTikTokenCounter(encoding string) (*TikTokenCounter, error) {
	tke, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding: %w", err)
	}
	return &TikTokenCounter{tke: tke}, nil
}

func (ttc *TikTokenCounter) Count(text string) int {
	return len(ttc.tke.Encode(text, nil, nil))
}

func main() {
	// Set log level to Info for more detailed output
	raggo.SetLogLevel(raggo.LogLevelInfo)

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	// Construct the path to the testdata directory
	sourceDir := filepath.Join(wd, "testdata")
	targetDir := filepath.Join(wd, "benchmark_data")

	// Create the target directory if it doesn't exist
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		log.Fatalf("Failed to create target directory: %v", err)
	}

	// Initialize components
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

	rateLimitedEmbedder, err := NewRateLimitedEmbedder(embedder)
	if err != nil {
		log.Fatalf("Failed to create rate-limited embedder: %v", err)
	}

	// Create a new ConcurrentPDFLoader
	loader := raggo.NewConcurrentPDFLoader(
		raggo.SetTimeout(5*time.Minute), // Increased timeout
		raggo.SetTempDir(targetDir),
	)

	// Run the benchmark
	desiredCount := 100 // Adjust this number as needed
	benchmarkPDFProcessing(loader, parser, chunker, rateLimitedEmbedder, sourceDir, targetDir, desiredCount)
}

func benchmarkPDFProcessing(loader raggo.ConcurrentPDFLoader, parser raggo.Parser, chunker raggo.Chunker, embedder *RateLimitedEmbedder, sourceDir, targetDir string, desiredCount int) {
	fmt.Printf("Starting PDF processing benchmark with %d files...\n", desiredCount)

	start := time.Now()

	// Load PDFs
	loadStart := time.Now()
	loadedFiles, err := loader.LoadPDFsConcurrent(context.Background(), sourceDir, targetDir, desiredCount)
	loadDuration := time.Since(loadStart)

	if err != nil {
		log.Printf("Warning: Encountered errors while loading PDF files: %v", err)
	}

	fmt.Printf("Attempted to load %d PDF files, successfully loaded %d in %v\n", desiredCount, len(loadedFiles), loadDuration)

	if len(loadedFiles) == 0 {
		log.Fatalf("No files were successfully loaded. Cannot continue benchmark.")
	}

	// Process and embed PDFs
	var wg sync.WaitGroup
	processStart := time.Now()
	errorCount := 0
	successCount := 0
	totalTokens := 0
	embedCount := 0

	for _, file := range loadedFiles {
		wg.Add(1)
		go func(filePath string) {
			defer wg.Done()
			tokens, embeds, err := processAndEmbedPDF(parser, chunker, embedder, filePath)
			if err != nil {
				log.Printf("Error processing %s: %v", filePath, err)
				errorCount++
			} else {
				successCount++
				totalTokens += tokens
				embedCount += embeds
			}
		}(file)
	}

	wg.Wait()
	processDuration := time.Since(processStart)

	totalDuration := time.Since(start)

	// Print benchmark results
	fmt.Printf("\nBenchmark Results:\n")
	fmt.Printf("Total files attempted: %d\n", desiredCount)
	fmt.Printf("Files successfully loaded: %d\n", len(loadedFiles))
	fmt.Printf("Files successfully processed: %d\n", successCount)
	fmt.Printf("Files with processing errors: %d\n", errorCount)
	fmt.Printf("Loading time: %v\n", loadDuration)
	fmt.Printf("Processing time: %v\n", processDuration)
	fmt.Printf("Total time: %v\n", totalDuration)
	fmt.Printf("Total tokens processed: %d\n", totalTokens)
	fmt.Printf("Total embeddings created: %d\n", embedCount)
	if successCount > 0 {
		fmt.Printf("Average time per successfully processed file: %v\n", totalDuration/time.Duration(successCount))
	}
	fmt.Printf("Average tokens per second: %.2f\n", float64(totalTokens)/processDuration.Seconds())
	fmt.Printf("Average embeddings per second: %.2f\n", float64(embedCount)/processDuration.Seconds())
}

func processAndEmbedPDF(parser raggo.Parser, chunker raggo.Chunker, embedder *RateLimitedEmbedder, filePath string) (int, int, error) {
	// Parse the PDF
	doc, err := parser.Parse(filePath)
	if err != nil {
		return 0, 0, fmt.Errorf("error parsing PDF: %w", err)
	}

	// Chunk the content
	chunks := chunker.Chunk(doc.Content)

	totalTokens := 0
	embedCount := 0

	// Embed each chunk
	for _, chunk := range chunks {
		tokens := embedder.tokenCounter.Count(chunk.Text)
		totalTokens += tokens

		_, err := embedder.Embed(context.Background(), chunk.Text)
		if err != nil {
			return totalTokens, embedCount, fmt.Errorf("error embedding chunk: %w", err)
		}
		embedCount++
	}

	log.Printf("Processed and embedded %s: %d characters, %d chunks, %d tokens", filePath, len(doc.Content), len(chunks), totalTokens)
	return totalTokens, embedCount, nil
}

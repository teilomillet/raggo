package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/teilomillet/goal"
	"github.com/teilomillet/raggo"
)

const EXPECTED_FILE_COUNT = 1000

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
		raggo.SetProvider("openai"),
		raggo.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.SetModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-4o-mini"),
		goal.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		goal.SetMaxTokens(2048),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	benchmarkPDFProcessing(parser, chunker, embedder, llm, targetDir)
}

func benchmarkPDFProcessing(parser raggo.Parser, chunker raggo.Chunker, embedder raggo.Embedder, llm goal.LLM, targetDir string) {
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

	for _, file := range files {
		wg.Add(1)
		go func(filePath string) {
			defer wg.Done()
			tokens, embeds, summaries, err := processAndEmbedPDF(parser, chunker, embedder, llm, filePath)
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

func processAndEmbedPDF(parser raggo.Parser, chunker raggo.Chunker, embedder raggo.Embedder, llm goal.LLM, filePath string) (int, int, int, error) {
	log.Printf("Processing file: %s", filePath)

	doc, err := parser.Parse(filePath)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("error parsing PDF: %w", err)
	}

	if len(doc.Content) == 0 {
		return 0, 0, 0, fmt.Errorf("parsed PDF content is empty")
	}

	log.Printf("Successfully parsed PDF: %s, Content length: %d", filePath, len(doc.Content))

	summaryPrompt := goal.NewPrompt(fmt.Sprintf("Summarize the following text in 2-3 sentences:\n\n%s", doc.Content))
	summary, err := llm.Generate(context.Background(), summaryPrompt)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("error generating summary: %w", err)
	}

	log.Printf("Generated summary for %s, Summary length: %d", filePath, len(summary))

	chunks := chunker.Chunk(summary)

	totalTokens := 0
	embedCount := 0

	for _, chunk := range chunks {
		totalTokens += len(chunk.Text) // Simple approximation

		_, err := embedder.Embed(context.Context(context.Background()), chunk.Text)
		if err != nil {
			return totalTokens, embedCount, 1, fmt.Errorf("error embedding chunk: %w", err)
		}
		embedCount++
	}

	log.Printf("Successfully processed %s: %d tokens, %d embeddings", filePath, totalTokens, embedCount)

	return totalTokens, embedCount, 1, nil
}

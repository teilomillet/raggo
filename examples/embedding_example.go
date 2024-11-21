package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/teilomillet/raggo"
)

func main() {
	// Set log level to Info
	raggo.SetLogLevel(raggo.LogLevelInfo)

	// Create a new Parser
	parser := raggo.NewParser()

	// Create a new Chunker
	chunker, err := raggo.NewChunker(
		raggo.ChunkSize(100),   // Set chunk size to 100 tokens
		raggo.ChunkOverlap(20), // Set chunk overlap to 20 tokens
		raggo.WithSentenceSplitter(raggo.SmartSentenceSplitter()),
		raggo.WithTokenCounter(raggo.NewDefaultTokenCounter()),
	)
	if err != nil {
		log.Fatalf("Failed to create chunker: %v", err)
	}

	// Create a new Embedder
	embedder, err := raggo.NewEmbedder(
		raggo.SetEmbedderProvider("openai"),
		raggo.SetEmbedderAPIKey(os.Getenv("OPENAI_API_KEY")), // Make sure to set this environment variable
		raggo.SetEmbedderModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	// Create an EmbeddingService
	embeddingService := raggo.NewEmbeddingService(embedder)

	// Get the path to a PDF file in the testdata directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	pdfPath := filepath.Join(wd, "testdata", "CV.pdf") // Make sure this file exists

	// Parse the PDF file
	doc, err := parser.Parse(pdfPath)
	if err != nil {
		log.Fatalf("Failed to parse PDF: %v", err)
	}

	fmt.Printf("Parsed document with %d characters\n", len(doc.Content))

	// Chunk the parsed content
	chunks := chunker.Chunk(doc.Content)

	fmt.Printf("Created %d chunks from the document\n", len(chunks))

	// Embed the chunks
	embeddedChunks, err := embeddingService.EmbedChunks(context.Background(), chunks)
	if err != nil {
		log.Fatalf("Failed to embed chunks: %v", err)
	}

	fmt.Printf("Successfully embedded %d chunks\n", len(embeddedChunks))

	// Print details of the first few embedded chunks
	for i, chunk := range embeddedChunks {
		if i >= 3 {
			break // Only print the first 3 chunks
		}
		fmt.Printf("Embedded Chunk %d:\n", i+1)
		fmt.Printf("  Text: %s\n", truncateString(chunk.Text, 50))
		fmt.Printf("  Embedding Vector Length: %d\n", len(chunk.Embeddings["default"]))
		fmt.Printf("  Metadata: %v\n", chunk.Metadata)
		fmt.Println()
	}
}

// truncateString truncates a string to a specified length, adding an ellipsis if truncated
func truncateString(s string, length int) string {
	if len(s) <= length {
		return s
	}
	return s[:length-3] + "..."
}

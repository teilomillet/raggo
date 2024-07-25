package main

import (
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

	// Print details of the first few chunks
	for i, chunk := range chunks {
		if i >= 5 {
			break // Only print the first 5 chunks
		}
		fmt.Printf("Chunk %d:\n", i+1)
		fmt.Printf("  Token Size: %d\n", chunk.TokenSize)
		fmt.Printf("  Start Sentence: %d\n", chunk.StartSentence)
		fmt.Printf("  End Sentence: %d\n", chunk.EndSentence)
		fmt.Printf("  Preview: %s\n", truncateString(chunk.Text, 100))
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

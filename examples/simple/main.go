package main

import (
	"context"
	"fmt"
	"log"
	"path/filepath"

	"github.com/teilomillet/raggo"
)

func main() {
	// Create a new SimpleRAG instance with default configuration
	config := raggo.DefaultConfig()
	config.Collection = "my_documents"
	
	rag, err := raggo.NewSimpleRAG(config)
	if err != nil {
		log.Fatalf("Failed to create SimpleRAG: %v", err)
	}
	defer rag.Close()

	// Get absolute path to docs directory
	docsPath := filepath.Join("examples", "chat", "docs")

	// Add all documents from the directory
	ctx := context.Background()
	err = rag.AddDocuments(ctx, docsPath)
	if err != nil {
		log.Fatalf("Failed to add documents: %v", err)
	}

	// Ask a question about PressureValve
	query := "What is MountainPass's PressureValve system and how did it help during Black Friday?"
	response, err := rag.Search(ctx, query)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}

	// Print response
	fmt.Printf("\nQuestion: %s\n", query)
	fmt.Printf("\nAnswer: %s\n", response)
}

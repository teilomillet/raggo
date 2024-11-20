package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/teilomillet/raggo"
)

func main() {
	// Example 1: Basic Contextual RAG Usage
	// This demonstrates the simplest way to use contextual RAG with default settings
	basicExample()

	// Example 2: Advanced Configuration
	// This shows how to customize the contextual RAG behavior
	advancedExample()
}

func basicExample() {
	fmt.Println("\n=== Basic Contextual RAG Example ===")

	// Initialize RAG with default settings
	// This automatically:
	// - Uses OpenAI's text-embedding-3-small model for embeddings
	// - Sets optimal chunk size and overlap for context preservation
	// - Configures reasonable TopK and similarity thresholds
	rag, err := raggo.NewDefaultContextualRAG("basic_contextual_docs")
	if err != nil {
		fmt.Printf("Failed to initialize RAG: %v\n", err)
		os.Exit(1)
	}
	defer rag.Close()

	// Add documents - the system will automatically:
	// - Split documents into semantic chunks
	// - Generate rich context for each chunk
	// - Store embeddings with contextual information
	docsPath := filepath.Join("examples", "docs")
	if err := rag.AddDocuments(context.Background(), docsPath); err != nil {
		fmt.Printf("Failed to add documents: %v\n", err)
		os.Exit(1)
	}

	// Simple search with automatic context enhancement
	query := "What are the key features of the product?"
	response, err := rag.Search(context.Background(), query)
	if err != nil {
		fmt.Printf("Failed to search: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nQuery: %s\nResponse: %s\n", query, response)
}

func advancedExample() {
	fmt.Println("\n=== Advanced Contextual RAG Example ===")

	// Create a custom configuration
	config := &raggo.ContextualRAGConfig{
		Collection:   "advanced_contextual_docs",
		Model:        "text-embedding-3-small", // Embedding model
		LLMModel:     "gpt-4o-mini",            // Model for context generation
		ChunkSize:    300,                      // Larger chunks for more context
		ChunkOverlap: 75,                       // 25% overlap for better continuity
		TopK:         5,                        // Number of similar chunks to retrieve
		MinScore:     0.7,                      // Higher threshold for better relevance
	}

	// Initialize RAG with custom configuration
	rag, err := raggo.NewContextualRAG(config)
	if err != nil {
		fmt.Printf("Failed to initialize RAG: %v\n", err)
		os.Exit(1)
	}
	defer rag.Close()

	// Add documents with enhanced context
	docsPath := filepath.Join("examples", "docs")
	if err := rag.AddDocuments(context.Background(), docsPath); err != nil {
		fmt.Printf("Failed to add documents: %v\n", err)
		os.Exit(1)
	}

	// Complex query demonstrating context-aware search
	query := "How does the system handle high load scenarios and what optimizations are in place?"
	response, err := rag.Search(context.Background(), query)
	if err != nil {
		fmt.Printf("Failed to search: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nQuery: %s\nResponse: %s\n", query, response)
}

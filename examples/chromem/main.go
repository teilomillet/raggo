package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/teilomillet/raggo"
)

func main() {
	// Enable debug logging
	raggo.SetLogLevel(raggo.LogLevelDebug)

	// Create a temporary directory for our documents
	tmpDir := "./data"
	err := os.MkdirAll(tmpDir, 0755)
	if err != nil {
		fmt.Printf("Error creating temp directory: %v\n", err)
		os.Exit(1)
	}

	// Create sample documents
	docs := map[string]string{
		"sky.txt":    "The sky is blue because of Rayleigh scattering.",
		"leaves.txt": "Leaves are green because chlorophyll absorbs red and blue light.",
	}

	for filename, content := range docs {
		err := os.WriteFile(filepath.Join(tmpDir, filename), []byte(content), 0644)
		if err != nil {
			fmt.Printf("Error writing file %s: %v\n", filename, err)
			os.Exit(1)
		}
	}

	// Initialize RAG with Chromem
	config := raggo.SimpleRAGConfig{
		Collection: "knowledge-base",
		DBType:     "chromem",
		DBAddress:  "./data/chromem.db",
		Model:      "text-embedding-3-small", // OpenAI embedding model
		APIKey:     os.Getenv("OPENAI_API_KEY"),
		Dimension:  1536, // text-embedding-3-small dimension
		// TopK is determined dynamically by the number of documents
	}

	raggo.Debug("Creating SimpleRAG with config", "config", config)

	rag, err := raggo.NewSimpleRAG(config)
	if err != nil {
		fmt.Printf("Error creating SimpleRAG: %v\n", err)
		os.Exit(1)
	}
	defer rag.Close()

	ctx := context.Background()

	// Add documents from the directory
	raggo.Debug("Adding documents from directory", "dir", tmpDir)
	err = rag.AddDocuments(ctx, tmpDir)
	if err != nil {
		fmt.Printf("Error adding documents: %v\n", err)
		os.Exit(1)
	}

	// Search for documents
	raggo.Debug("Searching for documents", "query", "Why is the sky blue?")
	response, err := rag.Search(ctx, "Why is the sky blue?")
	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Response: %s\n", response)
}

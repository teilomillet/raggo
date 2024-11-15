package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo"
)

func main() {
	// Create a custom LLM instance with specific configuration
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		gollm.SetMaxTokens(500),
		gollm.SetMaxRetries(5),
		gollm.SetRetryDelay(time.Second*3),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		fmt.Printf("Failed to create LLM: %v\n", err)
		os.Exit(1)
	}

	// Create RAG with custom LLM
	config := &raggo.ContextualRAGConfig{
		Collection: "my_contextual_docs",
		LLM:       llm,  // Use our custom LLM instance
	}

	rag, err := raggo.NewContextualRAG(config)
	if err != nil {
		fmt.Printf("Failed to initialize RAG: %v\n", err)
		os.Exit(1)
	}
	defer rag.Close()

	// Add documents
	docsPath := filepath.Join(os.Getenv("HOME"), "Desktop", "raggo", "examples", "chat", "docs")
	if err := rag.AddDocuments(context.Background(), docsPath); err != nil {
		fmt.Printf("Failed to add documents: %v\n", err)
		os.Exit(1)
	}

	// Wait a moment for the collection to be fully loaded
	time.Sleep(time.Second * 2)

	// Example query
	query := "What is MountainPass's PressureValve system and how did it help during Black Friday?"
	response, err := rag.Search(context.Background(), query)
	if err != nil {
		fmt.Printf("Failed to search: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(response)
}

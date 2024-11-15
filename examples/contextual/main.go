package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/teilomillet/raggo"
)

func main() {
	// Enable debug logging
	raggo.SetLogLevel(raggo.LogLevelOff)

	// Initialize RAG with default settings
	raggo.Debug("Initializing ContextualRAG", "collection", "my_contextual_docs")
	rag, err := raggo.NewDefaultContextualRAG("my_contextual_docs")
	if err != nil {
		raggo.Error("Failed to initialize RAG", "error", err)
		os.Exit(1)
	}
	defer rag.Close()

	// Add documents from the chat/docs directory
	docsPath := filepath.Join(os.Getenv("HOME"), "Desktop", "raggo", "examples", "chat", "docs")
	raggo.Debug("Adding documents", "path", docsPath)
	if err := rag.AddDocuments(context.Background(), docsPath); err != nil {
		raggo.Error("Failed to add documents", "error", err)
		os.Exit(1)
	}

	// Wait a moment for the collection to be fully loaded
	time.Sleep(time.Second * 2)

	// Example query
	query := "What is MountainPass's PressureValve system and how did it help during Black Friday?"
	raggo.Debug("Searching", "query", query)

	// Get response
	response, err := rag.Search(context.Background(), query)
	if err != nil {
		raggo.Error("Failed to search", "error", err)
		os.Exit(1)
	}

	fmt.Println(response)
}

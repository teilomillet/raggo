package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/teilomillet/raggo"
)

func main() {
	// Set log level to Warn to reduce output noise
	raggo.SetLogLevel(raggo.LogLevelWarn)

	// Create a new ConcurrentPDFLoader with custom options
	loader := raggo.NewConcurrentPDFLoader(
		raggo.SetTimeout(1*time.Minute),
		raggo.SetTempDir(os.TempDir()),
	)

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	// Specify the source directory containing PDF files
	sourceDir := filepath.Join(wd, "testdata")

	// Create a temporary directory for duplicated PDFs
	targetDir, err := os.MkdirTemp("", "raggo-pdf-test")
	if err != nil {
		log.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(targetDir)

	// Use the concurrent loader to load 10 PDF files (some may be duplicates)
	desiredCount := 1000
	start := time.Now()
	loadedFiles, err := loader.LoadPDFsConcurrent(context.Background(), sourceDir, targetDir, desiredCount)
	if err != nil {
		log.Fatalf("Error loading PDF files concurrently: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("Loaded %d PDF files concurrently in %v\n", len(loadedFiles), elapsed)
	for i, file := range loadedFiles {
		fmt.Printf("%d. %s\n", i+1, file)
	}
}


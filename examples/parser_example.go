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
	// Set log level to Info by default
	raggo.SetLogLevel(raggo.LogLevelInfo)

	parser := raggo.NewParser()
	loader := raggo.NewLoader()

	fmt.Println("Running examples with INFO level logging:")
	runExamples(parser, loader)

}

func runExamples(parser raggo.Parser, loader raggo.Loader) {
	// Example 1: Parse PDF file
	pdfExample(parser)

	// Example 2: Parse text file
	textExample(parser)

	// Example 3: Parse directory
	dirExample(loader)
}

func pdfExample(parser raggo.Parser) {
	fmt.Println("Example 1: Parsing PDF file")
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	pdfPath := filepath.Join(wd, "testdata", "CV.pdf")

	doc, err := parser.Parse(pdfPath)
	if err != nil {
		log.Printf("Error parsing PDF: %v\n", err)
		return
	}
	fmt.Printf("PDF parsed. Content length: %d\n", len(doc.Content))
}

func textExample(parser raggo.Parser) {
	fmt.Println("Example 2: Parsing text file")
	tempFile, err := os.CreateTemp("", "raggo-test-*.txt")
	if err != nil {
		log.Printf("Error creating temp file: %v\n", err)
		return
	}
	defer os.Remove(tempFile.Name())

	content := "This is a test file for raggo parser."
	if _, err := tempFile.WriteString(content); err != nil {
		log.Printf("Error writing to temp file: %v\n", err)
		return
	}
	tempFile.Close()

	doc, err := parser.Parse(tempFile.Name())
	if err != nil {
		log.Printf("Error parsing text file: %v\n", err)
		return
	}
	fmt.Printf("Text file parsed. Content length: %d\n", len(doc.Content))
}

func dirExample(loader raggo.Loader) {
	fmt.Println("Example 3: Loading directory")
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	testDataDir := filepath.Join(wd, "testdata")

	paths, err := loader.LoadDir(context.Background(), testDataDir)
	if err != nil {
		log.Printf("Error loading directory: %v\n", err)
	} else {
		fmt.Printf("Loaded %d files from directory\n", len(paths))
		for i, path := range paths {
			fmt.Printf("%d: %s\n", i+1, path)
		}
	}
}

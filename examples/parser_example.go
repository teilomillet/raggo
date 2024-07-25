package main

import (
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

	fmt.Println("Running examples with INFO level logging:")
	runExamples(parser)

}

func runExamples(parser raggo.Parser) {
	// Example 1: Parse PDF file
	pdfExample(parser)

	// Example 2: Parse text file
	textExample(parser)

	// Example 3: Parse directory
	dirExample(parser)
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

func dirExample(parser raggo.Parser) {
	fmt.Println("Example 3: Parsing directory")
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	testDataDir := filepath.Join(wd, "testdata")

	fileCount := 0
	err = filepath.Walk(testDataDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			_, err := parser.Parse(path)
			if err != nil {
				fmt.Printf("Error parsing %s: %v\n", path, err)
			} else {
				fileCount++
			}
		}
		return nil
	})

	if err != nil {
		log.Printf("Error walking directory: %v\n", err)
	} else {
		fmt.Printf("Parsed %d files in directory\n", fileCount)
	}
}


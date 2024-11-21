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
	// Set log level to Debug for more verbose output
	raggo.SetLogLevel(raggo.LogLevelWarn)

	// Create a new Loader with custom options
	loader := raggo.NewLoader(
		raggo.SetLoaderTimeout(1*time.Minute),
		raggo.SetTempDir(os.TempDir()),
	)

	// Example 1: Load from URL
	urlExample(loader)

	// Example 2: Load single file
	fileExample(loader)

	// Example 3: Load directory
	dirExample(loader)
}

func urlExample(loader raggo.Loader) {
	fmt.Println("Example 1: Loading from URL")
	url := "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
	urlPath, err := loader.LoadURL(context.Background(), url)
	if err != nil {
		log.Printf("Error loading URL: %v\n", err)
	} else {
		fmt.Printf("Successfully loaded URL to: %s\n", urlPath)
	}
	fmt.Println()
}

func fileExample(loader raggo.Loader) {
	fmt.Println("Example 2: Loading single file")
	// Create a temporary file for testing
	tempFile, err := os.CreateTemp("", "raggo-test-*.txt")
	if err != nil {
		log.Printf("Error creating temp file: %v\n", err)
		return
	}
	defer os.Remove(tempFile.Name())

	_, err = tempFile.WriteString("This is a test file for raggo loader.")
	if err != nil {
		log.Printf("Error writing to temp file: %v\n", err)
		return
	}
	tempFile.Close()

	filePath, err := loader.LoadFile(context.Background(), tempFile.Name())
	if err != nil {
		log.Printf("Error loading file: %v\n", err)
	} else {
		fmt.Printf("Successfully loaded file to: %s\n", filePath)
	}
	fmt.Println()
}

func dirExample(loader raggo.Loader) {
	fmt.Println("Example 3: Loading directory")
	// Create a temporary directory with some files for testing
	tempDir, err := os.MkdirTemp("", "raggo-test-dir")
	if err != nil {
		log.Printf("Error creating temp directory: %v\n", err)
		return
	}
	defer os.RemoveAll(tempDir)

	// Create a few test files in the directory
	for i := 1; i <= 3; i++ {
		fileName := filepath.Join(tempDir, fmt.Sprintf("test-file-%d.txt", i))
		err := os.WriteFile(fileName, []byte(fmt.Sprintf("This is test file %d", i)), 0644)
		if err != nil {
			log.Printf("Error creating test file: %v\n", err)
			return
		}
	}

	dirPaths, err := loader.LoadDir(context.Background(), tempDir)
	if err != nil {
		log.Printf("Error loading directory: %v\n", err)
	} else {
		fmt.Printf("Successfully loaded %d files from directory:\n", len(dirPaths))
		for _, path := range dirPaths {
			fmt.Printf("- %s\n", path)
		}
	}
}

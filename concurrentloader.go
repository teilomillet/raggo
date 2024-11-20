// Package raggo provides utilities for concurrent document loading and processing.
package raggo

import (
	"context"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/teilomillet/raggo/rag"
)

// ConcurrentPDFLoader extends the basic Loader interface with concurrent PDF processing
// capabilities. It provides efficient handling of multiple PDF files by:
//   - Loading files in parallel using goroutines
//   - Managing concurrent file operations safely
//   - Handling file duplication when needed
//   - Providing progress tracking and error handling
type ConcurrentPDFLoader interface {
	// Embeds the basic Loader interface
	Loader

	// LoadPDFsConcurrent loads a specified number of PDF files concurrently from a source directory.
	// If the source directory contains fewer files than the requested count, it automatically
	// duplicates existing PDFs to reach the desired number.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeout
	//   - sourceDir: Directory containing source PDF files
	//   - targetDir: Directory where duplicated PDFs will be stored
	//   - count: Desired number of PDF files to load
	//
	// Returns:
	//   - []string: Paths to all successfully loaded files
	//   - error: Any error encountered during the process
	//
	// Example usage:
	//   loader := raggo.NewConcurrentPDFLoader(raggo.SetTimeout(1*time.Minute))
	//   files, err := loader.LoadPDFsConcurrent(ctx, "source", "target", 10)
	LoadPDFsConcurrent(ctx context.Context, sourceDir string, targetDir string, count int) ([]string, error)
}

// concurrentPDFLoaderWrapper wraps the internal loader and adds concurrent PDF loading capability.
// It implements thread-safe operations and efficient resource management.
type concurrentPDFLoaderWrapper struct {
	internal *rag.Loader
}

// NewConcurrentPDFLoader creates a new ConcurrentPDFLoader with the given options.
// It supports all standard loader options plus concurrent processing capabilities.
//
// Options can include:
//   - SetTimeout: Maximum time for loading operations
//   - SetTempDir: Directory for temporary files
//   - SetRetryCount: Number of retries for failed operations
//
// Example:
//   loader := raggo.NewConcurrentPDFLoader(
//     raggo.SetTimeout(1*time.Minute),
//     raggo.SetTempDir(os.TempDir()),
//   )
func NewConcurrentPDFLoader(opts ...LoaderOption) ConcurrentPDFLoader {
	return &concurrentPDFLoaderWrapper{internal: rag.NewLoader(opts...)}
}

// LoadPDFsConcurrent implements the concurrent PDF loading strategy.
// It performs the following steps:
//  1. Lists all PDF files in the source directory
//  2. Creates the target directory if it doesn't exist
//  3. Duplicates PDFs if necessary to reach the desired count
//  4. Loads files concurrently using goroutines
//  5. Collects results and errors from concurrent operations
//
// The function uses channels for thread-safe communication and a WaitGroup
// to ensure all operations complete before returning.
func (clw *concurrentPDFLoaderWrapper) LoadPDFsConcurrent(ctx context.Context, sourceDir string, targetDir string, count int) ([]string, error) {
	pdfs, err := listPDFFiles(sourceDir)
	if err != nil {
		return nil, fmt.Errorf("failed to list PDF files in directory: %w", err)
	}

	if len(pdfs) == 0 {
		return nil, fmt.Errorf("no PDF files found in directory: %s", sourceDir)
	}

	// Create target directory if it doesn't exist
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create target directory: %w", err)
	}

	// Duplicate PDFs if necessary
	duplicatedPDFs, err := duplicatePDFs(pdfs, targetDir, count)
	if err != nil {
		return nil, fmt.Errorf("failed to duplicate PDFs: %w", err)
	}

	var wg sync.WaitGroup
	results := make(chan string, len(duplicatedPDFs))
	errors := make(chan error, len(duplicatedPDFs))

	for _, pdf := range duplicatedPDFs {
		wg.Add(1)
		go func(pdfPath string) {
			defer wg.Done()
			loadedPath, err := clw.internal.LoadFile(ctx, pdfPath)
			if err != nil {
				errors <- err
				return
			}
			results <- loadedPath
		}(pdf)
	}

	go func() {
		wg.Wait()
		close(results)
		close(errors)
	}()

	var loadedFiles []string
	var loadErrors []error

	for i := 0; i < len(duplicatedPDFs); i++ {
		select {
		case result := <-results:
			loadedFiles = append(loadedFiles, result)
		case err := <-errors:
			loadErrors = append(loadErrors, err)
		}
	}

	if len(loadErrors) > 0 {
		return loadedFiles, fmt.Errorf("encountered %d errors during loading", len(loadErrors))
	}

	return loadedFiles, nil
}

// LoadURL implements the Loader interface by loading a document from a URL.
// This method is inherited from the basic Loader interface.
func (clw *concurrentPDFLoaderWrapper) LoadURL(ctx context.Context, url string) (string, error) {
	return clw.internal.LoadURL(ctx, url)
}

// LoadFile implements the Loader interface by loading a single file.
// This method is inherited from the basic Loader interface.
func (clw *concurrentPDFLoaderWrapper) LoadFile(ctx context.Context, path string) (string, error) {
	return clw.internal.LoadFile(ctx, path)
}

// LoadDir implements the Loader interface by loading all files in a directory.
// This method is inherited from the basic Loader interface.
func (clw *concurrentPDFLoaderWrapper) LoadDir(ctx context.Context, dir string) ([]string, error) {
	return clw.internal.LoadDir(ctx, dir)
}

// listPDFFiles returns a list of all PDF files in the given directory.
// It recursively walks through the directory tree and identifies files
// with a .pdf extension (case-insensitive).
func listPDFFiles(dir string) ([]string, error) {
	var pdfs []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.ToLower(filepath.Ext(path)) == ".pdf" {
			pdfs = append(pdfs, path)
		}
		return nil
	})
	return pdfs, err
}

// duplicatePDFs duplicates the given PDF files to reach the desired count.
// If the number of source PDFs is less than the desired count, it creates
// copies with unique names by appending a counter to the original filename.
//
// The function ensures that:
//   - Each copy has a unique name
//   - The total number of files matches the desired count
//   - File copying is performed safely
func duplicatePDFs(pdfs []string, targetDir string, desiredCount int) ([]string, error) {
	var duplicatedPDFs []string
	numOriginalPDFs := len(pdfs)

	if numOriginalPDFs >= desiredCount {
		return pdfs[:desiredCount], nil
	}

	duplicationsNeeded := int(math.Ceil(float64(desiredCount) / float64(numOriginalPDFs)))

	for i := 0; i < duplicationsNeeded; i++ {
		for _, pdf := range pdfs {
			if len(duplicatedPDFs) >= desiredCount {
				break
			}

			newFileName := fmt.Sprintf("%s_copy%d%s", strings.TrimSuffix(filepath.Base(pdf), ".pdf"), i, ".pdf")
			newFilePath := filepath.Join(targetDir, newFileName)

			if err := copyFile(pdf, newFilePath); err != nil {
				return nil, fmt.Errorf("failed to copy file %s: %w", pdf, err)
			}

			duplicatedPDFs = append(duplicatedPDFs, newFilePath)
		}
	}

	return duplicatedPDFs, nil
}

// copyFile performs a safe copy of a file from src to dst.
// It handles:
//   - Opening source and destination files
//   - Proper resource cleanup with defer
//   - Efficient copying with io.Copy
func copyFile(src, dst string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	return err
}

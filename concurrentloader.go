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

// ConcurrentPDFLoader extends the basic Loader interface
type ConcurrentPDFLoader interface {
	Loader
	LoadPDFsConcurrent(ctx context.Context, sourceDir string, targetDir string, count int) ([]string, error)
}

// concurrentPDFLoaderWrapper wraps the internal loader and adds concurrent PDF loading capability
type concurrentPDFLoaderWrapper struct {
	internal *rag.Loader
}

// NewConcurrentPDFLoader creates a new ConcurrentPDFLoader with the given options
func NewConcurrentPDFLoader(opts ...LoaderOption) ConcurrentPDFLoader {
	return &concurrentPDFLoaderWrapper{internal: rag.NewLoader(opts...)}
}

// LoadPDFsConcurrent loads 'count' number of PDF files from the specified directory,
// duplicating files if necessary to reach the desired count
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

func (clw *concurrentPDFLoaderWrapper) LoadURL(ctx context.Context, url string) (string, error) {
	return clw.internal.LoadURL(ctx, url)
}

func (clw *concurrentPDFLoaderWrapper) LoadFile(ctx context.Context, path string) (string, error) {
	return clw.internal.LoadFile(ctx, path)
}

func (clw *concurrentPDFLoaderWrapper) LoadDir(ctx context.Context, dir string) ([]string, error) {
	return clw.internal.LoadDir(ctx, dir)
}

// listPDFFiles returns a list of all PDF files in the given directory
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

// duplicatePDFs duplicates the given PDF files to reach the desired count
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

// copyFile copies a file from src to dst
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

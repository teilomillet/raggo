// Package rag provides document loading functionality for the Raggo framework.
// The loader component handles various input sources including local files,
// directories, and URLs, with support for concurrent operations and
// configurable timeouts.
package rag

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Loader represents the internal loader implementation.
// It provides methods for loading documents from various sources
// with configurable HTTP client, timeout settings, and temporary
// storage management. The loader is designed to be thread-safe
// and can handle concurrent loading operations.
type Loader struct {
	client  *http.Client  // HTTP client for URL downloads
	timeout time.Duration // Timeout for operations
	tempDir string        // Directory for temporary files
	logger  Logger        // Logger for operation tracking
}

// NewLoader creates a new Loader with the given options.
// It initializes a loader with default settings and applies
// any provided options. Default settings include:
// - Standard HTTP client
// - 30-second timeout
// - System temporary directory
// - Global logger instance
func NewLoader(opts ...LoaderOption) *Loader {
	l := &Loader{
		client:  http.DefaultClient,
		timeout: 30 * time.Second,
		tempDir: os.TempDir(),
		logger:  GlobalLogger,
	}

	for _, opt := range opts {
		opt(l)
	}

	return l
}

// LoaderOption is a functional option for configuring a Loader.
// It follows the functional options pattern to provide a clean
// and extensible way to configure the loader.
type LoaderOption func(*Loader)

// WithHTTPClient sets a custom HTTP client for the Loader.
// This allows customization of the HTTP client used for URL downloads,
// enabling features like custom transport settings, proxies, or
// authentication mechanisms.
func WithHTTPClient(client *http.Client) LoaderOption {
	return func(l *Loader) {
		l.client = client
	}
}

// WithTimeout sets a custom timeout for the Loader.
// This timeout applies to all operations including:
// - URL downloads
// - File operations
// - Directory traversal
func WithTimeout(timeout time.Duration) LoaderOption {
	return func(l *Loader) {
		l.timeout = timeout
	}
}

// WithTempDir sets the temporary directory for downloaded files.
// This directory is used to store:
// - Downloaded files from URLs
// - Copies of local files for processing
// - Temporary files during directory operations
func WithTempDir(dir string) LoaderOption {
	return func(l *Loader) {
		l.tempDir = dir
	}
}

// WithLogger sets a custom logger for the Loader.
// The logger is used to track operations and debug issues
// across all loading operations.
func WithLogger(logger Logger) LoaderOption {
	return func(l *Loader) {
		l.logger = logger
	}
}

// LoadURL downloads a file from the given URL and stores it in the temporary directory.
// The function:
// 1. Creates a context with the configured timeout
// 2. Downloads the file using the HTTP client
// 3. Stores the file in the temporary directory
// 4. Returns the path to the downloaded file
//
// The downloaded file's name is derived from the URL's base name.
func (l *Loader) LoadURL(ctx context.Context, url string) (string, error) {
	l.logger.Debug("Starting LoadURL", "url", url)
	ctx, cancel := context.WithTimeout(ctx, l.timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		l.logger.Error("Failed to create request", "url", url, "error", err)
		return "", err
	}

	resp, err := l.client.Do(req)
	if err != nil {
		l.logger.Error("Failed to execute request", "url", url, "error", err)
		return "", err
	}
	defer resp.Body.Close()

	filename := filepath.Base(url)
	destPath := filepath.Join(l.tempDir, filename)

	out, err := os.Create(destPath)
	if err != nil {
		l.logger.Error("Failed to create file", "path", destPath, "error", err)
		return "", err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		l.logger.Error("Failed to write file content", "path", destPath, "error", err)
		return "", err
	}

	l.logger.Debug("Successfully loaded URL", "url", url, "path", destPath)
	return destPath, nil
}

// LoadFile copies a file to the temporary directory and returns its path.
// The function:
// 1. Verifies the source file exists
// 2. Creates a copy in the temporary directory
// 3. Returns the path to the copied file
//
// This ensures that the original file remains unchanged during processing.
func (l *Loader) LoadFile(ctx context.Context, path string) (string, error) {
	l.logger.Debug("Starting LoadFile", "path", path)

	_, err := os.Stat(path)
	if err != nil {
		l.logger.Error("File does not exist", "path", path, "error", err)
		return "", err
	}

	filename := filepath.Base(path)
	destPath := filepath.Join(l.tempDir, filename)

	src, err := os.Open(path)
	if err != nil {
		l.logger.Error("Failed to open source file", "path", path, "error", err)
		return "", err
	}
	defer src.Close()

	dest, err := os.Create(destPath)
	if err != nil {
		l.logger.Error("Failed to create destination file", "path", destPath, "error", err)
		return "", err
	}
	defer dest.Close()

	_, err = io.Copy(dest, src)
	if err != nil {
		l.logger.Error("Failed to copy file", "source", path, "destination", destPath, "error", err)
		return "", err
	}

	l.logger.Debug("Successfully loaded file", "source", path, "destination", destPath)
	return destPath, nil
}

// LoadDir recursively processes all files in a directory.
// The function:
// 1. Walks through the directory tree
// 2. Processes each file encountered
// 3. Returns paths to all processed files
//
// Files that fail to load are logged but don't stop the process.
// The function continues with the next file on error.
func (l *Loader) LoadDir(ctx context.Context, dir string) ([]string, error) {
	l.logger.Debug("Starting LoadDir", "dir", dir)

	var loadedFiles []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			l.logger.Error("Error accessing path", "path", path, "error", err)
			return err
		}
		if !info.IsDir() {
			l.logger.Debug("Processing file", "path", path)
			loadedPath, err := l.LoadFile(ctx, path)
			if err != nil {
				l.logger.Warn("Failed to load file", "path", path, "error", err)
				return nil // Continue with next file
			}
			loadedFiles = append(loadedFiles, loadedPath)
		}
		return nil
	})

	if err != nil {
		l.logger.Error("Error walking directory", "dir", dir, "error", err)
		return nil, err
	}

	l.logger.Debug("Successfully loaded directory", "dir", dir, "fileCount", len(loadedFiles))
	return loadedFiles, nil
}

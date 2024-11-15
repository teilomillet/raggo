// rag/load.go
package rag

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Loader represents the internal loader implementation
type Loader struct {
	client  *http.Client
	timeout time.Duration
	tempDir string
	logger  Logger
}

// NewLoader creates a new Loader with the given options
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

// LoaderOption is a functional option for configuring a Loader
type LoaderOption func(*Loader)

// WithHTTPClient sets a custom HTTP client for the Loader
func WithHTTPClient(client *http.Client) LoaderOption {
	return func(l *Loader) {
		l.client = client
	}
}

// WithTimeout sets a custom timeout for the Loader
func WithTimeout(timeout time.Duration) LoaderOption {
	return func(l *Loader) {
		l.timeout = timeout
	}
}

// WithTempDir sets the temporary directory for downloaded files
func WithTempDir(dir string) LoaderOption {
	return func(l *Loader) {
		l.tempDir = dir
	}
}

// WithLogger sets a custom logger for the Loader
func WithLogger(logger Logger) LoaderOption {
	return func(l *Loader) {
		l.logger = logger
	}
}

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

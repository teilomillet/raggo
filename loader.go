// Package raggo provides a high-level interface for document loading and processing
// in RAG (Retrieval-Augmented Generation) systems. The loader component handles
// various input sources with support for concurrent operations and configurable
// behaviors.
package raggo

import (
	"context"
	"net/http"
	"time"

	"github.com/teilomillet/raggo/rag"
)

// Loader represents the main interface for loading documents from various sources.
// It provides a unified API for handling:
//   - URLs: Download and process remote documents
//   - Files: Load and process local files
//   - Directories: Recursively process directory contents
//
// The interface is designed to be thread-safe and supports concurrent operations
// with configurable timeouts and error handling.
type Loader interface {
	// LoadURL downloads and processes a document from the given URL.
	// The function handles:
	//   - HTTP/HTTPS downloads
	//   - Timeout management
	//   - Temporary file storage
	//
	// Returns the processed content and any errors encountered.
	LoadURL(ctx context.Context, url string) (string, error)

	// LoadFile processes a local file at the given path.
	// The function:
	//   - Verifies file existence
	//   - Handles file reading
	//   - Manages temporary storage
	//
	// Returns the processed content and any errors encountered.
	LoadFile(ctx context.Context, path string) (string, error)

	// LoadDir recursively processes all files in a directory.
	// The function:
	//   - Walks the directory tree
	//   - Processes each file
	//   - Handles errors gracefully
	//
	// Returns paths to all processed files and any errors encountered.
	LoadDir(ctx context.Context, dir string) ([]string, error)
}

// loaderWrapper encapsulates the internal loader implementation
// providing a clean interface while maintaining all functionality.
type loaderWrapper struct {
	internal *rag.Loader
}

// LoaderOption is a functional option for configuring a Loader.
// It follows the functional options pattern to provide a clean
// and extensible configuration API.
//
// Common options include:
//   - WithHTTPClient: Custom HTTP client configuration
//   - SetLoaderTimeout: Operation timeout settings
//   - SetTempDir: Temporary storage location
type LoaderOption = rag.LoaderOption

// WithHTTPClient sets a custom HTTP client for the Loader.
// This enables customization of:
//   - Transport settings
//   - Proxy configuration
//   - Authentication mechanisms
//   - Connection pooling
//
// Example:
//
//	client := &http.Client{
//	    Timeout: 60 * time.Second,
//	    Transport: &http.Transport{
//	        MaxIdleConns: 10,
//	        IdleConnTimeout: 30 * time.Second,
//	    },
//	}
//	loader := NewLoader(WithHTTPClient(client))
func WithHTTPClient(client *http.Client) LoaderOption {
	return rag.WithHTTPClient(client)
}

// WithLoaderTimeout sets a custom timeout for all loader operations.
// The timeout applies to:
//   - URL downloads
//   - File operations
//   - Directory traversal
//
// Example:
//
//	// Set a 2-minute timeout for all operations
//	loader := NewLoader(WithLoaderTimeout(2 * time.Minute))
func WithLoaderTimeout(timeout time.Duration) LoaderOption {
	return rag.WithTimeout(timeout)
}

// SetTempDir sets the temporary directory for file operations.
// This directory is used for:
//   - Storing downloaded files
//   - Creating temporary copies
//   - Processing large documents
//
// Example:
//
//	// Use a custom temporary directory
//	loader := NewLoader(SetTempDir("/path/to/temp"))
func SetTempDir(dir string) LoaderOption {
	return rag.WithTempDir(dir)
}

// NewLoader creates a new Loader with the specified options.
// It initializes a loader with sensible defaults and applies
// any provided configuration options.
//
// Default settings:
//   - Standard HTTP client
//   - 30-second timeout
//   - System temporary directory
//
// Example:
//
//	loader := NewLoader(
//	    WithHTTPClient(customClient),
//	    WithLoaderTimeout(time.Minute),
//	    SetTempDir("/custom/temp"),
//	)
func NewLoader(opts ...LoaderOption) Loader {
	return &loaderWrapper{internal: rag.NewLoader(opts...)}
}

// LoadURL downloads and processes a document from the given URL.
// The function handles the entire download process including:
//   - Context and timeout management
//   - HTTP request execution
//   - Response processing
//   - Temporary file management
func (lw *loaderWrapper) LoadURL(ctx context.Context, url string) (string, error) {
	return lw.internal.LoadURL(ctx, url)
}

// LoadFile processes a local file at the given path.
// The function ensures safe file handling by:
//   - Verifying file existence
//   - Creating temporary copies
//   - Managing file resources
//   - Handling processing errors
func (lw *loaderWrapper) LoadFile(ctx context.Context, path string) (string, error) {
	return lw.internal.LoadFile(ctx, path)
}

// LoadDir recursively processes all files in a directory.
// The function provides robust directory handling:
//   - Recursive traversal
//   - Error tolerance (continues on file errors)
//   - Progress tracking
//   - Resource cleanup
func (lw *loaderWrapper) LoadDir(ctx context.Context, dir string) ([]string, error) {
	return lw.internal.LoadDir(ctx, dir)
}

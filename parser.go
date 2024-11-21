// Package raggo provides a flexible and extensible document parsing system
// for RAG (Retrieval-Augmented Generation) applications. The system supports
// multiple file formats and can be extended with custom parsers.
package raggo

import (
	"context"

	"github.com/teilomillet/raggo/rag"
)

// Document represents a parsed document with its content and metadata.
// The structure includes:
//   - Content: The extracted text from the document
//   - Metadata: Additional information about the document
//
// Example:
//
//	doc := Document{
//	    Content: "Extracted text content...",
//	    Metadata: map[string]string{
//	        "file_type": "pdf",
//	        "file_path": "/path/to/doc.pdf",
//	    },
//	}
type Document = rag.Document

// Parser defines the interface for document parsing implementations.
// Any type implementing this interface can be registered to handle
// specific file types. The interface is designed to be simple yet
// powerful enough to support various parsing strategies.
//
// Implementations must handle:
//   - File access and reading
//   - Content extraction
//   - Metadata collection
//   - Error handling
//
// Example:
//
//	// Create a new parser with default settings
//	parser := NewParser()
//
//	// Parse a document
//	doc, err := parser.Parse("document.pdf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Content: %s\n", doc.Content)
type Parser interface {
	// Parse processes a file and returns its content and metadata.
	// Returns an error if the parsing operation fails.
	Parse(filePath string) (Document, error)
}

// parserWrapper combines Parser and Loader capabilities into a single type.
// It implements both the Parser interface for document parsing and provides
// loading functionality through an embedded rag.Loader.
//
// The wrapper uses dependency injection to allow customization of both
// the parser and loader components, making it highly configurable and
// testable.
//
// Example usage with custom loader:
//
//	customLoader := rag.NewLoader(
//	    rag.WithTimeout(time.Minute),
//	    rag.WithTempDir("/custom/temp"),
//	)
//	
//	parser := NewParser(
//	    WithLoader(customLoader),
//	)
type parserWrapper struct {
    parser Parser       // Core parsing implementation
    loader *rag.Loader // Document loading capabilities
}

// ParserOption defines functional options for configuring a Parser.
// This follows the functional options pattern, allowing flexible and
// extensible configuration of the parser without breaking existing code.
//
// Custom options can be created by implementing this function type
// and modifying the parserWrapper fields as needed.
//
// Example creating a custom option:
//
//	func WithCustomTimeout(timeout time.Duration) ParserOption {
//	    return func(pw *parserWrapper) {
//	        pw.loader = rag.NewLoader(rag.WithTimeout(timeout))
//	    }
//	}
type ParserOption func(*parserWrapper)

// WithLoader injects a custom loader into the parser.
// This option allows you to provide a pre-configured rag.Loader
// instance with custom settings for timeout, temporary directory,
// HTTP client, or other loader-specific configurations.
//
// Example:
//
//	customLoader := rag.NewLoader(
//	    rag.WithTimeout(time.Minute),
//	    rag.WithTempDir("/custom/temp"),
//	)
//	
//	parser := NewParser(
//	    WithLoader(customLoader),
//	)
func WithLoader(loader *rag.Loader) ParserOption {
    return func(pw *parserWrapper) {
        pw.loader = loader
    }
}

// NewParser creates a new Parser with the given options.
// It initializes a parserWrapper with default settings and applies
// any provided configuration options. The resulting parser implements
// both document parsing and loading capabilities.
//
// Default configuration includes:
//   - Standard rag.Loader with default timeout and temp directory
//   - Default parser manager for handling various file types
//   - Built-in support for common file formats
//
// Example:
//
//	// Create parser with default settings
//	parser := NewParser()
//
//	// Create parser with custom loader
//	parser := NewParser(
//	    WithLoader(customLoader),
//	)
//
//	// Create parser with multiple options
//	parser := NewParser(
//	    WithLoader(customLoader),
//	    WithCustomOption(...),
//	)
func NewParser(opts ...ParserOption) Parser {
    pw := &parserWrapper{
        parser: rag.NewParserManager(),
        loader: rag.NewLoader(),
    }
    
    for _, opt := range opts {
        opt(pw)
    }
    
    return pw
}

// Parse implements the Parser interface by processing a file and extracting
// its content and metadata. It delegates the actual parsing to the underlying
// parser implementation while maintaining the option to preprocess files
// using the loader capabilities.
//
// Returns:
//   - Document: Contains the extracted content and metadata
//   - error: Any error encountered during parsing
//
// Example:
//
//	doc, err := parser.Parse("document.pdf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Content: %s\n", doc.Content)
func (pw *parserWrapper) Parse(filePath string) (Document, error) {
    return pw.parser.Parse(filePath)
}

// LoadDir implements the Loader interface by recursively processing
// all files in a directory. It delegates to the embedded loader's
// implementation while maintaining the parser's context.
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - dir: Directory path to process
//
// Returns:
//   - []string: Paths to all processed files
//   - error: Any error encountered during directory processing
//
// Example:
//
//	paths, err := parser.LoadDir(ctx, "/path/to/docs")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for _, path := range paths {
//	    fmt.Printf("Processed: %s\n", path)
//	}
func (pw *parserWrapper) LoadDir(ctx context.Context, dir string) ([]string, error) {
    return pw.loader.LoadDir(ctx, dir)
}

// LoadFile implements the Loader interface by processing a single file.
// It uses the embedded loader's implementation while maintaining the
// parser's context.
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - path: Path to the file to process
//
// Returns:
//   - string: Path to the processed file
//   - error: Any error encountered during file processing
//
// Example:
//
//	processedPath, err := parser.LoadFile(ctx, "document.pdf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Processed file at: %s\n", processedPath)
func (pw *parserWrapper) LoadFile(ctx context.Context, path string) (string, error) {
    return pw.loader.LoadFile(ctx, path)
}

// LoadURL implements the Loader interface by downloading and processing
// a file from a URL. It uses the embedded loader's implementation while
// maintaining the parser's context.
//
// Parameters:
//   - ctx: Context for cancellation and timeout
//   - url: URL of the file to download and process
//
// Returns:
//   - string: Path to the downloaded and processed file
//   - error: Any error encountered during download or processing
//
// Example:
//
//	processedPath, err := parser.LoadURL(ctx, "https://example.com/doc.pdf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Downloaded and processed file at: %s\n", processedPath)
func (pw *parserWrapper) LoadURL(ctx context.Context, url string) (string, error) {
    return pw.loader.LoadURL(ctx, url)
}

// SetFileTypeDetector customizes how file types are detected.
// This allows for sophisticated file type detection beyond simple
// extension matching.
//
// Example:
//
//	SetFileTypeDetector(parser, func(path string) string {
//	    // Custom logic to determine file type
//	    if strings.HasSuffix(path, ".md") {
//	        return "markdown"
//	    }
//	    return "unknown"
//	})
func SetFileTypeDetector(p Parser, detector func(string) string) {
	if pm, ok := p.(*rag.ParserManager); ok {
		pm.SetFileTypeDetector(detector)
	}
}

// WithParser adds a custom parser for a specific file type.
// This enables the parsing system to handle additional file formats
// through custom implementations.
//
// Example:
//
//	// Add support for markdown files
//	WithParser(parser, "markdown", &MarkdownParser{})
func WithParser(p Parser, fileType string, parser Parser) {
	if pm, ok := p.(*rag.ParserManager); ok {
		pm.AddParser(fileType, parser)
	}
}

// TextParser returns a new parser for plain text files.
// The text parser:
//   - Reads the entire file content
//   - Preserves text formatting
//   - Handles various encodings
//   - Provides basic metadata
//
// Example:
//
//	parser := TextParser()
//	doc, err := parser.Parse("document.txt")
func TextParser() Parser {
	return rag.NewTextParser()
}

// PDFParser returns a new parser for PDF documents.
// The PDF parser:
//   - Extracts text content from all pages
//   - Maintains text order
//   - Handles complex PDF structures
//   - Provides document metadata
//
// Example:
//
//	parser := PDFParser()
//	doc, err := parser.Parse("document.pdf")
func PDFParser() Parser {
	return rag.NewPDFParser()
}

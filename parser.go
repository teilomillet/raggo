// Package raggo provides a flexible and extensible document parsing system
// for RAG (Retrieval-Augmented Generation) applications. The system supports
// multiple file formats and can be extended with custom parsers.
package raggo

import (
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
type Parser interface {
	// Parse processes a file and returns its content and metadata.
	// Returns an error if the parsing operation fails.
	Parse(filePath string) (Document, error)
}

// NewParser creates a new Parser with default settings and handlers.
// The default configuration includes:
//   - PDF document support
//   - Plain text file support
//   - Extension-based file type detection
//
// Example:
//
//	parser := NewParser()
//	doc, err := parser.Parse("document.pdf")
func NewParser() Parser {
	return rag.NewParserManager()
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

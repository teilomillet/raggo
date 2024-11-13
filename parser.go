package raggo

import (
	"github.com/teilomillet/raggo/rag"
)

// Document represents a parsed document
type Document = rag.Document

// Parser defines the interface for parsing documents
type Parser interface {
	Parse(filePath string) (Document, error)
}

// NewParser creates a new Parser with default settings
func NewParser() Parser {
	return rag.NewParserManager()
}

// SetFileTypeDetector sets a custom file type detector
func SetFileTypeDetector(p Parser, detector func(string) string) {
	if pm, ok := p.(*rag.ParserManager); ok {
		pm.SetFileTypeDetector(detector)
	}
}

// WithParser adds a parser for a specific file type
func WithParser(p Parser, fileType string, parser Parser) {
	if pm, ok := p.(*rag.ParserManager); ok {
		pm.AddParser(fileType, parser)
	}
}

// TextParser returns a new text parser
func TextParser() Parser {
	return rag.NewTextParser()
}

// PDFParser returns a new PDF parser
func PDFParser() Parser {
	return rag.NewPDFParser()
}

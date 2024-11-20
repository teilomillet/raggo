// Package rag provides document parsing capabilities for various file formats.
// The parsing system is designed to be extensible, allowing users to add custom parsers
// for different file types while maintaining a consistent interface.
package rag

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ledongthuc/pdf"
)

// Document represents a parsed document with its content and associated metadata.
// The Content field contains the extracted text, while Metadata stores additional
// information about the document such as file type and path.
type Document struct {
	Content  string            // The extracted text content of the document
	Metadata map[string]string // Additional metadata about the document
}

// Parser defines the interface for document parsing implementations.
// Any type that implements this interface can be registered with the ParserManager
// to handle specific file types.
type Parser interface {
	// Parse processes a file at the given path and returns a Document.
	// It returns an error if the parsing operation fails.
	Parse(filePath string) (Document, error)
}

// ParserManager coordinates document parsing by managing different Parser implementations
// and routing files to the appropriate parser based on their type.
type ParserManager struct {
	// fileTypeDetector determines the file type based on the file path.
	fileTypeDetector func(string) string
	// parsers stores the registered parsers for different file types.
	parsers map[string]Parser
}

// NewParserManager creates a new ParserManager initialized with default settings
// and parsers for common file types (PDF and text files).
func NewParserManager() *ParserManager {
	pm := &ParserManager{
		fileTypeDetector: defaultFileTypeDetector,
		parsers:          make(map[string]Parser),
	}

	// Add default parsers
	pm.parsers["pdf"] = NewPDFParser()
	pm.parsers["text"] = NewTextParser()

	return pm
}

// Parse processes a document using the appropriate parser based on the file type.
// It uses the configured fileTypeDetector to determine the file type and then
// delegates to the corresponding parser. Returns an error if no suitable parser
// is found or if parsing fails.
func (pm *ParserManager) Parse(filePath string) (Document, error) {
	GlobalLogger.Debug("Starting to parse file", "path", filePath)
	fileType := pm.fileTypeDetector(filePath)
	parser, ok := pm.parsers[fileType]
	if !ok {
		GlobalLogger.Error("No parser available for file type", "type", fileType)
		return Document{}, fmt.Errorf("no parser available for file type: %s", fileType)
	}
	doc, err := parser.Parse(filePath)
	if err != nil {
		GlobalLogger.Error("Failed to parse document", "path", filePath, "error", err)
		return Document{}, err
	}
	GlobalLogger.Debug("Successfully parsed document", "path", filePath, "type", fileType)
	return doc, nil
}

// defaultFileTypeDetector determines file type based on file extension.
// Currently supports .pdf and .txt files, returning "unknown" for other extensions.
func defaultFileTypeDetector(filePath string) string {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".pdf":
		return "pdf"
	case ".txt":
		return "text"
	default:
		return "unknown"
	}
}

// SetFileTypeDetector allows customization of how file types are detected.
// This can be used to implement more sophisticated file type detection beyond
// simple extension matching.
func (pm *ParserManager) SetFileTypeDetector(detector func(string) string) {
	pm.fileTypeDetector = detector
}

// AddParser registers a new parser for a specific file type.
// This allows users to extend the system with custom parsers for additional
// file formats.
func (pm *ParserManager) AddParser(fileType string, parser Parser) {
	pm.parsers[fileType] = parser
}

// PDFParser implements the Parser interface for PDF files using the
// ledongthuc/pdf library for text extraction.
type PDFParser struct{}

// NewPDFParser creates a new PDFParser instance.
func NewPDFParser() *PDFParser {
	return &PDFParser{}
}

// Parse implements the Parser interface for PDF files.
// It extracts text content from the PDF and returns it along with basic metadata.
// Returns an error if the PDF cannot be processed.
func (p *PDFParser) Parse(filePath string) (Document, error) {
	GlobalLogger.Debug("Starting to parse PDF", "path", filePath)
	content, err := p.extractText(filePath)
	if err != nil {
		GlobalLogger.Error("Failed to extract text from PDF", "path", filePath, "error", err)
		return Document{}, fmt.Errorf("failed to extract text: %w", err)
	}
	GlobalLogger.Debug("Successfully parsed PDF", "path", filePath)
	return Document{
		Content: content,
		Metadata: map[string]string{
			"file_type": "pdf",
			"file_path": filePath,
		},
	}, nil
}

// extractText performs the actual text extraction from a PDF file.
// It processes the PDF page by page, concatenating the extracted text.
// Returns an error if any part of the extraction process fails.
func (p *PDFParser) extractText(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return "", fmt.Errorf("failed to get file info: %w", err)
	}

	reader, err := pdf.NewReader(file, fileInfo.Size())
	if err != nil {
		return "", fmt.Errorf("failed to create PDF reader: %w", err)
	}

	var textBuilder strings.Builder
	numPages := reader.NumPage()
	for i := 1; i <= numPages; i++ {
		page := reader.Page(i)
		if page.V.IsNull() {
			continue
		}
		content, err := page.GetPlainText(nil)
		if err != nil {
			return "", fmt.Errorf("failed to extract text from page %d: %w", i, err)
		}
		textBuilder.WriteString(content)
		textBuilder.WriteString("\n\n")
	}

	return textBuilder.String(), nil
}

// TextParser implements the Parser interface for plain text files.
type TextParser struct{}

// NewTextParser creates a new TextParser instance.
func NewTextParser() *TextParser {
	return &TextParser{}
}

// Parse implements the Parser interface for text files.
// It reads the entire file content and returns it along with basic metadata.
// Returns an error if the file cannot be read.
func (p *TextParser) Parse(filePath string) (Document, error) {
	GlobalLogger.Debug("Starting to parse text file", "path", filePath)
	content, err := os.ReadFile(filePath)
	if err != nil {
		GlobalLogger.Error("Failed to read text file", "path", filePath, "error", err)
		return Document{}, fmt.Errorf("failed to read file: %w", err)
	}
	GlobalLogger.Debug("Successfully parsed text file", "path", filePath)
	return Document{
		Content: string(content),
		Metadata: map[string]string{
			"file_type": "text",
			"file_path": filePath,
		},
	}, nil
}

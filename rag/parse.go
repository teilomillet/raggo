package rag

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ledongthuc/pdf"
)

// Document represents a parsed document
type Document struct {
	Content  string
	Metadata map[string]string
}

// Parser defines the interface for parsing documents
type Parser interface {
	Parse(filePath string) (Document, error)
}

// ParserManager is responsible for managing different parsers
type ParserManager struct {
	fileTypeDetector func(string) string
	parsers          map[string]Parser
}

// NewParserManager creates a new ParserManager with default settings
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

// Parse parses a document using the appropriate parser based on file type
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

// defaultFileTypeDetector is a simple file type detector based on file extension
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

// Parse parses a PDF file and returns its content
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

// SetFileTypeDetector sets a custom file type detector
func (pm *ParserManager) SetFileTypeDetector(detector func(string) string) {
	pm.fileTypeDetector = detector
}

// AddParser adds a parser for a specific file type
func (pm *ParserManager) AddParser(fileType string, parser Parser) {
	pm.parsers[fileType] = parser
}

// PDFParser is the implementation of Parser for PDF files
type PDFParser struct{}

// NewPDFParser creates a new PDFParser
func NewPDFParser() *PDFParser {
	return &PDFParser{}
}

// extractText extracts plain text from a PDF file
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

// TextParser is the implementation of Parser for text files
type TextParser struct{}

// NewTextParser creates a new TextParser
func NewTextParser() *TextParser {
	return &TextParser{}
}

// Parse parses a text file and returns its content
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

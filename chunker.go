package raggo

import (
	"github.com/teilomillet/raggo/internal/rag"
)

// Chunk represents a piece of text with metadata
type Chunk = rag.Chunk

// Chunker defines the interface for text chunking
type Chunker interface {
	Chunk(text string) []Chunk
}

// TokenCounter defines the interface for counting tokens in a string
type TokenCounter interface {
	Count(text string) int
}

// ChunkerOption is a function type for configuring Chunker
type ChunkerOption = rag.TextChunkerOption

// NewChunker creates a new Chunker with the given options
func NewChunker(options ...ChunkerOption) (Chunker, error) {
	return rag.NewTextChunker(options...)
}

// ChunkSize sets the chunk size
func ChunkSize(size int) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.ChunkSize = size
	}
}

// ChunkOverlap sets the chunk overlap
func ChunkOverlap(overlap int) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.ChunkOverlap = overlap
	}
}

// WithTokenCounter sets a custom token counter
func WithTokenCounter(counter TokenCounter) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.TokenCounter = counter
	}
}

// WithSentenceSplitter sets a custom sentence splitter function
func WithSentenceSplitter(splitter func(string) []string) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.SentenceSplitter = splitter
	}
}

// DefaultSentenceSplitter returns the default sentence splitter function
func DefaultSentenceSplitter() func(string) []string {
	return rag.DefaultSentenceSplitter
}

// SmartSentenceSplitter returns the smart sentence splitter function
func SmartSentenceSplitter() func(string) []string {
	return rag.SmartSentenceSplitter
}

// NewDefaultTokenCounter creates a new default token counter
func NewDefaultTokenCounter() TokenCounter {
	return &rag.DefaultTokenCounter{}
}

// NewTikTokenCounter creates a new TikToken counter with the specified encoding
func NewTikTokenCounter(encoding string) (TokenCounter, error) {
	return rag.NewTikTokenCounter(encoding)
}


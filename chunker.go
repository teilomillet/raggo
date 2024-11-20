// Package raggo provides a high-level interface for text chunking and token management,
// designed for use in retrieval-augmented generation (RAG) applications.
package raggo

import (
	"github.com/teilomillet/raggo/rag"
)

// Chunk represents a piece of text with associated metadata including its content,
// token count, and position within the original document. It tracks:
//   - The actual text content
//   - Number of tokens in the chunk
//   - Starting and ending sentence indices
type Chunk = rag.Chunk

// Chunker defines the interface for text chunking implementations.
// Implementations of this interface provide strategies for splitting text
// into semantically meaningful chunks while preserving context.
type Chunker interface {
	// Chunk splits the input text into a slice of Chunks according to the
	// implementation's strategy.
	Chunk(text string) []Chunk
}

// TokenCounter defines the interface for counting tokens in text.
// Different implementations can provide various tokenization strategies,
// from simple word-based counting to model-specific subword tokenization.
type TokenCounter interface {
	// Count returns the number of tokens in the given text according to
	// the implementation's tokenization strategy.
	Count(text string) int
}

// ChunkerOption is a function type for configuring Chunker instances.
// It follows the functional options pattern for clean and flexible configuration.
type ChunkerOption = rag.TextChunkerOption

// NewChunker creates a new Chunker with the given options.
// By default, it creates a TextChunker with:
//   - Chunk size: 200 tokens
//   - Chunk overlap: 50 tokens
//   - Default word-based token counter
//   - Basic sentence splitter
//
// Use the provided option functions to customize these settings.
func NewChunker(options ...ChunkerOption) (Chunker, error) {
	return rag.NewTextChunker(options...)
}

// ChunkSize sets the target size of each chunk in tokens.
// This determines how much text will be included in each chunk
// before starting a new one.
func ChunkSize(size int) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.ChunkSize = size
	}
}

// ChunkOverlap sets the number of tokens that should overlap between
// adjacent chunks. This helps maintain context across chunk boundaries
// and improves retrieval quality.
func ChunkOverlap(overlap int) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.ChunkOverlap = overlap
	}
}

// WithTokenCounter sets a custom token counter implementation.
// This allows you to use different tokenization strategies, such as:
//   - Word-based counting (DefaultTokenCounter)
//   - Model-specific tokenization (TikTokenCounter)
//   - Custom tokenization schemes
func WithTokenCounter(counter TokenCounter) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.TokenCounter = counter
	}
}

// WithSentenceSplitter sets a custom sentence splitter function.
// The function should take a string and return a slice of strings,
// where each string is a sentence. This allows for:
//   - Custom sentence boundary detection
//   - Language-specific splitting rules
//   - Special handling of abbreviations or formatting
func WithSentenceSplitter(splitter func(string) []string) ChunkerOption {
	return func(tc *rag.TextChunker) {
		tc.SentenceSplitter = splitter
	}
}

// DefaultSentenceSplitter returns the basic sentence splitter function
// that splits text on common punctuation marks (., !, ?).
// Suitable for simple English text without complex formatting.
func DefaultSentenceSplitter() func(string) []string {
	return rag.DefaultSentenceSplitter
}

// SmartSentenceSplitter returns an advanced sentence splitter that handles:
//   - Multiple punctuation marks
//   - Quoted sentences
//   - Parenthetical content
//   - Lists and enumerations
//
// Recommended for complex text with varied formatting and structure.
func SmartSentenceSplitter() func(string) []string {
	return rag.SmartSentenceSplitter
}

// NewDefaultTokenCounter creates a simple word-based token counter
// that splits text on whitespace. Suitable for basic use cases
// where exact token counts aren't critical.
func NewDefaultTokenCounter() TokenCounter {
	return &rag.DefaultTokenCounter{}
}

// NewTikTokenCounter creates a token counter using the tiktoken library,
// which implements the same tokenization used by OpenAI models.
// The encoding parameter specifies which tokenization model to use
// (e.g., "cl100k_base" for GPT-4, "p50k_base" for GPT-3).
func NewTikTokenCounter(encoding string) (TokenCounter, error) {
	return rag.NewTikTokenCounter(encoding)
}

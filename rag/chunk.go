// Package rag provides text chunking capabilities for processing documents into
// manageable pieces suitable for vector embedding and retrieval.
package rag

import (
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
)

// Chunk represents a piece of text with associated metadata for tracking its position
// and size within the original document.
type Chunk struct {
	// Text contains the actual content of the chunk
	Text string
	// TokenSize represents the number of tokens in this chunk
	TokenSize int
	// StartSentence is the index of the first sentence in this chunk
	StartSentence int
	// EndSentence is the index of the last sentence in this chunk (exclusive)
	EndSentence int
}

// Chunker defines the interface for text chunking implementations.
// Different implementations can provide various strategies for splitting text
// while maintaining context and semantic meaning.
type Chunker interface {
	// Chunk splits the input text into a slice of Chunks according to the
	// implementation's strategy.
	Chunk(text string) []Chunk
}

// TokenCounter defines the interface for counting tokens in a string.
// This abstraction allows for different tokenization strategies (e.g., words, subwords).
type TokenCounter interface {
	// Count returns the number of tokens in the given text according to the
	// implementation's tokenization strategy.
	Count(text string) int
}

// TextChunker provides an advanced implementation of the Chunker interface
// with support for overlapping chunks and custom tokenization.
type TextChunker struct {
	// ChunkSize is the target size of each chunk in tokens
	ChunkSize int
	// ChunkOverlap is the number of tokens that should overlap between adjacent chunks
	ChunkOverlap int
	// TokenCounter is used to count tokens in text segments
	TokenCounter TokenCounter
	// SentenceSplitter is a function that splits text into sentences
	SentenceSplitter func(string) []string
}

// NewTextChunker creates a new TextChunker with the given options.
// It uses sensible defaults if no options are provided:
// - ChunkSize: 200 tokens
// - ChunkOverlap: 50 tokens
// - TokenCounter: DefaultTokenCounter
// - SentenceSplitter: DefaultSentenceSplitter
func NewTextChunker(options ...TextChunkerOption) (*TextChunker, error) {
	tc := &TextChunker{
		ChunkSize:        200,
		ChunkOverlap:     50,
		TokenCounter:     &DefaultTokenCounter{},
		SentenceSplitter: DefaultSentenceSplitter,
	}

	for _, option := range options {
		option(tc)
	}

	return tc, nil
}

// TextChunkerOption is a function type for configuring TextChunker instances.
// This follows the functional options pattern for clean and flexible configuration.
type TextChunkerOption func(*TextChunker)

// Chunk splits the input text into chunks while preserving sentence boundaries
// and maintaining the specified overlap between chunks. The algorithm:
// 1. Splits the text into sentences
// 2. Builds chunks by adding sentences until the chunk size limit is reached
// 3. Creates overlap with previous chunk when starting a new chunk
// 4. Tracks token counts and sentence indices for each chunk
func (tc *TextChunker) Chunk(text string) []Chunk {
	sentences := tc.SentenceSplitter(text)
	var chunks []Chunk
	var currentChunk Chunk
	currentTokenCount := 0

	for i, sentence := range sentences {
		sentenceTokenCount := tc.TokenCounter.Count(sentence)

		if currentTokenCount+sentenceTokenCount > tc.ChunkSize && currentTokenCount > 0 {
			chunks = append(chunks, currentChunk)

			overlapStart := max(currentChunk.StartSentence, currentChunk.EndSentence-tc.estimateOverlapSentences(sentences, currentChunk.EndSentence, tc.ChunkOverlap))
			currentChunk = Chunk{
				Text:          strings.Join(sentences[overlapStart:i+1], " "),
				TokenSize:     0,
				StartSentence: overlapStart,
				EndSentence:   i + 1,
			}
			currentTokenCount = 0
			for j := overlapStart; j <= i; j++ {
				currentTokenCount += tc.TokenCounter.Count(sentences[j])
			}
		} else {
			if currentTokenCount == 0 {
				currentChunk.StartSentence = i
			}
			currentChunk.Text += sentence + " "
			currentChunk.EndSentence = i + 1
			currentTokenCount += sentenceTokenCount
		}
		currentChunk.TokenSize = currentTokenCount
	}

	if currentChunk.TokenSize > 0 {
		chunks = append(chunks, currentChunk)
	}

	return chunks
}

// estimateOverlapSentences calculates how many sentences from the end of the
// previous chunk should be included in the next chunk to achieve the desired
// token overlap.
func (tc *TextChunker) estimateOverlapSentences(sentences []string, endSentence, desiredOverlap int) int {
	overlapTokens := 0
	overlapSentences := 0
	for i := endSentence - 1; i >= 0 && overlapTokens < desiredOverlap; i-- {
		overlapTokens += tc.TokenCounter.Count(sentences[i])
		overlapSentences++
	}
	return overlapSentences
}

// DefaultSentenceSplitter provides a basic implementation for splitting text into sentences.
// It uses common punctuation marks (., !, ?) as sentence boundaries.
func DefaultSentenceSplitter(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
}

// SmartSentenceSplitter provides an advanced sentence splitting implementation that handles:
// - Multiple punctuation marks (., !, ?)
// - Common abbreviations
// - Quoted sentences
// - Parenthetical sentences
// - Lists and enumerations
func SmartSentenceSplitter(text string) []string {
	var sentences []string
	var currentSentence strings.Builder
	inQuote := false

	for _, r := range text {
		currentSentence.WriteRune(r)

		if r == '"' {
			inQuote = !inQuote
		}

		if (r == '.' || r == '!' || r == '?') && !inQuote {
			// Check if it's really the end of a sentence
			if len(sentences) > 0 || currentSentence.Len() > 1 {
				sentences = append(sentences, strings.TrimSpace(currentSentence.String()))
				currentSentence.Reset()
			}
		}
	}

	// Add any remaining text as a sentence
	if currentSentence.Len() > 0 {
		sentences = append(sentences, strings.TrimSpace(currentSentence.String()))
	}

	return sentences
}

// DefaultTokenCounter provides a simple word-based token counting implementation.
// It splits text on whitespace to approximate token counts. This is suitable
// for basic use cases but may not accurately reflect subword tokenization
// used by language models.
type DefaultTokenCounter struct{}

// Count returns the number of words in the text, using whitespace as a delimiter.
func (dtc *DefaultTokenCounter) Count(text string) int {
	return len(strings.Fields(text))
}

// TikTokenCounter provides accurate token counting using the tiktoken library,
// which implements the tokenization schemes used by OpenAI models.
type TikTokenCounter struct {
	tke *tiktoken.Tiktoken
}

// NewTikTokenCounter creates a new TikTokenCounter using the specified encoding.
// Common encodings include:
// - "cl100k_base" (GPT-4, ChatGPT)
// - "p50k_base" (GPT-3)
// - "r50k_base" (Codex)
func NewTikTokenCounter(encoding string) (*TikTokenCounter, error) {
	tke, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding: %w", err)
	}
	return &TikTokenCounter{tke: tke}, nil
}

// Count returns the exact number of tokens in the text according to the
// specified tiktoken encoding.
func (ttc *TikTokenCounter) Count(text string) int {
	return len(ttc.tke.Encode(text, nil, nil))
}

// max returns the larger of two integers.
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

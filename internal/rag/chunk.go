package rag

import (
	"fmt"
	"strings"

	"github.com/pkoukk/tiktoken-go"
)

// Chunk represents a piece of text with metadata
type Chunk struct {
	Text          string
	TokenSize     int
	StartSentence int
	EndSentence   int
}

// Chunker defines the interface for text chunking
type Chunker interface {
	Chunk(text string) []Chunk
}

// TokenCounter defines the interface for counting tokens in a string
type TokenCounter interface {
	Count(text string) int
}

// TextChunker is an implementation of Chunker with advanced features
type TextChunker struct {
	ChunkSize        int
	ChunkOverlap     int
	TokenCounter     TokenCounter
	SentenceSplitter func(string) []string
}

// NewTextChunker creates a new TextChunker with the given options
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

// TextChunkerOption is a function type for configuring TextChunker
type TextChunkerOption func(*TextChunker)

// Chunk splits the input text into chunks
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

// estimateOverlapSentences estimates the number of sentences needed for the desired token overlap
func (tc *TextChunker) estimateOverlapSentences(sentences []string, endSentence, desiredOverlap int) int {
	overlapTokens := 0
	overlapSentences := 0
	for i := endSentence - 1; i >= 0 && overlapTokens < desiredOverlap; i-- {
		overlapTokens += tc.TokenCounter.Count(sentences[i])
		overlapSentences++
	}
	return overlapSentences
}

// DefaultSentenceSplitter splits text into sentences (simplified version)
func DefaultSentenceSplitter(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return r == '.' || r == '!' || r == '?'
	})
}

// SmartSentenceSplitter is a more advanced sentence splitter that handles various punctuation and edge cases
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

// DefaultTokenCounter is a simple word-based token counter
type DefaultTokenCounter struct{}

func (dtc *DefaultTokenCounter) Count(text string) int {
	return len(strings.Fields(text))
}

// TikTokenCounter is a token counter that uses the tiktoken library
type TikTokenCounter struct {
	tke *tiktoken.Tiktoken
}

// NewTikTokenCounter creates a new TikTokenCounter with the specified encoding
func NewTikTokenCounter(encoding string) (*TikTokenCounter, error) {
	tke, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return nil, fmt.Errorf("failed to get encoding: %w", err)
	}
	return &TikTokenCounter{tke: tke}, nil
}

func (ttc *TikTokenCounter) Count(text string) int {
	return len(ttc.tke.Encode(text, nil, nil))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

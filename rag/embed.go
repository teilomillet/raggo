// Package rag provides functionality for converting text into vector embeddings
// using various embedding providers (e.g., OpenAI, Cohere, local models).
package rag

import (
	"context"
	"fmt"

	"github.com/teilomillet/raggo/rag/providers"
)

// EmbedderConfig holds the configuration for creating an Embedder instance.
// It supports multiple embedding providers and their specific options.
type EmbedderConfig struct {
	// Provider specifies the embedding service to use (e.g., "openai", "cohere")
	Provider string
	// Options contains provider-specific configuration parameters
	Options map[string]interface{}
}

// EmbedderOption is a function type for configuring the EmbedderConfig.
// It follows the functional options pattern for clean and flexible configuration.
type EmbedderOption func(*EmbedderConfig)

// SetProvider sets the provider for the Embedder.
// Common providers include:
// - "openai": OpenAI's text-embedding-ada-002 and other models
// - "cohere": Cohere's embedding models
// - "local": Local embedding models
func SetProvider(provider string) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Provider = provider
	}
}

// SetModel sets the specific model to use for embedding.
// The available models depend on the chosen provider.
// Examples:
// - OpenAI: "text-embedding-ada-002"
// - Cohere: "embed-multilingual-v2.0"
func SetModel(model string) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Options["model"] = model
	}
}

// SetAPIKey sets the authentication key for the embedding service.
// This is required for most cloud-based embedding providers.
func SetAPIKey(apiKey string) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Options["api_key"] = apiKey
	}
}

// SetOption sets a custom option for the Embedder.
// This allows for provider-specific configuration options
// that aren't covered by the standard options.
func SetOption(key string, value interface{}) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Options[key] = value
	}
}

// NewEmbedder creates a new Embedder instance based on the provided options.
// It uses the provider factory system to instantiate the appropriate embedder
// implementation. Returns an error if:
// - No provider is specified
// - The specified provider is not registered
// - The provider factory fails to create an embedder
func NewEmbedder(opts ...EmbedderOption) (providers.Embedder, error) {
	config := &EmbedderConfig{
		Options: make(map[string]interface{}),
	}
	for _, opt := range opts {
		opt(config)
	}
	if config.Provider == "" {
		return nil, fmt.Errorf("provider must be specified")
	}
	factory, err := providers.GetEmbedderFactory(config.Provider)
	if err != nil {
		return nil, err
	}
	return factory(config.Options)
}

// EmbeddedChunk represents a chunk of text along with its vector embeddings
// and associated metadata. This is the core data structure for storing
// and retrieving embedded content.
type EmbeddedChunk struct {
	// Text is the original text content that was embedded
	Text string `json:"text"`
	// Embeddings maps embedding types to their vector representations
	// Multiple embeddings can exist for different models or purposes
	Embeddings map[string][]float64 `json:"embeddings"`
	// Metadata stores additional information about the chunk
	// This can include source document info, timestamps, etc.
	Metadata map[string]interface{} `json:"metadata"`
}

// EmbeddingService handles the process of converting text chunks into
// vector embeddings. It encapsulates the embedding provider and provides
// a high-level interface for embedding operations.
type EmbeddingService struct {
	embedder providers.Embedder
}

// NewEmbeddingService creates a new embedding service with the specified embedder.
// The embedder must be properly configured and ready to generate embeddings.
func NewEmbeddingService(embedder providers.Embedder) *EmbeddingService {
	return &EmbeddingService{embedder: embedder}
}

// EmbedChunks processes a slice of text chunks and generates embeddings for each one.
// It handles the embedding process in sequence, with debug output for monitoring.
// The function:
// 1. Allocates space for the results
// 2. Processes each chunk through the embedder
// 3. Creates EmbeddedChunk instances with the results
// 4. Provides progress information via debug output
//
// Returns an error if any chunk fails to embed properly.
func (s *EmbeddingService) EmbedChunks(ctx context.Context, chunks []Chunk) ([]EmbeddedChunk, error) {
	embeddedChunks := make([]EmbeddedChunk, 0, len(chunks))

	// Debug output
	fmt.Printf("Processing %d chunks for embedding\n", len(chunks))

	for i, chunk := range chunks {
		// Debug output for each chunk
		fmt.Printf("Processing chunk %d/%d (length: %d)\n", i+1, len(chunks), len(chunk.Text))
		fmt.Printf("Chunk preview: %s\n", truncateString(chunk.Text, 100))

		embedding, err := s.embedder.Embed(ctx, chunk.Text)
		if err != nil {
			return nil, fmt.Errorf("error embedding chunk %d: %w", i+1, err)
		}

		embeddedChunk := EmbeddedChunk{
			Text: chunk.Text,
			Embeddings: map[string][]float64{
				"default": embedding,
			},
			Metadata: map[string]interface{}{
				"token_size":     chunk.TokenSize,
				"start_sentence": chunk.StartSentence,
				"end_sentence":   chunk.EndSentence,
				"chunk_index":    i,
			},
		}
		embeddedChunks = append(embeddedChunks, embeddedChunk)

		// Debug output for successful embedding
		fmt.Printf("Successfully embedded chunk %d (embedding size: %d)\n", i+1, len(embedding))
	}

	return embeddedChunks, nil
}

// truncateString shortens a string to the specified length, adding an ellipsis
// if the string was truncated. This is used for debug output to keep log
// messages readable.
func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

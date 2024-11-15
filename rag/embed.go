package rag

import (
	"context"
	"fmt"

	"github.com/teilomillet/raggo/rag/providers"
)

// EmbedderConfig holds the configuration for creating an Embedder
type EmbedderConfig struct {
	Provider string
	Options  map[string]interface{}
}

// EmbedderOption is a function type for configuring the EmbedderConfig
type EmbedderOption func(*EmbedderConfig)

// SetProvider sets the provider for the Embedder
func SetProvider(provider string) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Provider = provider
	}
}

// SetModel sets the model for the Embedder
func SetModel(model string) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Options["model"] = model
	}
}

// SetAPIKey sets the API key for the Embedder
func SetAPIKey(apiKey string) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Options["api_key"] = apiKey
	}
}

// SetOption sets a custom option for the Embedder
func SetOption(key string, value interface{}) EmbedderOption {
	return func(c *EmbedderConfig) {
		c.Options[key] = value
	}
}

// NewEmbedder creates a new Embedder instance based on the provided options
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

// EmbeddedChunk represents a chunk of text with its embeddings and metadata
type EmbeddedChunk struct {
	Text       string                 `json:"text"`
	Embeddings map[string][]float64   `json:"embeddings"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// EmbeddingService handles the embedding process
type EmbeddingService struct {
	embedder providers.Embedder
}

// NewEmbeddingService creates a new embedding service with a single embedder
func NewEmbeddingService(embedder providers.Embedder) *EmbeddingService {
	return &EmbeddingService{embedder: embedder}
}

// EmbedChunks embeds a slice of chunks
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

func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

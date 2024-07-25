package raggo

import (
	"context"

	"github.com/teilomillet/raggo/internal/rag"
	"github.com/teilomillet/raggo/internal/rag/providers"
)

// EmbeddedChunk represents a chunk of text with its embedding and metadata
type EmbeddedChunk = rag.EmbeddedChunk

// EmbedderOption is a function type for configuring the Embedder
type EmbedderOption = rag.EmbedderOption

// SetProvider sets the provider for the Embedder
func SetProvider(provider string) EmbedderOption {
	return rag.SetProvider(provider)
}

// SetModel sets the model for the Embedder
func SetModel(model string) EmbedderOption {
	return rag.SetModel(model)
}

// SetAPIKey sets the API key for the Embedder
func SetAPIKey(apiKey string) EmbedderOption {
	return rag.SetAPIKey(apiKey)
}

// SetOption sets a custom option for the Embedder
func SetOption(key string, value interface{}) EmbedderOption {
	return rag.SetOption(key, value)
}

// Embedder interface defines the contract for embedding implementations
type Embedder = providers.Embedder

// NewEmbedder creates a new Embedder instance based on the provided options
func NewEmbedder(opts ...EmbedderOption) (Embedder, error) {
	return rag.NewEmbedder(opts...)
}

// EmbeddingService handles the embedding process
type EmbeddingService struct {
	service *rag.EmbeddingService
}

// NewEmbeddingService creates a new embedding service
func NewEmbeddingService(embedder Embedder) *EmbeddingService {
	return &EmbeddingService{service: rag.NewEmbeddingService(embedder)}
}

// EmbedChunks embeds a slice of chunks
func (s *EmbeddingService) EmbedChunks(ctx context.Context, chunks []Chunk) ([]EmbeddedChunk, error) {
	return s.service.EmbedChunks(ctx, chunks)
}

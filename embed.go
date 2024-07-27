package raggo

import (
	"context"
	"fmt"

	"github.com/teilomillet/raggo/internal/rag"
	"github.com/teilomillet/raggo/internal/rag/providers"
)

// EmbeddedChunk represents a chunk of text with its embeddings and metadata
type EmbeddedChunk = rag.EmbeddedChunk

// EmbeddedChunk represents a chunk of text with its embeddings and metadata
//
//	type EmbeddedChunk struct {
//		Text       string
//		Embeddings map[string][]float64
//		Metadata   map[string]interface{}
//	}
//
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
	embedders map[string]Embedder
}

// NewEmbeddingService creates a new embedding service
func NewEmbeddingService(embedder Embedder) *EmbeddingService {
	return &EmbeddingService{
		embedders: map[string]Embedder{"default": embedder},
	}
}

// EmbedChunks embeds a slice of chunks
func (s *EmbeddingService) EmbedChunks(ctx context.Context, chunks []rag.Chunk) ([]rag.EmbeddedChunk, error) {
	embeddedChunks := make([]rag.EmbeddedChunk, 0, len(chunks))
	for _, chunk := range chunks {
		embeddings := make(map[string][]float64)
		for field, embedder := range s.embedders {
			embedding, err := embedder.Embed(ctx, chunk.Text)
			if err != nil {
				return nil, fmt.Errorf("error embedding chunk for field %s: %w", field, err)
			}
			embeddings[field] = embedding
		}
		embeddedChunk := rag.EmbeddedChunk{
			Text:       chunk.Text,
			Embeddings: embeddings,
			Metadata: map[string]interface{}{
				"token_size":     chunk.TokenSize,
				"start_sentence": chunk.StartSentence,
				"end_sentence":   chunk.EndSentence,
			},
		}
		embeddedChunks = append(embeddedChunks, embeddedChunk)
	}
	return embeddedChunks, nil
}

// Package raggo provides a high-level interface for text embedding and retrieval
// operations in RAG (Retrieval-Augmented Generation) systems. It simplifies the
// process of converting text into vector embeddings using various providers.
package raggo

import (
	"context"
	"fmt"

	"github.com/teilomillet/raggo/rag"
	"github.com/teilomillet/raggo/rag/providers"
)

// EmbeddedChunk represents a chunk of text with its embeddings and metadata.
// It serves as the core data structure for storing and retrieving embedded content
// in the RAG system.
//
// Structure:
//   - Text: The original text content that was embedded
//   - Embeddings: Vector representations from different models/providers
//   - Metadata: Additional context and information about the chunk
//
// Example:
//
//	chunk := EmbeddedChunk{
//	    Text: "Sample text content",
//	    Embeddings: map[string][]float64{
//	        "default": []float64{0.1, 0.2, 0.3},
//	    },
//	    Metadata: map[string]interface{}{
//	        "source": "document1.txt",
//	        "timestamp": time.Now(),
//	    },
//	}
type EmbeddedChunk = rag.EmbeddedChunk

// EmbedderOption is a function type for configuring the Embedder.
// It follows the functional options pattern to provide a clean and
// flexible configuration API.
//
// Common options include:
//   - SetEmbedderProvider: Choose the embedding service provider
//   - SetEmbedderModel: Select the specific embedding model
//   - SetEmbedderAPIKey: Configure authentication
//   - SetOption: Set custom provider-specific options
type EmbedderOption = rag.EmbedderOption

// SetEmbedderProvider sets the provider for the Embedder.
// Supported providers include:
//   - "openai": OpenAI's text-embedding-ada-002 and other models
//   - "cohere": Cohere's embedding models
//   - "local": Local embedding models (if configured)
//
// Example:
//
//	embedder, err := NewEmbedder(
//	    SetEmbedderProvider("openai"),
//	    SetEmbedderModel("text-embedding-ada-002"),
//	)
func SetEmbedderProvider(provider string) EmbedderOption {
	return rag.SetProvider(provider)
}

// SetEmbedderModel sets the specific model to use for embedding.
// Available models depend on the chosen provider:
//   - OpenAI: "text-embedding-ada-002" (recommended)
//   - Cohere: "embed-multilingual-v2.0"
//   - Local: Depends on configured models
//
// Example:
//
//	embedder, err := NewEmbedder(
//	    SetEmbedderProvider("openai"),
//	    SetEmbedderModel("text-embedding-ada-002"),
//	)
func SetEmbedderModel(model string) EmbedderOption {
	return rag.SetModel(model)
}

// SetEmbedderAPIKey sets the authentication key for the embedding service.
// This is required for most cloud-based embedding providers.
//
// Security Note: Store API keys securely and never commit them to version control.
// Consider using environment variables or secure key management systems.
//
// Example:
//
//	embedder, err := NewEmbedder(
//	    SetEmbedderProvider("openai"),
//	    SetEmbedderAPIKey(os.Getenv("OPENAI_API_KEY")),
//	)
func SetEmbedderAPIKey(apiKey string) EmbedderOption {
	return rag.SetAPIKey(apiKey)
}

// SetOption sets a custom option for the Embedder.
// This allows for provider-specific configuration that isn't covered
// by the standard options.
//
// Example:
//
//	embedder, err := NewEmbedder(
//	    SetEmbedderProvider("openai"),
//	    SetOption("timeout", 30*time.Second),
//	    SetOption("max_retries", 3),
//	)
func SetOption(key string, value interface{}) EmbedderOption {
	return rag.SetOption(key, value)
}

// Embedder interface defines the contract for embedding implementations.
// This allows for different embedding providers to be used interchangeably.
type Embedder = providers.Embedder

// NewEmbedder creates a new Embedder instance based on the provided options.
// It handles provider selection and configuration, returning a ready-to-use
// embedding interface.
//
// Returns an error if:
//   - No provider is specified
//   - The provider is not supported
//   - Configuration is invalid
//   - Authentication fails
//
// Example:
//
//	embedder, err := NewEmbedder(
//	    SetEmbedderProvider("openai"),
//	    SetEmbedderModel("text-embedding-ada-002"),
//	    SetEmbedderAPIKey(os.Getenv("OPENAI_API_KEY")),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
func NewEmbedder(opts ...EmbedderOption) (Embedder, error) {
	return rag.NewEmbedder(opts...)
}

// EmbeddingService handles the embedding process for text content.
// It supports multiple embedders for different fields or purposes,
// allowing for flexible embedding strategies.
type EmbeddingService struct {
	embedders map[string]Embedder
}

// NewEmbeddingService creates a new embedding service with the specified embedder
// as the default embedding provider.
//
// Example:
//
//	embedder, _ := NewEmbedder(SetEmbedderProvider("openai"))
//	service := NewEmbeddingService(embedder)
func NewEmbeddingService(embedder Embedder) *EmbeddingService {
	return &EmbeddingService{
		embedders: map[string]Embedder{"default": embedder},
	}
}

// EmbedChunks processes a slice of text chunks and generates embeddings for each one.
// It supports multiple embedding fields per chunk, using different embedders
// for each field if configured.
//
// The function:
//   1. Processes each chunk through configured embedders
//   2. Combines embeddings from all fields
//   3. Preserves chunk metadata
//   4. Handles errors for individual chunks
//
// Example:
//
//	chunks := []rag.Chunk{
//	    {Text: "First chunk", TokenSize: 10},
//	    {Text: "Second chunk", TokenSize: 12},
//	}
//	embedded, err := service.EmbedChunks(ctx, chunks)
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

// Embed generates embeddings for a single text string using the default embedder.
// This is a convenience method for simple embedding operations.
//
// Example:
//
//	text := "Sample text to embed"
//	embedding, err := service.Embed(ctx, text)
//	if err != nil {
//	    log.Fatal(err)
//	}
func (s *EmbeddingService) Embed(ctx context.Context, text string) ([]float64, error) {
	// Get the default embedder
	embedder, ok := s.embedders["default"]
	if !ok {
		return nil, fmt.Errorf("no default embedder found")
	}

	// Get embedding using the default embedder
	embedding, err := embedder.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("error embedding text: %w", err)
	}

	return embedding, nil
}

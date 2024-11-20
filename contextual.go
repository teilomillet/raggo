// Package raggo provides advanced Retrieval-Augmented Generation (RAG) capabilities
// with contextual awareness and memory management.
package raggo

import (
	"context"
	"fmt"
)

// ContextualStoreOptions configures how documents are processed and stored with
// contextual information. It provides settings for:
//   - Vector database collection management
//   - Document chunking and processing
//   - Embedding model configuration
//   - Batch processing controls
//
// This configuration is designed to optimize the balance between processing
// efficiency and context preservation.
type ContextualStoreOptions struct {
	// Collection specifies the vector database collection name
	Collection string

	// APIKey is the authentication key for the embedding provider
	APIKey string

	// ChunkSize determines the size of text chunks in tokens
	// Larger chunks preserve more context but use more memory
	ChunkSize int

	// ChunkOverlap controls how much text overlaps between chunks
	// More overlap helps maintain context across chunk boundaries
	ChunkOverlap int

	// BatchSize sets how many documents to process simultaneously
	// Higher values increase throughput but use more memory
	BatchSize int

	// ModelName specifies which language model to use for context generation
	// This model enriches chunks with additional contextual information
	ModelName string
}

// StoreWithContext processes documents and stores them with enhanced contextual information.
// It uses a combination of:
//   - Semantic chunking for document segmentation
//   - Language model enrichment for context generation
//   - Vector embedding for efficient retrieval
//   - Batch processing for performance
//
// The function automatically handles:
//   - Default configuration values
//   - Resource management
//   - Error handling and reporting
//   - Context-aware processing
//
// Example usage:
//
//	opts := raggo.ContextualStoreOptions{
//	    Collection: "my_docs",
//	    APIKey:     os.Getenv("OPENAI_API_KEY"),
//	    ChunkSize:  512,
//	    BatchSize:  100,
//	}
//	
//	err := raggo.StoreWithContext(ctx, "path/to/docs", opts)
func StoreWithContext(ctx context.Context, source string, opts ContextualStoreOptions) error {
	// Use default values if not specified
	if opts.ChunkSize == 0 {
		opts.ChunkSize = 512
	}
	if opts.ChunkOverlap == 0 {
		opts.ChunkOverlap = 64
	}
	if opts.BatchSize == 0 {
		opts.BatchSize = 100
	}
	if opts.ModelName == "" {
		opts.ModelName = "gpt-4o-mini"
	}

	// Initialize RAG with context-aware configuration
	rag, err := NewRAG(
		WithMilvus(opts.Collection),
		WithOpenAI(opts.APIKey),
		func(c *RAGConfig) {
			c.ChunkSize = opts.ChunkSize
			c.ChunkOverlap = opts.ChunkOverlap
			c.BatchSize = opts.BatchSize
		},
	)
	if err != nil {
		return fmt.Errorf("failed to initialize RAG: %w", err)
	}
	defer rag.Close()

	// Process and store documents with enhanced context
	return rag.ProcessWithContext(ctx, source, opts.ModelName)
}

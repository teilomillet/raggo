package raggo

import (
	"context"
	"fmt"
)

// ContextualStoreOptions holds settings for contextual document processing
type ContextualStoreOptions struct {
	Collection   string
	APIKey       string
	ChunkSize    int
	ChunkOverlap int
	BatchSize    int
	ModelName    string
}

// StoreWithContext processes documents and stores them with contextual information
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

	// Initialize RAG
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

	// Process and store documents with context
	return rag.ProcessWithContext(ctx, source, opts.ModelName)
}

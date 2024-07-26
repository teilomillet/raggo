package rag

import (
	"context"
	"fmt"
)

// Store defines the interface for vector database operations
type Store interface {
	// SaveEmbeddings stores multiple embedded chunks
	SaveEmbeddings(ctx context.Context, collectionName string, chunks []EmbeddedChunk) error

	// Search finds the most similar chunks to the given embedding
	Search(ctx context.Context, collectionName string, vector []float64, topK int) ([]SearchResult, error)

	// Close releases any resources held by the store
	Close() error
}

// StoreConfig holds the configuration for creating a Store
type StoreConfig struct {
	Type    string
	Options map[string]interface{}
}

// StoreFactory is a function type for creating Store instances
type StoreFactory func(StoreConfig) (Store, error)

var storeRegistry = make(map[string]StoreFactory)

// RegisterStore registers a new store type with its factory function
func RegisterStore(name string, factory StoreFactory) {
	storeRegistry[name] = factory
}

// NewStore creates a new Store instance based on the provided configuration
func NewStore(cfg StoreConfig) (Store, error) {
	factory, ok := storeRegistry[cfg.Type]
	if !ok {
		return nil, fmt.Errorf("unsupported store type: %s", cfg.Type)
	}
	return factory(cfg)
}

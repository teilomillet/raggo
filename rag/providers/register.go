package providers

import (
	"context"
	"fmt"
	"sync"
)

// EmbedderFactory is a function type that creates a new Embedder
type EmbedderFactory func(config map[string]interface{}) (Embedder, error)

var (
	embedderFactories = make(map[string]EmbedderFactory)
	mu                sync.RWMutex
)

// RegisterEmbedder registers a new embedder factory
func RegisterEmbedder(name string, factory EmbedderFactory) {
	mu.Lock()
	defer mu.Unlock()
	embedderFactories[name] = factory
}

// GetEmbedderFactory returns the factory for the given embedder name
func GetEmbedderFactory(name string) (EmbedderFactory, error) {
	mu.RLock()
	defer mu.RUnlock()
	factory, ok := embedderFactories[name]
	if !ok {
		return nil, fmt.Errorf("embedder not found: %s", name)
	}
	return factory, nil
}

// Embedder interface defines the contract for embedding implementations
type Embedder interface {
	// Embed generates embeddings for the given text
	Embed(ctx context.Context, text string) ([]float64, error)

	// GetDimension returns the dimension of the embeddings for the current model
	GetDimension() (int, error)
}

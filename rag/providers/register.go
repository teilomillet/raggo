// Package providers implements a flexible system for managing different embedding
// service providers in the Raggo framework. Each provider offers unique capabilities
// for converting text into vector representations that capture semantic meaning.
// The registration system allows new providers to be easily added and configured
// while maintaining a consistent interface for the rest of the system.
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

// Provider defines the interface that all embedding providers must implement.
// This abstraction ensures that different providers can be used interchangeably
// while providing their own specific implementation details. A provider is
// responsible for converting text into vector representations that can be used
// for semantic similarity search.
type Provider interface {
	// Embed converts a slice of text inputs into their vector representations.
	// The method should handle batching and rate limiting internally. It returns
	// a slice of vectors, where each vector corresponds to the input text at the
	// same index. An error is returned if the embedding process fails.
	Embed(inputs []string) ([][]float32, error)

	// Close releases any resources held by the provider, such as API connections
	// or cached data. This method should be called when the provider is no longer
	// needed to prevent resource leaks.
	Close() error
}

// Config holds the configuration settings for an embedding provider.
// Different providers may use different subsets of these settings, but
// the configuration structure remains consistent to simplify provider
// management and initialization.
type Config struct {
	// APIKey is used for authentication with the provider's service.
	// For local models, this may be left empty.
	APIKey string

	// Model specifies which embedding model to use. Each provider may
	// offer multiple models with different characteristics.
	Model string

	// BatchSize determines how many texts can be embedded in a single API call.
	// This helps optimize performance and manage rate limits.
	BatchSize int

	// Dimension specifies the size of the output vectors. This must match
	// the chosen model's output dimension.
	Dimension int

	// Additional provider-specific settings can be added here
	Settings map[string]interface{}
}

// registry maintains a thread-safe map of provider factories. Each factory
// is a function that creates a new instance of a specific provider type
// using the provided configuration.
type registry struct {
	mu        sync.RWMutex
	factories map[string]func(cfg *Config) (Provider, error)
}

// The global registry instance that maintains all registered provider factories.
var globalRegistry = &registry{
	factories: make(map[string]func(cfg *Config) (Provider, error)),
}

// Register adds a new provider factory to the global registry. The factory
// function should create and configure a new instance of the provider when
// called. If a provider with the same name already exists, it will be
// overwritten, allowing for provider updates and replacements.
func Register(name string, factory func(cfg *Config) (Provider, error)) {
	globalRegistry.mu.Lock()
	defer globalRegistry.mu.Unlock()
	globalRegistry.factories[name] = factory
}

// Get retrieves a provider factory from the registry and creates a new provider
// instance using the supplied configuration. If the requested provider is not
// found in the registry, an error is returned. This method is thread-safe and
// can be called from multiple goroutines.
func Get(name string, cfg *Config) (Provider, error) {
	globalRegistry.mu.RLock()
	factory, ok := globalRegistry.factories[name]
	globalRegistry.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("provider not found: %s", name)
	}

	return factory(cfg)
}

// List returns the names of all registered providers. This is useful for
// discovering available providers and validating provider names before
// attempting to create instances.
func List() []string {
	globalRegistry.mu.RLock()
	defer globalRegistry.mu.RUnlock()

	providers := make([]string, 0, len(globalRegistry.factories))
	for name := range globalRegistry.factories {
		providers = append(providers, name)
	}
	return providers
}

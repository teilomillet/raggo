// Package providers includes this example to demonstrate how to implement
// new embedding providers for the Raggo framework. This file shows the
// recommended patterns and best practices for creating a provider that
// integrates seamlessly with the system.
package providers

import (
	"fmt"
)

// ExampleProvider demonstrates how to implement a new embedding provider.
// Replace this with your actual provider implementation. Your provider
// should handle:
// - Connection management
// - Resource cleanup
// - Error handling
// - Rate limiting (if applicable)
// - Batching (if supported)
type ExampleProvider struct {
	// Add fields needed by your provider
	apiKey    string
	model     string
	dimension int
	// Add any connection or state management fields
	client interface{}
}

// NewExampleProvider shows how to create a new provider instance.
// Your initialization function should:
// 1. Validate the configuration
// 2. Set up any connections or resources
// 3. Initialize internal state
// 4. Return a fully configured provider
func NewExampleProvider(cfg *Config) (*ExampleProvider, error) {
	// Validate required configuration
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	// Initialize your provider
	provider := &ExampleProvider{
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		dimension: cfg.Dimension,
	}

	// Set up any connections or resources
	// Example:
	// client, err := yourapi.NewClient(cfg.APIKey)
	// if err != nil {
	//     return nil, fmt.Errorf("failed to create client: %w", err)
	// }
	// provider.client = client

	return provider, nil
}

// Embed generates embeddings for a batch of input texts.
// Your implementation should:
// 1. Validate the inputs
// 2. Prepare the batch request
// 3. Call your embedding service
// 4. Handle errors appropriately
// 5. Return the vector representations as float32 arrays
//
// Note: The method accepts a slice of strings and returns a slice of float32 vectors
// to match the Provider interface requirements.
func (p *ExampleProvider) Embed(texts []string) ([][]float32, error) {
	// Validate input
	if len(texts) == 0 {
		return nil, fmt.Errorf("empty input texts")
	}

	// Initialize result slice
	result := make([][]float32, len(texts))

	// Process each text in the batch
	for i, text := range texts {
		if text == "" {
			return nil, fmt.Errorf("empty text at position %d", i)
		}

		// Call your embedding service
		// Example:
		// response, err := p.client.CreateEmbedding(&Request{
		//     Text:  text,
		//     Model: p.model,
		// })
		// if err != nil {
		//     return nil, fmt.Errorf("embedding creation failed for text %d: %w", i, err)
		// }

		// For this example, return a mock vector
		mockVector := make([]float32, p.dimension)
		for j := range mockVector {
			mockVector[j] = 0.1 // Replace with actual embedding values
		}
		result[i] = mockVector
	}

	return result, nil
}

// GetDimension demonstrates how to implement the dimension reporting method.
// Your implementation should:
// 1. Return the correct dimension for your model
// 2. Handle any model-specific variations
// 3. Return an error if the dimension cannot be determined
func (p *ExampleProvider) GetDimension() (int, error) {
	if p.dimension == 0 {
		return 0, fmt.Errorf("dimension not set")
	}
	return p.dimension, nil
}

// Close demonstrates how to implement resource cleanup.
// Your implementation should:
// 1. Close any open connections
// 2. Release any held resources
// 3. Handle cleanup errors appropriately
func (p *ExampleProvider) Close() error {
	// Clean up your resources
	// Example:
	// if p.client != nil {
	//     if err := p.client.Close(); err != nil {
	//         return fmt.Errorf("failed to close client: %w", err)
	//     }
	// }
	return nil
}

func init() {
	// Register your provider with a unique name
	Register("example", func(cfg *Config) (Provider, error) {
		return NewExampleProvider(cfg)
	})
}

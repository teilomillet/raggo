// Package config provides a flexible configuration management system for the Raggo
// Retrieval-Augmented Generation (RAG) framework. It handles configuration loading,
// validation, and persistence with support for multiple sources:
//   - Configuration files (JSON)
//   - Environment variables
//   - Programmatic defaults
//
// The package implements a hierarchical configuration system where settings can be
// overridden in the following order (highest to lowest precedence):
//   1. Environment variables
//   2. Configuration file
//   3. Default values
package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Config holds all configuration for the RAG system. It provides a centralized
// way to manage settings across different components of the system.
//
// Configuration categories:
//   - Provider settings: Embedding and service providers
//   - Collection settings: Vector database collections
//   - Search settings: Retrieval and ranking parameters
//   - Document processing: Text chunking and batching
//   - Vector store: Database-specific configuration
//   - System settings: Timeouts, retries, and headers
type Config struct {
	// Provider settings configure the embedding and service providers
	Provider string            // Service provider (e.g., "milvus", "openai")
	Model    string            // Model identifier for embeddings
	APIKeys  map[string]string // API keys for different providers

	// Collection settings define the vector database structure
	Collection string // Name of the vector collection

	// Search settings control retrieval behavior and ranking
	SearchStrategy      string                 // Search method (e.g., "dense", "hybrid")
	DefaultTopK        int                    // Default number of results to return
	DefaultMinScore    float64                // Minimum similarity score threshold
	DefaultSearchParams map[string]interface{} // Additional search parameters
	EnableReRanking    bool                   // Enable result re-ranking
	RRFConstant        float64                // Reciprocal Rank Fusion constant

	// Document processing settings for text handling
	DefaultChunkSize    int // Size of text chunks
	DefaultChunkOverlap int // Overlap between consecutive chunks
	DefaultBatchSize    int // Number of items per processing batch
	DefaultIndexType    string // Type of vector index (e.g., "HNSW")

	// Vector store settings for database configuration
	VectorDBConfig map[string]interface{} // Database-specific settings

	// Timeouts and retries for system operations
	Timeout    time.Duration // Operation timeout
	MaxRetries int          // Maximum retry attempts

	// Additional settings for extended functionality
	ExtraHeaders map[string]string // Additional HTTP headers
}

// LoadConfig loads configuration from multiple sources, combining them according
// to the precedence rules. It automatically searches for configuration files in
// standard locations and applies environment variable overrides.
//
// Configuration file search paths:
//   1. $RAGGO_CONFIG environment variable
//   2. ~/.raggo/config.json
//   3. ~/.config/raggo/config.json
//   4. ./raggo.json
//
// Environment variable overrides:
//   - RAGGO_PROVIDER: Service provider
//   - RAGGO_MODEL: Model identifier
//   - RAGGO_COLLECTION: Collection name
//   - RAGGO_API_KEY: Default API key
//
// Example usage:
//
//	cfg, err := config.LoadConfig()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Printf("Using provider: %s\n", cfg.Provider)
func LoadConfig() (*Config, error) {
	// Default configuration with production-ready settings
	cfg := &Config{
		Provider:            "milvus",           // Fast, open-source vector database
		Model:              "text-embedding-3-small", // Latest OpenAI embedding model
		Collection:         "documents",         // Default collection name
		SearchStrategy:     "dense",            // Pure vector similarity search
		DefaultTopK:        5,                  // Conservative number of results
		DefaultMinScore:    0.7,                // High confidence threshold
		DefaultChunkSize:   512,                // Balanced chunk size
		DefaultChunkOverlap: 50,                // Moderate overlap
		DefaultBatchSize:   100,                // Efficient batch size
		DefaultIndexType:   "HNSW",             // Fast approximate search
		DefaultSearchParams: map[string]interface{}{
			"ef": 64,                           // HNSW search depth
		},
		EnableReRanking: false,                 // Disabled by default
		RRFConstant:     60,                    // Standard RRF constant
		Timeout:         30 * time.Second,      // Conservative timeout
		MaxRetries:      3,                     // Reasonable retry count
		APIKeys:         make(map[string]string),
		ExtraHeaders:    make(map[string]string),
		VectorDBConfig:  make(map[string]interface{}),
	}

	// Try to load from config file
	configFile := os.Getenv("RAGGO_CONFIG")
	if configFile == "" {
		// Try default locations
		home, err := os.UserHomeDir()
		if err == nil {
			candidates := []string{
				filepath.Join(home, ".raggo", "config.json"),
				filepath.Join(home, ".config", "raggo", "config.json"),
				"raggo.json",
			}

			for _, candidate := range candidates {
				if _, err := os.Stat(candidate); err == nil {
					configFile = candidate
					break
				}
			}
		}
	}

	if configFile != "" {
		data, err := os.ReadFile(configFile)
		if err == nil {
			if err := json.Unmarshal(data, cfg); err != nil {
				return nil, err
			}
		}
	}

	// Override with environment variables
	if provider := os.Getenv("RAGGO_PROVIDER"); provider != "" {
		cfg.Provider = provider
	}
	if model := os.Getenv("RAGGO_MODEL"); model != "" {
		cfg.Model = model
	}
	if collection := os.Getenv("RAGGO_COLLECTION"); collection != "" {
		cfg.Collection = collection
	}
	if apiKey := os.Getenv("RAGGO_API_KEY"); apiKey != "" {
		cfg.APIKeys[cfg.Provider] = apiKey
	}

	return cfg, nil
}

// Save persists the configuration to a JSON file at the specified path.
// It creates any necessary parent directories and sets appropriate file
// permissions.
//
// Example usage:
//
//	cfg := &Config{
//	    Provider: "milvus",
//	    Model:    "text-embedding-3-small",
//	}
//	err := cfg.Save("~/.raggo/config.json")
//	if err != nil {
//	    log.Fatal(err)
//	}
func (c *Config) Save(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}

	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

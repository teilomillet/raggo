package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// Config holds all configuration for the RAG system
type Config struct {
	// Provider settings
	Provider string
	Model    string
	APIKeys  map[string]string

	// Collection settings
	Collection string

	// Search settings
	SearchStrategy      string
	DefaultTopK        int
	DefaultMinScore    float64
	DefaultSearchParams map[string]interface{}
	EnableReRanking    bool
	RRFConstant        float64

	// Document processing settings
	DefaultChunkSize    int
	DefaultChunkOverlap int
	DefaultBatchSize    int
	DefaultIndexType    string

	// Vector store settings
	VectorDBConfig map[string]interface{}

	// Timeouts and retries
	Timeout    time.Duration
	MaxRetries int

	// Additional settings
	ExtraHeaders map[string]string
}

// LoadConfig loads configuration from a file or environment
func LoadConfig() (*Config, error) {
	// Default configuration
	cfg := &Config{
		Provider:            "milvus",
		Model:              "text-embedding-3-small",
		Collection:         "documents",
		SearchStrategy:     "dense",
		DefaultTopK:        5,
		DefaultMinScore:    0.7,
		DefaultChunkSize:   512,
		DefaultChunkOverlap: 50,
		DefaultBatchSize:   100,
		DefaultIndexType:   "HNSW",
		DefaultSearchParams: map[string]interface{}{
			"ef": 64,
		},
		EnableReRanking: false,
		RRFConstant:     60,
		Timeout:         30 * time.Second,
		MaxRetries:      3,
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

// Save saves the configuration to a file
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

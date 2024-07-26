package rag

import (
	"context"
	"fmt"
)

// VectorDB represents a vector database
type VectorDB interface {
	SaveEmbeddings(ctx context.Context, collectionName string, chunks []EmbeddedChunk) error
	Search(ctx context.Context, collectionName string, query []float64, limit int, param SearchParam) ([]SearchResult, error)
	Close() error
}

// VectorDBConfig holds the configuration for creating a VectorDB
type VectorDBConfig struct {
	Type      string
	Address   string
	Dimension int
	Options   map[string]interface{}
}

// VectorDBOption is a function type for configuring VectorDBConfig
type VectorDBOption func(*VectorDBConfig)

// SetVectorDBType sets the type of vector database
func SetVectorDBType(dbType string) VectorDBOption {
	return func(c *VectorDBConfig) {
		c.Type = dbType
	}
}

// SetVectorDBAddress sets the address for the vector database
func SetVectorDBAddress(address string) VectorDBOption {
	return func(c *VectorDBConfig) {
		c.Address = address
	}
}

// SetVectorDBDimension sets the dimension for the vector database
func SetVectorDBDimension(dimension int) VectorDBOption {
	return func(c *VectorDBConfig) {
		c.Dimension = dimension
	}
}

// SetVectorDBOption sets a custom option for the vector database
func SetVectorDBOption(key string, value interface{}) VectorDBOption {
	return func(c *VectorDBConfig) {
		if c.Options == nil {
			c.Options = make(map[string]interface{})
		}
		c.Options[key] = value
	}
}

// VectorDBFactory is a function type for creating VectorDB instances
type VectorDBFactory func(VectorDBConfig) (VectorDB, error)

var vectorDBRegistry = make(map[string]VectorDBFactory)

// RegisterVectorDB registers a new vector database type with its factory function
func RegisterVectorDB(name string, factory VectorDBFactory) {
	vectorDBRegistry[name] = factory
}

// NewVectorDB creates a new VectorDB instance based on the provided configuration
func NewVectorDB(opts ...VectorDBOption) (VectorDB, error) {
	config := &VectorDBConfig{
		Dimension: 1536, // Default dimension
		Options:   make(map[string]interface{}),
	}
	for _, opt := range opts {
		opt(config)
	}

	if config.Type == "" {
		return nil, fmt.Errorf("vector database type must be specified")
	}

	factory, ok := vectorDBRegistry[config.Type]
	if !ok {
		return nil, fmt.Errorf("unsupported vector database type: %s", config.Type)
	}

	return factory(*config)
}

// SearchParam interface for search-related parameters
type SearchParam interface {
	Params() map[string]interface{}
}

// DefaultSearchParam provides a basic implementation of SearchParam
type DefaultSearchParam struct {
	params map[string]interface{}
}

func NewDefaultSearchParam() *DefaultSearchParam {
	return &DefaultSearchParam{
		params: make(map[string]interface{}),
	}
}

func (d *DefaultSearchParam) Params() map[string]interface{} {
	return d.params
}

func (d *DefaultSearchParam) SetParam(key string, value interface{}) {
	d.params[key] = value
}

// SearchResult represents a single search result
type SearchResult struct {
	ID       interface{}
	Score    float64
	Text     string
	Metadata map[string]interface{}
}

package rag

import (
	"context"
	"fmt"
)

// VectorDB represents a vector database
type VectorDB interface {
	SaveEmbeddings(ctx context.Context, collectionName string, chunks []EmbeddedChunk) error
	Search(ctx context.Context, collectionName string, query []float64, limit int, param SearchParam) ([]SearchResult, error)
	HybridSearch(ctx context.Context, collectionName string, queries map[string][]float64, fields []string, limit int, param SearchParam) ([]SearchResult, error)
	ValidateQueryFields(ctx context.Context, collectionName string, queryFields []string) error
	Close() error
}

// VectorDBConfig holds the configuration for creating a VectorDB
type VectorDBConfig struct {
	Type      string
	Address   string
	Dimension int
	Options   map[string]interface{}
}

// VectorDBFactory is a function type for creating VectorDB instances
type VectorDBFactory func(VectorDBConfig) (VectorDB, error)

var vectorDBRegistry = make(map[string]VectorDBFactory)

// RegisterVectorDB registers a new vector database type with its factory function
func RegisterVectorDB(name string, factory VectorDBFactory) {
	vectorDBRegistry[name] = factory
}

// NewVectorDB creates a new VectorDB instance based on the provided configuration
func NewVectorDB(config VectorDBConfig) (VectorDB, error) {
	if config.Type == "" {
		return nil, fmt.Errorf("vector database type must be specified")
	}

	factory, ok := vectorDBRegistry[config.Type]
	if !ok {
		return nil, fmt.Errorf("unsupported vector database type: %s", config.Type)
	}

	return factory(config)
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
	Fields   map[string]interface{} // New field to store additional returned fields
}

// VectorDBError represents an error that occurred during a vector database operation
type VectorDBError struct {
	Op  string
	Err error
}

func (e *VectorDBError) Error() string {
	return fmt.Sprintf("vectordb operation %s failed: %v", e.Op, e.Err)
}

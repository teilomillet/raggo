package raggo

import (
	"context"

	"github.com/teilomillet/raggo/internal/rag"
	_ "github.com/teilomillet/raggo/internal/rag/vectordb"
)

// VectorDB represents a vector database
type VectorDB interface {
	SaveEmbeddings(ctx context.Context, collectionName string, chunks []EmbeddedChunk) error
	Search(ctx context.Context, collectionName string, query []float64, limit int, param SearchParam) ([]SearchResult, error)
	HybridSearch(ctx context.Context, collectionName string, queries map[string][]float64, fields []string, limit int, param SearchParam) ([]SearchResult, error)
	ValidateQueryFields(ctx context.Context, collectionName string, queryFields []string) error
	Close() error
}

// SearchParam interface for search-related parameters
type SearchParam = rag.SearchParam

// SearchResult represents a single search result
type SearchResult = rag.SearchResult

// NewDefaultSearchParam creates a new DefaultSearchParam
func NewDefaultSearchParam() SearchParam {
	return rag.NewDefaultSearchParam()
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

// NewVectorDB creates a new VectorDB instance based on the provided configuration
func NewVectorDB(opts ...VectorDBOption) (VectorDB, error) {
	config := &VectorDBConfig{
		Dimension: 1536, // Default dimension
		Options:   make(map[string]interface{}),
	}
	for _, opt := range opts {
		opt(config)
	}

	internalDB, err := rag.NewVectorDB(rag.VectorDBConfig{
		Type:      config.Type,
		Address:   config.Address,
		Dimension: config.Dimension,
		Options:   config.Options,
	})
	if err != nil {
		return nil, err
	}

	return &vectorDBWrapper{internalDB}, nil
}

// vectorDBWrapper wraps the internal VectorDB to match the public interface
type vectorDBWrapper struct {
	internal rag.VectorDB
}

func (w *vectorDBWrapper) SaveEmbeddings(ctx context.Context, collectionName string, chunks []EmbeddedChunk) error {
	return w.internal.SaveEmbeddings(ctx, collectionName, chunks)
}

func (w *vectorDBWrapper) Search(ctx context.Context, collectionName string, query []float64, limit int, param SearchParam) ([]SearchResult, error) {
	return w.internal.Search(ctx, collectionName, query, limit, param)
}

func (w *vectorDBWrapper) HybridSearch(ctx context.Context, collectionName string, queries map[string][]float64, fields []string, limit int, param SearchParam) ([]SearchResult, error) {
	return w.internal.HybridSearch(ctx, collectionName, queries, fields, limit, param)
}

func (w *vectorDBWrapper) ValidateQueryFields(ctx context.Context, collectionName string, queryFields []string) error {
	return w.internal.ValidateQueryFields(ctx, collectionName, queryFields)
}

func (w *vectorDBWrapper) Close() error {
	return w.internal.Close()
}

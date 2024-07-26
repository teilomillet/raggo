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
	Close() error
}

// VectorDBOption is a function type for configuring VectorDB
type VectorDBOption = rag.VectorDBOption

// SetType sets the type of vector database
func SetType(dbType string) VectorDBOption {
	return rag.SetVectorDBType(dbType)
}

// SetAddress sets the address for the vector database
func SetAddress(address string) VectorDBOption {
	return rag.SetVectorDBAddress(address)
}

// SetDimension sets the dimension for the vector database
func SetDimension(dimension int) VectorDBOption {
	return rag.SetVectorDBDimension(dimension)
}

// SetOption sets a custom option for the vector database
func SetVectorDBOption(key string, value interface{}) VectorDBOption {
	return rag.SetVectorDBOption(key, value)
}

// SetMetric sets the metric type for vector similarity calculation
func SetMetric(metric string) VectorDBOption {
	return rag.SetVectorDBOption("metric", metric)
}

// SetIndexType sets the index type for Milvus
func SetIndexType(indexType string) VectorDBOption {
	return rag.SetVectorDBOption("index_type", indexType)
}

// SetIndexParams sets the index parameters for Milvus
func SetIndexParams(params map[string]interface{}) VectorDBOption {
	return rag.SetVectorDBOption("index_params", params)
}

// SetSearchParams sets the search parameters for Milvus
func SetSearchParams(params map[string]interface{}) VectorDBOption {
	return rag.SetVectorDBOption("search_params", params)
}

// NewVectorDB creates a new VectorDB instance based on the provided configuration
func NewVectorDB(opts ...VectorDBOption) (VectorDB, error) {
	return rag.NewVectorDB(opts...)
}

// SearchParam interface for search-related parameters
type SearchParam = rag.SearchParam

// DefaultSearchParam provides a basic implementation of SearchParam
type DefaultSearchParam = rag.DefaultSearchParam

// NewDefaultSearchParam creates a new DefaultSearchParam
func NewDefaultSearchParam() *DefaultSearchParam {
	return rag.NewDefaultSearchParam()
}

// SearchResult represents a single search result
type SearchResult = rag.SearchResult


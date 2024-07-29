package vectordb

import (
	"context"
)

type VectorDBConfig struct {
	Address    string
	Dimension  int
	IndexType  string
	MetricType string
	TopK       int
}

type SearchResult struct {
	ID       int64
	Score    float32
	Metadata map[string]interface{}
}

type VectorDB interface {
	Connect(ctx context.Context, config VectorDBConfig) error
	Disconnect(ctx context.Context) error
	CreateIndex(ctx context.Context, collectionName string) error
	LoadCollection(ctx context.Context, collectionName string) error
	CreateCollection(ctx context.Context, name string) error
	HasCollection(ctx context.Context, name string) (bool, error)
	DropCollection(ctx context.Context, name string) error
	Insert(ctx context.Context, collectionName string, vectors [][]float32, metadata []map[string]interface{}) error
	Search(ctx context.Context, collectionName string, queryVector []float32) ([]SearchResult, error)
	HybridSearch(ctx context.Context, collectionName string, queryVectors [][]float32) ([]SearchResult, error)
}

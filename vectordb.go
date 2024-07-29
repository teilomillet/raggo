package raggo

import (
	"context"
	"fmt"

	"github.com/teilomillet/raggo/internal/rag/vectordb"
)

// VectorDBManager provides a high-level interface for vector database operations
type VectorDBManager struct {
	db vectordb.VectorDB
}

// SearchResult represents the result of a vector search operation
type SearchResult = vectordb.SearchResult

// VectorDBConfig represents the configuration for the vector database
type VectorDBConfig struct {
	DBType     string
	Address    string
	Dimension  int
	IndexType  string
	MetricType string
	TopK       int
}

// ManagerOption is a function that modifies the VectorDBConfig
type ManagerOption func(*VectorDBConfig)

// WithAddress sets the address for the vector database
func WithAddress(address string) ManagerOption {
	return func(c *VectorDBConfig) {
		c.Address = address
	}
}

// WithDimension sets the dimension for the vectors
func WithDimension(dimension int) ManagerOption {
	return func(c *VectorDBConfig) {
		c.Dimension = dimension
	}
}

// WithIndexType sets the index type for the vector database
func WithIndexType(indexType string) ManagerOption {
	return func(c *VectorDBConfig) {
		c.IndexType = indexType
	}
}

// WithMetricType sets the metric type for the vector database
func WithMetricType(metricType string) ManagerOption {
	return func(c *VectorDBConfig) {
		c.MetricType = metricType
	}
}

// WithTopK sets the number of top results to return
func WithTopK(topK int) ManagerOption {
	return func(c *VectorDBConfig) {
		c.TopK = topK
	}
}

// NewVectorDBManager creates a new VectorDBManager with the specified database type and options
func NewVectorDBManager(dbType string, opts ...ManagerOption) (*VectorDBManager, error) {
	config := VectorDBConfig{
		DBType:     dbType,
		Dimension:  128, // Default values
		IndexType:  "HNSW",
		MetricType: "L2",
		TopK:       5,
	}

	for _, opt := range opts {
		opt(&config)
	}

	var db vectordb.VectorDB

	switch config.DBType {
	case "memory":
		db = &vectordb.InMemoryDB{}
	case "milvus":
		db = &vectordb.MilvusDB{}
	default:
		return nil, fmt.Errorf("unsupported database type: %s", config.DBType)
	}

	internalConfig := vectordb.VectorDBConfig{
		Address:    config.Address,
		Dimension:  config.Dimension,
		IndexType:  config.IndexType,
		MetricType: config.MetricType,
		TopK:       config.TopK,
	}

	ctx := context.Background()
	if err := db.Connect(ctx, internalConfig); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	return &VectorDBManager{db: db}, nil
}

// CreateCollection creates a new collection in the vector database
func (m *VectorDBManager) CreateCollection(name string) error {
	ctx := context.Background()
	return m.db.CreateCollection(ctx, name)
}

// InsertVectors inserts vectors into the specified collection
func (m *VectorDBManager) InsertVectors(collectionName string, vectors [][]float32, metadata []map[string]interface{}) error {
	ctx := context.Background()
	return m.db.Insert(ctx, collectionName, vectors, metadata)
}

// Search performs a vector search in the specified collection
func (m *VectorDBManager) Search(collectionName string, queryVector []float32) ([]vectordb.SearchResult, error) {
	ctx := context.Background()
	return m.db.Search(ctx, collectionName, queryVector)
}

// HybridSearch performs a hybrid search using multiple query vectors
func (m *VectorDBManager) HybridSearch(collectionName string, queryVectors [][]float32) ([]SearchResult, error) {
	ctx := context.Background()
	fmt.Printf("Performing hybrid search with %d query vectors\n", len(queryVectors))
	results, err := m.db.HybridSearch(ctx, collectionName, queryVectors)
	if err != nil {
		return nil, fmt.Errorf("hybrid search failed: %w", err)
	}
	fmt.Printf("Hybrid search returned %d results\n", len(results))
	return results, nil
}

// Close disconnects from the vector database
func (m *VectorDBManager) Close() error {
	ctx := context.Background()
	return m.db.Disconnect(ctx)
}

// EnsureCollection creates a collection if it doesn't exist
func (m *VectorDBManager) EnsureCollection(name string) error {
	ctx := context.Background()
	exists, err := m.db.HasCollection(ctx, name)
	if err != nil {
		return fmt.Errorf("failed to check collection existence: %w", err)
	}
	if !exists {
		if err := m.db.CreateCollection(ctx, name); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		if err := m.db.CreateIndex(ctx, name); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}
	}
	return m.db.LoadCollection(ctx, name)
}

// DeleteCollection drops a collection from the vector database
func (m *VectorDBManager) DeleteCollection(name string) error {
	ctx := context.Background()
	return m.db.DropCollection(ctx, name)
}


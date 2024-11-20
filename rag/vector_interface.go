// Package rag provides a unified interface for interacting with vector databases,
// offering a clean abstraction layer for vector similarity search operations.
package rag

import (
	"context"
	"fmt"
	"time"
)

// VectorDB defines the standard interface that all vector database implementations must implement.
// It provides operations for managing collections, inserting data, and performing vector similarity searches.
type VectorDB interface {
	// Connect establishes a connection to the vector database.
	Connect(ctx context.Context) error
	
	// Close terminates the connection to the vector database.
	Close() error
	
	// HasCollection checks if a collection with the given name exists.
	HasCollection(ctx context.Context, name string) (bool, error)
	
	// DropCollection removes a collection and all its data.
	DropCollection(ctx context.Context, name string) error
	
	// CreateCollection creates a new collection with the specified schema.
	CreateCollection(ctx context.Context, name string, schema Schema) error
	
	// Insert adds new records to the specified collection.
	Insert(ctx context.Context, collectionName string, data []Record) error
	
	// Flush ensures all pending writes are committed to storage.
	Flush(ctx context.Context, collectionName string) error
	
	// CreateIndex builds an index on the specified field to optimize search operations.
	CreateIndex(ctx context.Context, collectionName, field string, index Index) error
	
	// LoadCollection loads a collection into memory for faster access.
	LoadCollection(ctx context.Context, name string) error
	
	// Search performs a vector similarity search in the specified collection.
	Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error)
	
	// HybridSearch combines vector similarity search with additional filtering or reranking.
	HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error)
	
	// SetColumnNames configures the column names for the database operations.
	SetColumnNames(names []string)
}

// SearchParam defines the parameters for vector similarity search operations.
type SearchParam struct {
	// MetricType specifies the distance metric to use (e.g., "L2", "IP", "COSINE")
	MetricType string
	// Params contains additional search parameters specific to the database implementation
	Params     map[string]interface{}
}

// Schema defines the structure of a collection in the vector database.
type Schema struct {
	// Name is the identifier for the schema
	Name        string
	// Description provides additional information about the schema
	Description string
	// Fields defines the structure of the data in the collection
	Fields      []Field
}

// Field represents a single field in a schema, which can be a vector or scalar value.
type Field struct {
	// Name is the identifier for the field
	Name       string
	// DataType specifies the type of data stored in the field
	DataType   string
	// PrimaryKey indicates if this field is the primary key
	PrimaryKey bool
	// AutoID indicates if the field value should be automatically generated
	AutoID     bool
	// Dimension specifies the size of the vector (for vector fields)
	Dimension  int
	// MaxLength specifies the maximum length for variable-length fields
	MaxLength  int
}

// Record represents a single data entry in the vector database.
type Record struct {
	// Fields maps field names to their values
	Fields map[string]interface{}
}

// Vector represents a mathematical vector as a slice of float64 values.
type Vector []float64

// Index defines the parameters for building an index on a field.
type Index struct {
	// Type specifies the type of index to build (e.g., "IVF", "IVFPQ")
	Type       string
	// Metric specifies the distance metric to use for the index
	Metric     string
	// Parameters contains additional index parameters specific to the database implementation
	Parameters map[string]interface{}
}

// SearchResult represents a single result from a vector similarity search.
type SearchResult struct {
	// ID is the identifier for the result
	ID     int64
	// Score is the similarity score for the result
	Score  float64
	// Fields contains additional information about the result
	Fields map[string]interface{}
}

// Config defines the configuration for a vector database connection.
type Config struct {
	// Type specifies the type of vector database to connect to (e.g., "milvus", "memory")
	Type        string
	// Address specifies the address of the vector database
	Address     string
	// MaxPoolSize specifies the maximum number of connections to the database
	MaxPoolSize int
	// Timeout specifies the timeout for database operations
	Timeout     time.Duration
	// Parameters contains additional configuration parameters specific to the database implementation
	Parameters  map[string]interface{}
}

// Option defines a function that can be used to configure a Config.
type Option func(*Config)

// SetType sets the type of vector database to connect to.
func (c *Config) SetType(dbType string) *Config {
	c.Type = dbType
	return c
}

// SetAddress sets the address of the vector database.
func (c *Config) SetAddress(address string) *Config {
	c.Address = address
	return c
}

// SetMaxPoolSize sets the maximum number of connections to the database.
func (c *Config) SetMaxPoolSize(size int) *Config {
	c.MaxPoolSize = size
	return c
}

// SetTimeout sets the timeout for database operations.
func (c *Config) SetTimeout(timeout time.Duration) *Config {
	c.Timeout = timeout
	return c
}

// NewVectorDB creates a new VectorDB instance based on the provided configuration.
func NewVectorDB(cfg *Config) (VectorDB, error) {
	switch cfg.Type {
	case "milvus":
		return newMilvusDB(cfg)
	case "memory":
		return newMemoryDB(cfg)
	case "chromem":
		return newChromemDB(cfg)
	default:
		return nil, fmt.Errorf("unsupported database type: %s", cfg.Type)
	}
}

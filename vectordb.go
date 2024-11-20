// Package raggo provides a high-level abstraction over various vector database
// implementations. This file defines the VectorDB type, which wraps the lower-level
// rag.VectorDB interface with additional functionality and type safety.
package raggo

import (
	"context"
	"fmt"
	"time"

	"github.com/teilomillet/raggo/rag"
)

// VectorDB provides a high-level interface for vector database operations.
// It wraps the lower-level rag.VectorDB interface and adds:
// - Type-safe configuration
// - Connection management
// - Dimension tracking
// - Database type information
type VectorDB struct {
	db        rag.VectorDB // Underlying vector database implementation
	dbType    string      // Type of database (e.g., "milvus", "memory")
	address   string      // Connection address
	dimension int         // Vector dimension
}

// Config holds configuration options for VectorDB instances.
// It provides a clean way to configure database connections
// without exposing implementation details.
type Config struct {
	Type        string        // Database type (e.g., "milvus", "memory")
	Address     string        // Connection address
	MaxPoolSize int           // Maximum number of connections
	Timeout     time.Duration // Operation timeout
	Dimension   int           // Vector dimension
}

// Option is a function type for configuring VectorDB instances.
// It follows the functional options pattern for clean and flexible configuration.
type Option func(*Config)

// WithType sets the database type.
// Supported types:
// - "milvus": Production-grade vector database
// - "memory": In-memory database for testing
// - "chromem": Chrome-based persistent storage
func WithType(dbType string) Option {
	return func(c *Config) {
		c.Type = dbType
	}
}

// WithAddress sets the database connection address.
// Examples:
// - Milvus: "localhost:19530"
// - Memory: "" (no address needed)
// - ChromeM: "./data/vectors.db"
func WithAddress(address string) Option {
	return func(c *Config) {
		c.Address = address
	}
}

// WithMaxPoolSize sets the maximum number of database connections.
// This is particularly relevant for Milvus and other client-server databases.
func WithMaxPoolSize(size int) Option {
	return func(c *Config) {
		c.MaxPoolSize = size
	}
}

// WithTimeout sets the operation timeout duration.
// This affects all database operations including:
// - Connection attempts
// - Search operations
// - Insert operations
func WithTimeout(timeout time.Duration) Option {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

// WithDimension sets the dimension of vectors to be stored.
// This must match the dimension of your embedding model:
// - text-embedding-3-small: 1536
// - text-embedding-ada-002: 1536
// - Cohere embed-multilingual-v3.0: 1024
func WithDimension(dimension int) Option {
	return func(c *Config) {
		c.Dimension = dimension
	}
}

// NewVectorDB creates a new vector database connection with the specified options.
// The function:
// 1. Applies all configuration options
// 2. Creates the appropriate database implementation
// 3. Sets up the connection (but doesn't connect yet)
//
// Returns an error if:
// - The database type is unsupported
// - The configuration is invalid
// - The database implementation fails to initialize
func NewVectorDB(opts ...Option) (*VectorDB, error) {
	cfg := &Config{}
	for _, opt := range opts {
		opt(cfg)
	}
	ragDB, err := rag.NewVectorDB(&rag.Config{
		Type:        cfg.Type,
		Address:     cfg.Address,
		MaxPoolSize: cfg.MaxPoolSize,
		Timeout:     cfg.Timeout,
		Parameters: map[string]interface{}{
			"dimension": cfg.Dimension,
		},
	})
	if err != nil {
		return nil, err
	}
	return &VectorDB{
		db:        ragDB,
		dbType:    cfg.Type,
		address:   cfg.Address,
		dimension: cfg.Dimension,
	}, nil
}

// Connect establishes a connection to the vector database.
// This method must be called before any database operations.
func (vdb *VectorDB) Connect(ctx context.Context) error {
	return vdb.db.Connect(ctx)
}

// Close closes the vector database connection.
// This method should be called when the database is no longer needed.
func (vdb *VectorDB) Close() error {
	return vdb.db.Close()
}

// HasCollection checks if a collection exists in the database.
// Returns true if the collection exists, false otherwise.
func (vdb *VectorDB) HasCollection(ctx context.Context, name string) (bool, error) {
	return vdb.db.HasCollection(ctx, name)
}

// CreateCollection creates a new collection in the database.
// The schema defines the structure of the collection.
func (vdb *VectorDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	return vdb.db.CreateCollection(ctx, name, rag.Schema(schema))
}

// DropCollection drops a collection from the database.
// Returns an error if the collection does not exist.
func (vdb *VectorDB) DropCollection(ctx context.Context, name string) error {
	return vdb.db.DropCollection(ctx, name)
}

// Insert inserts a batch of records into a collection.
// The records must match the schema of the collection.
func (vdb *VectorDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	fmt.Printf("Inserting %d records into collection: %s\n", len(data), collectionName)

	ragRecords := make([]rag.Record, len(data))
	for i, record := range data {
		ragRecords[i] = rag.Record(record)
	}
	return vdb.db.Insert(ctx, collectionName, ragRecords)
}

// Flush flushes the pending operations in a collection.
// This method is used to ensure that all pending operations are written to disk.
func (vdb *VectorDB) Flush(ctx context.Context, collectionName string) error {
	return vdb.db.Flush(ctx, collectionName)
}

// CreateIndex creates an index on a field in a collection.
// The index type defines the type of index to create.
func (vdb *VectorDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	return vdb.db.CreateIndex(ctx, collectionName, field, rag.Index(index))
}

// LoadCollection loads a collection from disk.
// This method is used to load a collection that was previously created.
func (vdb *VectorDB) LoadCollection(ctx context.Context, name string) error {
	return vdb.db.LoadCollection(ctx, name)
}

// Search searches for vectors in a collection.
// The search parameters define the search criteria.
// Returns a list of search results.
func (vdb *VectorDB) Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error) {
	fmt.Printf("Searching in collection %s for top %d results with metric type %s\n", collectionName, topK, metricType)

	results, err := vdb.db.Search(ctx, collectionName, vectors, topK, metricType, searchParams)
	if err != nil {
		return nil, err
	}
	return convertSearchResults(results), nil
}

// HybridSearch performs a hybrid search in a collection.
// The search parameters define the search criteria.
// The reranker is used to rerank the search results.
// Returns a list of search results.
func (vdb *VectorDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	fmt.Printf("Performing hybrid search in collection %s for top %d results with metric type %s\n", collectionName, topK, metricType)

	results, err := vdb.db.HybridSearch(ctx, collectionName, vectors, topK, metricType, searchParams, reranker)
	if err != nil {
		return nil, err
	}
	return convertSearchResults(results), nil
}

// convertSearchResults converts a list of rag.SearchResult to a list of SearchResult.
func convertSearchResults(ragResults []rag.SearchResult) []SearchResult {
	results := make([]SearchResult, len(ragResults))
	for i, r := range ragResults {
		results[i] = SearchResult(r)
	}
	return results
}

// SetColumnNames sets the column names for a collection.
// This method is used to set the column names for a collection.
func (vdb *VectorDB) SetColumnNames(names []string) {
	vdb.db.SetColumnNames(names)
}

// Type returns the type of the vector database.
func (vdb *VectorDB) Type() string {
	return vdb.dbType
}

// Address returns the address of the vector database.
func (vdb *VectorDB) Address() string {
	return vdb.address
}

// Dimension returns the dimension of the vectors in the database.
func (vdb *VectorDB) Dimension() int {
	return vdb.dimension
}

// Types to match the internal rag package
type Schema = rag.Schema
type Field = rag.Field
type Record = rag.Record
type Vector = rag.Vector
type Index = rag.Index
type SearchResult = rag.SearchResult

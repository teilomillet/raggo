// Package rag provides retrieval-augmented generation capabilities.
package rag

import (
	"context"
	"fmt"
	"sync"
)

// ExampleDB demonstrates how to implement a new vector database provider.
// This template shows the minimum required functionality and common patterns.
//
// Key Features to Implement:
// - Thread-safe operations
// - Vector similarity search
// - Collection management
// - Data persistence (if applicable)
// - Error handling and logging
type ExampleDB struct {
	// Configuration
	config *Config

	// Connection state
	isConnected bool

	// Collection management
	collections map[string]interface{} // Replace interface{} with your collection type
	mu         sync.RWMutex           // Protects concurrent access to collections

	// Search configuration
	columnNames []string // Names of columns to retrieve in search results
	dimension   int      // Vector dimension for embeddings
}

// newExampleDB creates a new ExampleDB instance with the given configuration.
// Initialize your database connection and any required resources here.
func newExampleDB(cfg *Config) (*ExampleDB, error) {
	// Get dimension from config parameters (example)
	dimension, ok := cfg.Parameters["dimension"].(int)
	if !ok {
		dimension = 1536 // Default dimension
	}

	return &ExampleDB{
		config:      cfg,
		collections: make(map[string]interface{}),
		dimension:   dimension,
	}, nil
}

// Connect establishes a connection to the database.
// Implement your connection logic here.
func (db *ExampleDB) Connect(ctx context.Context) error {
	GlobalLogger.Debug("Connecting to example database", "address", db.config.Address)
	
	// Add your connection logic here
	// Example:
	// client, err := yourdb.Connect(db.config.Address)
	// if err != nil {
	//     return fmt.Errorf("failed to connect: %w", err)
	// }

	db.isConnected = true
	return nil
}

// Close terminates the database connection.
// Clean up any resources here.
func (db *ExampleDB) Close() error {
	if !db.isConnected {
		return nil
	}
	
	// Add your cleanup logic here
	db.isConnected = false
	return nil
}

// HasCollection checks if a collection exists.
func (db *ExampleDB) HasCollection(ctx context.Context, name string) (bool, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	_, exists := db.collections[name]
	return exists, nil
}

// CreateCollection initializes a new collection.
func (db *ExampleDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Validate collection doesn't exist
	if _, exists := db.collections[name]; exists {
		return fmt.Errorf("collection %s already exists", name)
	}

	// Initialize your collection here
	// Example:
	// collection, err := db.client.CreateCollection(name, schema)
	// if err != nil {
	//     return fmt.Errorf("failed to create collection: %w", err)
	// }
	// db.collections[name] = collection

	return nil
}

// DropCollection removes a collection.
func (db *ExampleDB) DropCollection(ctx context.Context, name string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	delete(db.collections, name)
	return nil
}

// Insert adds new records to a collection.
func (db *ExampleDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	// Example vector conversion if needed:
	// vectors := make([][]float32, len(data))
	// for i, record := range data {
	//     vectors[i] = toFloat32Slice(record.Vector)
	// }

	// Add your insert logic here
	return nil
}

// Search performs vector similarity search.
func (db *ExampleDB) Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error) {
	// Example implementation steps:
	// 1. Convert vectors if needed
	// 2. Perform search
	// 3. Format results

	return nil, fmt.Errorf("not implemented")
}

// HybridSearch combines vector and keyword search (optional).
func (db *ExampleDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	return nil, fmt.Errorf("hybrid search not supported")
}

// Additional optional methods:

// Flush ensures data persistence (if applicable).
func (db *ExampleDB) Flush(ctx context.Context, collectionName string) error {
	return nil
}

// CreateIndex builds search indexes (if applicable).
func (db *ExampleDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	return nil
}

// LoadCollection prepares a collection for searching (if needed).
func (db *ExampleDB) LoadCollection(ctx context.Context, name string) error {
	return nil
}

// SetColumnNames configures which fields to return in search results.
func (db *ExampleDB) SetColumnNames(names []string) {
	db.columnNames = names
}

// Helper functions

// exampleToFloat32Slice converts vectors if your database needs a different format.
// Note: If you need float32 conversion, consider using the existing toFloat32Slice
// function from the rag package instead of implementing your own.
func exampleToFloat32Slice(v Vector) []float32 {
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = float32(val)
	}
	return result
}

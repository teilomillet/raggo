// Package rag provides an in-memory vector database implementation that serves
// as a lightweight solution for vector similarity search. It's ideal for testing,
// prototyping, and small-scale applications that don't require persistence.
package rag

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
)

// MemoryDB implements the VectorDB interface using in-memory storage.
// It provides thread-safe operations for managing collections and performing
// vector similarity searches without the need for external database systems.
type MemoryDB struct {
	// collections stores all vector collections in memory
	collections map[string]*Collection
	// mu provides thread-safety for concurrent operations
	mu sync.RWMutex
	// columnNames specifies which fields to include in search results
	columnNames []string
}

// Collection represents a named set of records with a defined schema.
// It's the basic unit of organization in the memory database.
type Collection struct {
	// Schema defines the structure of records in this collection
	Schema Schema
	// Data holds the actual records in the collection
	Data []Record
}

// newMemoryDB creates a new in-memory vector database instance.
// It initializes an empty collection map and returns a ready-to-use database.
func newMemoryDB(cfg *Config) (*MemoryDB, error) {
	return &MemoryDB{
		collections: make(map[string]*Collection),
	}, nil
}

// Connect is a no-op for the in-memory database as no connection is needed.
// It's implemented to satisfy the VectorDB interface.
func (m *MemoryDB) Connect(ctx context.Context) error {
	return nil
}

// Close is a no-op for the in-memory database as no cleanup is needed.
// It's implemented to satisfy the VectorDB interface.
func (m *MemoryDB) Close() error {
	return nil
}

// HasCollection checks if a collection with the given name exists in the database.
// This operation is thread-safe and uses a read lock.
func (m *MemoryDB) HasCollection(ctx context.Context, name string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, exists := m.collections[name]
	return exists, nil
}

// DropCollection removes a collection and all its data from the database.
// This operation is thread-safe and uses a write lock.
func (m *MemoryDB) DropCollection(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.collections, name)
	return nil
}

// CreateCollection creates a new collection with the specified schema.
// Returns an error if a collection with the same name already exists.
// This operation is thread-safe and uses a write lock.
func (m *MemoryDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.collections[name]; exists {
		return fmt.Errorf("collection %s already exists", name)
	}
	m.collections[name] = &Collection{Schema: schema}
	return nil
}

// Insert adds new records to the specified collection.
// Returns an error if the collection doesn't exist.
// This operation is thread-safe and uses a write lock.
func (m *MemoryDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	collection, exists := m.collections[collectionName]
	if !exists {
		return fmt.Errorf("collection %s does not exist", collectionName)
	}
	collection.Data = append(collection.Data, data...)
	return nil
}

// Flush is a no-op for the in-memory database as all operations are immediate.
// It's implemented to satisfy the VectorDB interface.
func (m *MemoryDB) Flush(ctx context.Context, collectionName string) error {
	return nil
}

// CreateIndex is a no-op for the in-memory database as it uses linear search.
// Future implementations could add indexing for better performance.
func (m *MemoryDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	return nil
}

// LoadCollection is a no-op for the in-memory database as all data is always loaded.
// It's implemented to satisfy the VectorDB interface.
func (m *MemoryDB) LoadCollection(ctx context.Context, name string) error {
	return nil
}

// Search performs vector similarity search in the specified collection.
// It supports different distance metrics and returns the top K most similar vectors.
// The search process:
// 1. Validates the collection exists
// 2. Computes distances between query vectors and stored vectors
// 3. Sorts results by similarity score
// 4. Returns the top K results with specified fields
func (m *MemoryDB) Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error) {
	// The implementation remains largely the same, but now we can use metricType and searchParams
	// For simplicity, we'll ignore these new parameters in this example
	collection, exists := m.collections[collectionName]
	if !exists {
		return nil, fmt.Errorf("collection %s does not exist", collectionName)
	}

	var results []SearchResult

	for _, record := range collection.Data {
		for fieldName, searchVector := range vectors {
			if v, ok := record.Fields[fieldName].(Vector); ok {
				distance := m.calculateDistance(searchVector, v, metricType)
				fields := make(map[string]interface{})
				for _, name := range m.columnNames {
					if value, exists := record.Fields[name]; exists {
						fields[name] = value
					}
				}
				results = append(results, SearchResult{
					ID:     record.Fields["ID"].(int64),
					Score:  distance,
					Fields: fields,
				})
				break
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return results, nil
}

// HybridSearch performs a multi-vector similarity search with optional reranking.
// It's similar to Search but supports searching across multiple vector fields
// and combining the results. The process:
// 1. Validates the collection exists
// 2. Computes distances for each vector field
// 3. Combines distances using average
// 4. Sorts and returns top K results
func (m *MemoryDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	// The implementation remains largely the same, but now we can use metricType, searchParams, and reranker
	// For simplicity, we'll ignore these new parameters in this example
	collection, exists := m.collections[collectionName]
	if !exists {
		return nil, fmt.Errorf("collection %s does not exist", collectionName)
	}

	var results []SearchResult
	for _, record := range collection.Data {
		var totalDistance float64
		var fieldsMatched int
		for fieldName, searchVector := range vectors {
			if v, ok := record.Fields[fieldName].(Vector); ok {
				totalDistance += m.calculateDistance(searchVector, v, metricType)
				fieldsMatched++
			}
		}

		if fieldsMatched == len(vectors) {
			fields := make(map[string]interface{})
			for _, name := range m.columnNames {
				if value, exists := record.Fields[name]; exists {
					fields[name] = value
				}
			}
			results = append(results, SearchResult{
				ID:     record.Fields["ID"].(int64),
				Score:  totalDistance / float64(len(vectors)),
				Fields: fields,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	return results, nil
}

// calculateDistance computes the distance between two vectors using the specified metric.
// Supported metrics:
// - "L2": Euclidean distance (default)
// - "IP": Inner product (negative, as larger means more similar)
// Returns a float64 representing the distance/similarity score.
func (m *MemoryDB) calculateDistance(a, b Vector, metricType string) float64 {
	var sum float64
	switch metricType {
	case "L2":
		for i := range a {
			diff := a[i] - b[i]
			sum += diff * diff
		}
		return math.Sqrt(sum)
	case "IP":
		for i := range a {
			sum += a[i] * b[i]
		}
		return -sum // Negative because larger IP means closer vectors
	default:
		// Default to L2
		for i := range a {
			diff := a[i] - b[i]
			sum += diff * diff
		}
		return math.Sqrt(sum)
	}
}

// euclideanDistance computes the L2 (Euclidean) distance between two vectors.
// This is a helper function used by calculateDistance when metricType is "L2".
func euclideanDistance(a, b Vector) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// SetColumnNames configures which fields should be included in search results.
// This allows for selective field retrieval to optimize response size.
func (m *MemoryDB) SetColumnNames(names []string) {
	m.columnNames = names
}

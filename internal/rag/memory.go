// File: memory.go

package rag

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
)

type MemoryDB struct {
	collections map[string]*Collection
	mu          sync.RWMutex
	columnNames []string
}

type Collection struct {
	Schema Schema
	Data   []Record
}

func newMemoryDB(cfg *Config) (*MemoryDB, error) {
	return &MemoryDB{
		collections: make(map[string]*Collection),
	}, nil
}

func (m *MemoryDB) Connect(ctx context.Context) error {
	return nil // No-op for in-memory database
}

func (m *MemoryDB) Close() error {
	return nil // No-op for in-memory database
}

func (m *MemoryDB) HasCollection(ctx context.Context, name string) (bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	_, exists := m.collections[name]
	return exists, nil
}

func (m *MemoryDB) DropCollection(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.collections, name)
	return nil
}

func (m *MemoryDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.collections[name]; exists {
		return fmt.Errorf("collection %s already exists", name)
	}
	m.collections[name] = &Collection{Schema: schema}
	return nil
}

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

func (m *MemoryDB) Flush(ctx context.Context, collectionName string) error {
	return nil // No-op for in-memory database
}

func (m *MemoryDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	return nil // No-op for in-memory database, we'll use linear search
}

func (m *MemoryDB) LoadCollection(ctx context.Context, name string) error {
	return nil // No-op for in-memory database
}

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

func euclideanDistance(a, b Vector) float64 {
	var sum float64
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func (m *MemoryDB) SetColumnNames(names []string) {
	m.columnNames = names
}

package vectordb

import (
	"context"
	"fmt"
	"math"
	"sort"
)

type InMemoryDB struct {
	collections map[string]struct {
		vector1 [][]float32
		vector2 [][]float32
	}
	config VectorDBConfig
}

func (m *InMemoryDB) Connect(ctx context.Context, config VectorDBConfig) error {
	m.config = config
	m.collections = make(map[string]struct {
		vector1 [][]float32
		vector2 [][]float32
	})
	return nil
}

func (m *InMemoryDB) Disconnect(ctx context.Context) error {
	return nil
}

func (m *InMemoryDB) CreateCollection(ctx context.Context, name string) error {
	m.collections[name] = struct {
		vector1 [][]float32
		vector2 [][]float32
	}{
		vector1: make([][]float32, 0),
		vector2: make([][]float32, 0),
	}
	return nil
}

func (m *InMemoryDB) HasCollection(ctx context.Context, name string) (bool, error) {

	_, exists := m.collections[name]
	return exists, nil
}

func (m *InMemoryDB) LoadCollection(ctx context.Context, collectionName string) error {
	// No-op for in-memory implementation
	return nil
}

func (m *InMemoryDB) DropCollection(ctx context.Context, name string) error {
	delete(m.collections, name)
	return nil
}

func (m *InMemoryDB) CreateIndex(ctx context.Context, collectionName string) error {
	// No-op for in-memory implementation
	return nil
}

func (m *InMemoryDB) Insert(ctx context.Context, collectionName string, vectors [][]float32, metadata []map[string]interface{}) error {
	collection := m.collections[collectionName]
	for _, vector := range vectors {
		collection.vector1 = append(collection.vector1, vector)
		collection.vector2 = append(collection.vector2, vector)
	}
	m.collections[collectionName] = collection
	return nil
}

func (m *InMemoryDB) Search(ctx context.Context, collectionName string, queryVector []float32) ([]SearchResult, error) {
	collection := m.collections[collectionName]
	results := make([]SearchResult, 0, m.config.TopK)

	for i := range collection.vector1 {
		distance := euclideanDistance(queryVector, collection.vector1[i])
		results = append(results, SearchResult{
			ID:    int64(i),
			Score: distance,
			Metadata: map[string]interface{}{
				"vector1": collection.vector1[i],
				"vector2": collection.vector2[i],
			},
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	if len(results) > m.config.TopK {
		results = results[:m.config.TopK]
	}

	return results, nil
}

func (m *InMemoryDB) HybridSearch(ctx context.Context, collectionName string, queryVectors [][]float32) ([]SearchResult, error) {
	fmt.Printf("InMemoryDB: Performing hybrid search with %d query vectors\n", len(queryVectors))
	collection := m.collections[collectionName]
	results := make([]SearchResult, 0, m.config.TopK)

	if len(queryVectors) == 0 {
		return results, nil
	}

	for i := range collection.vector1 {
		var totalDistance float32
		if len(queryVectors) > 0 {
			totalDistance += euclideanDistance(queryVectors[0], collection.vector1[i])
		}
		if len(queryVectors) > 1 {
			totalDistance += euclideanDistance(queryVectors[1], collection.vector2[i])
		}
		results = append(results, SearchResult{
			ID:    int64(i),
			Score: totalDistance,
			Metadata: map[string]interface{}{
				"vector1": collection.vector1[i],
				"vector2": collection.vector2[i],
			},
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score
	})

	if len(results) > m.config.TopK {
		results = results[:m.config.TopK]
	}

	fmt.Printf("InMemoryDB: Returning %d results\n", len(results))
	return results, nil
}

func euclideanDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

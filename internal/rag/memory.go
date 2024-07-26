package rag

import (
	"context"
	"math"
	"sort"
)

type MemoryStore struct {
	chunks map[string][]EmbeddedChunk
}

func NewMemoryStore(config VectorDBConfig) (VectorDB, error) {
	GlobalLogger.Debug("Creating new MemoryStore")
	return &MemoryStore{
		chunks: make(map[string][]EmbeddedChunk),
	}, nil
}

func (m *MemoryStore) SaveEmbeddings(ctx context.Context, collectionName string, chunks []EmbeddedChunk) error {
	for i := range chunks {
		chunks[i].Embedding = normalizeVector(chunks[i].Embedding)
	}
	GlobalLogger.Debug("Saving embeddings to MemoryStore", "collectionName", collectionName, "chunkCount", len(chunks))
	m.chunks[collectionName] = append(m.chunks[collectionName], chunks...)
	return nil
}

func normalizeVector(vector []float64) []float64 {
	var sum float64
	for _, v := range vector {
		sum += v * v
	}
	magnitude := math.Sqrt(sum)
	if magnitude == 0 {
		return vector // Avoid division by zero
	}
	normalized := make([]float64, len(vector))
	for i, v := range vector {
		normalized[i] = v / magnitude
	}
	return normalized
}

func innerProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func (m *MemoryStore) Search(ctx context.Context, collectionName string, query []float64, topK int, param SearchParam) ([]SearchResult, error) {
	GlobalLogger.Debug("Searching in MemoryStore", "collectionName", collectionName, "topK", topK)
	collection, ok := m.chunks[collectionName]
	if !ok {
		GlobalLogger.Warn("Collection not found in MemoryStore", "collectionName", collectionName)
		return nil, nil
	}

	results := make([]SearchResult, 0, len(collection))
	for i, chunk := range collection {
		score := cosineSimilarity(query, chunk.Embedding)
		results = append(results, SearchResult{
			ID:       int64(i),
			Text:     chunk.Text,
			Score:    score,
			Metadata: chunk.Metadata,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	GlobalLogger.Debug("Search completed", "resultCount", len(results))
	return results, nil
}

func (m *MemoryStore) Close() error {
	GlobalLogger.Debug("Closing MemoryStore")
	m.chunks = nil
	return nil
}

func cosineSimilarity(a, b []float64) float64 {
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

func init() {
	RegisterVectorDB("memory", NewMemoryStore)
}

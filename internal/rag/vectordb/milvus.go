package vectordb

import (
	"context"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	idCol, keyCol, embeddingCol = "ID", "key", "vector"
)

type MilvusDB struct {
	client client.Client
	config VectorDBConfig
}

func (m *MilvusDB) Connect(ctx context.Context, config VectorDBConfig) error {
	m.config = config
	c, err := client.NewClient(ctx, client.Config{
		Address: config.Address,
	})
	if err != nil {
		return fmt.Errorf("failed to connect to Milvus: %w", err)
	}
	m.client = c
	return nil
}

func (m *MilvusDB) Disconnect(ctx context.Context) error {
	return m.client.Close()
}

func (m *MilvusDB) CreateCollection(ctx context.Context, name string) error {
	schema := entity.NewSchema().
		WithName(name).
		WithDescription("Collection created by RagGo").
		WithField(entity.NewField().WithName("ID").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("key").WithDataType(entity.FieldTypeInt64)).
		WithField(entity.NewField().WithName("vector1").WithDataType(entity.FieldTypeFloatVector).WithDim(int64(m.config.Dimension))).
		WithField(entity.NewField().WithName("vector2").WithDataType(entity.FieldTypeFloatVector).WithDim(int64(m.config.Dimension)))

	return m.client.CreateCollection(ctx, schema, entity.DefaultShardNumber)
}

func (m *MilvusDB) DropCollection(ctx context.Context, name string) error {
	return m.client.DropCollection(ctx, name)
}

func (m *MilvusDB) HasCollection(ctx context.Context, name string) (bool, error) {
	return m.client.HasCollection(ctx, name)
}

func (m *MilvusDB) LoadCollection(ctx context.Context, collectionName string) error {
	err := m.client.LoadCollection(ctx, collectionName, false)
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}
	return nil
}

func (m *MilvusDB) CreateIndex(ctx context.Context, collectionName string) error {
	idx, err := entity.NewIndexHNSW(entity.L2, 16, 256)
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}
	if err := m.client.CreateIndex(ctx, collectionName, "vector1", idx, false); err != nil {
		return fmt.Errorf("failed to create index for vector1: %w", err)
	}
	if err := m.client.CreateIndex(ctx, collectionName, "vector2", idx, false); err != nil {
		return fmt.Errorf("failed to create index for vector2: %w", err)
	}
	return nil
}

func (m *MilvusDB) Insert(ctx context.Context, collectionName string, vectors [][]float32, metadata []map[string]interface{}) error {
	keyList := make([]int64, len(vectors))
	for i := range keyList {
		keyList[i] = int64(i)
	}
	keyColData := entity.NewColumnInt64("key", keyList)
	embeddingColData1 := entity.NewColumnFloatVector("vector1", m.config.Dimension, vectors)
	embeddingColData2 := entity.NewColumnFloatVector("vector2", m.config.Dimension, vectors)

	_, err := m.client.Insert(ctx, collectionName, "", keyColData, embeddingColData1, embeddingColData2)
	if err != nil {
		return fmt.Errorf("failed to insert data: %w", err)
	}

	return m.client.Flush(ctx, collectionName, false)
}

func (m *MilvusDB) Search(ctx context.Context, collectionName string, queryVector []float32) ([]SearchResult, error) {
	// Load the collection if not already loaded
	err := m.client.LoadCollection(ctx, collectionName, false)
	if err != nil {
		return nil, fmt.Errorf("failed to load collection: %w", err)
	}

	sp, err := entity.NewIndexHNSWSearchParam(30)
	if err != nil {
		return nil, fmt.Errorf("failed to create search params: %w", err)
	}

	vec2search := []entity.Vector{entity.FloatVector(queryVector)}

	result, err := m.client.Search(ctx, collectionName, nil, "", []string{"ID", "key", "vector1", "vector2"}, vec2search,
		"vector1", entity.L2, m.config.TopK, sp)
	if err != nil {
		return nil, fmt.Errorf("failed to search collection: %w", err)
	}

	var searchResults []SearchResult
	for _, rs := range result {
		for i := 0; i < rs.ResultCount; i++ {
			id, _ := rs.IDs.GetAsInt64(i)
			score := rs.Scores[i]
			vector1, _ := rs.Fields.GetColumn("vector1").Get(i)
			vector2, _ := rs.Fields.GetColumn("vector2").Get(i)
			searchResults = append(searchResults, SearchResult{
				ID:    id,
				Score: score,
				Metadata: map[string]interface{}{
					"vector1": vector1,
					"vector2": vector2,
				},
			})
		}
	}
	return searchResults, nil
}

func (m *MilvusDB) HybridSearch(ctx context.Context, collectionName string, queryVectors [][]float32) ([]SearchResult, error) {
	fmt.Printf("MilvusDB: Performing hybrid search with %d query vectors\n", len(queryVectors))
	sp, err := entity.NewIndexHNSWSearchParam(30)
	if err != nil {
		return nil, fmt.Errorf("failed to create search params: %w", err)
	}

	vec2search1 := []entity.Vector{entity.FloatVector(queryVectors[0])}
	vec2search2 := []entity.Vector{entity.FloatVector(queryVectors[1])}

	result, err := m.client.HybridSearch(ctx, collectionName, nil, m.config.TopK, []string{"key", "vector1", "vector2"},
		client.NewRRFReranker(), []*client.ANNSearchRequest{
			client.NewANNSearchRequest("vector1", entity.L2, "", vec2search1, sp, m.config.TopK),
			client.NewANNSearchRequest("vector2", entity.L2, "", vec2search2, sp, m.config.TopK),
		})
	if err != nil {
		return nil, fmt.Errorf("failed to perform hybrid search: %w", err)
	}

	var searchResults []SearchResult
	for _, rs := range result {
		for i := 0; i < rs.ResultCount; i++ {
			id, _ := rs.IDs.GetAsInt64(i)
			score := rs.Scores[i]
			embedding1, _ := rs.Fields.GetColumn("vector1").Get(i)
			embedding2, _ := rs.Fields.GetColumn("vector2").Get(i)
			searchResults = append(searchResults, SearchResult{
				ID:    id,
				Score: score,
				Metadata: map[string]interface{}{
					"vector1": embedding1,
					"vector2": embedding2,
				},
			})
		}
	}

	fmt.Printf("MilvusDB: Returning %d results\n", len(searchResults))
	return searchResults, nil
}

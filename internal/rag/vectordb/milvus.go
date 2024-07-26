package vectordb

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/teilomillet/raggo/internal/rag"
)

type MilvusDB struct {
	client       client.Client
	address      string
	dimension    int
	loaded       map[string]bool
	indexType    string
	indexParams  map[string]interface{}
	searchParams map[string]interface{}
	metric       entity.MetricType // New field for metric
}

func NewMilvusDB(config rag.VectorDBConfig) (rag.VectorDB, error) {
	rag.GlobalLogger.Debug("Creating new MilvusDB", "address", config.Address, "dimension", config.Dimension)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	c, err := client.NewClient(ctx, client.Config{Address: config.Address})
	if err != nil {
		rag.GlobalLogger.Error("Failed to connect to Milvus", "error", err)
		return nil, fmt.Errorf("failed to connect to Milvus: %w", err)
	}

	indexType := "IVF_FLAT"
	if it, ok := config.Options["index_type"].(string); ok {
		indexType = it
	}

	indexParams := map[string]interface{}{
		"nlist":          1024,
		"M":              16,
		"efConstruction": 500,
	}
	if ip, ok := config.Options["index_params"].(map[string]interface{}); ok {
		for k, v := range ip {
			indexParams[k] = v
		}
	}

	searchParams := map[string]interface{}{
		"nprobe": 16,
	}
	if sp, ok := config.Options["search_params"].(map[string]interface{}); ok {
		for k, v := range sp {
			searchParams[k] = v
		}
	}
	metric := entity.L2
	if metricStr, ok := config.Options["metric"].(string); ok {
		switch strings.ToLower(metricStr) {
		case "ip", "inner_product":
			metric = entity.IP
		case "l2":
			metric = entity.L2
		default:
			return nil, fmt.Errorf("unsupported metric: %s", metricStr)
		}
	}

	return &MilvusDB{
		client:       c,
		address:      config.Address,
		dimension:    config.Dimension,
		loaded:       make(map[string]bool),
		indexType:    indexType,
		indexParams:  indexParams,
		searchParams: searchParams,
		metric:       metric,
	}, nil
}

func normalizeVector[T float32 | float64](vector []T) []T {
	var sum T
	for _, v := range vector {
		sum += v * v
	}
	magnitude := T(math.Sqrt(float64(sum)))
	if magnitude == 0 {
		return vector // Avoid division by zero
	}
	normalized := make([]T, len(vector))
	for i, v := range vector {
		normalized[i] = v / magnitude
	}
	return normalized
}

func (m *MilvusDB) ensureCollectionLoaded(ctx context.Context, collectionName string) error {
	if m.loaded[collectionName] {
		return nil
	}

	err := m.loadCollectionWithRetry(ctx, collectionName, 5)
	if err != nil {
		return err
	}

	m.loaded[collectionName] = true
	return nil
}

func (m *MilvusDB) ensureCollectionDimension(ctx context.Context, collectionName string) error {
	collInfo, err := m.client.DescribeCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to describe collection: %w", err)
	}

	var currentDim int64
	for _, field := range collInfo.Schema.Fields {
		if field.Name == "embedding" {
			currentDim, _ = strconv.ParseInt(field.TypeParams["dim"], 10, 64)
			break
		}
	}

	if currentDim != int64(m.dimension) {
		rag.GlobalLogger.Warn("Collection dimension mismatch, recreating collection",
			"collectionName", collectionName,
			"currentDim", currentDim,
			"expectedDim", m.dimension)

		// Drop the existing collection
		err = m.client.DropCollection(ctx, collectionName)
		if err != nil {
			return fmt.Errorf("failed to drop collection: %w", err)
		}

		// Recreate the collection
		return m.createCollectionIfNotExists(ctx, collectionName)
	}

	return nil
}

func (m *MilvusDB) loadCollectionWithRetry(ctx context.Context, collectionName string, maxRetries int) error {
	for i := 0; i < maxRetries; i++ {
		rag.GlobalLogger.Debug("Attempting to load collection", "collectionName", collectionName, "attempt", i+1)

		// Wait for index to be ready before loading
		err := m.waitForIndex(ctx, collectionName)
		if err != nil {
			rag.GlobalLogger.Debug("Failed to wait for index, retrying", "error", err)
			time.Sleep(time.Second * time.Duration(i+1))
			continue
		}

		err = m.client.LoadCollection(ctx, collectionName, false)
		if err == nil {
			rag.GlobalLogger.Debug("Collection loaded successfully", "collectionName", collectionName)
			return nil
		}
		rag.GlobalLogger.Debug("Failed to load collection, retrying", "error", err)
		time.Sleep(time.Second * time.Duration(i+1))
	}
	return fmt.Errorf("failed to load collection after %d attempts", maxRetries)
}

func (m *MilvusDB) unloadCollection(ctx context.Context, collectionName string) error {
	rag.GlobalLogger.Debug("Unloading collection", "collectionName", collectionName)
	err := m.client.ReleaseCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to unload collection: %w", err)
	}
	return nil
}

func (m *MilvusDB) waitForIndex(ctx context.Context, collectionName string) error {
	rag.GlobalLogger.Debug("Waiting for index to be ready", "collectionName", collectionName)
	startTime := time.Now()
	for {
		if time.Since(startTime) > 5*time.Minute {
			return fmt.Errorf("timeout waiting for index to be ready")
		}

		indexes, err := m.client.DescribeIndex(ctx, collectionName, "embedding")
		if err != nil {
			rag.GlobalLogger.Warn("Failed to describe index", "error", err)
			time.Sleep(5 * time.Second)
			continue
		}

		if len(indexes) > 0 {
			rag.GlobalLogger.Debug("Index is ready", "collectionName", collectionName)
			return nil
		}

		time.Sleep(5 * time.Second)
	}
}

func (m *MilvusDB) createCollectionIfNotExists(ctx context.Context, collectionName string) error {
	has, err := m.client.HasCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to check if collection exists: %w", err)
	}

	if !has {
		rag.GlobalLogger.Debug("Collection does not exist, creating", "collectionName", collectionName, "dimension", m.dimension)
		schema := entity.NewSchema().
			WithName(collectionName).
			WithDescription("Collection for storing text embeddings").
			WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true)).
			WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(int64(m.dimension))).
			WithField(entity.NewField().WithName("text").WithDataType(entity.FieldTypeVarChar).WithMaxLength(65535)).
			WithField(entity.NewField().WithName("token_size").WithDataType(entity.FieldTypeInt32)).
			WithField(entity.NewField().WithName("start_sentence").WithDataType(entity.FieldTypeInt32)).
			WithField(entity.NewField().WithName("end_sentence").WithDataType(entity.FieldTypeInt32))

		err = m.client.CreateCollection(ctx, schema, entity.DefaultShardNumber)
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		rag.GlobalLogger.Debug("Collection created successfully", "collectionName", collectionName, "dimension", m.dimension)
	}

	return nil
}

func (m *MilvusDB) indexExists(ctx context.Context, collectionName, fieldName string) (bool, error) {
	indexes, err := m.client.DescribeIndex(ctx, collectionName, fieldName)
	if err != nil {
		if strings.Contains(err.Error(), "index doesn't exist") {
			return false, nil
		}
		return false, fmt.Errorf("failed to describe index: %w", err)
	}
	return len(indexes) > 0, nil
}

func (m *MilvusDB) createIndex(ctx context.Context, collectionName string) error {
	rag.GlobalLogger.Debug("Creating index for collection", "collectionName", collectionName, "indexType", m.indexType)

	var idx entity.Index
	var err error
	switch m.indexType {
	case "IVF_FLAT":
		idx, err = entity.NewIndexIvfFlat(m.metric, m.indexParams["nlist"].(int))
	case "HNSW":
		idx, err = entity.NewIndexHNSW(m.metric, m.indexParams["M"].(int), m.indexParams["efConstruction"].(int))
	default:
		return fmt.Errorf("unsupported index type: %s", m.indexType)
	}

	if err != nil {
		return fmt.Errorf("failed to create index object: %w", err)
	}

	err = m.client.CreateIndex(ctx, collectionName, "embedding", idx, false)
	if err != nil {
		if strings.Contains(err.Error(), "index already exist") {
			rag.GlobalLogger.Debug("Index already exists", "collectionName", collectionName)
			return nil
		}
		return fmt.Errorf("failed to create index in Milvus: %w", err)
	}

	rag.GlobalLogger.Debug("Index creation completed", "collectionName", collectionName)
	return nil
}
func (m *MilvusDB) createIndexWithRetry(ctx context.Context, collectionName string, maxRetries int) error {
	var err error
	for i := 0; i < maxRetries; i++ {
		rag.GlobalLogger.Debug("Attempting to create index", "attempt", i+1, "collectionName", collectionName)

		err = m.createIndex(ctx, collectionName)
		if err == nil {
			rag.GlobalLogger.Debug("Index created successfully", "collectionName", collectionName)
			return nil
		}
		rag.GlobalLogger.Warn("Failed to create index", "attempt", i+1, "error", err)
		time.Sleep(time.Second * time.Duration(i+1))
	}
	return fmt.Errorf("failed to create index after %d attempts: %w", maxRetries, err)
}
func (m *MilvusDB) createScalarIndex(ctx context.Context, collectionName, fieldName string) error {
	rag.GlobalLogger.Debug("Creating scalar index for field", "collectionName", collectionName, "fieldName", fieldName)
	idx := entity.NewScalarIndex()
	err := m.client.CreateIndex(ctx, collectionName, fieldName, idx, false)
	if err != nil {
		return fmt.Errorf("failed to create scalar index: %w", err)
	}
	rag.GlobalLogger.Debug("Scalar index created successfully", "collectionName", collectionName, "fieldName", fieldName)
	return nil
}

func (m *MilvusDB) SaveEmbeddings(ctx context.Context, collectionName string, chunks []rag.EmbeddedChunk) error {
	if len(chunks) == 0 {
		return fmt.Errorf("no chunks to save")
	}

	if len(chunks[0].Embedding) != m.dimension {
		return fmt.Errorf("embedding dimension mismatch: expected %d, got %d", m.dimension, len(chunks[0].Embedding))
	}

	// Step 1: Ensure collection exists
	rag.GlobalLogger.Debug("Ensuring collection exists", "collectionName", collectionName)
	err := m.createCollectionIfNotExists(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to create collection: %w", err)
	}

	// Step 2: Insert data
	rag.GlobalLogger.Debug("Inserting data into collection", "collectionName", collectionName, "chunkCount", len(chunks))
	ids := make([]int64, len(chunks))
	embeddings := make([][]float32, len(chunks))
	texts := make([]string, len(chunks))
	tokenSizes := make([]int32, len(chunks))
	startSentences := make([]int32, len(chunks))
	endSentences := make([]int32, len(chunks))

	for i, chunk := range chunks {
		ids[i] = int64(i)
		embeddings[i] = toFloat32Slice(normalizeVector(chunk.Embedding))
		texts[i] = chunk.Text
		tokenSizes[i] = int32(chunk.Metadata["token_size"].(int))
		startSentences[i] = int32(chunk.Metadata["start_sentence"].(int))
		endSentences[i] = int32(chunk.Metadata["end_sentence"].(int))
	}

	_, err = m.client.Insert(ctx, collectionName, "",
		entity.NewColumnInt64("id", ids),
		entity.NewColumnFloatVector("embedding", m.dimension, embeddings),
		entity.NewColumnVarChar("text", texts),
		entity.NewColumnInt32("token_size", tokenSizes),
		entity.NewColumnInt32("start_sentence", startSentences),
		entity.NewColumnInt32("end_sentence", endSentences),
	)
	if err != nil {
		return fmt.Errorf("failed to insert chunks: %w", err)
	}

	// Step 3: Flush the collection
	rag.GlobalLogger.Debug("Flushing collection", "collectionName", collectionName)
	err = m.client.Flush(ctx, collectionName, false)
	if err != nil {
		return fmt.Errorf("failed to flush collection: %w", err)
	}

	// Step 4: Create index with retry
	rag.GlobalLogger.Debug("Creating index for collection", "collectionName", collectionName)
	err = m.createIndexWithRetry(ctx, collectionName, 10) // Increased retry attempts to 10
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	// Step 5: Load the collection
	rag.GlobalLogger.Debug("Loading collection", "collectionName", collectionName)
	err = m.loadCollectionWithRetry(ctx, collectionName, 5)
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	rag.GlobalLogger.Debug("Embeddings saved successfully", "collectionName", collectionName)
	return nil
}

func (m *MilvusDB) Search(ctx context.Context, collectionName string, query []float64, limit int, param rag.SearchParam) ([]rag.SearchResult, error) {
	rag.GlobalLogger.Debug("Searching in MilvusDB", "collectionName", collectionName, "limit", limit, "queryDimension", len(query))

	err := m.ensureCollectionLoaded(ctx, collectionName)
	if err != nil {
		return nil, fmt.Errorf("failed to load collection: %w", err)
	}

	searchParams := m.searchParams
	if param != nil {
		for k, v := range param.Params() {
			searchParams[k] = v
		}
	}

	var sp entity.SearchParam
	switch m.indexType {
	case "IVF_FLAT":
		sp, err = entity.NewIndexIvfFlatSearchParam(searchParams["nprobe"].(int))
	case "HNSW":
		sp, err = entity.NewIndexHNSWSearchParam(searchParams["ef"].(int))
	default:
		return nil, fmt.Errorf("unsupported index type: %s", m.indexType)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to create search param: %w", err)
	}

	vector := toFloat32Slice(normalizeVector(query))
	rag.GlobalLogger.Debug("Search query", "normalizedVector", vector[:5]) // Log only first 5 elements

	searchResults, err := m.client.Search(
		ctx,
		collectionName,
		[]string{},
		"",
		[]string{"id", "text", "token_size", "start_sentence", "end_sentence"},
		[]entity.Vector{entity.FloatVector(vector)},
		"embedding",
		m.metric, // Use the metric specified in the MilvusDB instance
		limit,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to perform search: %w", err)
	}

	rag.GlobalLogger.Debug("Raw Milvus search results", "resultCount", len(searchResults), "firstResultCount", searchResults[0].ResultCount)
	if len(searchResults) > 0 {
		rag.GlobalLogger.Debug("Top search result",
			"score", searchResults[0].Scores[0],
			"resultCount", searchResults[0].ResultCount)
	}

	results, err := m.processResults(searchResults, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to process search results: %w", err)
	}

	return results, nil
}
func (m *MilvusDB) fallbackSearch(ctx context.Context, collectionName string, query []float64, limit int, param rag.SearchParam) ([]rag.SearchResult, error) {
	rag.GlobalLogger.Debug("Performing brute-force fallback search")

	minDistanceThreshold := 0.01
	if thresholdParam, ok := param.Params()["min_distance"]; ok {
		minDistanceThreshold = thresholdParam.(float64)
	}

	// Fetch all vectors from the collection
	allVectors, err := m.getAllVectors(ctx, collectionName)
	if err != nil {
		rag.GlobalLogger.Warn("Failed to fetch all vectors, proceeding with partial results", "error", err)
		// Continue with an empty map if we couldn't fetch any vectors
		allVectors = make(map[int64][]float64)
	}

	rag.GlobalLogger.Debug("Retrieved vectors for fallback search", "vectorCount", len(allVectors))

	// Perform brute-force search
	results := make([]rag.SearchResult, 0, limit)
	for id, vector := range allVectors {
		similarity := cosineSimilarity(query, vector)
		if similarity < minDistanceThreshold {
			continue
		}
		results = append(results, rag.SearchResult{
			ID:    id,
			Score: similarity,
			// Note: We're not fetching the text and other metadata here.
			// You may want to add a separate function to fetch this information if needed.
		})
	}

	// Sort results
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top results
	if len(results) > limit {
		results = results[:limit]
	}

	rag.GlobalLogger.Debug("Fallback search completed", "resultCount", len(results))
	return results, nil
}
func (m *MilvusDB) getAllVectors(ctx context.Context, collectionName string) (map[int64][]float64, error) {
	rag.GlobalLogger.Debug("Fetching all vectors from collection", "collectionName", collectionName)

	// Create a dummy query vector
	dummyQuery := make([]float32, m.dimension)
	for i := range dummyQuery {
		dummyQuery[i] = 1.0 // Set all values to 1.0
	}

	// Create a search parameter
	sp, err := entity.NewIndexIvfFlatSearchParam(16) // Use a reasonable nprobe value
	if err != nil {
		return nil, fmt.Errorf("failed to create search param: %w", err)
	}

	const maxTopK = 16384
	searchResults, err := m.client.Search(
		ctx,
		collectionName,
		[]string{},
		"",
		[]string{"id", "embedding"},
		[]entity.Vector{entity.FloatVector(dummyQuery)},
		"embedding",
		entity.L2,
		maxTopK,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch all vectors: %w", err)
	}

	if len(searchResults) == 0 || searchResults[0].ResultCount == 0 {
		return nil, errors.New("no vectors found in collection")
	}

	allVectors := make(map[int64][]float64)
	for i := 0; i < searchResults[0].ResultCount; i++ {
		id, err := searchResults[0].IDs.GetAsInt64(i)
		if err != nil {
			rag.GlobalLogger.Warn("Failed to get ID", "error", err)
			continue
		}
		embedding, err := searchResults[0].Fields.GetColumn("embedding").Get(i)
		if err != nil {
			rag.GlobalLogger.Warn("Failed to get embedding", "error", err)
			continue
		}
		allVectors[id] = toFloat64Slice(embedding.([]float32))
	}

	rag.GlobalLogger.Debug("Fetched all vectors", "vectorCount", len(allVectors))
	return allVectors, nil
}

func toFloat32Slice(slice []float64) []float32 {
	float32Slice := make([]float32, len(slice))
	for i, v := range slice {
		float32Slice[i] = float32(v)
	}
	return float32Slice
}

func toFloat64Slice(slice []float32) []float64 {
	float64Slice := make([]float64, len(slice))
	for i, v := range slice {
		float64Slice[i] = float64(v)
	}
	return float64Slice
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		rag.GlobalLogger.Warn("Vector dimensions do not match", "lenA", len(a), "lenB", len(b))
		return 0
	}
	var dotProduct, magnitudeA, magnitudeB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		magnitudeA += a[i] * a[i]
		magnitudeB += b[i] * b[i]
	}
	magnitude := math.Sqrt(magnitudeA) * math.Sqrt(magnitudeB)
	if magnitude == 0 {
		return 0
	}
	return dotProduct / magnitude
}

func (m *MilvusDB) processResults(searchResults []client.SearchResult, limit int) ([]rag.SearchResult, error) {
	if len(searchResults) == 0 || searchResults[0].ResultCount == 0 {
		return nil, errors.New("no search results found")
	}

	var results []rag.SearchResult
	for _, result := range searchResults {
		for i := 0; i < result.ResultCount; i++ {
			id, _ := result.IDs.GetAsInt64(i)
			text, _ := result.Fields.GetColumn("text").Get(i)
			tokenSize, _ := result.Fields.GetColumn("token_size").Get(i)
			startSentence, _ := result.Fields.GetColumn("start_sentence").Get(i)
			endSentence, _ := result.Fields.GetColumn("end_sentence").Get(i)

			score := float64(result.Scores[i])
			if m.metric == entity.L2 {
				// For L2, lower scores are better, so we invert the score
				score = 1 / (1 + score)
			}

			results = append(results, rag.SearchResult{
				ID:    id,
				Text:  text.(string),
				Score: score,
				Metadata: map[string]interface{}{
					"token_size":     tokenSize,
					"start_sentence": startSentence,
					"end_sentence":   endSentence,
				},
			})

			rag.GlobalLogger.Debug("Processing search result",
				"id", id,
				"text", text,
				"rawScore", result.Scores[i],
				"normalizedScore", score,
				"tokenSize", tokenSize,
				"startSentence", startSentence,
				"endSentence", endSentence)
		}
	}

	// Sort results based on score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score < results[j].Score // For L2 distance, lower is better
	})

	if len(results) < limit {
		rag.GlobalLogger.Warn("Fewer results returned than requested", "requested", limit, "returned", len(results))
	} else if len(results) > limit {
		results = results[:limit]
	}

	rag.GlobalLogger.Debug("Search results processed", "resultCount", len(results))
	return results, nil
}

func (m *MilvusDB) Close() error {
	rag.GlobalLogger.Debug("Closing MilvusDB connection")
	return m.client.Close()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func init() {
	rag.RegisterVectorDB("milvus", NewMilvusDB)
}


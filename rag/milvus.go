// Package rag provides retrieval-augmented generation capabilities.
package rag

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// MilvusDB implements a vector database interface using Milvus.
// It provides high-performance vector similarity search with:
// - HNSW indexing for fast approximate nearest neighbor search
// - Hybrid search combining multiple vector fields
// - Flexible schema definition and data types
// - Batch operations for efficient data management
type MilvusDB struct {
	client      client.Client        // Milvus client connection
	config      *Config             // Database configuration
	columnNames []string            // Names of columns to retrieve in search results
}

// newMilvusDB creates a new MilvusDB instance with the given configuration.
// Note: This doesn't establish the connection - call Connect() separately.
func newMilvusDB(cfg *Config) (*MilvusDB, error) {
	return &MilvusDB{config: cfg}, nil
}

// Connect establishes a connection to the Milvus server.
// It uses the address specified in the configuration and returns an error
// if the connection cannot be established.
func (m *MilvusDB) Connect(ctx context.Context) error {
	GlobalLogger.Debug("Attempting to connect to Milvus", "address", m.config.Address)
	
	c, err := client.NewClient(ctx, client.Config{
		Address: m.config.Address,
	})
	if err != nil {
		GlobalLogger.Error("Failed to connect to Milvus", "error", err, "address", m.config.Address)
		return fmt.Errorf("failed to connect to Milvus at %s: %w\nPlease ensure Milvus is running (e.g., with 'docker-compose up -d')", m.config.Address, err)
	}
	
	m.client = c
	GlobalLogger.Debug("Successfully connected to Milvus")
	return nil
}

// Close terminates the connection to the Milvus server.
// It should be called when the database is no longer needed.
func (m *MilvusDB) Close() error {
	return m.client.Close()
}

// HasCollection checks if a collection with the given name exists.
func (m *MilvusDB) HasCollection(ctx context.Context, name string) (bool, error) {
	return m.client.HasCollection(ctx, name)
}

// DropCollection removes a collection and all its data from the database.
// Warning: This operation is irreversible.
func (m *MilvusDB) DropCollection(ctx context.Context, name string) error {
	return m.client.DropCollection(ctx, name)
}

// CreateCollection creates a new collection with the specified schema.
// The schema defines:
// - Field names and types (including vector fields)
// - Primary key configuration
// - Auto-ID settings
// - Vector dimensions
// - VARCHAR field lengths
func (m *MilvusDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	milvusSchema := entity.NewSchema().WithName(name).WithDescription(schema.Description)
	for _, field := range schema.Fields {
		f := entity.NewField().WithName(field.Name).WithDataType(m.convertDataType(field.DataType))

		if field.PrimaryKey {
			f.WithIsPrimaryKey(true)
		}
		if field.AutoID {
			f.WithIsAutoID(true)
		}
		if field.DataType == "float_vector" {
			f.WithDim(int64(field.Dimension))
		}
		if field.DataType == "varchar" && field.MaxLength > 0 {
			f.WithMaxLength(int64(field.MaxLength))
		}
		milvusSchema.WithField(f)
	}

	GlobalLogger.Debug("Creating collection", "name", name, "schema", fmt.Sprintf("%+v", milvusSchema))
	return m.client.CreateCollection(ctx, milvusSchema, entity.DefaultShardNumber)
}

// Insert adds new records to a collection.
// It handles multiple data types and automatically creates appropriate columns.
// The function:
// 1. Creates columns for each field type
// 2. Appends values to respective columns
// 3. Performs batch insertion for efficiency
func (m *MilvusDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	columns := make(map[string]entity.Column)
	for _, record := range data {
		for fieldName, fieldValue := range record.Fields {
			if _, ok := columns[fieldName]; !ok {
				col := m.createColumn(fieldName, fieldValue)
				columns[fieldName] = col
				GlobalLogger.Debug("Created column", "field", fieldName, "type", fmt.Sprintf("%T", col))
			}
			m.appendToColumn(columns[fieldName], fieldValue)
		}
	}

	columnList := make([]entity.Column, 0, len(columns))
	for fieldName, col := range columns {
		columnList = append(columnList, col)
		GlobalLogger.Debug("Inserting column", "field", fieldName, "type", fmt.Sprintf("%T", col), "values", col.Len())
	}

	_, err := m.client.Insert(ctx, collectionName, "", columnList...)
	if err != nil {
		GlobalLogger.Error("Failed to insert data", "collection", collectionName, "error", err)
	}
	return err
}

// Flush ensures all inserted data is persisted to disk.
// This is important to call before searching newly inserted data.
func (m *MilvusDB) Flush(ctx context.Context, collectionName string) error {
	return m.client.Flush(ctx, collectionName, false)
}

// CreateIndex builds an index on a specified field to optimize search performance.
// Currently supports:
// - HNSW index type with configurable M and efConstruction parameters
// - Different metric types (L2, IP, etc.)
func (m *MilvusDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	var idx entity.Index
	var err error

	switch index.Type {
	case "HNSW":
		idx, err = entity.NewIndexHNSW(m.convertMetricType(index.Metric), index.Parameters["M"].(int), index.Parameters["efConstruction"].(int))
	default:
		return fmt.Errorf("unsupported index type: %s", index.Type)
	}

	if err != nil {
		return err
	}

	return m.client.CreateIndex(ctx, collectionName, field, idx, false)
}

// LoadCollection loads a collection into memory for searching.
// This must be called before performing searches on the collection.
func (m *MilvusDB) LoadCollection(ctx context.Context, name string) error {
	return m.client.LoadCollection(ctx, name, false)
}

// Search performs vector similarity search on a single field.
// Parameters:
// - vectors: Map of field name to vector values
// - topK: Number of results to return
// - metricType: Distance metric (L2, IP, etc.)
// - searchParams: Index-specific search parameters
func (m *MilvusDB) Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error) {
	// Assume we're searching only one field for simplicity
	var fieldName string
	var vector Vector
	for f, v := range vectors {
		fieldName = f
		vector = v
		break
	}

	floatVector := make([]float32, len(vector))
	for i, v := range vector {
		floatVector[i] = float32(v)
	}

	sp, err := m.createSearchParam(searchParams)
	if err != nil {
		return nil, err
	}

	result, err := m.client.Search(ctx, collectionName, nil, "", m.columnNames,
		[]entity.Vector{entity.FloatVector(floatVector)},
		fieldName, m.convertMetricType(metricType), topK, sp)
	if err != nil {
		return nil, err
	}

	return m.wrapSearchResults(result), nil
}

// HybridSearch performs search across multiple vector fields with reranking.
// It combines results using:
// 1. Individual ANN searches on each vector field
// 2. Reranking of combined results (default: RRF reranker)
// 3. Final top-K selection
func (m *MilvusDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	limit := topK
	subRequests := make([]*client.ANNSearchRequest, 0, len(vectors))

	sp, err := m.createSearchParam(searchParams)
	if err != nil {
		return nil, err
	}

	for fieldName, vector := range vectors {
		floatVector := make([]float32, len(vector))
		for i, v := range vector {
			floatVector[i] = float32(v)
		}
		subRequests = append(subRequests, client.NewANNSearchRequest(fieldName, m.convertMetricType(metricType), "", []entity.Vector{entity.FloatVector(floatVector)}, sp, topK))
	}

	var milvusReranker client.Reranker
	if reranker == nil {
		// Use default reranker if none is provided
		milvusReranker = client.NewRRFReranker()
	} else {
		var ok bool
		milvusReranker, ok = reranker.(client.Reranker)
		if !ok {
			return nil, fmt.Errorf("invalid reranker type")
		}
	}

	result, err := m.client.HybridSearch(ctx, collectionName, nil, limit, m.columnNames, milvusReranker, subRequests)
	if err != nil {
		return nil, err
	}

	return m.wrapSearchResults(result), nil
}

// createSearchParam creates search parameters for the specified index type.
// Currently supports HNSW index with 'ef' parameter for search-time optimization.
func (m *MilvusDB) createSearchParam(params map[string]interface{}) (entity.SearchParam, error) {
	if params["type"] == "HNSW" {
		ef, ok := params["ef"].(int)
		if !ok {
			return nil, fmt.Errorf("invalid ef parameter for HNSW search")
		}
		return entity.NewIndexHNSWSearchParam(ef)
	}
	// Add more search param types as needed
	return nil, fmt.Errorf("unsupported search param type")
}

// convertMetricType converts string metric types to Milvus entity.MetricType.
// Supported types: L2, IP (Inner Product), COSINE, etc.
func (m *MilvusDB) convertMetricType(metricType string) entity.MetricType {
	switch metricType {
	case "L2":
		return entity.L2
	case "IP":
		return entity.IP
	default:
		return entity.L2 // Default to L2 if unknown
	}
}

// convertDataType converts string data types to Milvus entity.FieldType.
// Supports: int64, float, string, float_vector, etc.
func (m *MilvusDB) convertDataType(dataType string) entity.FieldType {
	switch dataType {
	case "int64":
		return entity.FieldTypeInt64
	case "float_vector":
		return entity.FieldTypeFloatVector
	case "varchar":
		return entity.FieldTypeVarChar
	default:
		return entity.FieldTypeNone
	}
}

// createColumn creates a new column with appropriate type based on the field value.
// Handles: Int64, Float32, String, FloatVector, etc.
func (m *MilvusDB) createColumn(fieldName string, fieldValue interface{}) entity.Column {
	switch v := fieldValue.(type) {
	case int64:
		return entity.NewColumnInt64(fieldName, []int64{})
	case []float64:
		return entity.NewColumnFloatVector(fieldName, len(v), [][]float32{})
	case []float32:
		return entity.NewColumnFloatVector(fieldName, len(v), [][]float32{})
	case string:
		return entity.NewColumnVarChar(fieldName, []string{})
	case map[string]interface{}:
		// For metadata fields, we just create a varchar column
		// The actual JSON conversion happens in appendToColumn
		return entity.NewColumnVarChar(fieldName, []string{})
	default:
		panic(fmt.Sprintf("unsupported field type for %s: %T", fieldName, fieldValue))
	}
}

// SetColumnNames sets the list of column names to retrieve in search results.
func (m *MilvusDB) SetColumnNames(names []string) {
	m.columnNames = names
}

// appendToColumn adds a value to the appropriate type of column.
// Handles type conversion and validation for different field types.
func (m *MilvusDB) appendToColumn(col entity.Column, value interface{}) {
	switch c := col.(type) {
	case *entity.ColumnInt64:
		c.AppendValue(value.(int64))
	case *entity.ColumnFloatVector:
		var floatVector []float32
		switch v := value.(type) {
		case []float64:
			floatVector = make([]float32, len(v))
			for i, val := range v {
				floatVector[i] = float32(val)
			}
		case []float32:
			floatVector = v
		default:
			panic(fmt.Sprintf("unsupported vector type: %T", value))
		}
		c.AppendValue(floatVector)
	case *entity.ColumnVarChar:
		switch v := value.(type) {
		case string:
			c.AppendValue(v)
		case map[string]interface{}:
			// Handle metadata by converting to JSON string
			jsonStr, err := json.Marshal(v)
			if err != nil {
				panic(fmt.Sprintf("failed to marshal metadata: %v", err))
			}
			c.AppendValue(string(jsonStr))
		default:
			panic(fmt.Sprintf("unsupported varchar value type: %T", value))
		}
	default:
		panic(fmt.Sprintf("unsupported column type: %T", col))
	}
}

// wrapSearchResults converts Milvus search results to the internal SearchResult format.
// It extracts scores, IDs, and field values from the Milvus results.
func (m *MilvusDB) wrapSearchResults(result []client.SearchResult) []SearchResult {
	var searchResults []SearchResult
	for _, rs := range result {
		for i := 0; i < rs.ResultCount; i++ {
			id, _ := rs.IDs.GetAsInt64(i)
			fields := make(map[string]interface{})

			for _, fieldName := range m.columnNames {
				if column := rs.Fields.GetColumn(fieldName); column != nil {
					if value, err := column.Get(i); err == nil {
						fields[fieldName] = value
					}
				}
			}

			searchResults = append(searchResults, SearchResult{
				ID:     id,
				Score:  float64(rs.Scores[i]),
				Fields: fields,
			})
		}
	}
	return searchResults
}

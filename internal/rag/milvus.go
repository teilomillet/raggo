// File: milvus.go

package rag

import (
	"context"
	"fmt"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type MilvusDB struct {
	client      client.Client
	config      *Config
	columnNames []string
}

func newMilvusDB(cfg *Config) (*MilvusDB, error) {
	return &MilvusDB{config: cfg}, nil
}

func (m *MilvusDB) Connect(ctx context.Context) error {
	c, err := client.NewClient(ctx, client.Config{
		Address: m.config.Address,
	})
	if err != nil {
		return err
	}
	m.client = c
	return nil
}

func (m *MilvusDB) Close() error {
	return m.client.Close()
}

func (m *MilvusDB) HasCollection(ctx context.Context, name string) (bool, error) {
	return m.client.HasCollection(ctx, name)
}

func (m *MilvusDB) DropCollection(ctx context.Context, name string) error {
	return m.client.DropCollection(ctx, name)
}

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
		if field.DataType == "varchar" {
			f.WithMaxLength(int64(field.MaxLength))
		}
		milvusSchema.WithField(f)
	}
	return m.client.CreateCollection(ctx, milvusSchema, entity.DefaultShardNumber)
}

func (m *MilvusDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	columns := make(map[string]entity.Column)
	for _, record := range data {
		for fieldName, fieldValue := range record.Fields {
			if _, ok := columns[fieldName]; !ok {
				col := m.createColumn(fieldName, fieldValue)
				columns[fieldName] = col
				fmt.Printf("Created column for field %s with type %T\n", fieldName, col)
			}
			m.appendToColumn(columns[fieldName], fieldValue)
		}
	}

	columnList := make([]entity.Column, 0, len(columns))
	for fieldName, col := range columns {
		columnList = append(columnList, col)
		fmt.Printf("Inserting column %s with type %T and %d values\n", fieldName, col, col.Len())
	}

	_, err := m.client.Insert(ctx, collectionName, "", columnList...)
	if err != nil {
		fmt.Printf("Error inserting data: %v\n", err)
	}
	return err
}

func (m *MilvusDB) Flush(ctx context.Context, collectionName string) error {
	return m.client.Flush(ctx, collectionName, false)
}

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

func (m *MilvusDB) LoadCollection(ctx context.Context, name string) error {
	return m.client.LoadCollection(ctx, name, false)
}

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

func (m *MilvusDB) createColumn(fieldName string, fieldValue interface{}) entity.Column {
	switch v := fieldValue.(type) {
	case int64:
		return entity.NewColumnInt64(fieldName, []int64{})
	case []float64:
		return entity.NewColumnFloatVector(fieldName, len(v), [][]float32{})
	case string:
		return entity.NewColumnVarChar(fieldName, []string{}) // Changed this line
	default:
		panic(fmt.Sprintf("unsupported field type for %s: %T", fieldName, fieldValue))
	}
}

func (m *MilvusDB) SetColumnNames(names []string) {
	m.columnNames = names
}

func (m *MilvusDB) appendToColumn(col entity.Column, value interface{}) {
	switch c := col.(type) {
	case *entity.ColumnInt64:
		c.AppendValue(value.(int64))
	case *entity.ColumnFloatVector:
		floatVector := make([]float32, len(value.([]float64)))
		for i, v := range value.([]float64) {
			floatVector[i] = float32(v)
		}
		c.AppendValue(floatVector)
	case *entity.ColumnVarChar:
		c.AppendValue(value.(string))
	default:
		panic(fmt.Sprintf("unsupported column type: %T", col))
	}
}

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

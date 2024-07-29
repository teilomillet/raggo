// File: vectordb.go

package raggo

import (
	"context"
	"fmt"
	"github.com/teilomillet/raggo/internal/rag"
	"time"
)

type VectorDB struct {
	db rag.VectorDB
}

type Config struct {
	Type        string
	Address     string
	MaxPoolSize int
	Timeout     time.Duration
}

type Option func(*Config)

func WithType(dbType string) Option {
	return func(c *Config) {
		c.Type = dbType
	}
}

func WithAddress(address string) Option {
	return func(c *Config) {
		c.Address = address
	}
}

func WithMaxPoolSize(size int) Option {
	return func(c *Config) {
		c.MaxPoolSize = size
	}
}

func WithTimeout(timeout time.Duration) Option {
	return func(c *Config) {
		c.Timeout = timeout
	}
}

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
	})
	if err != nil {
		return nil, err
	}
	return &VectorDB{db: ragDB}, nil
}

func (vdb *VectorDB) Connect(ctx context.Context) error {
	return vdb.db.Connect(ctx)
}

func (vdb *VectorDB) Close() error {
	return vdb.db.Close()
}

func (vdb *VectorDB) HasCollection(ctx context.Context, name string) (bool, error) {
	return vdb.db.HasCollection(ctx, name)
}

func (vdb *VectorDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	return vdb.db.CreateCollection(ctx, name, rag.Schema(schema))
}

func (vdb *VectorDB) DropCollection(ctx context.Context, name string) error {
	return vdb.db.DropCollection(ctx, name)
}

func (vdb *VectorDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	fmt.Printf("Inserting %d records into collection: %s\n", len(data), collectionName)

	ragRecords := make([]rag.Record, len(data))
	for i, record := range data {
		ragRecords[i] = rag.Record(record)
	}
	return vdb.db.Insert(ctx, collectionName, ragRecords)
}

func (vdb *VectorDB) Flush(ctx context.Context, collectionName string) error {
	return vdb.db.Flush(ctx, collectionName)
}

func (vdb *VectorDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	return vdb.db.CreateIndex(ctx, collectionName, field, rag.Index(index))
}

func (vdb *VectorDB) LoadCollection(ctx context.Context, name string) error {
	return vdb.db.LoadCollection(ctx, name)
}

func (vdb *VectorDB) Search(ctx context.Context, collectionName string, vector Vector, topK int) ([]SearchResult, error) {
	fmt.Printf("Searching in collection %s for top %d results\n", collectionName, topK)

	results, err := vdb.db.Search(ctx, collectionName, rag.Vector(vector), topK)
	if err != nil {
		return nil, err
	}
	return convertSearchResults(results), nil
}

func (vdb *VectorDB) HybridSearch(ctx context.Context, collectionName string, fieldName string, vectors []Vector, topK int) ([]SearchResult, error) {
	fmt.Printf("Performing hybrid search in collection %s, field %s for top %d results\n", collectionName, fieldName, topK)

	ragVectors := make([]rag.Vector, len(vectors))
	for i, v := range vectors {
		ragVectors[i] = rag.Vector(v)
	}
	results, err := vdb.db.HybridSearch(ctx, collectionName, fieldName, ragVectors, topK)
	if err != nil {
		return nil, err
	}
	return convertSearchResults(results), nil
}

func convertSearchResults(ragResults []rag.SearchResult) []SearchResult {
	results := make([]SearchResult, len(ragResults))
	for i, r := range ragResults {
		results[i] = SearchResult(r)
	}
	return results
}

func (vdb *VectorDB) SetColumnNames(names []string) {
	vdb.db.SetColumnNames(names)
}

// Types to match the internal rag package
type Schema = rag.Schema
type Field = rag.Field
type Record = rag.Record
type Vector = rag.Vector
type Index = rag.Index
type SearchResult = rag.SearchResult


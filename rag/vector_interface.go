// File: vectordb.go

package rag

import (
	"context"
	"fmt"
	"time"
)

type VectorDB interface {
	Connect(ctx context.Context) error
	Close() error
	HasCollection(ctx context.Context, name string) (bool, error)
	DropCollection(ctx context.Context, name string) error
	CreateCollection(ctx context.Context, name string, schema Schema) error
	Insert(ctx context.Context, collectionName string, data []Record) error
	Flush(ctx context.Context, collectionName string) error
	CreateIndex(ctx context.Context, collectionName, field string, index Index) error
	LoadCollection(ctx context.Context, name string) error
	Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error)
	HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error)
	SetColumnNames(names []string)
}

type SearchParam struct {
	MetricType string
	Params     map[string]interface{}
}

type Schema struct {
	Name        string
	Description string
	Fields      []Field
}

type Field struct {
	Name       string
	DataType   string
	PrimaryKey bool
	AutoID     bool
	Dimension  int
	MaxLength  int
}

type Record struct {
	Fields map[string]interface{}
}

type Vector []float64

type Index struct {
	Type       string
	Metric     string
	Parameters map[string]interface{}
}

type SearchResult struct {
	ID     int64
	Score  float64
	Fields map[string]interface{}
}

type Config struct {
	Type        string
	Address     string
	MaxPoolSize int
	Timeout     time.Duration
	Parameters  map[string]interface{}
}

type Option func(*Config)

func (c *Config) SetType(dbType string) *Config {
	c.Type = dbType
	return c
}

func (c *Config) SetAddress(address string) *Config {
	c.Address = address
	return c
}

func (c *Config) SetMaxPoolSize(size int) *Config {
	c.MaxPoolSize = size
	return c
}

func (c *Config) SetTimeout(timeout time.Duration) *Config {
	c.Timeout = timeout
	return c
}

func NewVectorDB(cfg *Config) (VectorDB, error) {
	switch cfg.Type {
	case "milvus":
		return newMilvusDB(cfg)
	case "memory":
		return newMemoryDB(cfg)
	case "chromem":
		return newChromemDB(cfg)
	default:
		return nil, fmt.Errorf("unsupported database type: %s", cfg.Type)
	}
}

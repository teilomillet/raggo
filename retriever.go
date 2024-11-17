package raggo

import (
	"context"
	"fmt"
	"os"
	"time"
)

// Retriever handles semantic search operations with a reusable configuration
type Retriever struct {
	config   *RetrieverConfig
	vectorDB *VectorDB
	embedder Embedder
	ready    bool
}

// RetrieverConfig holds settings for the retrieval process
type RetrieverConfig struct {
	// Core settings
	Collection string
	TopK       int
	MinScore   float64
	UseHybrid  bool
	Columns    []string

	// Vector DB settings
	DBType    string
	DBAddress string

	// Embedding settings
	Provider string
	Model    string
	APIKey   string

	// Advanced settings
	MetricType   string
	Timeout      time.Duration
	SearchParams map[string]interface{}
	OnResult     func(SearchResult)
	OnError      func(error)
}

// RetrieverResult represents a single retrieved result
type RetrieverResult struct {
	Content    string                 `json:"content"`
	Score      float64                `json:"score"`
	Metadata   map[string]interface{} `json:"metadata"`
	Source     string                 `json:"source"`
	ChunkIndex int                    `json:"chunk_index"`
}

func defaultRetrieverConfig() *RetrieverConfig {
	return &RetrieverConfig{
		Collection: "documents",
		TopK:       5,
		MinScore:   0.7,
		UseHybrid:  true,
		Columns:    []string{"Text", "Metadata"},
		DBType:     "milvus",
		DBAddress:  "localhost:19530",
		Provider:   "openai",
		Model:      "text-embedding-3-small",
		APIKey:     os.Getenv("OPENAI_API_KEY"),
		MetricType: "L2",
		Timeout:    30 * time.Second,
		SearchParams: map[string]interface{}{
			"type": "HNSW",
			"ef":   64,
		},
	}
}

// RetrieverOption configures the retriever
type RetrieverOption func(*RetrieverConfig)

// NewRetriever creates a new Retriever with the given options
func NewRetriever(opts ...RetrieverOption) (*Retriever, error) {
	cfg := defaultRetrieverConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	r := &Retriever{config: cfg}
	if err := r.initialize(); err != nil {
		return nil, err
	}

	return r, nil
}

func (r *Retriever) initialize() error {
	var err error

	r.vectorDB, err = NewVectorDB(
		WithType(r.config.DBType),
		WithAddress(r.config.DBAddress),
		WithTimeout(r.config.Timeout),
	)
	if err != nil {
		return fmt.Errorf("failed to create vector store: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), r.config.Timeout)
	defer cancel()

	if err := r.vectorDB.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to vector store: %w", err)
	}

	r.embedder, err = NewEmbedder(
		SetProvider(r.config.Provider),
		SetModel(r.config.Model),
		SetAPIKey(r.config.APIKey),
	)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}

	r.ready = true
	return nil
}

func (r *Retriever) Close() error {
	if r.vectorDB != nil {
		return r.vectorDB.Close()
	}
	return nil
}

// Retrieve finds similar content for the given query
func (r *Retriever) Retrieve(ctx context.Context, query string) ([]RetrieverResult, error) {
	if !r.ready {
		return nil, fmt.Errorf("retriever not properly initialized")
	}

	queryEmbedding, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to create query embedding: %w", err)
	}

	r.vectorDB.SetColumnNames(r.config.Columns)
	vectors := map[string]Vector{"Embedding": queryEmbedding}

	var searchResults []SearchResult
	var searchErr error

	if r.config.UseHybrid {
		searchResults, searchErr = r.vectorDB.HybridSearch(
			ctx,
			r.config.Collection,
			vectors,
			r.config.TopK,
			r.config.MetricType,
			r.config.SearchParams,
			nil,
		)
	} else {
		searchResults, searchErr = r.vectorDB.Search(
			ctx,
			r.config.Collection,
			vectors,
			r.config.TopK,
			r.config.MetricType,
			r.config.SearchParams,
		)
	}

	if searchErr != nil {
		return nil, fmt.Errorf("search failed: %w", searchErr)
	}

	results := make([]RetrieverResult, 0, len(searchResults))
	for _, result := range searchResults {
		if result.Score < r.config.MinScore {
			continue
		}

		content, _ := result.Fields["Text"].(string)
		metadata, _ := result.Fields["Metadata"].(map[string]interface{})

		match := RetrieverResult{
			Content:  content,
			Score:    result.Score,
			Metadata: metadata,
		}

		if metadata != nil {
			match.Source, _ = metadata["source"].(string)
			match.ChunkIndex, _ = metadata["chunk"].(int)
		}

		if r.config.OnResult != nil {
			r.config.OnResult(result)
		}

		results = append(results, match)
	}

	return results, nil
}

// GetVectorDB returns the underlying vector database instance
func (r *Retriever) GetVectorDB() *VectorDB {
	return r.vectorDB
}

// Configuration options

func WithRetrieveCollection(name string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Collection = name
	}
}

func WithTopK(k int) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.TopK = k
	}
}

func WithMinScore(score float64) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.MinScore = score
	}
}

func WithRetrieveDB(dbType, address string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.DBType = dbType
		c.DBAddress = address
	}
}

func WithRetrieveEmbedding(provider, model, key string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Provider = provider
		c.Model = model
		c.APIKey = key
	}
}

func WithHybrid(enabled bool) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.UseHybrid = enabled
	}
}

func WithColumns(columns ...string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Columns = columns
	}
}

func WithRetrieveCallbacks(onResult func(SearchResult), onError func(error)) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.OnResult = onResult
		c.OnError = onError
	}
}

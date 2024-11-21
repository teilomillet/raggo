// Package raggo implements a sophisticated document retrieval system that combines
// vector similarity search with optional reranking strategies. The retriever
// component serves as the core engine for finding and ranking relevant documents
// based on semantic similarity and other configurable criteria.
//
// Key features:
//   - Semantic similarity search using vector embeddings
//   - Hybrid search combining vector and keyword matching
//   - Configurable reranking strategies
//   - Flexible result filtering and scoring
//   - Extensible callback system for result processing
package raggo

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"
)

// Retriever handles semantic search operations with a reusable configuration.
// It provides a high-level interface for performing vector similarity searches
// and managing search results. The Retriever maintains connections to the
// vector database and embedding service throughout its lifecycle.
type Retriever struct {
	config   *RetrieverConfig // Configuration for retrieval operations
	vectorDB *VectorDB        // Connection to vector database
	embedder Embedder         // Embedding service client
	ready    bool             // Initialization status
}

// RetrieverConfig holds settings for the retrieval process. It provides
// fine-grained control over search behavior, database connections, and
// result processing.
type RetrieverConfig struct {
	// Core settings define the basic search behavior
	Collection string   // Name of the vector collection to search
	TopK       int      // Maximum number of results to return
	MinScore   float64  // Minimum similarity score threshold
	UseHybrid  bool     // Enable hybrid search (vector + keyword)
	Columns    []string // Columns to retrieve from the database

	// Vector DB settings configure the database connection
	DBType    string // Type of vector database (e.g., "milvus")
	DBAddress string // Database connection address
	Dimension int    // Embedding vector dimension

	// Embedding settings configure the embedding service
	Provider string // Embedding provider (e.g., "openai")
	Model    string // Model name for embeddings
	APIKey   string // Authentication key

	// Advanced settings provide additional control
	MetricType   string                 // Distance metric (e.g., "L2", "IP")
	Timeout      time.Duration          // Operation timeout
	SearchParams map[string]interface{} // Additional search parameters
	OnResult     func(SearchResult)     // Callback for each result
	OnError      func(error)            // Error handling callback
}

// RetrieverResult represents a single retrieved result with its metadata
// and relevance information. It provides a structured way to access both
// the content and context of each search result.
type RetrieverResult struct {
	Content    string                 `json:"content"`     // Retrieved text content
	Score      float64                `json:"score"`       // Similarity score
	Metadata   map[string]interface{} `json:"metadata"`    // Associated metadata
	Source     string                 `json:"source"`      // Source identifier
	ChunkIndex int                    `json:"chunk_index"` // Position in source
}

// NewRetriever creates a new Retriever with the given options. It initializes
// the necessary connections and validates the configuration.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithRetrieveCollection("documents"),
//	    WithTopK(5),
//	    WithMinScore(0.7),
//	    WithRetrieveDB("milvus", "localhost:19530"),
//	    WithRetrieveEmbedding("openai", "text-embedding-3-small", os.Getenv("OPENAI_API_KEY")),
//	)
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

// RetrieverOption configures the retriever using the functional options pattern.
// This allows for flexible and extensible configuration while maintaining
// backward compatibility.
type RetrieverOption func(*RetrieverConfig)

// Retrieve finds similar content for the given query using vector similarity
// search. It handles the complete retrieval pipeline:
//  1. Query embedding generation
//  2. Vector similarity search
//  3. Result filtering and processing
//  4. Metadata enrichment
//
// Example:
//
//	results, err := retriever.Retrieve(ctx, "How does photosynthesis work?")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	for _, result := range results {
//	    fmt.Printf("Score: %.2f, Content: %s\n", result.Score, result.Content)
//	}
func (r *Retriever) Retrieve(ctx context.Context, query string) ([]RetrieverResult, error) {
	if !r.ready {
		return nil, fmt.Errorf("retriever not properly initialized")
	}

	// Try a search with topK=1 first to check if collection has any documents
	queryEmbedding, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to create query embedding: %w", err)
	}

	r.vectorDB.SetColumnNames(r.config.Columns)
	vectors := map[string]Vector{"Embedding": queryEmbedding}

	// Try to get one result to check if collection has documents
	var initialResults []SearchResult
	var searchErr error
	if r.config.UseHybrid {
		initialResults, searchErr = r.vectorDB.HybridSearch(
			ctx,
			r.config.Collection,
			vectors,
			1,
			r.config.MetricType,
			r.config.SearchParams,
			nil,
		)
	} else {
		initialResults, searchErr = r.vectorDB.Search(
			ctx,
			r.config.Collection,
			vectors,
			1,
			r.config.MetricType,
			r.config.SearchParams,
		)
	}

	// If we get a "no results" error or empty results, collection is empty
	if searchErr != nil || len(initialResults) == 0 {
		log.Printf("Warning: Collection '%s' appears to be empty, returning no results", r.config.Collection)
		return []RetrieverResult{}, nil
	}

	// Now try with requested topK
	var searchResults []SearchResult
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

	// If we get an error but we know there are documents, the TopK might be too high
	if searchErr != nil {
		log.Printf("Warning: Failed to retrieve %d results, trying with %d results (collection might have fewer documents)", 
			r.config.TopK, len(initialResults))
		
		// Fall back to what we know works
		searchResults = initialResults
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

	if len(results) < r.config.TopK {
		log.Printf("Info: Returned %d results (fewer than requested TopK=%d)", len(results), r.config.TopK)
	}

	return results, nil
}

// GetVectorDB returns the underlying vector database instance.
// This provides access to lower-level database operations when needed.
func (r *Retriever) GetVectorDB() *VectorDB {
	return r.vectorDB
}

// WithRetrieveCollection sets the collection name for retrieval operations.
// The collection must exist in the vector database.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithRetrieveCollection("scientific_papers"),
//	)
func WithRetrieveCollection(name string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Collection = name
	}
}

// WithTopK sets the maximum number of results to return.
// The actual number of results may be less if MinScore filtering is applied.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithTopK(10), // Return top 10 results
//	)
func WithTopK(k int) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.TopK = k
	}
}

// WithMinScore sets the minimum similarity score threshold.
// Results with scores below this threshold will be filtered out.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithMinScore(0.8), // Only return high-confidence matches
//	)
func WithMinScore(score float64) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.MinScore = score
	}
}

// WithRetrieveDB configures the vector database connection.
// Supports various vector database implementations.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithRetrieveDB("milvus", "localhost:19530"),
//	)
func WithRetrieveDB(dbType, address string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.DBType = dbType
		c.DBAddress = address
	}
}

// WithRetrieveEmbedding configures the embedding service.
// Supports multiple embedding providers and models.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithRetrieveEmbedding(
//	        "openai",
//	        "text-embedding-3-small",
//	        os.Getenv("OPENAI_API_KEY"),
//	    ),
//	)
func WithRetrieveEmbedding(provider, model, key string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Provider = provider
		c.Model = model
		c.APIKey = key
	}
}

// WithHybrid enables or disables hybrid search.
// Hybrid search combines vector similarity with keyword matching.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithHybrid(true), // Enable hybrid search
//	)
func WithHybrid(enabled bool) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.UseHybrid = enabled
	}
}

// WithColumns specifies which columns to retrieve from the database.
// This can optimize performance by only fetching needed fields.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithColumns("Text", "Metadata", "Source"),
//	)
func WithColumns(columns ...string) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Columns = columns
	}
}

// WithRetrieveDimension sets the embedding vector dimension.
// This must match the dimension of your chosen embedding model.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithRetrieveDimension(1536), // OpenAI embedding dimension
//	)
func WithRetrieveDimension(dimension int) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.Dimension = dimension
	}
}

// WithRetrieveCallbacks sets result and error handling callbacks.
// These callbacks are called during the retrieval process.
//
// Example:
//
//	retriever, err := NewRetriever(
//	    WithRetrieveCallbacks(
//	        func(result SearchResult) {
//	            log.Printf("Found result: %v\n", result)
//	        },
//	        func(err error) {
//	            log.Printf("Error: %v\n", err)
//	        },
//	    ),
//	)
func WithRetrieveCallbacks(onResult func(SearchResult), onError func(error)) RetrieverOption {
	return func(c *RetrieverConfig) {
		c.OnResult = onResult
		c.OnError = onError
	}
}

// defaultRetrieverConfig returns a RetrieverConfig with production-ready defaults.
// These defaults are chosen to provide good performance while being
// conservative with resource usage.
//
// Default settings include:
//   - Top 10 results
//   - Minimum score of 0.7
//   - L2 distance metric
//   - 30-second timeout
//   - Standard column set (Text, Metadata)
func defaultRetrieverConfig() *RetrieverConfig {
	return &RetrieverConfig{
		Collection: "documents",
		TopK:       5,
		MinScore:   0.7,
		UseHybrid:  true,
		Columns:    []string{"Text", "Metadata"},
		DBType:     "milvus",
		DBAddress:  "localhost:19530",
		Dimension:  128,
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

func (r *Retriever) initialize() error {
	var err error

	r.vectorDB, err = NewVectorDB(
		WithType(r.config.DBType),
		WithAddress(r.config.DBAddress),
		WithTimeout(r.config.Timeout),
		WithDimension(r.config.Dimension),
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
		SetEmbedderProvider(r.config.Provider),
		SetEmbedderModel(r.config.Model),
		SetEmbedderAPIKey(r.config.APIKey),
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

// Package raggo implements a comprehensive Retrieval-Augmented Generation (RAG) system
// that enhances language models with the ability to access and reason over external
// knowledge. The system seamlessly integrates vector similarity search with natural
// language processing to provide accurate and contextually relevant responses.
//
// The package offers two main interfaces:
//   - RAG: A full-featured implementation with extensive configuration options
//   - SimpleRAG: A streamlined interface for basic use cases
//
// The RAG system works by:
// 1. Processing documents into semantic chunks
// 2. Storing document vectors in a configurable database
// 3. Finding relevant context through similarity search
// 4. Generating responses that combine retrieved context with queries
//
// Key Features:
//   - Multiple vector database support (Milvus, in-memory, Chrome)
//   - Intelligent document chunking and embedding
//   - Hybrid search capabilities
//   - Context-aware retrieval
//   - Configurable LLM integration
//
// Example Usage:
//
//	config := raggo.DefaultRAGConfig()
//	config.APIKey = os.Getenv("OPENAI_API_KEY")
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetProvider("openai"),
//	    raggo.SetModel("text-embedding-3-small"),
//	    raggo.WithMilvus("my_documents"),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Add documents
//	err = rag.LoadDocuments(context.Background(), "path/to/docs")
//
//	// Query the system
//	results, err := rag.Query(context.Background(), "your question here")
package raggo

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo/rag"
)

// RAGConfig holds the complete configuration for a RAG system. It provides
// fine-grained control over all aspects of the system's operation, from database
// settings to search parameters. The configuration is designed to be flexible
// enough to accommodate various use cases while maintaining sensible defaults.
type RAGConfig struct {
	// Database settings control how documents are stored and indexed
	DBType      string // Vector database type (e.g., "milvus", "memory")
	DBAddress   string // Database connection address
	Collection  string // Name of the vector collection
	AutoCreate  bool   // Automatically create collection if it doesn't exist
	IndexType   string // Type of vector index (e.g., "HNSW", "IVF")
	IndexMetric string // Distance metric for similarity (e.g., "L2", "IP")

	// Processing settings determine how documents are handled
	ChunkSize    int // Size of text chunks in tokens
	ChunkOverlap int // Overlap between consecutive chunks
	BatchSize    int // Number of documents to process in parallel

	// Embedding settings configure vector generation
	Provider string // Embedding provider (e.g., "openai", "cohere")
	Model    string // Embedding model name
	LLMModel string // Language model for text generation
	APIKey   string // API key for the provider

	// Search settings control retrieval behavior
	TopK      int     // Number of results to retrieve
	MinScore  float64 // Minimum similarity score threshold
	UseHybrid bool    // Whether to use hybrid search

	// System settings affect operational behavior
	Timeout time.Duration // Operation timeout
	TempDir string        // Directory for temporary files
	Debug   bool          // Enable debug logging

	// Search parameters for fine-tuning
	SearchParams map[string]interface{} // Provider-specific search parameters
}

// RAGOption is a function that modifies RAGConfig.
// It follows the functional options pattern for clean and flexible configuration.
type RAGOption func(*RAGConfig)

// RAG provides a comprehensive interface for document processing and retrieval.
// It coordinates the interaction between multiple components:
// - Vector database for efficient similarity search
// - Embedding service for semantic vector generation
// - Document processor for text chunking and enrichment
// - Language model for context-aware response generation
//
// The system is designed to be:
// - Thread-safe for concurrent operations
// - Memory-efficient when processing large documents
// - Extensible through custom implementations
// - Configurable for different use cases
type RAG struct {
	db       *VectorDB         // Vector database connection
	embedder *EmbeddingService // Service for generating embeddings
	config   *RAGConfig        // System configuration
}

// DefaultRAGConfig returns a default RAG configuration.
// It provides a reasonable set of default values for most use cases.
func DefaultRAGConfig() *RAGConfig {
	return &RAGConfig{
		DBType:       "milvus",
		DBAddress:    "localhost:19530",
		Collection:   "documents",
		AutoCreate:   true,
		IndexType:    "HNSW",
		IndexMetric:  "L2",
		ChunkSize:    512,
		ChunkOverlap: 64,
		BatchSize:    100,
		Provider:     "openai",
		Model:        "text-embedding-3-small",
		LLMModel:     "gpt-4o-mini", // Update to latest model
		APIKey:       os.Getenv("OPENAI_API_KEY"),
		TopK:         5,
		MinScore:     0.7,
		UseHybrid:    true,
		Timeout:      5 * time.Minute,
		TempDir:      os.TempDir(),
		SearchParams: map[string]interface{}{
			"nprobe": 10,
			"ef":     64,
			"type":   "HNSW",
		},
	}
}

// Common options
// SetProvider sets the embedding provider for the RAG system.
// Supported providers include "openai", "cohere", and others depending on implementation.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetProvider("openai"),
//	)
func SetProvider(provider string) RAGOption {
	return func(c *RAGConfig) {
		c.Provider = provider
	}
}

// SetModel specifies the embedding model to use for vector generation.
// The model should be compatible with the chosen provider.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetModel("text-embedding-3-small"),
//	)
func SetModel(model string) RAGOption {
	return func(c *RAGConfig) {
		c.Model = model
	}
}

// SetAPIKey configures the API key for the chosen provider.
// This key should have appropriate permissions for embedding and LLM operations.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
//	)
func SetAPIKey(key string) RAGOption {
	return func(c *RAGConfig) {
		c.APIKey = key
	}
}

// SetCollection specifies the name of the vector collection to use.
// This collection will store document embeddings and metadata.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetCollection("my_documents"),
//	)
func SetCollection(name string) RAGOption {
	return func(c *RAGConfig) {
		c.Collection = name
	}
}

// SetSearchStrategy configures the search approach for document retrieval.
// Supported strategies include "simple" for pure vector search and
// "hybrid" for combined vector and keyword search.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetSearchStrategy("hybrid"),
//	)
func SetSearchStrategy(strategy string) RAGOption {
	return func(c *RAGConfig) {
		c.UseHybrid = strategy == "hybrid"
	}
}

// SetDBAddress configures the connection address for the vector database.
// Format depends on the database type (e.g., "localhost:19530" for Milvus).
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetDBAddress("localhost:19530"),
//	)
func SetDBAddress(address string) RAGOption {
	return func(c *RAGConfig) {
		c.DBAddress = address
	}
}

// SetChunkSize configures the size of text chunks in tokens.
// Larger chunks provide more context but may reduce retrieval precision.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetChunkSize(512),
//	)
func SetChunkSize(size int) RAGOption {
	return func(c *RAGConfig) {
		c.ChunkSize = size
	}
}

// SetChunkOverlap specifies the overlap between consecutive chunks in tokens.
// Overlap helps maintain context across chunk boundaries.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetChunkOverlap(50),
//	)
func SetChunkOverlap(overlap int) RAGOption {
	return func(c *RAGConfig) {
		c.ChunkOverlap = overlap
	}
}

// SetTopK configures the number of similar documents to retrieve.
// Higher values provide more context but may introduce noise.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetTopK(5),
//	)
func SetTopK(k int) RAGOption {
	return func(c *RAGConfig) {
		c.TopK = k
	}
}

// SetMinScore sets the minimum similarity score threshold for retrieval.
// Documents with scores below this threshold are filtered out.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetMinScore(0.7),
//	)
func SetMinScore(score float64) RAGOption {
	return func(c *RAGConfig) {
		c.MinScore = score
	}
}

// SetTimeout configures the maximum duration for operations.
// This affects database operations, embedding generation, and LLM calls.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetTimeout(30 * time.Second),
//	)
func SetTimeout(timeout time.Duration) RAGOption {
	return func(c *RAGConfig) {
		c.Timeout = timeout
	}
}

// SetDebug enables or disables debug logging.
// When enabled, the system will output detailed operation information.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.SetDebug(true),
//	)
func SetDebug(debug bool) RAGOption {
	return func(c *RAGConfig) {
		c.Debug = debug
	}
}

// WithOpenAI is a convenience function that configures the RAG system
// to use OpenAI's embedding and language models.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
//	)
func WithOpenAI(apiKey string) RAGOption {
	return func(c *RAGConfig) {
		c.Provider = "openai"
		c.Model = "text-embedding-3-small"
		c.APIKey = apiKey
	}
}

// WithMilvus is a convenience function that configures the RAG system
// to use Milvus as the vector database with the specified collection.
//
// Example:
//
//	rag, err := raggo.NewRAG(
//	    raggo.WithMilvus("my_documents"),
//	)
func WithMilvus(collection string) RAGOption {
	return func(c *RAGConfig) {
		c.DBType = "milvus"
		c.DBAddress = "localhost:19530"
		c.Collection = collection
	}
}

// NewRAG creates a new RAG instance.
// It takes a variable number of RAGOption functions to configure the system.
func NewRAG(opts ...RAGOption) (*RAG, error) {
	cfg := DefaultRAGConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	rag := &RAG{config: cfg}
	if err := rag.initialize(); err != nil {
		return nil, err
	}

	return rag, nil
}

func (r *RAG) initialize() error {
	var err error

	// Initialize vector database
	r.db, err = NewVectorDB(
		WithType(r.config.DBType),
		WithAddress(r.config.DBAddress),
		WithTimeout(r.config.Timeout),
	)
	if err != nil {
		return fmt.Errorf("failed to create vector store: %w", err)
	}

	// Initialize embedder
	embedder, err := NewEmbedder(
		SetEmbedderProvider(r.config.Provider),
		SetEmbedderModel(r.config.Model),
		SetEmbedderAPIKey(r.config.APIKey),
	)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}
	r.embedder = NewEmbeddingService(embedder)

	return r.db.Connect(context.Background())
}

// LoadDocuments processes and stores documents in the vector database.
// It handles various document formats and automatically chunks text
// based on the configured chunk size and overlap.
//
// The source parameter can be a file path or directory. When a directory
// is provided, all supported documents within it are processed recursively.
//
// Example:
//
//	err := rag.LoadDocuments(ctx, "path/to/docs")
func (r *RAG) LoadDocuments(ctx context.Context, source string) error {
	loader := NewLoader(SetTempDir(r.config.TempDir))
	chunker, err := NewChunker(
		ChunkSize(r.config.ChunkSize),
		ChunkOverlap(r.config.ChunkOverlap),
	)
	if err != nil {
		return err
	}

	// Process source
	var paths []string
	if info, err := os.Stat(source); err == nil {
		if info.IsDir() {
			paths, err = loader.LoadDir(ctx, source)
		} else {
			path, err := loader.LoadFile(ctx, source)
			if err == nil {
				paths = []string{path}
			}
		}
		if err != nil {
			return fmt.Errorf("failed to load source: %w", err)
		}
	} else {
		return fmt.Errorf("invalid source: %s", source)
	}

	// Ensure collection exists
	if r.config.AutoCreate {
		if err := r.ensureCollection(ctx); err != nil {
			return err
		}
	}

	// Process documents
	for _, path := range paths {
		if err := r.processDocument(ctx, path, chunker); err != nil {
			return err
		}
	}

	return nil
}

// storeEnrichedChunks stores chunks with their context.
// It takes a context, enriched chunks, and a source path as input.
func (r *RAG) storeEnrichedChunks(ctx context.Context, enrichedChunks []string, source string) error {
	// Convert strings to Chunks
	chunks := make([]rag.Chunk, len(enrichedChunks))
	for i, text := range enrichedChunks {
		chunks[i] = rag.Chunk{
			Text:      text,
			TokenSize: len(text), // Simplified token count
		}
	}

	// Create embeddings
	embeddedChunks, err := r.embedder.EmbedChunks(ctx, chunks)
	if err != nil {
		return fmt.Errorf("failed to embed enriched chunks: %w", err)
	}

	// Convert to records
	records := make([]Record, len(embeddedChunks))
	for i, chunk := range embeddedChunks {
		records[i] = Record{
			Fields: map[string]interface{}{
				"Embedding": chunk.Embeddings["default"],
				"Text":      chunk.Text,
				"Metadata": map[string]interface{}{
					"source":     source,
					"chunk":      i,
					"total":      len(enrichedChunks),
					"token_size": chunk.Metadata["token_size"],
				},
			},
		}
	}

	// Store in database
	return r.db.Insert(ctx, r.config.Collection, records)
}

// ProcessWithContext processes and stores documents with additional contextual information.
// It takes a context, source path, and an optional LLM model as input.
func (r *RAG) ProcessWithContext(ctx context.Context, source string, llmModel string) error {
	Debug("Processing source:", source)

	// Ensure collection exists
	exists, err := r.db.HasCollection(ctx, r.config.Collection)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if !exists {
		// Create collection with schema
		schema := Schema{
			Name: r.config.Collection,
			Fields: []Field{
				{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
				{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
				{Name: "Text", DataType: "varchar", MaxLength: 65535},
				{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
			},
		}

		if err := r.db.CreateCollection(ctx, r.config.Collection, schema); err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}

		// Create index
		index := Index{
			Type:   r.config.IndexType,
			Metric: r.config.IndexMetric,
			Parameters: map[string]interface{}{
				"M":              16,
				"efConstruction": 256,
			},
		}

		if err := r.db.CreateIndex(ctx, r.config.Collection, "Embedding", index); err != nil {
			return fmt.Errorf("failed to create index: %w", err)
		}

		if err := r.db.LoadCollection(ctx, r.config.Collection); err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}
	}

	// Parse the document directly
	parser := NewParser()

	// Add a custom file type detector
	SetFileTypeDetector(parser, func(filePath string) string {
		ext := filepath.Ext(filePath)
		Debug("Detected file type for", filePath, ":", ext)
		switch ext {
		case ".txt":
			return "text"
		case ".pdf":
			return "pdf"
		// Add more file types as needed
		default:
			return "unknown"
		}
	})

	doc, err := parser.Parse(source)
	if err != nil {
		return fmt.Errorf("failed to parse document: %w", err)
	}

	// Create chunks
	chunker, _ := NewChunker(
		ChunkSize(r.config.ChunkSize),
		ChunkOverlap(r.config.ChunkOverlap),
	)
	chunks := chunker.Chunk(doc.Content)

	Debug("Number of chunks created:", len(chunks))

	// Use the provided llmModel if specified, otherwise use the config model
	modelToUse := r.config.LLMModel
	if llmModel != "" {
		modelToUse = llmModel
	}

	// Generate context for each chunk using LLM
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel(modelToUse),
		gollm.SetAPIKey(r.config.APIKey),
	)
	if err != nil {
		return fmt.Errorf("failed to initialize LLM: %w", err)
	}

	// Process chunks with context in batches
	for i := 0; i < len(chunks); i += r.config.BatchSize {
		end := min(i+r.config.BatchSize, len(chunks))
		batch := chunks[i:end]

		Debug("Processing batch", i/r.config.BatchSize+1, "of", (len(chunks)-1)/r.config.BatchSize+1)

		// Generate context and combine with text for batch
		enrichedChunks := make([]string, len(batch))
		for j, chunk := range batch {
			context, err := generateChunkContext(ctx, llm, doc.Content, chunk.Text)
			if err != nil {
				return fmt.Errorf("failed to generate context: %w", err)
			}

			// Combine context and content into a single enriched text
			enrichedChunks[j] = fmt.Sprintf("%s\n\nContent:\n%s", context, chunk.Text)

			Debug("Chunk", i+j+1, "of", len(chunks))
			Debug("Original content:", truncateString(chunk.Text, 100))
			Debug("Generated context:", context)
			Debug("Enriched content:", truncateString(enrichedChunks[j], 200))
		}

		// Create records with enriched text
		records := make([]Record, len(batch))
		for j := range batch {
			// Create embeddings for enriched text
			embedding, err := r.embedder.Embed(ctx, enrichedChunks[j])
			if err != nil {
				return fmt.Errorf("failed to embed text: %w", err)
			}

			records[j] = Record{
				Fields: map[string]interface{}{
					"Text":      enrichedChunks[j],
					"Embedding": embedding,
					"Metadata": map[string]interface{}{
						"source":     source,
						"chunk":      i + j,
						"total":      len(chunks),
						"token_size": len(enrichedChunks[j]),
					},
				},
			}
		}

		// Store in database
		if err := r.db.Insert(ctx, r.config.Collection, records); err != nil {
			return fmt.Errorf("failed to store chunks: %w", err)
		}
	}

	return nil
}

// Add this helper function if it doesn't exist
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func generateChunkContext(ctx context.Context, llm gollm.LLM, document, chunk string) (string, error) {
	documentContextPrompt := fmt.Sprintf("<document> %s </document>", document)
	chunkContextPrompt := fmt.Sprintf(`Analyze the following chunk from a larger document:
<chunk> %s </chunk>

Your task is to craft a concise, highly specific context (1-2 sentences) for this chunk. The context should:
1. Reflect the unique content and ideas presented in the chunk.
2. Relate the chunk's information to the broader themes of the document.
3. Be formulated in a way that enhances semantic search and retrieval.
4. Stand independently without relying on phrases like "This chunk" or "This section".
5. Use varied, natural language that avoids repetitive structures.

Provide only the context, without any introductory phrases or explanations.`, chunk)

	prompt := fmt.Sprintf("%s\n\n%s", documentContextPrompt, chunkContextPrompt)

	return llm.Generate(ctx, gollm.NewPrompt(prompt))
}

// Query performs a retrieval operation using the configured search strategy.
// It returns a slice of RetrieverResult containing relevant document chunks
// and their similarity scores.
//
// The query parameter should be a natural language question or statement.
// The system will convert it to a vector and find similar documents.
//
// Example:
//
//	results, err := rag.Query(ctx, "How does feature X work?")
func (r *RAG) Query(ctx context.Context, query string) ([]RetrieverResult, error) {
	if !r.config.UseHybrid {
		return r.simpleSearch(ctx, query)
	}
	return r.hybridSearch(ctx, query)
}

// Close releases all resources held by the RAG system, including
// database connections and embedding service clients.
//
// It should be called when the RAG system is no longer needed.
//
// Example:
//
//	defer rag.Close()
func (r *RAG) Close() error {
	if r.db != nil {
		return r.db.Close()
	}
	return nil
}

// Internal helper methods
func (r *RAG) ensureCollection(ctx context.Context) error {
	exists, err := r.db.HasCollection(ctx, r.config.Collection)
	if err != nil {
		return err
	}

	if !exists {
		schema := Schema{
			Name: r.config.Collection,
			Fields: []Field{
				{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
				{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
				{Name: "Text", DataType: "varchar", MaxLength: 65535},
				{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
			},
		}

		if err := r.db.CreateCollection(ctx, r.config.Collection, schema); err != nil {
			return err
		}

		index := Index{
			Type:   r.config.IndexType,
			Metric: r.config.IndexMetric,
			Parameters: map[string]interface{}{
				"M":              16,
				"efConstruction": 256,
			},
		}

		if err := r.db.CreateIndex(ctx, r.config.Collection, "Embedding", index); err != nil {
			return err
		}

		return r.db.LoadCollection(ctx, r.config.Collection)
	}

	return nil
}

func (r *RAG) processDocument(ctx context.Context, path string, chunker Chunker) error { // Changed: use interface
	parser := NewParser()
	doc, err := parser.Parse(path)
	if err != nil {
		return fmt.Errorf("failed to parse document: %w", err)
	}

	chunks := chunker.Chunk(doc.Content)
	embeddedChunks, err := r.embedder.EmbedChunks(ctx, chunks)
	if err != nil {
		return fmt.Errorf("failed to embed chunks: %w", err)
	}

	records := make([]Record, len(embeddedChunks))
	for i, chunk := range embeddedChunks {
		records[i] = Record{
			Fields: map[string]interface{}{
				"Embedding": chunk.Embeddings["default"],
				"Text":      chunk.Text,
				"Metadata": map[string]interface{}{
					"source":     path,
					"chunk":      i,
					"total":      len(chunks),
					"token_size": chunk.Metadata["token_size"],
				},
			},
		}
	}

	return r.db.Insert(ctx, r.config.Collection, records)
}

func (r *RAG) simpleSearch(ctx context.Context, query string) ([]RetrieverResult, error) {
	embedding, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	r.db.SetColumnNames([]string{"Text", "Metadata"})
	vectors := map[string]Vector{"Embedding": embedding}

	results, err := r.db.Search(
		ctx,
		r.config.Collection,
		vectors,
		r.config.TopK,
		r.config.IndexMetric,
		r.config.SearchParams,
	)
	if err != nil {
		return nil, err
	}

	return r.processResults(results), nil
}

func (r *RAG) hybridSearch(ctx context.Context, query string) ([]RetrieverResult, error) {
	embedding, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	r.db.SetColumnNames([]string{"Text", "Metadata"})
	vectors := map[string]Vector{"Embedding": embedding}

	results, err := r.db.HybridSearch(
		ctx,
		r.config.Collection,
		vectors,
		r.config.TopK,
		r.config.IndexMetric,
		r.config.SearchParams, // Use the config's search params
		nil,
	)
	if err != nil {
		return nil, err
	}

	return r.processResults(results), nil
}

func (r *RAG) processResults(results []SearchResult) []RetrieverResult {
	processed := make([]RetrieverResult, 0, len(results))

	for _, result := range results {
		if result.Score < r.config.MinScore {
			continue
		}

		content, _ := result.Fields["Text"].(string)
		metadata, _ := result.Fields["Metadata"].(map[string]interface{})

		retResult := RetrieverResult{
			Content:  content,
			Score:    result.Score,
			Metadata: metadata,
		}

		if metadata != nil {
			retResult.Source, _ = metadata["source"].(string)
			retResult.ChunkIndex, _ = metadata["chunk"].(int)
		}

		processed = append(processed, retResult)
	}

	return processed
}

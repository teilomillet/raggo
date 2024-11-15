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

// RAGConfig holds all RAG settings to avoid name collision with VectorDB Config
type RAGConfig struct {
	// Database settings
	DBType      string
	DBAddress   string
	Collection  string
	AutoCreate  bool
	IndexType   string
	IndexMetric string

	// Processing settings
	ChunkSize    int
	ChunkOverlap int
	BatchSize    int

	// Embedding settings
	Provider string
	Model    string // For embeddings
	LLMModel string // For LLM operations
	APIKey   string

	// Search settings
	TopK      int
	MinScore  float64
	UseHybrid bool

	// System settings
	Timeout time.Duration
	TempDir string
	Debug   bool

	// Search parameters
	SearchParams map[string]interface{} // Add this field
}

// RAGOption modifies RAGConfig
type RAGOption func(*RAGConfig)

// RAG provides a unified interface for document processing and retrieval
type RAG struct {
	db       *VectorDB
	embedder *EmbeddingService
	config   *RAGConfig
}

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
func WithOpenAI(apiKey string) RAGOption {
	return func(c *RAGConfig) {
		c.Provider = "openai"
		c.Model = "text-embedding-3-small"
		c.APIKey = apiKey
	}
}

func WithMilvus(collection string) RAGOption {
	return func(c *RAGConfig) {
		c.DBType = "milvus"
		c.DBAddress = "localhost:19530"
		c.Collection = collection
	}
}

// NewRAG creates a new RAG instance
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
		SetProvider(r.config.Provider),
		SetModel(r.config.Model),
		SetAPIKey(r.config.APIKey),
	)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}
	r.embedder = NewEmbeddingService(embedder)

	return r.db.Connect(context.Background())
}

// LoadDocuments processes and stores documents
func (r *RAG) LoadDocuments(ctx context.Context, source string) error {
	loader := NewLoader(SetTempDir(r.config.TempDir))
	chunker, err := NewChunker(
		ChunkSize(r.config.ChunkSize), // Changed: use correct option functions
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

// storeEnrichedChunks stores chunks with their context
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

// ProcessWithContext processes and stores documents with additional contextual information
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

// Query performs RAG retrieval and returns results
func (r *RAG) Query(ctx context.Context, query string) ([]RetrieverResult, error) {
	if !r.config.UseHybrid {
		return r.simpleSearch(ctx, query)
	}
	return r.hybridSearch(ctx, query)
}

// Close releases resources
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

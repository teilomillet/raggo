package raggo

import (
	"context"
	"fmt"
	"os"
	"time"
)

// RegisterConfig holds all configuration for the registration process
type RegisterConfig struct {
	// Storage settings
	VectorDBType   string            // e.g., "milvus"
	VectorDBConfig map[string]string // Connection settings
	CollectionName string            // Target collection
	AutoCreate     bool              // Create collection if missing

	// Processing settings
	ChunkSize      int
	ChunkOverlap   int
	BatchSize      int
	TempDir        string
	MaxConcurrency int
	Timeout        time.Duration

	// Embedding settings
	EmbeddingProvider string // e.g., "openai"
	EmbeddingModel    string // e.g., "text-embedding-3-small"
	EmbeddingKey      string

	// Callbacks
	OnProgress func(processed, total int)
	OnError    func(error)
}

// defaultConfig returns a configuration with sensible defaults
func defaultConfig() *RegisterConfig {
	return &RegisterConfig{
		VectorDBType:      "milvus",
		VectorDBConfig:    map[string]string{"address": "localhost:19530"},
		CollectionName:    "documents",
		AutoCreate:        true,
		ChunkSize:         512,
		ChunkOverlap:      64,
		BatchSize:         100,
		TempDir:           os.TempDir(),
		MaxConcurrency:    4,
		Timeout:           5 * time.Minute,
		EmbeddingProvider: "openai",
		EmbeddingModel:    "text-embedding-3-small",
		EmbeddingKey:      os.Getenv("OPENAI_API_KEY"),
		OnProgress: func(processed, total int) {
			Info(fmt.Sprintf("Progress: %d/%d", processed, total))
		},
		OnError: func(err error) {
			Error(fmt.Sprintf("Error: %v", err))
		},
	}
}

// RegisterOption is a function that modifies RegisterConfig
type RegisterOption func(*RegisterConfig)

// Register processes documents and stores them in a vector database.
// It accepts various sources: file paths, directory paths, or URLs.
func Register(ctx context.Context, source string, opts ...RegisterOption) error {
	// Initialize configuration
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(cfg)
	}

	// Create loader
	loader := NewLoader(
		SetTempDir(cfg.TempDir),
		SetTimeout(cfg.Timeout),
	)

	// Create chunker
	chunker, err := NewChunker(
		ChunkSize(cfg.ChunkSize),
		ChunkOverlap(cfg.ChunkOverlap),
	)
	if err != nil {
		return fmt.Errorf("failed to create chunker: %w", err)
	}

	// Create embedder
	embedder, err := NewEmbedder(
		SetProvider(cfg.EmbeddingProvider),
		SetModel(cfg.EmbeddingModel),
		SetAPIKey(cfg.EmbeddingKey),
	)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}

	// Create vector store
	vectorDB, err := NewVectorDB(
		WithType(cfg.VectorDBType),
		WithAddress(cfg.VectorDBConfig["address"]),
		WithTimeout(cfg.Timeout),
	)
	if err != nil {
		return fmt.Errorf("failed to create vector store: %w", err)
	}
	defer vectorDB.Close()

	if err := vectorDB.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to vector store: %w", err)
	}

	// Create collection if needed
	if cfg.AutoCreate {
		exists, _ := vectorDB.HasCollection(ctx, cfg.CollectionName)
		if !exists {
			schema := Schema{
				Name: cfg.CollectionName,
				Fields: []Field{
					{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
					{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
					{Name: "Text", DataType: "varchar", MaxLength: 65535},     // Added MaxLength
					{Name: "Metadata", DataType: "varchar", MaxLength: 65535}, // Added MaxLength
				},
			}

			// Create collection with schema
			if err := vectorDB.CreateCollection(ctx, cfg.CollectionName, schema); err != nil {
				return fmt.Errorf("failed to create collection: %w", err)
			}

			// Create index for vector field
			index := Index{
				Type:   "HNSW",
				Metric: "L2",
				Parameters: map[string]interface{}{
					"M":              16,
					"efConstruction": 256,
				},
			}
			if err := vectorDB.CreateIndex(ctx, cfg.CollectionName, "Embedding", index); err != nil {
				return fmt.Errorf("failed to create index: %w", err)
			}

			// Load collection
			if err := vectorDB.LoadCollection(ctx, cfg.CollectionName); err != nil {
				return fmt.Errorf("failed to load collection: %w", err)
			}
		}
	}
	// Process source
	var paths []string
	var loadErr error // Changed to use a different variable name
	if info, err := os.Stat(source); err == nil {
		if info.IsDir() {
			paths, loadErr = loader.LoadDir(ctx, source)
		} else {
			var path string
			path, loadErr = loader.LoadFile(ctx, source)
			if loadErr == nil {
				paths = []string{path}
			}
		}
		if loadErr != nil {
			return fmt.Errorf("failed to load source: %w", loadErr)
		}
	} else if isURL(source) {
		path, loadErr := loader.LoadURL(ctx, source)
		if loadErr != nil {
			return fmt.Errorf("failed to load URL: %w", loadErr)
		}
		paths = []string{path}
	} else {
		return fmt.Errorf("invalid source: %s", source)
	}

	// Create embedding service
	embeddingService := NewEmbeddingService(embedder)

	// Process files
	for i, path := range paths {
		// Parse content
		parser := NewParser()
		doc, err := parser.Parse(path)
		if err != nil {
			cfg.OnError(fmt.Errorf("failed to parse %s: %w", path, err))
			continue
		}

		// Create chunks
		chunks := chunker.Chunk(doc.Content)

		// Create embeddings
		embeddedChunks, err := embeddingService.EmbedChunks(ctx, chunks)
		if err != nil {
			cfg.OnError(fmt.Errorf("failed to embed chunks from %s: %w", path, err))
			continue
		}

		// Convert to records
		records := make([]Record, len(embeddedChunks))
		for j, chunk := range embeddedChunks {
			records[j] = Record{
				Fields: map[string]interface{}{
					"Embedding": chunk.Embeddings["default"],
					"Text":      chunk.Text,
					"Metadata": map[string]interface{}{
						"source":     path,
						"chunk":      j,
						"total":      len(chunks),
						"token_size": chunk.Metadata["token_size"],
					},
				},
			}
		}

		// Insert into vector store
		if err := vectorDB.Insert(ctx, cfg.CollectionName, records); err != nil {
			cfg.OnError(fmt.Errorf("failed to insert records from %s: %w", path, err))
			continue
		}

		cfg.OnProgress(i+1, len(paths))
	}

	return nil
}

// Configuration options

func WithVectorDB(dbType string, config map[string]string) RegisterOption {
	return func(c *RegisterConfig) {
		c.VectorDBType = dbType
		c.VectorDBConfig = config
	}
}

func WithCollection(name string, autoCreate bool) RegisterOption {
	return func(c *RegisterConfig) {
		c.CollectionName = name
		c.AutoCreate = autoCreate
	}
}

func WithChunking(size, overlap int) RegisterOption {
	return func(c *RegisterConfig) {
		c.ChunkSize = size
		c.ChunkOverlap = overlap
	}
}

func WithEmbedding(provider, model, key string) RegisterOption {
	return func(c *RegisterConfig) {
		c.EmbeddingProvider = provider
		c.EmbeddingModel = model
		c.EmbeddingKey = key
	}
}

func WithConcurrency(max int) RegisterOption {
	return func(c *RegisterConfig) {
		c.MaxConcurrency = max
	}
}

// Helper functions

func isURL(s string) bool {
	return len(s) > 8 && (s[:7] == "http://" || s[:8] == "https://")
}

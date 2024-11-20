package raggo

import (
	"context"
	"fmt"
	"os"
	"strconv"
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
		OnProgress:        func(processed, total int) { Debug("Progress", "processed", processed, "total", total) },
		OnError:          func(err error) { Error("Error during registration", "error", err) },
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

	Debug("Initializing registration", "source", source, "config", cfg)

	// Create loader
	Debug("Creating loader")
	loader := NewLoader(
		SetTempDir(cfg.TempDir),
		SetTimeout(cfg.Timeout),
	)

	// Create chunker
	Debug("Creating chunker")
	chunker, err := NewChunker(
		ChunkSize(cfg.ChunkSize),
		ChunkOverlap(cfg.ChunkOverlap),
	)
	if err != nil {
		return fmt.Errorf("failed to create chunker: %w", err)
	}

	// Create embedder
	Debug("Creating embedder")
	embedder, err := NewEmbedder(
		SetProvider(cfg.EmbeddingProvider),
		SetModel(cfg.EmbeddingModel),
		SetAPIKey(cfg.EmbeddingKey),
	)
	if err != nil {
		return fmt.Errorf("failed to create embedder: %w", err)
	}

	// Get embedding dimension
	Debug("Getting embedding dimension")
	dimension, err := embedder.GetDimension()
	if err != nil {
		return fmt.Errorf("failed to get embedding dimension: %w", err)
	}
	Debug("Embedding dimension", "dimension", dimension)

	// Create vector store
	Debug("Creating vector store")
	// Get dimension from config or use the one from embedder
	configDimension := 0
	if dimStr := cfg.VectorDBConfig["dimension"]; dimStr != "" {
		if dim, err := strconv.Atoi(dimStr); err == nil {
			configDimension = dim
		}
	}
	if configDimension == 0 {
		configDimension = dimension
	}
	Debug("Using dimension", "dimension", configDimension)

	vectorDB, err := NewVectorDB(
		WithType(cfg.VectorDBType),
		WithAddress(cfg.VectorDBConfig["address"]),
		WithTimeout(cfg.Timeout),
		WithDimension(configDimension),
	)
	if err != nil {
		return fmt.Errorf("failed to create vector store: %w", err)
	}
	defer vectorDB.Close()

	Debug("Connecting to vector store")
	if err := vectorDB.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to vector store: %w", err)
	}

	// Create collection if needed
	if cfg.AutoCreate {
		Debug("Checking collection existence", "collection", cfg.CollectionName)
		exists, _ := vectorDB.HasCollection(ctx, cfg.CollectionName)
		if !exists {
			Debug("Creating collection", "collection", cfg.CollectionName)
			schema := Schema{
				Name: cfg.CollectionName,
				Fields: []Field{
					{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
					{Name: "Embedding", DataType: "float_vector", Dimension: dimension},
					{Name: "Text", DataType: "varchar", MaxLength: 65535},
					{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
				},
			}

			// Create collection with schema
			if err := vectorDB.CreateCollection(ctx, cfg.CollectionName, schema); err != nil {
				return fmt.Errorf("failed to create collection: %w", err)
			}

			// Create index for vector field
			Debug("Creating index")
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
			Debug("Loading collection")
			if err := vectorDB.LoadCollection(ctx, cfg.CollectionName); err != nil {
				return fmt.Errorf("failed to load collection: %w", err)
			}
		}
	}

	// Process source
	Debug("Processing source", "source", source)
	var paths []string
	var loadErr error
	if info, err := os.Stat(source); err == nil {
		if info.IsDir() {
			Debug("Loading directory")
			paths, loadErr = loader.LoadDir(ctx, source)
		} else {
			Debug("Loading file")
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
		Debug("Loading URL")
		path, loadErr := loader.LoadURL(ctx, source)
		if loadErr != nil {
			return fmt.Errorf("failed to load URL: %w", loadErr)
		}
		paths = []string{path}
	} else {
		return fmt.Errorf("invalid source: %s", source)
	}

	// Create embedding service
	Debug("Creating embedding service")
	embeddingService := NewEmbeddingService(embedder)

	// Process files
	Debug("Processing files", "count", len(paths))
	for i, path := range paths {
		Debug("Processing file", "path", path, "index", i+1, "total", len(paths))

		// Parse content
		parser := NewParser()
		doc, err := parser.Parse(path)
		if err != nil {
			cfg.OnError(fmt.Errorf("failed to parse %s: %w", path, err))
			continue
		}

		// Create chunks
		Debug("Creating chunks")
		chunks := chunker.Chunk(doc.Content)
		Debug("Created chunks", "count", len(chunks))

		// Create embeddings
		Debug("Creating embeddings")
		embeddedChunks, err := embeddingService.EmbedChunks(ctx, chunks)
		if err != nil {
			cfg.OnError(fmt.Errorf("failed to embed chunks from %s: %w", path, err))
			continue
		}
		Debug("Created embeddings", "count", len(embeddedChunks))

		// Convert to records
		Debug("Converting to records")
		records := make([]Record, len(embeddedChunks))
		for j, chunk := range embeddedChunks {
			embedding, ok := chunk.Embeddings["default"]
			if !ok || len(embedding) == 0 {
				cfg.OnError(fmt.Errorf("missing or empty embedding for chunk %d in %s", j, path))
				continue
			}
			
			// Convert []float64 to []float32 for ChromemDB
			embedding32 := make([]float32, len(embedding))
			for i, v := range embedding {
				embedding32[i] = float32(v)
			}
			
			records[j] = Record{
				Fields: map[string]interface{}{
					"Embedding": embedding32,
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
		Debug("Inserting records", "count", len(records))
		if err := vectorDB.Insert(ctx, cfg.CollectionName, records); err != nil {
			cfg.OnError(fmt.Errorf("failed to insert records from %s: %w", path, err))
			continue
		}

		cfg.OnProgress(i+1, len(paths))
	}

	Debug("Registration complete")
	return nil
}

// Configuration options

func WithVectorDB(dbType string, config map[string]string) RegisterOption {
	return func(c *RegisterConfig) {
		c.VectorDBType = dbType
		if config == nil {
			config = make(map[string]string)
		}
		// Ensure dimension is preserved in VectorDBConfig
		if config["dimension"] == "" {
			config["dimension"] = "1536" // Default dimension
		}
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

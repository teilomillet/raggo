// Package raggo provides a comprehensive registration system for vector database
// implementations in RAG (Retrieval-Augmented Generation) applications. This
// package enables dynamic registration and management of vector databases with
// support for concurrent operations, configurable processing, and extensible
// architecture.
package raggo

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/teilomillet/raggo/rag"
)

// RegisterConfig holds the complete configuration for document registration
// and vector database setup. It provides fine-grained control over all aspects
// of the registration process.
type RegisterConfig struct {
	// Storage settings control vector database configuration
	VectorDBType   string            // Type of vector database (e.g., "milvus", "pinecone")
	VectorDBConfig map[string]string // Database-specific configuration parameters
	CollectionName string            // Name of the collection to store vectors
	AutoCreate     bool              // Automatically create collection if missing

	// Processing settings define how documents are handled
	ChunkSize      int           // Size of text chunks for processing
	ChunkOverlap   int           // Overlap between consecutive chunks
	BatchSize      int           // Number of items to process in each batch
	TempDir        string        // Directory for temporary files
	MaxConcurrency int           // Maximum number of concurrent operations
	Timeout        time.Duration // Operation timeout duration

	// Embedding settings configure the embedding generation
	EmbeddingProvider string // Embedding service provider (e.g., "openai")
	EmbeddingModel    string // Specific model to use for embeddings
	EmbeddingKey      string // Authentication key for embedding service

	// Callbacks for monitoring and error handling
	OnProgress func(processed, total int) // Called to report progress
	OnError    func(error)                // Called when errors occur
}

// defaultConfig returns a RegisterConfig initialized with production-ready
// default values. These defaults are carefully chosen to provide good
// performance while being conservative with resource usage.
//
// Default settings include:
//   - Milvus vector database on localhost
//   - 512-token chunks with 64-token overlap
//   - 100 items per batch
//   - 4 concurrent operations
//   - 5-minute timeout
//   - OpenAI's text-embedding-3-small model
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
		OnError:           func(err error) { Error("Error during registration", "error", err) },
	}
}

// RegisterOption is a function type for modifying RegisterConfig.
// It follows the functional options pattern to provide a clean and
// extensible way to configure the registration process.
type RegisterOption func(*RegisterConfig)

// Register processes documents from various sources and stores them in a vector
// database. It handles the entire pipeline from document loading to vector storage:
//  1. Document loading from files, directories, or URLs
//  2. Text chunking and preprocessing
//  3. Embedding generation
//  4. Vector database storage
//
// The process is highly configurable through RegisterOptions and supports
// progress monitoring and error handling through callbacks.
//
// Example:
//
//	err := Register(ctx, "docs/",
//	    WithVectorDB("milvus", map[string]string{
//	        "address": "localhost:19530",
//	    }),
//	    WithCollection("technical_docs", true),
//	    WithChunking(512, 64),
//	    WithEmbedding("openai", "text-embedding-3-small", os.Getenv("OPENAI_API_KEY")),
//	    WithConcurrency(4),
//	)
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
		SetLoaderTimeout(cfg.Timeout),
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
		SetEmbedderProvider(cfg.EmbeddingProvider),
		SetEmbedderModel(cfg.EmbeddingModel),
		SetEmbedderAPIKey(cfg.EmbeddingKey),
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

// WithVectorDB configures the vector database settings for registration.
// It specifies the database type and its configuration parameters.
//
// Supported database types include:
//   - "milvus": Milvus vector database
//   - "pinecone": Pinecone vector database
//   - "qdrant": Qdrant vector database
//
// Example:
//
//	Register(ctx, "docs/",
//	    WithVectorDB("milvus", map[string]string{
//	        "address": "localhost:19530",
//	        "user": "default",
//	        "password": "password",
//	    }),
//	)
func WithVectorDB(dbType string, config map[string]string) RegisterOption {
	return func(cfg *RegisterConfig) {
		cfg.VectorDBType = dbType
		if config == nil {
			config = make(map[string]string)
		}
		// Ensure dimension is preserved in VectorDBConfig
		if config["dimension"] == "" {
			config["dimension"] = "1536" // Default dimension
		}
		cfg.VectorDBConfig = config
	}
}

// WithCollection sets the collection name and auto-creation behavior.
// When autoCreate is true, the collection will be created if it doesn't
// exist, including appropriate indexes for vector similarity search.
//
// Example:
//
//	Register(ctx, "docs/",
//	    WithCollection("technical_docs", true),
//	)
func WithCollection(name string, autoCreate bool) RegisterOption {
	return func(cfg *RegisterConfig) {
		cfg.CollectionName = name
		cfg.AutoCreate = autoCreate
	}
}

// WithChunking configures the text chunking parameters for document processing.
// The size parameter determines the length of each chunk, while overlap
// specifies how much text should be shared between consecutive chunks.
//
// Example:
//
//	Register(ctx, "docs/",
//	    WithChunking(512, 64), // 512-token chunks with 64-token overlap
//	)
func WithChunking(size, overlap int) RegisterOption {
	return func(cfg *RegisterConfig) {
		cfg.ChunkSize = size
		cfg.ChunkOverlap = overlap
	}
}

// WithEmbedding configures the embedding generation settings.
// It specifies the provider, model, and authentication key for
// generating vector embeddings from text.
//
// Supported providers:
//   - "openai": OpenAI's embedding models
//   - "cohere": Cohere's embedding models
//   - "local": Local embedding models
//
// Example:
//
//	Register(ctx, "docs/",
//	    WithEmbedding("openai",
//	        "text-embedding-3-small",
//	        os.Getenv("OPENAI_API_KEY"),
//	    ),
//	)
func WithEmbedding(provider, model, key string) RegisterOption {
	return func(cfg *RegisterConfig) {
		cfg.EmbeddingProvider = provider
		cfg.EmbeddingModel = model
		cfg.EmbeddingKey = key
	}
}

// WithConcurrency sets the maximum number of concurrent operations
// during document processing. This affects:
//   - Document loading
//   - Chunk processing
//   - Embedding generation
//   - Vector storage
//
// Example:
//
//	Register(ctx, "docs/",
//	    WithConcurrency(8), // Process up to 8 items concurrently
//	)
func WithConcurrency(max int) RegisterOption {
	return func(cfg *RegisterConfig) {
		cfg.MaxConcurrency = max
	}
}

// isURL determines if a string represents a valid URL.
// It checks for common URL schemes (http, https, ftp).
func isURL(s string) bool {
	return len(s) > 8 && (s[:7] == "http://" || s[:8] == "https://")
}

// dbRegistry maintains a thread-safe registry of vector database implementations.
// It provides a central location for registering and retrieving database
// implementations, ensuring thread-safe access in concurrent environments.
//
// The registry supports:
//   - Dynamic registration of new implementations
//   - Thread-safe access to implementations
//   - Runtime discovery of available databases
type dbRegistry struct {
	mu        sync.RWMutex
	factories map[string]func(cfg *Config) (rag.VectorDB, error)
}

// registry is the global instance of dbRegistry that maintains all registered
// vector database implementations. It is initialized when the package is loaded
// and should be accessed through the package-level registration functions.
var registry = &dbRegistry{
	factories: make(map[string]func(cfg *Config) (rag.VectorDB, error)),
}

// RegisterVectorDB registers a new vector database implementation.
// It allows third-party implementations to be integrated into the
// Raggo ecosystem at runtime.
//
// Example:
//
//	RegisterVectorDB("custom_db", func(cfg *Config) (rag.VectorDB, error) {
//	    return NewCustomDB(cfg)
//	})
func RegisterVectorDB(dbType string, factory func(cfg *Config) (rag.VectorDB, error)) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	registry.factories[dbType] = factory
}

// GetVectorDB retrieves a vector database implementation from the registry.
// It returns an error if the requested implementation is not found or
// if creation fails.
//
// Example:
//
//	db, err := GetVectorDB("milvus", &Config{
//	    Address: "localhost:19530",
//	})
func GetVectorDB(dbType string, cfg *Config) (rag.VectorDB, error) {
	registry.mu.RLock()
	factory, ok := registry.factories[dbType]
	registry.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("vector database type not registered: %s", dbType)
	}

	return factory(cfg)
}

// ListRegisteredDBs returns a list of all registered vector database types.
// This is useful for discovering available implementations and validating
// configuration options.
//
// Example:
//
//	dbs := ListRegisteredDBs()
//	for _, db := range dbs {
//	    fmt.Printf("Supported database: %s\n", db)
//	}
func ListRegisteredDBs() []string {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	types := make([]string, 0, len(registry.factories))
	for dbType := range registry.factories {
		types = append(types, dbType)
	}
	return types
}

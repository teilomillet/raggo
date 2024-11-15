package raggo

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
)

// SimpleRAG provides a minimal interface for RAG operations
type SimpleRAG struct {
	retriever  *Retriever
	collection string
	apiKey     string
	model      string
	vectorDB   *VectorDB
	llm        gollm.LLM
}

// SimpleRAGConfig holds configuration for SimpleRAG
type SimpleRAGConfig struct {
	Collection   string
	APIKey       string
	Model        string
	ChunkSize    int
	ChunkOverlap int
	TopK         int
	MinScore     float64
	LLMModel     string
}

// DefaultConfig returns a default configuration
func DefaultConfig() SimpleRAGConfig {
	return SimpleRAGConfig{
		Collection:   "documents",
		Model:        "text-embedding-3-small",
		ChunkSize:    200,
		ChunkOverlap: 50,
		TopK:         5,
		MinScore:     0.1,
		LLMModel:     "gpt-4o-mini",
	}
}

// NewSimpleRAG creates a new SimpleRAG instance with minimal configuration
func NewSimpleRAG(config SimpleRAGConfig) (*SimpleRAG, error) {
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
		if config.APIKey == "" {
			return nil, fmt.Errorf("OpenAI API key is required")
		}
	}

	if config.Collection == "" {
		config.Collection = DefaultConfig().Collection
	}

	if config.Model == "" {
		config.Model = DefaultConfig().Model
	}

	if config.LLMModel == "" {
		config.LLMModel = DefaultConfig().LLMModel
	}

	// Initialize LLM
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel(config.LLMModel),
		gollm.SetAPIKey(config.APIKey),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize LLM: %w", err)
	}

	// Initialize vector database
	vectorDB, err := NewVectorDB(
		WithType("milvus"),
		WithAddress("localhost:19530"),
		WithTimeout(5*time.Minute),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector database: %w", err)
	}

	// Connect to the database
	ctx := context.Background()
	err = vectorDB.Connect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to vector database: %w", err)
	}

	// Check and drop existing collection
	exists, err := vectorDB.HasCollection(ctx, config.Collection)
	if err != nil {
		return nil, fmt.Errorf("failed to check collection: %w", err)
	}
	if exists {
		log.Println("Dropping existing collection")
		err = vectorDB.DropCollection(ctx, config.Collection)
		if err != nil {
			return nil, fmt.Errorf("failed to drop collection: %w", err)
		}
	}

	// Create retriever with configured options
	retriever, err := NewRetriever(
		WithRetrieveDB("milvus", "localhost:19530"),
		WithRetrieveCollection(config.Collection),
		WithTopK(config.TopK),
		WithMinScore(config.MinScore),
		WithHybrid(false), // Start with simple search
		WithRetrieveEmbedding(
			"openai",
			config.Model,
			config.APIKey,
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create retriever: %w", err)
	}

	return &SimpleRAG{
		retriever:  retriever,
		collection: config.Collection,
		apiKey:     config.APIKey,
		model:      config.Model,
		vectorDB:   vectorDB,
		llm:        llm,
	}, nil
}

// AddDocuments adds documents to the vector database
func (s *SimpleRAG) AddDocuments(ctx context.Context, source string) error {
	if ctx == nil {
		ctx = context.Background()
	}

	log.Printf("Adding documents from source: %s", source)

	// Check if source is a directory
	fileInfo, err := os.Stat(source)
	if err != nil {
		return fmt.Errorf("failed to stat source: %w", err)
	}

	if fileInfo.IsDir() {
		// Read all files in directory
		files, err := os.ReadDir(source)
		if err != nil {
			return fmt.Errorf("failed to read directory: %w", err)
		}

		// Process each file
		for _, file := range files {
			if !file.IsDir() && strings.HasSuffix(file.Name(), ".txt") {
				filePath := filepath.Join(source, file.Name())
				err := Register(ctx, filePath,
					WithCollection(s.collection, true),
					WithChunking(DefaultConfig().ChunkSize, DefaultConfig().ChunkOverlap),
					WithEmbedding("openai", s.model, s.apiKey),
				)
				if err != nil {
					return fmt.Errorf("failed to add document %s: %w", file.Name(), err)
				}
				log.Printf("Successfully processed file: %s", file.Name())
			}
		}
	} else {
		// Register single file
		err := Register(ctx, source,
			WithCollection(s.collection, true),
			WithChunking(DefaultConfig().ChunkSize, DefaultConfig().ChunkOverlap),
			WithEmbedding("openai", s.model, s.apiKey),
		)
		if err != nil {
			return fmt.Errorf("failed to add document: %w", err)
		}
	}

	// Create and load index
	err = s.vectorDB.CreateIndex(ctx, s.collection, "Embedding", Index{
		Type:   "HNSW",
		Metric: "L2",
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 256,
		},
	})
	if err != nil {
		return fmt.Errorf("failed to create index: %w", err)
	}

	// Load the collection
	err = s.vectorDB.LoadCollection(ctx, s.collection)
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	log.Printf("Successfully added documents from: %s", source)
	return nil
}

// Search performs a semantic search query and generates a response
func (s *SimpleRAG) Search(ctx context.Context, query string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	log.Printf("Performing search with query: %s", query)

	results, err := s.retriever.Retrieve(ctx, query)
	if err != nil {
		return "", fmt.Errorf("failed to search: %w", err)
	}

	log.Printf("Found %d results", len(results))

	// Prepare context from results
	var contexts []string
	for _, result := range results {
		contexts = append(contexts, result.Content)
	}

	// Generate response using LLM
	prompt := fmt.Sprintf(`Here are some relevant sections from our documentation:

%s

Based on this information, please answer the following question: %s

If the information isn't found in the provided context, please say so clearly.`,
		strings.Join(contexts, "\n\n---\n\n"),
		query,
	)

	resp, err := s.llm.Generate(ctx, gollm.NewPrompt(prompt))
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	return resp, nil
}

// Close releases resources
func (s *SimpleRAG) Close() error {
	if s.vectorDB != nil {
		s.vectorDB.Close()
	}
	return s.retriever.Close()
}

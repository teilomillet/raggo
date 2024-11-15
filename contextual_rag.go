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

// ContextualRAG provides a simplified interface for contextual RAG operations
type ContextualRAG struct {
	rag       *RAG
	retriever *Retriever
	llm       gollm.LLM
	llmModel  string
}

// ContextualRAGConfig holds configuration for ContextualRAG
type ContextualRAGConfig struct {
	Collection   string
	APIKey       string
	Model        string     // Embedding model
	LLMModel     string     // LLM model for context generation
	LLM          gollm.LLM  // Optional custom LLM instance
	ChunkSize    int
	ChunkOverlap int
	TopK         int
	MinScore     float64
}

// DefaultContextualConfig returns a default configuration
func DefaultContextualConfig() ContextualRAGConfig {
	return ContextualRAGConfig{
		Collection:   "contextual_docs",
		APIKey:       "",
		Model:        "text-embedding-3-small",
		LLMModel:     "gpt-4o-mini",
		ChunkSize:    200,    // Increased for better context
		ChunkOverlap: 50,     // Reasonable overlap
		TopK:         10,     // Number of results to return
		MinScore:     0.0,    // No minimum score to allow for more flexible matching
	}
}

// NewContextualRAG creates a new ContextualRAG instance with custom configuration
func NewContextualRAG(config *ContextualRAGConfig) (*ContextualRAG, error) {
	// Start with default configuration
	defaultConfig := DefaultContextualConfig()
	
	if config == nil {
		config = &defaultConfig
	} else {
		// Merge with defaults for any unset values
		if config.APIKey == "" {
			config.APIKey = defaultConfig.APIKey
		}
		if config.Model == "" {
			config.Model = defaultConfig.Model
		}
		if config.LLMModel == "" {
			config.LLMModel = defaultConfig.LLMModel
		}
		if config.ChunkSize == 0 {
			config.ChunkSize = defaultConfig.ChunkSize
		}
		if config.ChunkOverlap == 0 {
			config.ChunkOverlap = defaultConfig.ChunkOverlap
		}
		if config.TopK == 0 {
			config.TopK = defaultConfig.TopK
		}
		if config.MinScore == 0 {
			config.MinScore = defaultConfig.MinScore
		}
	}

	// Try to get API key from env if not set
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
		if config.APIKey == "" {
			return nil, fmt.Errorf("OpenAI API key is required: set it directly or via OPENAI_API_KEY environment variable")
		}
	}

	return initializeRAG(config)
}

// NewDefaultContextualRAG creates a new ContextualRAG instance with minimal configuration
func NewDefaultContextualRAG(collection string) (*ContextualRAG, error) {
	config := DefaultContextualConfig()
	config.Collection = collection
	
	// Get API key from environment
	config.APIKey = os.Getenv("OPENAI_API_KEY")
	if config.APIKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable is required")
	}
	
	return initializeRAG(&config)
}

// initializeRAG handles the actual initialization of the RAG system
func initializeRAG(config *ContextualRAGConfig) (*ContextualRAG, error) {
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

	// Create base RAG with optimized search settings
	ragConfig := DefaultRAGConfig()
	ragConfig.Collection = config.Collection
	ragConfig.Model = config.Model
	ragConfig.APIKey = config.APIKey
	ragConfig.ChunkSize = config.ChunkSize
	ragConfig.ChunkOverlap = config.ChunkOverlap
	ragConfig.TopK = config.TopK
	ragConfig.MinScore = config.MinScore
	ragConfig.UseHybrid = true
	ragConfig.SearchParams = map[string]interface{}{
		"nprobe": 10,
		"ef":     64,
		"type":   "HNSW",
	}

	// Initialize RAG with basic settings
	ragOpts := []RAGOption{
		WithOpenAI(config.APIKey),
		WithMilvus(config.Collection),
	}

	rag, err := NewRAG(ragOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create RAG: %w", err)
	}

	// Initialize Retriever with functional options
	retriever, err := NewRetriever(
		WithRetrieveDB("milvus", "localhost:19530"),
		WithRetrieveCollection(config.Collection),
		WithTopK(config.TopK),
		WithMinScore(config.MinScore),
		WithHybrid(true),
		WithRetrieveEmbedding(
			"openai",
			config.Model,
			config.APIKey,
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create retriever: %w", err)
	}

	// Initialize LLM if not provided
	var llm gollm.LLM
	if config.LLM != nil {
		llm = config.LLM
	} else {
		llm, err = gollm.NewLLM(
			gollm.SetProvider("openai"),
			gollm.SetModel(config.LLMModel),
			gollm.SetAPIKey(config.APIKey),
			gollm.SetMaxTokens(200),
			gollm.SetMaxRetries(3),
			gollm.SetRetryDelay(time.Second*2),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create LLM: %w", err)
		}
	}

	return &ContextualRAG{
		rag:       rag,
		retriever: retriever,
		llm:       llm,
		llmModel:  config.LLMModel,
	}, nil
}

// AddDocuments adds documents to the vector database with contextual information
func (c *ContextualRAG) AddDocuments(ctx context.Context, source string) error {
	if ctx == nil {
		ctx = context.Background()
	}

	log.Printf("Processing documents from: %s", source)

	// Check if source is a directory
	fileInfo, err := os.Stat(source)
	if err != nil {
		return fmt.Errorf("failed to stat source: %w", err)
	}

	// Process files with context
	if fileInfo.IsDir() {
		// Read all files in directory
		files, err := os.ReadDir(source)
		if err != nil {
			return fmt.Errorf("failed to read directory: %w", err)
		}

		// Process each file
		for _, file := range files {
			if !file.IsDir() {
				filePath := filepath.Join(source, file.Name())

				// Process file with context
				if err := c.rag.ProcessWithContext(ctx, filePath, c.llmModel); err != nil {
					return fmt.Errorf("failed to process file %s: %w", file.Name(), err)
				}
				log.Printf("Successfully processed file: %s", file.Name())
			}
		}
	} else {
		// Process single file with context
		if err := c.rag.ProcessWithContext(ctx, source, c.llmModel); err != nil {
			return fmt.Errorf("failed to process file: %w", err)
		}
	}

	// Create and load index
	err = c.rag.db.CreateIndex(ctx, c.rag.config.Collection, "Embedding", Index{
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
	err = c.rag.db.LoadCollection(ctx, c.rag.config.Collection)
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	log.Printf("Successfully added documents from: %s", source)
	return nil
}

// Search performs a semantic search query and returns a natural language response
func (c *ContextualRAG) Search(ctx context.Context, query string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	log.Printf("Searching for: %s", query)

	// Generate context for the query to improve search relevance
	queryContext, err := c.generateContext(ctx, query)
	if err != nil {
		log.Printf("Warning: Failed to generate query context: %v", err)
		// Continue with original query if context generation fails
		queryContext = query
	} else {
		log.Printf("Enhanced query with context: %s", queryContext)
	}

	// Get search results using retriever with both original query and context
	results, err := c.retriever.Retrieve(ctx, queryContext)
	if err != nil {
		return "", fmt.Errorf("search failed: %w", err)
	}

	if len(results) == 0 {
		log.Printf("No results found for query: %s", query)
		return "I could not find any relevant information to answer your question.", nil
	}

	log.Printf("Found %d results", len(results))

	// Build context from search results
	var contextBuilder strings.Builder
	contextBuilder.WriteString("Based on the following information:\n\n")

	// Track total relevance score
	totalScore := 0.0
	for i, result := range results {
		log.Printf("Result %d: Score=%.3f Source=%s", i+1, result.Score, filepath.Base(result.Source))
		contextBuilder.WriteString(fmt.Sprintf("%d. %s\n", i+1, result.Content))
		if result.Source != "" {
			contextBuilder.WriteString(fmt.Sprintf("   Source: %s (Score: %.3f)\n", filepath.Base(result.Source), result.Score))
		}
		totalScore += result.Score
	}

	avgScore := totalScore / float64(len(results))
	log.Printf("Average relevance score: %.3f", avgScore)

	contextBuilder.WriteString("\nPlease provide a comprehensive answer to this question: " + query)

	// Create a prompt for the LLM
	prompt := gollm.NewPrompt(contextBuilder.String())

	// Generate response using LLM
	response, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	return response, nil
}

// generateContext uses the LLM to generate a richer context for the query
func (c *ContextualRAG) generateContext(ctx context.Context, query string) (string, error) {
	prompt := gollm.NewPrompt(fmt.Sprintf(
		"Given this search query: '%s'\n"+
			"Generate a more detailed version that includes relevant context and related terms "+
			"to improve semantic search. Keep the enhanced query concise but comprehensive.",
		query))

	enhancedQuery, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		return "", err
	}

	return enhancedQuery, nil
}

// Close releases resources
func (c *ContextualRAG) Close() error {
	return c.rag.Close()
}

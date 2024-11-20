// Package raggo provides advanced Retrieval-Augmented Generation (RAG) capabilities
// with contextual awareness and memory management.
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

// ContextualRAG provides a high-level interface for context-aware document processing
// and retrieval. It enhances traditional RAG systems by:
//   - Maintaining semantic relationships between document chunks
//   - Generating rich contextual metadata for improved retrieval
//   - Supporting customizable chunking and embedding strategies
//   - Providing flexible LLM integration for response generation
//
// Example usage:
//
//	// Create with default settings
//	rag, err := raggo.NewDefaultContextualRAG("my_docs")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer rag.Close()
//	
//	// Add documents with automatic context generation
//	err = rag.AddDocuments(context.Background(), "path/to/docs")
//	
//	// Perform context-aware search
//	response, err := rag.Search(context.Background(), "How does feature X work?")
type ContextualRAG struct {
	rag       *RAG
	retriever *Retriever
	llm       gollm.LLM
	llmModel  string
}

// ContextualRAGConfig provides fine-grained control over the RAG system's behavior.
// It allows customization of:
//   - Document processing (chunk size, overlap)
//   - Embedding generation (model selection)
//   - Retrieval strategy (top-k, similarity threshold)
//   - LLM integration (custom instance, model selection)
//
// Example configuration:
//
//	config := &raggo.ContextualRAGConfig{
//	    Collection:   "technical_docs",
//	    Model:        "text-embedding-3-small",
//	    ChunkSize:    300,     // Larger chunks for more context
//	    ChunkOverlap: 50,      // Overlap for context continuity
//	    TopK:         5,       // Number of relevant chunks
//	    MinScore:     0.7,     // Similarity threshold
//	}
type ContextualRAGConfig struct {
	// Collection specifies the vector database collection name
	Collection string

	// APIKey for authentication with the embedding/LLM provider
	APIKey string

	// Model specifies the embedding model for vector generation
	Model string

	// LLMModel specifies the language model for context generation
	LLMModel string

	// LLM allows using a custom LLM instance with specific configuration
	LLM gollm.LLM

	// ChunkSize controls the size of document segments (in tokens)
	// Larger values preserve more context but increase processing time
	ChunkSize int

	// ChunkOverlap determines how much text overlaps between chunks
	// Higher values help maintain context across chunk boundaries
	ChunkOverlap int

	// TopK specifies how many similar chunks to retrieve
	// Adjust based on needed context breadth
	TopK int

	// MinScore sets the minimum similarity threshold for retrieval
	// Higher values increase precision but may reduce recall
	MinScore float64
}

// DefaultContextualConfig returns a balanced configuration suitable for
// most use cases. It provides:
//   - Reasonable chunk sizes for context preservation
//   - Modern embedding model selection
//   - Conservative similarity thresholds
//   - Efficient batch processing settings
func DefaultContextualConfig() ContextualRAGConfig {
	return ContextualRAGConfig{
		Collection:   "contextual_docs",
		Model:        "text-embedding-3-small",
		LLMModel:     "gpt-4o-mini",
		ChunkSize:    200,    // Balanced for most documents
		ChunkOverlap: 50,     // 25% overlap for context
		TopK:         10,     // Reasonable number of results
		MinScore:     0.0,    // No minimum for flexible matching
	}
}

// NewContextualRAG creates a new ContextualRAG instance with custom configuration.
// It provides advanced control over:
//   - Document processing behavior
//   - Embedding generation
//   - Retrieval strategies
//   - LLM integration
//
// The function will:
//   - Merge provided config with defaults
//   - Validate settings
//   - Initialize vector store
//   - Set up LLM integration
//
// Example:
//
//	config := &raggo.ContextualRAGConfig{
//	    Collection: "my_docs",
//	    ChunkSize:  300,
//	    TopK:       5,
//	}
//	rag, err := raggo.NewContextualRAG(config)
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

// NewDefaultContextualRAG creates a new instance with production-ready defaults.
// It's ideal for quick setup while maintaining good performance.
//
// The function:
//   - Uses environment variables for API keys
//   - Sets optimal processing parameters
//   - Configures reliable retrieval settings
//
// Example:
//
//	rag, err := raggo.NewDefaultContextualRAG("my_collection")
//	if err != nil {
//	    log.Fatal(err)
//	}
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

// AddDocuments processes and stores documents with contextual awareness.
// The function:
//   - Splits documents into semantic chunks
//   - Generates rich contextual metadata
//   - Creates and stores embeddings
//   - Maintains relationships between chunks
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - source: Path to document or directory
//
// Example:
//
//	err := rag.AddDocuments(ctx, "path/to/docs")
func (r *ContextualRAG) AddDocuments(ctx context.Context, source string) error {
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
				if err := r.rag.ProcessWithContext(ctx, filePath, r.llmModel); err != nil {
					return fmt.Errorf("failed to process file %s: %w", file.Name(), err)
				}
				log.Printf("Successfully processed file: %s", file.Name())
			}
		}
	} else {
		// Process single file with context
		if err := r.rag.ProcessWithContext(ctx, source, r.llmModel); err != nil {
			return fmt.Errorf("failed to process file: %w", err)
		}
	}

	// Create and load index
	err = r.rag.db.CreateIndex(ctx, r.rag.config.Collection, "Embedding", Index{
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
	err = r.rag.db.LoadCollection(ctx, r.rag.config.Collection)
	if err != nil {
		return fmt.Errorf("failed to load collection: %w", err)
	}

	log.Printf("Successfully added documents from: %s", source)
	return nil
}

// Search performs context-aware retrieval and generates a natural language response.
// The process:
//   1. Analyzes query for context requirements
//   2. Retrieves relevant document chunks
//   3. Synthesizes information with context preservation
//   4. Generates a coherent response
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - query: Natural language query string
//
// Example:
//
//	response, err := rag.Search(ctx, "How does the system handle errors?")
func (r *ContextualRAG) Search(ctx context.Context, query string) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	log.Printf("Searching for: %s", query)

	// Generate context for the query to improve search relevance
	queryContext, err := r.generateContext(ctx, query)
	if err != nil {
		log.Printf("Warning: Failed to generate query context: %v", err)
		// Continue with original query if context generation fails
		queryContext = query
	} else {
		log.Printf("Enhanced query with context: %s", queryContext)
	}

	// Get search results using retriever with both original query and context
	results, err := r.retriever.Retrieve(ctx, queryContext)
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
	response, err := r.llm.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %w", err)
	}

	return response, nil
}

// generateContext uses the LLM to generate a richer context for the query
func (r *ContextualRAG) generateContext(ctx context.Context, query string) (string, error) {
	prompt := gollm.NewPrompt(fmt.Sprintf(
		"Given this search query: '%s'\n"+
			"Generate a more detailed version that includes relevant context and related terms "+
			"to improve semantic search. Keep the enhanced query concise but comprehensive.",
		query))

	enhancedQuery, err := r.llm.Generate(ctx, prompt)
	if err != nil {
		return "", err
	}

	return enhancedQuery, nil
}

// Close releases all resources held by the ContextualRAG instance.
// Always defer Close() after creating a new instance.
func (r *ContextualRAG) Close() error {
	return r.rag.Close()
}

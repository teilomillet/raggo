// Package raggo provides advanced context-aware retrieval and memory management
// capabilities for RAG (Retrieval-Augmented Generation) systems.
package raggo

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
)

// MemoryContextOptions configures the behavior of the contextual memory system.
// It provides fine-grained control over memory storage, retrieval, and relevance
// assessment through a set of configurable parameters.
type MemoryContextOptions struct {
	// TopK determines how many relevant memories to retrieve for context
	// Higher values provide more context but may introduce noise
	TopK int

	// MinScore sets the minimum similarity threshold for memory retrieval
	// Higher values ensure more relevant but fewer memories
	MinScore float64

	// Collection specifies the vector database collection for storing memories
	Collection string

	// DBType specifies the vector database type (e.g., "chromem", "milvus")
	DBType string

	// DBAddress specifies the vector database address (e.g., "./.chromem/name" or "localhost:19530")
	DBAddress string

	// UseHybrid enables hybrid search (vector + keyword matching)
	// Note: Some vector databases like ChromaDB don't support hybrid search
	UseHybrid bool

	// IncludeScore determines whether to include relevance scores in results
	IncludeScore bool

	// StoreLastN limits memory storage to the N most recent interactions
	// Use 0 for unlimited storage
	StoreLastN int

	// StoreRAGInfo enables storage of RAG-enhanced context information
	// This provides richer context by storing processed and enhanced memories
	StoreRAGInfo bool
}

// MemoryContext provides intelligent context management for RAG systems by:
//   - Storing and retrieving relevant past interactions
//   - Enriching queries with historical context
//   - Managing memory lifecycle and relevance
//
// It integrates with vector databases for efficient similarity search and
// supports configurable memory management strategies.
type MemoryContext struct {
	retriever     *Retriever
	options       MemoryContextOptions
	lastStoreTime time.Time
	lastMemoryLen int
}

// MemoryTopK configures the number of relevant memories to retrieve.
// Higher values provide more comprehensive context but may impact performance.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryTopK(5),  // Retrieve top 5 relevant memories
//	)
func MemoryTopK(k int) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.TopK = k
	}
}

// MemoryMinScore sets the similarity threshold for memory retrieval.
// Memories with scores below this threshold are considered irrelevant.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryMinScore(0.7),  // Only retrieve highly similar memories
//	)
func MemoryMinScore(score float64) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.MinScore = score
	}
}

// MemoryCollection specifies the vector database collection for storing memories.
// Different collections can be used to separate contexts for different use cases.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryCollection("support_chat"),  // Store support chat memories
//	)
func MemoryCollection(collection string) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.Collection = collection
	}
}

// MemoryVectorDB configures the vector database type and address.
// This allows choosing between different vector store implementations and their locations.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryVectorDB("chromem", "./.chromem/chat"),  // Use Chromem store
//	)
func MemoryVectorDB(dbType, address string) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.DBType = dbType
		o.DBAddress = address
	}
}

// MemoryHybridSearch enables or disables hybrid search (vector + keyword matching).
// Note that some vector databases like ChromemDB don't support hybrid search.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryHybridSearch(false),  // Disable hybrid search for ChromaDB
//	)
func MemoryHybridSearch(enabled bool) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.UseHybrid = enabled
	}
}

// MemoryScoreInclusion controls whether similarity scores are included in results.
// Useful for debugging or implementing custom relevance filtering.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryScoreInclusion(true),  // Include similarity scores
//	)
func MemoryScoreInclusion(include bool) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.IncludeScore = include
	}
}

// MemoryStoreLastN limits memory storage to the N most recent interactions.
// This helps manage memory usage and maintain relevant context windows.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryStoreLastN(100),  // Keep last 100 interactions
//	)
func MemoryStoreLastN(n int) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.StoreLastN = n
	}
}

// MemoryStoreRAGInfo enables storage of RAG-enhanced context information.
// This provides richer context by storing processed and enhanced memories.
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryStoreRAGInfo(true),  // Store enhanced context
//	)
func MemoryStoreRAGInfo(store bool) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.StoreRAGInfo = store
	}
}

// NewMemoryContext creates a new memory context manager with the specified options.
// It initializes the underlying vector store and configures memory management.
//
// The function supports multiple configuration options through functional parameters:
//   - TopK: Number of relevant memories to retrieve
//   - MinScore: Similarity threshold for relevance
//   - Collection: Vector store collection name
//   - StoreLastN: Recent memory retention limit
//   - StoreRAGInfo: Enhanced context storage
//
// Example:
//
//	ctx, err := raggo.NewMemoryContext(apiKey,
//	    raggo.MemoryTopK(5),
//	    raggo.MemoryMinScore(0.7),
//	    raggo.MemoryCollection("chat_memory"),
//	    raggo.MemoryStoreLastN(100),
//	    raggo.MemoryStoreRAGInfo(true),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer ctx.Close()
func NewMemoryContext(apiKey string, opts ...func(*MemoryContextOptions)) (*MemoryContext, error) {
	// Default options
	options := MemoryContextOptions{
		TopK:         3,
		MinScore:     0.7,
		Collection:   "memory_store",
		DBType:       "milvus",          // Default to Milvus
		DBAddress:    "localhost:19530", // Default Milvus address
		UseHybrid:    false,
		IncludeScore: false,
		StoreLastN:   0,
		StoreRAGInfo: false,
	}

	// Apply custom options
	for _, opt := range opts {
		opt(&options)
	}

	// Initialize retriever with config
	retriever, err := NewRetriever(
		WithRetrieveCollection(options.Collection),
		WithTopK(options.TopK),
		WithMinScore(options.MinScore),
		WithRetrieveEmbedding("openai", "text-embedding-3-small", apiKey),
		WithRetrieveDB(options.DBType, options.DBAddress),
		WithHybrid(options.UseHybrid),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize retriever: %w", err)
	}

	// Ensure collection exists with proper schema
	ctx := context.Background()
	db := retriever.GetVectorDB()
	exists, err := db.HasCollection(ctx, options.Collection)
	if err != nil {
		return nil, fmt.Errorf("failed to check collection: %w", err)
	}

	if !exists {
		// Create collection with proper schema
		schema := Schema{
			Name:        options.Collection,
			Description: "Memory context collection for RAG",
			Fields: []Field{
				{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
				{Name: "Embedding", DataType: "float_vector", Dimension: 1536}, // text-embedding-3-small dimension
				{Name: "Text", DataType: "varchar", MaxLength: 65535},
				{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
			},
		}

		if err := db.CreateCollection(ctx, options.Collection, schema); err != nil {
			return nil, fmt.Errorf("failed to create collection: %w", err)
		}

		// Create index for vector search
		index := Index{
			Type:   "HNSW",
			Metric: "COSINE",
			Parameters: map[string]interface{}{
				"M":              16,
				"efConstruction": 256,
			},
		}

		if err := db.CreateIndex(ctx, options.Collection, "Embedding", index); err != nil {
			return nil, fmt.Errorf("failed to create index: %w", err)
		}

		if err := db.LoadCollection(ctx, options.Collection); err != nil {
			return nil, fmt.Errorf("failed to load collection: %w", err)
		}
	}

	return &MemoryContext{
		retriever:     retriever,
		options:       options,
		lastStoreTime: time.Time{},
		lastMemoryLen: 0,
	}, nil
}

// shouldStore determines whether to store the given memory based on configured rules.
// It checks:
//   - StoreLastN limits
//   - Time-based storage policies
//   - Memory content validity
func (m *MemoryContext) shouldStore(memory []gollm.MemoryMessage) bool {
	newLen := len(memory)
	timeSinceLastStore := time.Since(m.lastStoreTime)
	messagesDiff := newLen - m.lastMemoryLen

	return messagesDiff >= m.options.StoreLastN/2 ||
		timeSinceLastStore > 5*time.Minute
}

// StoreMemory explicitly stores messages in the memory context.
// It processes and indexes the messages for later retrieval.
//
// Example:
//
//	err := ctx.StoreMemory(context.Background(), []gollm.MemoryMessage{
//	    {Role: "user", Content: "How does feature X work?"},
//	    {Role: "assistant", Content: "Feature X works by..."},
//	})
func (m *MemoryContext) StoreMemory(ctx context.Context, messages []gollm.MemoryMessage) error {
	if len(messages) == 0 {
		return nil
	}

	var memoryContent string
	for _, msg := range messages {
		memoryContent += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
	}

	// TODO: Implement document storage through the vector store
	// For now, we'll just store the content through the retriever
	_, err := m.retriever.Retrieve(ctx, memoryContent)
	if err != nil {
		return fmt.Errorf("failed to store memory: %w", err)
	}

	return nil
}

// StoreLastN stores only the most recent N messages from the memory.
// This helps maintain a sliding window of relevant context.
//
// Example:
//
//	err := ctx.StoreLastN(context.Background(), messages, 10)  // Keep last 10 messages
func (m *MemoryContext) StoreLastN(ctx context.Context, memory []gollm.MemoryMessage, n int) error {
	if !m.shouldStore(memory) {
		return nil
	}

	start := len(memory) - n
	if start < 0 {
		start = 0
	}

	err := m.StoreMemory(ctx, memory[start:])
	if err == nil {
		m.lastStoreTime = time.Now()
		m.lastMemoryLen = len(memory)
	}
	return err
}

// EnhancePrompt enriches a prompt with relevant context from memory.
// It retrieves and integrates past interactions to provide better context.
//
// Example:
//
//	enhanced, err := ctx.EnhancePrompt(context.Background(), prompt, messages)
func (m *MemoryContext) EnhancePrompt(ctx context.Context, prompt *gollm.Prompt, memory []gollm.MemoryMessage) (*gollm.Prompt, error) {
	relevantContext, err := m.retrieveContext(ctx, prompt.Input)
	if err != nil {
		return prompt, fmt.Errorf("failed to retrieve context: %w", err)
	}

	enhancedPrompt := gollm.NewPrompt(
		prompt.Input,
		gollm.WithSystemPrompt(prompt.SystemPrompt, gollm.CacheTypeEphemeral),
		gollm.WithContext(strings.Join(relevantContext, "\n")),
	)

	if m.options.StoreRAGInfo && len(relevantContext) > 0 {
		contextMsg := gollm.MemoryMessage{
			Role:    "system",
			Content: "Retrieved Context:\n" + strings.Join(relevantContext, "\n"),
		}
		memory = append(memory, contextMsg)
	}

	return enhancedPrompt, nil
}

// retrieveContext retrieves relevant context from stored memories.
// It uses vector similarity search to find the most relevant past interactions.
func (m *MemoryContext) retrieveContext(ctx context.Context, input string) ([]string, error) {
	results, err := m.retriever.Retrieve(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to search context: %w", err)
	}

	var relevantContext []string
	for _, result := range results {
		if result.Content != "" {
			if m.options.IncludeScore {
				relevantContext = append(relevantContext, fmt.Sprintf("[Score: %.2f] %s", result.Score, result.Content))
			} else {
				relevantContext = append(relevantContext, result.Content)
			}
		}
	}

	return relevantContext, nil
}

// Close releases resources held by the memory context.
// Always defer Close() after creating a new memory context.
func (m *MemoryContext) Close() error {
	return m.retriever.Close()
}

// GetRetriever returns the underlying retriever for advanced configuration.
// This provides access to low-level retrieval settings and operations.
func (m *MemoryContext) GetRetriever() *Retriever {
	return m.retriever
}

// GetOptions returns the current context options configuration.
// Useful for inspecting or copying the current settings.
func (m *MemoryContext) GetOptions() MemoryContextOptions {
	return m.options
}

// UpdateOptions allows updating context options at runtime.
// This enables dynamic reconfiguration of memory management behavior.
//
// Example:
//
//	ctx.UpdateOptions(
//	    raggo.MemoryTopK(10),      // Increase context breadth
//	    raggo.MemoryMinScore(0.8), // Raise relevance threshold
//	)
func (m *MemoryContext) UpdateOptions(opts ...func(*MemoryContextOptions)) {
	options := m.GetOptions()
	for _, opt := range opts {
		opt(&options)
	}
	m.options = options

	// Create new retriever with updated config
	if retriever, err := NewRetriever(
		WithRetrieveCollection(options.Collection),
		WithTopK(options.TopK),
		WithMinScore(options.MinScore),
	); err == nil {
		m.retriever = retriever
	}
}

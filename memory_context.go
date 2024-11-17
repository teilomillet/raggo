package raggo

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/teilomillet/gollm"
)

// MemoryContextOptions configures how the memory context works
type MemoryContextOptions struct {
	TopK         int
	MinScore     float64
	Collection   string
	IncludeScore bool
	StoreLastN   int
	StoreRAGInfo bool // Whether to store RAG-enhanced context in memory
}

// MemoryContext provides contextual memory for gollm using RAG
type MemoryContext struct {
	retriever     *Retriever
	options       MemoryContextOptions
	lastStoreTime time.Time
	lastMemoryLen int
}

// MemoryTopK sets the number of relevant memories to retrieve
func MemoryTopK(k int) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.TopK = k
	}
}

// MemoryMinScore sets the minimum similarity score threshold
func MemoryMinScore(score float64) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.MinScore = score
	}
}

// MemoryCollection sets the RAG collection name
func MemoryCollection(collection string) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.Collection = collection
	}
}

// MemoryScoreInclusion controls whether to include relevance scores
func MemoryScoreInclusion(include bool) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.IncludeScore = include
	}
}

// MemoryStoreLastN sets the number of recent messages to store
func MemoryStoreLastN(n int) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.StoreLastN = n
	}
}

// MemoryStoreRAGInfo controls whether to store RAG-enhanced context in memory
func MemoryStoreRAGInfo(store bool) func(*MemoryContextOptions) {
	return func(o *MemoryContextOptions) {
		o.StoreRAGInfo = store
	}
}

// NewMemoryContext creates a new memory context provider that works with gollm
func NewMemoryContext(apiKey string, opts ...func(*MemoryContextOptions)) (*MemoryContext, error) {
	// Default options
	options := MemoryContextOptions{
		TopK:         3,
		MinScore:     0.7,
		Collection:   "memory_store",
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
		WithRetrieveDB("milvus", "localhost:19530"), // Add default DB config
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
			Name: options.Collection,
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

// shouldStore determines whether to store the given memory
func (m *MemoryContext) shouldStore(memory []gollm.MemoryMessage) bool {
	newLen := len(memory)
	timeSinceLastStore := time.Since(m.lastStoreTime)
	messagesDiff := newLen - m.lastMemoryLen

	return messagesDiff >= m.options.StoreLastN/2 ||
		timeSinceLastStore > 5*time.Minute
}

// StoreMemory explicitly stores messages in the memory context
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

// StoreLastN stores the last N messages from the memory
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

// EnhancePrompt enriches a prompt with relevant context from memory
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

// retrieveContext retrieves relevant context from RAG
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

// Close releases resources
func (m *MemoryContext) Close() error {
	return m.retriever.Close()
}

// GetRetriever returns the underlying retriever instance for advanced configuration
func (m *MemoryContext) GetRetriever() *Retriever {
	return m.retriever
}

// GetOptions returns the current context options
func (m *MemoryContext) GetOptions() MemoryContextOptions {
	return m.options
}

// UpdateOptions allows updating context options at runtime
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

# Memory Context Documentation

## Overview
The Memory Context system provides an enhanced way to manage and utilize contextual information in RAG applications. It enables the storage and retrieval of previous interactions, document contexts, and related information to improve the quality of responses.

## Core Components

### MemoryContext Struct
The main component that manages contextual memory:

```go
type MemoryContext struct {
    retriever  *RAG
    config     *MemoryConfig
    lastN      int
    storeRAG   bool
}
```

### Configuration
Memory context can be configured using various options:

```go
type MemoryConfig struct {
    Collection  string  // Vector DB collection name
    TopK        int     // Number of similar contexts to retrieve
    MinScore    float64 // Minimum similarity score
    StoreLastN  int     // Number of recent interactions to store
    StoreRAG    bool    // Whether to store RAG information
}
```

## Key Features

### 1. Memory Management
- Store and retrieve recent interactions
- Configure the number of interactions to maintain
- Automatic cleanup of old memories

### 2. Context Enhancement
- Enrich queries with relevant historical context
- Maintain conversation coherence
- Support for multi-turn interactions

### 3. RAG Integration
- Seamless integration with the RAG system
- Enhanced document retrieval with historical context
- Configurable similarity thresholds

## Usage Examples

### Basic Memory Context Setup
```go
memoryContext, err := raggo.NewMemoryContext(apiKey,
    raggo.MemoryCollection("tech_docs"),
    raggo.MemoryTopK(5),
    raggo.MemoryMinScore(0.01),
    raggo.MemoryStoreLastN(10),
    raggo.MemoryStoreRAGInfo(true),
)
if err != nil {
    log.Fatal(err)
}
```

### Using Memory Context in Applications
```go
// Store new memory
err = memoryContext.Store(ctx, "user query", "system response")

// Retrieve relevant memories
memories, err := memoryContext.Retrieve(ctx, "current query")

// Process with context
response, err := memoryContext.ProcessWithContext(ctx, "user query")
```

## Best Practices

1. **Memory Configuration**
   - Set appropriate `StoreLastN` based on your use case
   - Configure `TopK` and `MinScore` for optimal context retrieval
   - Enable `StoreRAG` for enhanced context awareness

2. **Performance Considerations**
   - Monitor memory usage with large conversation histories
   - Use appropriate batch sizes for memory operations
   - Implement cleanup strategies for old memories

3. **Context Quality**
   - Regularly evaluate the quality of retrieved contexts
   - Adjust similarity thresholds based on application needs
   - Consider implementing context filtering mechanisms

## Advanced Features

### Custom Memory Processing
Implement custom memory processing logic:

```go
memoryContext.SetProcessor(func(ctx context.Context, memory Memory) (string, error) {
    // Custom processing logic
    return processedMemory, nil
})
```

### Memory Filtering
Apply filters to retrieved memories:

```go
memoryContext.SetFilter(func(memory Memory) bool {
    // Custom filtering logic
    return shouldIncludeMemory
})
```

## Example Use Cases

### 1. Chatbot Enhancement
```go
// Initialize memory context for chat
memoryContext, _ := raggo.NewMemoryContext(apiKey,
    raggo.MemoryCollection("chat_history"),
    raggo.MemoryStoreLastN(20),
)

// Process chat messages with context
for {
    response, _ := memoryContext.ProcessWithContext(ctx, userMessage)
    memoryContext.Store(ctx, userMessage, response)
}
```

### 2. Document Q&A System
```go
// Initialize memory context for document Q&A
memoryContext, _ := raggo.NewMemoryContext(apiKey,
    raggo.MemoryCollection("doc_qa"),
    raggo.MemoryTopK(3),
    raggo.MemoryStoreRAGInfo(true),
)

// Process document queries with context
response, _ := memoryContext.ProcessWithContext(ctx, userQuery)
```

## Integration with Vector Databases

The memory context system integrates seamlessly with vector databases for efficient storage and retrieval of contextual information:

```go
// Configure vector database integration
retriever := memoryContext.GetRetriever()
if err := retriever.GetVectorDB().LoadCollection(ctx, "tech_docs"); err != nil {
    log.Fatal(err)
}
```

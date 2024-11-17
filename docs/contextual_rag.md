# Contextual RAG Documentation

## Overview
The Contextual RAG implementation extends the basic RAG functionality by adding sophisticated context management capabilities. It enables the system to maintain and utilize contextual information across queries, making responses more coherent and contextually relevant.

## Core Components

### ContextualRAG Struct
```go
type ContextualRAG struct {
    rag           *RAG
    contextStore  map[string]string
    contextSize   int
    contextWindow int
}
```

### Configuration
The Contextual RAG system can be configured with various options:

```go
type ContextualConfig struct {
    // Base RAG configuration
    RAGConfig *RAGConfig

    // Context settings
    ContextSize   int  // Maximum size of stored context
    ContextWindow int  // Window size for context consideration
    UseMemory    bool // Whether to use memory for context
}
```

## Key Features

### 1. Context Management
- Maintains conversation history and context
- Configurable context window and size
- Automatic context pruning and relevance scoring

### 2. Enhanced Query Processing
- Context-aware query understanding
- Historical context integration
- Improved response coherence

### 3. Memory Integration
- Optional memory storage for long-term context
- Configurable memory retention
- Context-based memory retrieval

## Usage Examples

### Basic Setup
```go
contextualRAG, err := raggo.NewContextualRAG(
    raggo.WithBaseRAG(baseRAG),
    raggo.WithContextSize(5),
    raggo.WithContextWindow(3),
    raggo.WithMemory(true),
)
if err != nil {
    log.Fatal(err)
}
```

### Processing Queries with Context
```go
// Process a query with context
response, err := contextualRAG.ProcessQuery(ctx, "What is the latest development?")

// Add context explicitly
contextualRAG.AddContext("topic", "AI Development")
response, err = contextualRAG.ProcessQuery(ctx, "What are the challenges?")
```

## Best Practices

1. **Context Management**
   - Set appropriate context size based on use case
   - Regularly clean up old context
   - Monitor context relevance scores

2. **Query Processing**
   - Structure queries to leverage context
   - Use context hints when available
   - Monitor context window effectiveness

3. **Memory Usage**
   - Enable memory for long-running conversations
   - Configure memory retention appropriately
   - Implement context cleanup strategies

## Advanced Features

### Custom Context Processing
```go
contextualRAG.SetContextProcessor(func(context, query string) string {
    // Custom context processing logic
    return processedContext
})
```

### Context Filtering
```go
contextualRAG.SetContextFilter(func(context string) bool {
    // Custom filtering logic
    return isRelevant
})
```

## Example Use Cases

### 1. Multi-turn Conversations
```go
// Initialize contextual RAG for conversation
contextualRAG, _ := raggo.NewContextualRAG(
    raggo.WithContextSize(10),
    raggo.WithMemory(true),
)

// Process conversation turns
for {
    response, _ := contextualRAG.ProcessQuery(ctx, userQuery)
    contextualRAG.AddContext("conversation", response)
}
```

### 2. Document Analysis with Context
```go
// Initialize for document analysis
contextualRAG, _ := raggo.NewContextualRAG(
    raggo.WithContextWindow(5),
    raggo.WithDocumentContext(true),
)

// Process document sections
for _, section := range sections {
    contextualRAG.AddContext("document", section)
    analysis, _ := contextualRAG.ProcessQuery(ctx, "Analyze this section")
}
```

## Integration with Memory Systems

The contextual RAG system can be integrated with various memory systems:

```go
// Configure memory integration
memorySystem := raggo.NewMemorySystem(
    raggo.WithMemorySize(1000),
    raggo.WithMemoryType("semantic"),
)

contextualRAG.SetMemorySystem(memorySystem)
```

## Performance Considerations

1. **Context Size**
   - Larger context sizes increase memory usage
   - Monitor context processing overhead
   - Balance context size with response time

2. **Memory Usage**
   - Implement context cleanup strategies
   - Monitor memory consumption
   - Use appropriate indexing for context retrieval

3. **Query Processing**
   - Optimize context matching algorithms
   - Cache frequently used contexts
   - Implement context relevance scoring

## Error Handling

```go
// Handle context-related errors
if err := contextualRAG.ProcessQuery(ctx, query); err != nil {
    switch err.(type) {
    case *ContextSizeError:
        // Handle context size exceeded
    case *ContextProcessingError:
        // Handle processing error
    default:
        // Handle other errors
    }
}
```

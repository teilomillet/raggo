# RagGo Library Documentation Summary

## Overview
RagGo is a comprehensive Go library for implementing Retrieval-Augmented Generation (RAG) systems. It provides multiple implementations to suit different use cases, from simple document retrieval to complex context-aware applications.

## Components Overview

### 1. Core RAG (`rag.go`)
The foundation of the library, providing basic RAG functionality:
- Document processing and embedding
- Vector database integration
- Configurable search parameters
- Extensible architecture

[Detailed Documentation →](./rag.md)

### 2. Simple RAG (`simple_rag.go`)
A streamlined implementation for basic use cases:
- Minimal setup required
- Basic document processing
- Simple search functionality
- Ideal for straightforward applications

[Detailed Documentation →](./simple_rag.md)

### 3. Contextual RAG (`contextual_rag.go`)
Advanced implementation with context management:
- Conversation history tracking
- Context-aware responses
- Memory integration
- Suitable for complex applications

[Detailed Documentation →](./contextual_rag.md)

### 4. Memory Context (`memory_context.go`)
Enhanced memory management system:
- Long-term memory storage
- Context retention
- Configurable memory policies
- Integration with RAG systems

[Detailed Documentation →](./memory_context.md)

## Quick Start Guide

### 1. Basic Usage
```go
// Initialize basic RAG
rag, err := raggo.NewRAG(
    raggo.WithOpenAI(apiKey),
    raggo.WithMilvus("documents"),
)

// Process documents
err = rag.LoadDocuments(ctx, "path/to/docs")

// Query
results, err := rag.Query(ctx, "your query")
```

### 2. Simple RAG Usage
```go
// Initialize simple RAG
simpleRAG, err := raggo.NewSimpleRAG(
    raggo.WithVectorDB("milvus", "localhost:19530"),
    raggo.WithEmbeddings("openai", apiKey),
)

// Process and query
err = simpleRAG.AddDocument(ctx, "document.txt")
results, err := simpleRAG.Query(ctx, "query")
```

### 3. Contextual RAG Usage
```go
// Initialize contextual RAG
contextualRAG, err := raggo.NewContextualRAG(
    raggo.WithBaseRAG(baseRAG),
    raggo.WithContextSize(5),
    raggo.WithMemory(true),
)

// Process with context
response, err := contextualRAG.ProcessQuery(ctx, "query")
```

## Feature Comparison

| Feature                    | Simple RAG | Core RAG | Contextual RAG |
|---------------------------|------------|----------|----------------|
| Document Processing       | Basic      | Advanced | Advanced       |
| Context Management        | No         | Basic    | Advanced       |
| Memory Integration        | No         | Optional | Yes            |
| Search Capabilities       | Basic      | Advanced | Advanced       |
| Setup Complexity         | Low        | Medium   | High           |
| Resource Requirements    | Low        | Medium   | High           |

## Common Use Cases

### 1. Document Q&A
Best Implementation: Simple RAG
```go
simpleRAG, _ := raggo.NewSimpleRAG(
    raggo.WithCollection("qa_docs"),
    raggo.WithTopK(1),
)
```

### 2. Chatbot with Memory
Best Implementation: Contextual RAG
```go
contextualRAG, _ := raggo.NewContextualRAG(
    raggo.WithContextSize(10),
    raggo.WithMemory(true),
)
```

### 3. Document Analysis
Best Implementation: Core RAG
```go
rag, _ := raggo.NewRAG(
    raggo.WithChunkSize(512),
    raggo.WithHybridSearch(true),
)
```

## Example Implementations

### 1. Memory-Enhanced Chatbot
A sophisticated chatbot implementation demonstrating advanced RAG capabilities:
- Document-based knowledge integration
- Context-aware responses
- Memory retention across conversations
- Interactive CLI interface

[Detailed Documentation →](./chatbot_example.md)

```go
// Initialize components
memoryContext, err := raggo.NewMemoryContext(apiKey,
    raggo.MemoryCollection("tech_docs"),
    raggo.MemoryTopK(5),
    raggo.MemoryMinScore(0.01),
    raggo.MemoryStoreLastN(10),
)

// Process documents
for _, doc := range docs {
    content, err := os.ReadFile(doc)
    err = memoryContext.Store(ctx, filepath.Base(doc), string(content))
}

// Interactive chat loop
for {
    query := getUserInput()
    response, err := memoryContext.ProcessWithContext(ctx, query)
    fmt.Printf("\nResponse: %s\n", response)
}
```

Key Features:
- Vector database integration (Milvus)
- OpenAI LLM integration
- Document processing and storage
- Context-aware query processing
- Memory management
- Error handling and logging

[View Full Example →](../examples/memory_enhancer_example.go)

## Best Practices

### 1. Configuration
- Use appropriate chunk sizes (default: 512)
- Configure TopK based on use case
- Set reasonable timeouts
- Use environment variables for API keys

### 2. Performance
- Implement batch processing
- Monitor API usage
- Use connection pooling
- Cache frequent queries

### 3. Error Handling
- Implement proper error handling
- Use context for cancellation
- Close resources properly
- Monitor system resources

## Integration Examples

### 1. HTTP Server
```go
func handleQuery(w http.ResponseWriter, r *http.Request) {
    rag := getRAGInstance() // Get appropriate RAG instance
    results, err := rag.Query(r.Context(), r.URL.Query().Get("q"))
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(results)
}
```

### 2. CLI Application
```go
func main() {
    rag := initializeRAG() // Initialize appropriate RAG
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        fmt.Print("Query: ")
        if !scanner.Scan() {
            break
        }
        results, _ := rag.Query(context.Background(), scanner.Text())
        fmt.Printf("Results: %v\n", results)
    }
}
```

## Dependencies

- Vector Database: Milvus (default)
- Embedding Service: OpenAI (default)
- Go version: 1.16+

## Getting Started

1. **Installation**
   ```bash
   go get github.com/teilomillet/raggo
   ```

2. **Basic Setup**
   ```go
   import "github.com/teilomillet/raggo"
   ```

3. **Environment Variables**
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Code style
- Testing requirements
- Pull request process
- Documentation standards

## Resources

- [Examples Directory](../examples/)
- [API Reference](./api.md)
- [FAQ](./faq.md)
- [Troubleshooting Guide](./troubleshooting.md)

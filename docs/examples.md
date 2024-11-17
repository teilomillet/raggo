# RagGo Examples Documentation

This document provides detailed explanations of the example implementations in the RagGo library.

## Memory Enhancer Example

### Overview
The Memory Enhancer example demonstrates how to create a RAG system with enhanced memory capabilities for processing technical documentation and maintaining context across interactions.

### Key Components
```go
// Main components used in the example
- Vector Database (Milvus)
- Memory Context
- LLM Integration (OpenAI)
```

### Implementation Details

1. **Setup and Initialization**
```go
// Initialize LLM
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o-mini"),
    gollm.SetAPIKey(apiKey),
    gollm.SetLogLevel(gollm.LogLevelInfo),
)

// Initialize Vector Database
vectorDB, err := raggo.NewVectorDB(
    raggo.WithType("milvus"),
    raggo.WithAddress("localhost:19530"),
)

// Create Memory Context
memoryContext, err := raggo.NewMemoryContext(apiKey,
    raggo.MemoryCollection("tech_docs"),
    raggo.MemoryTopK(5),
    raggo.MemoryMinScore(0.01),
    raggo.MemoryStoreLastN(10),
    raggo.MemoryStoreRAGInfo(true),
)
```

2. **Document Processing**
```go
// Load and process technical documentation
docsDir := filepath.Join("examples", "chat", "docs")
docs := []string{
    filepath.Join(docsDir, "microservices.txt"),
    filepath.Join(docsDir, "vector_databases.txt"),
    // ... additional documents
}

for _, doc := range docs {
    content, err := os.ReadFile(doc)
    if err != nil {
        log.Printf("Warning: Failed to read %s: %v", doc, err)
        continue
    }
    // Store document content as memory
    err = memoryContext.Store(ctx, filepath.Base(doc), string(content))
    if err != nil {
        log.Printf("Warning: Failed to store %s: %v", doc, err)
    }
}
```

3. **Interactive Query Processing**
```go
// Process user queries with context
scanner := bufio.NewScanner(os.Stdin)
for {
    fmt.Print("\nEnter your question (or 'quit' to exit): ")
    if !scanner.Scan() {
        break
    }
    
    query := scanner.Text()
    if strings.ToLower(query) == "quit" {
        break
    }

    response, err := memoryContext.ProcessWithContext(ctx, query)
    if err != nil {
        log.Printf("Error processing query: %v", err)
        continue
    }

    fmt.Printf("\nResponse: %s\n", response)
}
```

## Best Practices Demonstrated

1. **Error Handling**
   - Proper error checking and logging
   - Graceful handling of document loading failures
   - User-friendly error messages

2. **Resource Management**
   - Proper initialization and cleanup of resources
   - Use of context for cancellation
   - Cleanup of vector database collections

3. **User Interaction**
   - Clear user prompts and instructions
   - Graceful exit handling
   - Informative response formatting

## Running the Example

1. **Prerequisites**
   ```bash
   # Set up environment variables
   export OPENAI_API_KEY=your_api_key

   # Ensure Milvus is running
   docker-compose up -d
   ```

2. **Running the Example**
   ```bash
   go run examples/memory_enhancer_example.go
   ```

3. **Example Interactions**
   ```
   Enter your question: What are microservices?
   Response: [Detailed response about microservices based on loaded documentation]

   Enter your question: How do vector databases work?
   Response: [Context-aware response about vector databases]
   ```

## Customization Options

1. **Document Sources**
   - Modify the `docs` slice to include different document sources
   - Adjust document processing logic for different file types

2. **Memory Settings**
   - Adjust `TopK` for different numbers of similar contexts
   - Modify `MinScore` for stricter/looser similarity matching
   - Change `StoreLastN` for different memory retention

3. **Model Configuration**
   - Change the LLM model for different capabilities
   - Adjust logging levels for debugging
   - Modify vector database settings

## Additional Examples

### Contextual Example
Located in `examples/contextual/main.go`, this example demonstrates:
- Advanced context management
- Multi-turn conversations
- Context-aware document processing

### Basic RAG Example
Located in `examples/basic/main.go`, this example shows:
- Simple RAG setup
- Basic document processing
- Query handling without memory enhancement

# Memory-Enhanced Chatbot Example Documentation

## Overview
The Memory Enhancer Example demonstrates how to build a sophisticated chatbot using RagGo's RAG and Memory Context features. This implementation showcases advanced capabilities like context retention, document-based knowledge, and interactive conversations.

## Features
- Document-based knowledge integration
- Context-aware responses
- Memory retention across conversations
- Interactive command-line interface
- Vector database integration
- Hybrid search capabilities

## Implementation Details

### 1. Setup and Configuration
```go
// Initialize LLM with OpenAI
llm, err := gollm.NewLLM(
    gollm.SetProvider("openai"),
    gollm.SetModel("gpt-4o-mini"),
    gollm.SetAPIKey(apiKey),
    gollm.SetLogLevel(gollm.LogLevelInfo),
)

// Configure Vector Database
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

### 2. Document Management
```go
// Define document sources
docsDir := filepath.Join("examples", "chat", "docs")
docs := []string{
    filepath.Join(docsDir, "microservices.txt"),
    filepath.Join(docsDir, "vector_databases.txt"),
    filepath.Join(docsDir, "rag_systems.txt"),
    filepath.Join(docsDir, "golang_basics.txt"),
    filepath.Join(docsDir, "embeddings.txt"),
}

// Process and store documents
for _, doc := range docs {
    content, err := os.ReadFile(doc)
    if err != nil {
        log.Printf("Warning: Failed to read %s: %v", doc, err)
        continue
    }
    err = memoryContext.Store(ctx, filepath.Base(doc), string(content))
    if err != nil {
        log.Printf("Warning: Failed to store %s: %v", doc, err)
    }
}
```

### 3. Interactive Chat Loop
```go
// Initialize scanner for user input
scanner := bufio.NewScanner(os.Stdin)

// Main chat loop
for {
    fmt.Print("\nEnter your question (or 'quit' to exit): ")
    if !scanner.Scan() {
        break
    }
    
    query := scanner.Text()
    if strings.ToLower(query) == "quit" {
        break
    }

    // Process query with context
    response, err := memoryContext.ProcessWithContext(ctx, query)
    if err != nil {
        log.Printf("Error processing query: %v", err)
        continue
    }

    fmt.Printf("\nResponse: %s\n", response)
}
```

## Usage Guide

### Prerequisites
1. Environment Setup
   ```bash
   # Set OpenAI API key
   export OPENAI_API_KEY=your_api_key

   # Start Milvus database
   docker-compose up -d
   ```

2. Document Preparation
   - Place your knowledge base documents in the `examples/chat/docs` directory
   - Supported formats: `.txt` files
   - Recommended document size: 1-10KB per file

### Running the Example
```bash
# Navigate to the project directory
cd raggo

# Run the example
go run examples/memory_enhancer_example.go
```

### Example Interactions
```
Enter your question: What are microservices?
Response: Based on the documentation, microservices are a software architecture pattern where applications are built as a collection of small, independent services. Each service runs in its own process and communicates through well-defined APIs...

Enter your question: How do they handle data?
Response: [Context-aware response about microservice data handling, building on the previous question]
```

## Configuration Options

### Memory Context Settings
```go
memoryContext, err := raggo.NewMemoryContext(apiKey,
    // Collection name for vector storage
    raggo.MemoryCollection("tech_docs"),
    
    // Number of similar contexts to retrieve
    raggo.MemoryTopK(5),
    
    // Minimum similarity score (0-1)
    raggo.MemoryMinScore(0.01),
    
    // Number of recent interactions to store
    raggo.MemoryStoreLastN(10),
    
    // Store RAG information for better context
    raggo.MemoryStoreRAGInfo(true),
)
```

### Vector Database Settings
```go
vectorDB, err := raggo.NewVectorDB(
    // Database type
    raggo.WithType("milvus"),
    
    // Connection address
    raggo.WithAddress("localhost:19530"),
)
```

## Best Practices

### 1. Document Organization
- Split large documents into smaller, focused files
- Use clear, descriptive filenames
- Maintain consistent document format
- Regular updates to knowledge base

### 2. Memory Management
- Configure appropriate `TopK` for your use case
- Set `MinScore` based on required accuracy
- Adjust `StoreLastN` based on conversation length
- Monitor memory usage

### 3. Error Handling
- Implement graceful error recovery
- Log errors appropriately
- Provide user-friendly error messages
- Handle connection issues

### 4. Performance Optimization
- Batch process documents when possible
- Monitor API rate limits
- Use appropriate chunk sizes
- Implement caching if needed

## Customization Guide

### 1. Adding New Document Types
```go
// Example: Add PDF support
func processPDFDocument(path string) (string, error) {
    // PDF processing logic
    return content, nil
}
```

### 2. Custom Response Processing
```go
// Example: Add response formatting
func formatResponse(response string) string {
    // Response formatting logic
    return formattedResponse
}
```

### 3. Extended Commands
```go
// Example: Add command handling
switch strings.ToLower(query) {
case "help":
    showHelp()
case "stats":
    showStats()
default:
    processQuery(query)
}
```

## Troubleshooting

### Common Issues and Solutions

1. **Connection Errors**
   ```
   Error: Failed to connect to Milvus
   Solution: Ensure Milvus is running and accessible
   ```

2. **API Rate Limits**
   ```
   Error: OpenAI API rate limit exceeded
   Solution: Implement rate limiting or increase limits
   ```

3. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Adjust batch sizes and memory settings
   ```

## Advanced Features

### 1. Conversation History
- Maintains context across multiple queries
- Enables follow-up questions
- Provides coherent conversation flow

### 2. Document Context
- Integrates knowledge from multiple sources
- Provides source-based responses
- Maintains document relevance

### 3. Memory Enhancement
- Improves response accuracy
- Enables learning from interactions
- Provides personalized responses

## Example Extensions

### 1. Web Interface
```go
// Example: Add HTTP endpoint
func handleChatRequest(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    response, err := memoryContext.ProcessWithContext(r.Context(), query)
    // Handle response
}
```

### 2. Slack Integration
```go
// Example: Add Slack bot
func handleSlackMessage(event *slack.MessageEvent) {
    response, _ := memoryContext.ProcessWithContext(context.Background(), event.Text)
    // Send response to Slack
}
```

## Metrics and Monitoring

### 1. Performance Metrics
- Response time tracking
- Memory usage monitoring
- API call statistics

### 2. Quality Metrics
- Response relevance scores
- Context utilization rates
- User satisfaction metrics

## Future Enhancements

1. **Planned Features**
   - Multi-language support
   - Advanced document processing
   - Enhanced context management

2. **Potential Improvements**
   - Response caching
   - Query optimization
   - Advanced error recovery

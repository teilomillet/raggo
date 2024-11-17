# Simple RAG Documentation

## Overview
The Simple RAG implementation provides a streamlined, easy-to-use interface for basic RAG operations. It's designed for straightforward use cases where advanced context management isn't required, offering a balance between functionality and simplicity.

## Core Components

### SimpleRAG Struct
```go
type SimpleRAG struct {
    embedder    *EmbeddingService
    vectorStore *VectorDB
    config      *SimpleConfig
}
```

### Configuration
Simple configuration options for basic RAG operations:

```go
type SimpleConfig struct {
    // Vector store settings
    VectorDBType    string
    VectorDBAddress string
    Collection      string

    // Embedding settings
    EmbeddingModel string
    APIKey         string

    // Search settings
    TopK     int
    MinScore float64
}
```

## Key Features

### 1. Simplified Document Processing
- Straightforward document ingestion
- Basic chunking and embedding
- Direct vector storage

### 2. Basic Search Functionality
- Simple similarity search
- Configurable result count
- Basic relevance scoring

### 3. Minimal Setup Required
- Default configurations
- Automatic resource management
- Simplified API

## Usage Examples

### Basic Setup
```go
simpleRAG, err := raggo.NewSimpleRAG(
    raggo.WithVectorDB("milvus", "localhost:19530"),
    raggo.WithEmbeddings("openai", apiKey),
    raggo.WithTopK(3),
)
if err != nil {
    log.Fatal(err)
}
```

### Document Processing
```go
// Process a single document
err = simpleRAG.AddDocument(ctx, "document.txt")

// Process multiple documents
err = simpleRAG.AddDocuments(ctx, []string{
    "doc1.txt",
    "doc2.txt",
})
```

### Query Processing
```go
// Simple query
results, err := simpleRAG.Query(ctx, "How does this work?")

// Query with custom parameters
results, err = simpleRAG.QueryWithParams(ctx, "How does this work?", 
    raggo.WithResultCount(5),
    raggo.WithMinScore(0.5),
)
```

## Best Practices

1. **Document Management**
   - Keep documents reasonably sized
   - Use appropriate file formats
   - Maintain clean document structure

2. **Query Optimization**
   - Keep queries focused and specific
   - Use appropriate TopK values
   - Monitor query performance

3. **Resource Management**
   - Close resources when done
   - Monitor memory usage
   - Use batch processing for large datasets

## Example Use Cases

### 1. Document Q&A System
```go
// Initialize simple RAG for Q&A
simpleRAG, _ := raggo.NewSimpleRAG(
    raggo.WithCollection("qa_docs"),
    raggo.WithTopK(1),
)

// Add documentation
simpleRAG.AddDocument(ctx, "documentation.txt")

// Process questions
answer, _ := simpleRAG.Query(ctx, "What is the installation process?")
```

### 2. Basic Search System
```go
// Initialize for search
simpleRAG, _ := raggo.NewSimpleRAG(
    raggo.WithTopK(5),
    raggo.WithMinScore(0.7),
)

// Add searchable content
simpleRAG.AddDocuments(ctx, []string{
    "content1.txt",
    "content2.txt",
})

// Search content
results, _ := simpleRAG.Query(ctx, "search term")
```

## Performance Tips

1. **Document Processing**
   - Use batch processing for multiple documents
   - Monitor embedding API usage
   - Implement rate limiting for API calls

2. **Query Optimization**
   - Cache frequent queries
   - Use appropriate TopK values
   - Monitor query latency

3. **Resource Usage**
   - Implement connection pooling
   - Monitor memory consumption
   - Use appropriate batch sizes

## Error Handling

```go
// Basic error handling
if err := simpleRAG.AddDocument(ctx, "doc.txt"); err != nil {
    switch err.(type) {
    case *FileError:
        // Handle file-related errors
    case *ProcessingError:
        // Handle processing errors
    default:
        // Handle other errors
    }
}
```

## Integration Examples

### With HTTP Server
```go
func handleQuery(w http.ResponseWriter, r *http.Request) {
    query := r.URL.Query().Get("q")
    results, err := simpleRAG.Query(r.Context(), query)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    json.NewEncoder(w).Encode(results)
}
```

### With CLI Application
```go
func main() {
    simpleRAG, _ := raggo.NewSimpleRAG(/* config */)
    scanner := bufio.NewScanner(os.Stdin)
    
    for {
        fmt.Print("Enter query: ")
        if !scanner.Scan() {
            break
        }
        
        results, _ := simpleRAG.Query(context.Background(), scanner.Text())
        fmt.Printf("Results: %v\n", results)
    }
}
```

## Limitations and Considerations

1. **No Advanced Context Management**
   - Limited to single-query operations
   - No conversation history
   - No context window management

2. **Basic Search Only**
   - No hybrid search capabilities
   - Limited to vector similarity
   - Basic relevance scoring

3. **Limited Customization**
   - Fixed chunking strategy
   - Basic embedding options
   - Simple configuration options

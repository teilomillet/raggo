# RAG (Retrieval-Augmented Generation) Documentation

## Overview
The RAG (Retrieval-Augmented Generation) package provides a powerful and flexible system for document processing, embedding, storage, and retrieval. It integrates with vector databases and language models to enable context-aware document processing and intelligent information retrieval.

## Core Components

### RAG Struct
The main component that provides a unified interface for document processing and retrieval:

```go
type RAG struct {
    db       *VectorDB
    embedder *EmbeddingService
    config   *RAGConfig
}
```

### Configuration
The `RAGConfig` struct holds all RAG settings:

```go
type RAGConfig struct {
    // Database settings
    DBType      string
    DBAddress   string
    Collection  string
    AutoCreate  bool
    IndexType   string
    IndexMetric string

    // Processing settings
    ChunkSize    int
    ChunkOverlap int
    BatchSize    int

    // Embedding settings
    Provider  string
    Model     string    // For embeddings
    LLMModel  string    // For LLM operations
    APIKey    string

    // Search settings
    TopK      int
    MinScore  float64
    UseHybrid bool

    // System settings
    Timeout    time.Duration
    TempDir    string
    Debug      bool
}
```

## Key Features

### 1. Document Processing
- Chunking documents with configurable size and overlap
- Enriching chunks with contextual information
- Batch processing for efficient handling of large documents

### 2. Search and Retrieval
- Simple vector similarity search
- Hybrid search combining vector and keyword-based approaches
- Configurable search parameters (TopK, MinScore)

### 3. Vector Database Integration
- Default support for Milvus
- Extensible design for other vector databases
- Automatic collection creation and management

### 4. Embedding Services
- Integration with OpenAI embeddings
- Configurable embedding models
- Extensible for other embedding providers

## Usage Examples

### Basic RAG Setup
```go
rag, err := raggo.NewRAG(
    raggo.WithOpenAI(apiKey),
    raggo.WithMilvus("documents"),
    raggo.SetChunkSize(512),
    raggo.SetTopK(5),
)
if err != nil {
    log.Fatal(err)
}
defer rag.Close()
```

### Loading Documents
```go
ctx := context.Background()
err = rag.LoadDocuments(ctx, "path/to/documents")
if err != nil {
    log.Fatal(err)
}
```

### Querying
```go
results, err := rag.Query(ctx, "your search query")
if err != nil {
    log.Fatal(err)
}
```

## Best Practices

1. **Configuration**
   - Use appropriate chunk sizes based on your content (default: 512)
   - Adjust TopK and MinScore based on your use case
   - Enable hybrid search for better results when appropriate

2. **Performance**
   - Use batch processing for large document sets
   - Configure appropriate timeouts
   - Monitor vector database performance

3. **Error Handling**
   - Always handle errors appropriately
   - Use context for cancellation and timeouts
   - Close resources using defer

4. **Security**
   - Never hardcode API keys
   - Use environment variables for sensitive configuration
   - Implement appropriate access controls for your vector database

## Advanced Features

### Context-Aware Processing
The RAG system can enrich document chunks with contextual information:

```go
err = rag.ProcessWithContext(ctx, "path/to/documents", "gpt-4")
```

### Custom Search Parameters
Configure specific search parameters for your use case:

```go
rag, err := raggo.NewRAG(
    raggo.SetSearchParams(map[string]interface{}{
        "nprobe": 10,
        "ef":     64,
        "type":   "HNSW",
    }),
)
```

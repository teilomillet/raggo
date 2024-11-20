# Raggo - Retrieval Augmented Generation Library

> A flexible RAG (Retrieval Augmented Generation) library for Go, designed to make document processing and context-aware AI interactions simple and efficient.

<p align="center">
  <strong>🔍 Smart Document Search • 💬 Context-Aware Responses • 🤖 Intelligent RAG</strong>
</p>

[![Go Reference](https://pkg.go.dev/badge/github.com/teilomillet/raggo.svg)](https://pkg.go.dev/github.com/teilomillet/raggo)
[![Go Report Card](https://goreportcard.com/badge/github.com/teilomillet/raggo)](https://goreportcard.com/report/github.com/teilomillet/raggo)
[![License](https://img.shields.io/github/license/teilomillet/raggo)](https://github.com/teilomillet/raggo/blob/main/LICENSE)


## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"github.com/teilomillet/raggo"
)

func main() {
	// Initialize RAG with default settings
	rag, err := raggo.NewSimpleRAG(raggo.DefaultConfig())
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer rag.Close()

	// Add documents from a directory
	err = rag.AddDocuments(context.Background(), "./docs")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// Search with natural language
	response, _ := rag.Search(context.Background(), "What are the key features?")
	fmt.Printf("Answer: %s\n", response)
}
```

## Configuration

Raggo provides a flexible configuration system that can be loaded from multiple sources (environment variables, JSON files, or programmatic defaults):

```go
// Load configuration (automatically checks standard paths)
cfg, err := config.LoadConfig()
if err != nil {
    log.Fatal(err)
}

// Or create a custom configuration
cfg := &config.Config{
    Provider:   "milvus",           // Vector store provider
    Model:      "text-embedding-3-small",
    Collection: "my_documents",
    
    // Search settings
    DefaultTopK:     5,      // Number of similar chunks to retrieve
    DefaultMinScore: 0.7,    // Similarity threshold
    
    // Document processing
    DefaultChunkSize:    300,  // Size of text chunks
    DefaultChunkOverlap: 50,   // Overlap between chunks
}

// Create RAG instance with config
rag, err := raggo.NewSimpleRAG(cfg)
```

Configuration can be saved for reuse:
```go
err := cfg.Save("~/.raggo/config.json")
```

Environment variables (take precedence over config files):
- `RAGGO_PROVIDER`: Service provider
- `RAGGO_MODEL`: Model identifier
- `RAGGO_COLLECTION`: Collection name
- `RAGGO_API_KEY`: Default API key



## Table of Contents

### Part 1: Core Components
1. [Quick Start](#quick-start)
2. [Building Blocks](#building-blocks)
   - [Document Loading](#document-loading)
   - [Text Parsing](#text-parsing)
   - [Text Chunking](#text-chunking)
   - [Embeddings](#embeddings)
   - [Vector Storage](#vector-storage)

### Part 2: RAG Implementations
1. [Simple RAG](#simple-rag)
   - [Basic Usage](#basic-usage)
   - [Document Q&A](#document-qa)
   - [Configuration](#configuration)
2. [Contextual RAG](#contextual-rag)
   - [Advanced Features](#advanced-features)
   - [Context Window](#context-window)
   - [Hybrid Search](#hybrid-search)
3. [Memory Context](#memory-context)
   - [Chat Applications](#chat-applications)
   - [Memory Management](#memory-management)
   - [Context Enhancement](#context-enhancement)
4. [Advanced Use Cases](#advanced-use-cases)
   - [Full Processing Pipeline](#full-processing-pipeline)
   - [Concurrent Processing](#concurrent-processing)
   - [Rate Limiting](#rate-limiting)

## Part 1: Core Components

### Quick Start

#### Prerequisites
```bash
# Set API key
export OPENAI_API_KEY=your-api-key

# Install Raggo
go get github.com/teilomillet/raggo
```

### Building Blocks

#### Document Loading
```go
loader := raggo.NewLoader(raggo.SetTimeout(1*time.Minute))
doc, err := loader.LoadURL(context.Background(), "https://example.com/doc.pdf")
```

#### Text Parsing
```go
parser := raggo.NewParser()
doc, err := parser.Parse("document.pdf")
```

#### Text Chunking
```go
chunker := raggo.NewChunker(raggo.ChunkSize(100))
chunks := chunker.Chunk(doc.Content)
```

#### Embeddings
```go
embedder := raggo.NewEmbedder(
    raggo.SetProvider("openai"),
    raggo.SetModel("text-embedding-3-small"),
)
```

#### Vector Storage
```go
db := raggo.NewVectorDB(raggo.WithMilvus("collection"))
```

## Part 2: RAG Implementations

### Simple RAG
Best for straightforward document Q&A:

```go
package main

import (
    "context"
    "log"
    "github.com/teilomillet/raggo"
)

func main() {
    // Initialize SimpleRAG
    rag, err := raggo.NewSimpleRAG(raggo.SimpleRAGConfig{
        Collection: "docs",
        Model:      "text-embedding-3-small",
        ChunkSize:  300,
        TopK:       3,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer rag.Close()

    // Add documents
    err = rag.AddDocuments(context.Background(), "./documents")
    if err != nil {
        log.Fatal(err)
    }

    // Search with different strategies
    basicResponse, _ := rag.Search(context.Background(), "What is the main feature?")
    hybridResponse, _ := rag.SearchHybrid(context.Background(), "How does it work?", 0.7)
    
    log.Printf("Basic Search: %s\n", basicResponse)
    log.Printf("Hybrid Search: %s\n", hybridResponse)
}
```

### Contextual RAG
For complex document understanding and context-aware responses:

```go
package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/teilomillet/raggo"
)

func main() {
	// Initialize RAG with default settings
	rag, err := raggo.NewDefaultContextualRAG("basic_contextual_docs")
	if err != nil {
		fmt.Printf("Failed to initialize RAG: %v\n", err)
		os.Exit(1)
	}
	defer rag.Close()

	// Add documents - the system will automatically:
	// - Split documents into semantic chunks
	// - Generate rich context for each chunk
	// - Store embeddings with contextual information
	docsPath := filepath.Join("examples", "docs")
	if err := rag.AddDocuments(context.Background(), docsPath); err != nil {
		fmt.Printf("Failed to add documents: %v\n", err)
		os.Exit(1)
	}

	// Simple search with automatic context enhancement
	query := "What are the key features of the product?"
	response, err := rag.Search(context.Background(), query)
	if err != nil {
		fmt.Printf("Failed to search: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nQuery: %s\nResponse: %s\n", query, response)
}
```

### Advanced Configuration

```go
// Create a custom configuration
config := &raggo.ContextualRAGConfig{
	Collection:   "advanced_contextual_docs",
	Model:        "text-embedding-3-small", // Embedding model
	LLMModel:     "gpt-4o-mini",           // Model for context generation
	ChunkSize:    300,                      // Larger chunks for more context
	ChunkOverlap: 75,                       // 25% overlap for better continuity
	TopK:         5,                        // Number of similar chunks to retrieve
	MinScore:     0.7,                      // Higher threshold for better relevance
}

// Initialize RAG with custom configuration
rag, err := raggo.NewContextualRAG(config)
if err != nil {
	log.Fatalf("Failed to initialize RAG: %v", err)
}
defer rag.Close()
```

### Memory Context
For chat applications and long-term context retention:

```go
package main

import (
    "context"
    "log"
    "github.com/teilomillet/raggo"
    "github.com/teilomillet/gollm"
)

func main() {
    // Initialize Memory Context
    memoryCtx, err := raggo.NewMemoryContext(
        os.Getenv("OPENAI_API_KEY"),
        raggo.MemoryTopK(5),
        raggo.MemoryCollection("chat"),
        raggo.MemoryStoreLastN(100),
        raggo.MemoryMinScore(0.7),
    )
    if err != nil {
        log.Fatal(err)
    }
    defer memoryCtx.Close()

    // Initialize Contextual RAG
    rag, err := raggo.NewContextualRAG(&raggo.ContextualRAGConfig{
        Collection: "docs",
        Model:     "text-embedding-3-small",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer rag.Close()

    // Example chat interaction
    messages := []gollm.MemoryMessage{
        {Role: "user", Content: "How does the authentication system work?"},
    }
    
    // Store conversation
    err = memoryCtx.StoreMemory(context.Background(), messages)
    if err != nil {
        log.Fatal(err)
    }
    
    // Get enhanced response with context
    prompt := &gollm.Prompt{Messages: messages}
    enhanced, _ := memoryCtx.EnhancePrompt(context.Background(), prompt, messages)
    response, _ := rag.Search(context.Background(), enhanced.Messages[0].Content)
    
    log.Printf("Response: %s\n", response)
}
```

### Advanced Use Cases

#### Full Processing Pipeline
Process large document sets with rate limiting and concurrent processing:

```go
package main

import (
    "context"
    "log"
    "sync"
    "time"
    "github.com/teilomillet/raggo"
    "golang.org/x/time/rate"
)

const (
    GPT_RPM_LIMIT   = 5000    // Requests per minute
    GPT_TPM_LIMIT   = 4000000 // Tokens per minute
    MAX_CONCURRENT  = 10      // Max concurrent goroutines
)

func main() {
    // Initialize components
    parser := raggo.NewParser()
    chunker := raggo.NewChunker(raggo.ChunkSize(500))
    embedder := raggo.NewEmbedder(
        raggo.SetProvider("openai"),
        raggo.SetModel("text-embedding-3-small"),
    )

    // Create rate limiters
    limiter := rate.NewLimiter(rate.Limit(GPT_RPM_LIMIT/60), GPT_RPM_LIMIT)
    
    // Process documents concurrently
    var wg sync.WaitGroup
    semaphore := make(chan struct{}, MAX_CONCURRENT)

    files, _ := filepath.Glob("./documents/*.pdf")
    for _, file := range files {
        wg.Add(1)
        semaphore <- struct{}{} // Acquire semaphore
        
        go func(file string) {
            defer wg.Done()
            defer func() { <-semaphore }() // Release semaphore
            
            // Wait for rate limit
            limiter.Wait(context.Background())
            
            // Process document
            doc, _ := parser.Parse(file)
            chunks := chunker.Chunk(doc.Content)
            embeddings, _ := embedder.CreateEmbeddings(chunks)
            
            log.Printf("Processed %s: %d chunks\n", file, len(chunks))
        }(file)
    }
    
    wg.Wait()
}
```

## Best Practices

### Resource Management
- Always use `defer Close()`
- Monitor memory usage
- Clean up old data

### Performance
- Use concurrent processing for large datasets
- Configure appropriate chunk sizes
- Enable hybrid search when needed

### Context Management
- Use Memory Context for chat applications
- Configure context window size
- Clean up old memories periodically

## Examples

Check `/examples` for more:
- Basic usage: `/examples/simple/`
- Context-aware: `/examples/contextual/`
- Chat applications: `/examples/chat/`
- Memory usage: `/examples/memory_enhancer_example.go`
- Full pipeline: `/examples/full_process.go`
- Benchmarks: `/examples/process_embedding_benchmark.go`

## License

MIT License - see [LICENSE](LICENSE) file

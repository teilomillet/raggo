# Raggo

Raggo helps your Go programs answer questions by looking through documents. Think of it as a smart assistant that reads your documents and answers questions about them.

[![Go Reference](https://pkg.go.dev/badge/github.com/teilomillet/raggo.svg)](https://pkg.go.dev/github.com/teilomillet/raggo)
[![Go Report Card](https://goreportcard.com/badge/github.com/teilomillet/raggo)](https://goreportcard.com/report/github.com/teilomillet/raggo)
[![License](https://img.shields.io/github/license/teilomillet/raggo)](https://github.com/teilomillet/raggo/blob/main/LICENSE)

## Getting Started

### 1. Install Raggo
```bash
go get github.com/teilomillet/raggo
```

### 2. Set up Milvus
Raggo uses [Milvus](https://milvus.io/) as its vector database (required). Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md) to set it up. Once installed, Raggo will automatically connect to Milvus at `localhost:19530`.

### 3. Set your OpenAI API key
```bash
export OPENAI_API_KEY=your-api-key
```

## Simple Examples

### 1. Ask a Question
This is the simplest way to use Raggo. Just create a RAG and ask a question:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "github.com/teilomillet/raggo"
)

func main() {
    // Create a new RAG
    rag, err := raggo.NewDefaultSimpleRAG("my_first_rag")
    if err != nil {
        log.Fatal(err)
    }

    // Ask a question
    answer, err := rag.Search(context.Background(), "What is a RAG system?")
    if err != nil {
        log.Fatal(err)
    }

    // Print the answer
    fmt.Println(answer)
}
```

### 2. Add Your Own Documents
Now let's add some documents and ask questions about them:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "github.com/teilomillet/raggo"
)

func main() {
    // Create a new RAG
    rag, err := raggo.NewDefaultSimpleRAG("my_docs_rag")
    if err != nil {
        log.Fatal(err)
    }

    // Add documents from a folder
    err = rag.AddDocuments(context.Background(), "./my_documents")
    if err != nil {
        log.Fatal(err)
    }

    // Ask a question about your documents
    answer, err := rag.Search(context.Background(), "What do my documents say about project deadlines?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(answer)
}
```

### 3. Smart Answers with Context
For better answers that remember context:

```go
package main

import (
    "context"
    "fmt"
    "log"
    "github.com/teilomillet/raggo"
)

func main() {
    // Create a RAG that remembers context
    rag, err := raggo.NewDefaultContextualRAG("my_smart_rag")
    if err != nil {
        log.Fatal(err)
    }

    // Add your documents
    err = rag.AddDocuments(context.Background(), "./my_documents")
    if err != nil {
        log.Fatal(err)
    }

    // Ask related questions
    ctx := context.Background()
    
    answer1, _ := rag.Search(ctx, "What is our company's return policy?")
    fmt.Println("First answer:", answer1)
    
    answer2, _ := rag.Search(ctx, "What about for electronics?")
    fmt.Println("Second answer:", answer2)
}
```

## What's Next?

Once you're comfortable with these basics, you can:
- Use custom language models
- Process documents in parallel
- Add vector database support
- Customize document processing
- Add visualization tools

Check out our [Advanced Examples](#advanced-examples) to learn more.

## Advanced Examples

### Basic Examples
- `simple/`: Basic RAG implementation
- `contextual/`: Contextual RAG with custom LLM
- `chat/`: Chat-based RAG system

### Advanced Examples
- `full_process.go`: Complete production pipeline
- `recruit_example.go`: Resume processing system
- `vectordb_example.go`: Vector database integration
- `tsne_example.go`: Embedding visualization

### Performance Examples
- `process_embedding_benchmark.go`: Performance testing
- `concurrent_loader_example.go`: Concurrent processing
- `rate_limiting_example.go`: API rate limiting

## Best Practices

### Production Deployment
1. Implement proper error handling and retries
2. Use rate limiting for API calls
3. Monitor system resources
4. Implement logging and metrics

### Performance Optimization
1. Enable concurrent processing
2. Use batch operations
3. Implement caching strategies
4. Monitor and optimize resource usage

### Security
1. Secure API keys and credentials
2. Implement proper access controls
3. Validate and sanitize inputs
4. Monitor for abuse

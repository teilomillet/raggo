package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/teilomillet/raggo"
)

func main() {
	// Set log level to Debug for more detailed output
	raggo.SetLogLevel(raggo.LogLevelDebug)

	// Create a new Parser
	parser := raggo.NewParser()

	// Create a new Chunker
	chunker, err := raggo.NewChunker(
		raggo.ChunkSize(100),
		raggo.ChunkOverlap(20),
		raggo.WithSentenceSplitter(raggo.SmartSentenceSplitter()),
		raggo.WithTokenCounter(raggo.NewDefaultTokenCounter()),
	)
	if err != nil {
		log.Fatalf("Failed to create chunker: %v", err)
	}

	// Create a new Embedder
	embedder, err := raggo.NewEmbedder(
		raggo.SetProvider("openai"),
		raggo.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.SetModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	// Create an EmbeddingService
	embeddingService := raggo.NewEmbeddingService(embedder)

	// Get the path to a PDF file in the testdata directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	pdfPath := filepath.Join(wd, "testdata", "CV.pdf")

	// Parse the PDF file
	doc, err := parser.Parse(pdfPath)
	if err != nil {
		log.Fatalf("Failed to parse PDF: %v", err)
	}

	fmt.Printf("Parsed document with %d characters\n", len(doc.Content))

	// Chunk the parsed content
	chunks := chunker.Chunk(doc.Content)

	fmt.Printf("Created %d chunks from the document\n", len(chunks))

	// Embed the chunks
	embeddedChunks, err := embeddingService.EmbedChunks(context.Background(), chunks)
	if err != nil {
		log.Fatalf("Failed to embed chunks: %v", err)
	}

	fmt.Printf("Successfully embedded %d chunks\n", len(embeddedChunks))

	// Create VectorDB instances for memory, Milvus with L2, and Milvus with IP
	memoryDB, err := raggo.NewVectorDB(
		raggo.SetType("memory"),
		raggo.SetDimension(1536),
	)
	if err != nil {
		log.Fatalf("Failed to create memory vector database: %v", err)
	}
	defer memoryDB.Close()

	milvusDBL2, err := raggo.NewVectorDB(
		raggo.SetType("milvus"),
		raggo.SetAddress("localhost:19530"),
		raggo.SetDimension(1536),
		raggo.SetMetric("l2"),
		raggo.SetIndexType("IVF_FLAT"),
		raggo.SetIndexParams(map[string]interface{}{
			"nlist": 1024,
		}),
		raggo.SetSearchParams(map[string]interface{}{
			"nprobe": 16,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to create Milvus L2 vector database: %v", err)
	}
	defer milvusDBL2.Close()

	milvusDBIP, err := raggo.NewVectorDB(
		raggo.SetType("milvus"),
		raggo.SetAddress("localhost:19530"),
		raggo.SetDimension(1536),
		raggo.SetMetric("ip"),
		raggo.SetIndexType("IVF_FLAT"),
		raggo.SetIndexParams(map[string]interface{}{
			"nlist": 1024,
		}),
		raggo.SetSearchParams(map[string]interface{}{
			"nprobe": 16,
		}),
	)
	if err != nil {
		log.Fatalf("Failed to create Milvus IP vector database: %v", err)
	}
	defer milvusDBIP.Close()

	// Save embeddings to all databases
	collectionName := "test_collection"
	err = memoryDB.SaveEmbeddings(context.Background(), collectionName, embeddedChunks)
	if err != nil {
		log.Fatalf("Failed to save embeddings to memory database: %v", err)
	}

	err = milvusDBL2.SaveEmbeddings(context.Background(), collectionName+"_l2", embeddedChunks)
	if err != nil {
		log.Fatalf("Failed to save embeddings to Milvus L2 database: %v", err)
	}

	err = milvusDBIP.SaveEmbeddings(context.Background(), collectionName+"_ip", embeddedChunks)
	if err != nil {
		log.Fatalf("Failed to save embeddings to Milvus IP database: %v", err)
	}

	// Perform a search using all databases
	query := embeddedChunks[0].Embedding // Use the first chunk as a query
	limit := 5
	searchParam := raggo.NewDefaultSearchParam()

	memoryResults, err := memoryDB.Search(context.Background(), collectionName, query, limit, searchParam)
	if err != nil {
		log.Fatalf("Failed to search memory database: %v", err)
	}

	milvusL2Results, err := milvusDBL2.Search(context.Background(), collectionName+"_l2", query, limit, searchParam)
	if err != nil {
		log.Fatalf("Failed to search Milvus L2 database: %v", err)
	}

	milvusIPResults, err := milvusDBIP.Search(context.Background(), collectionName+"_ip", query, limit, searchParam)
	if err != nil {
		log.Fatalf("Failed to search Milvus IP database: %v", err)
	}

	// Compare results
	fmt.Println("\nMemory Search Results:")
	printResults(memoryResults)

	fmt.Println("\nMilvus L2 Search Results:")
	printResults(milvusL2Results)

	fmt.Println("\nMilvus IP Search Results:")
	printResults(milvusIPResults)

	compareResults(memoryResults, milvusL2Results, milvusIPResults)
}

func compareResults(memoryResults, milvusL2Results, milvusIPResults []raggo.SearchResult) {
	fmt.Println("\nComparing Memory, Milvus L2, and Milvus IP results:")
	for i := 0; i < len(memoryResults) && i < len(milvusL2Results) && i < len(milvusIPResults); i++ {
		memoryResult := memoryResults[i]
		milvusL2Result := milvusL2Results[i]
		milvusIPResult := milvusIPResults[i]
		fmt.Printf("%d. Memory: (Score: %.4f, ID: %v), Milvus L2: (Score: %.4f, ID: %v), Milvus IP: (Score: %.4f, ID: %v)\n",
			i+1, memoryResult.Score, memoryResult.ID, milvusL2Result.Score, milvusL2Result.ID, milvusIPResult.Score, milvusIPResult.ID)
		if memoryResult.ID != milvusL2Result.ID || memoryResult.ID != milvusIPResult.ID {
			fmt.Println("   *** Results differ ***")
		}
	}
}

func printResults(results []raggo.SearchResult) {
	for i, result := range results {
		fmt.Printf("%d. Score: %f, Text: %s\n", i+1, result.Score, truncateString(result.Text, 50))
	}
}

func truncateString(s string, length int) string {
	if len(s) <= length {
		return s
	}
	return s[:length-3] + "..."
}

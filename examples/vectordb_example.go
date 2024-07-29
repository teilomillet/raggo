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

	// Get the path to the testdata directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}
	testdataPath := filepath.Join(wd, "testdata")

	// Process all files in the testdata directory
	var allEmbeddedChunks []raggo.EmbeddedChunk
	err = filepath.Walk(testdataPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			fmt.Printf("Processing file: %s\n", path)

			// Parse the file
			doc, err := parser.Parse(path)
			if err != nil {
				log.Printf("Failed to parse file %s: %v", path, err)
				return nil // Continue with next file
			}

			fmt.Printf("Parsed document with %d characters\n", len(doc.Content))

			// Chunk the parsed content
			chunks := chunker.Chunk(doc.Content)

			fmt.Printf("Created %d chunks from the document\n", len(chunks))

			// Embed the chunks
			embeddedChunks, err := embeddingService.EmbedChunks(context.Background(), chunks)
			if err != nil {
				log.Printf("Failed to embed chunks from file %s: %v", path, err)
				return nil // Continue with next file
			}

			fmt.Printf("Successfully embedded %d chunks\n", len(embeddedChunks))

			allEmbeddedChunks = append(allEmbeddedChunks, embeddedChunks...)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Failed to walk through testdata directory: %v", err)
	}

	fmt.Printf("Total embedded chunks from all files: %d\n", len(allEmbeddedChunks))

	// Create VectorDBManager instances for memory and Milvus
	memoryManager, err := raggo.NewVectorDBManager("memory",
		raggo.WithDimension(1536),
		raggo.WithTopK(5),
	)
	if err != nil {
		log.Fatalf("Failed to create memory VectorDBManager: %v", err)
	}
	defer memoryManager.Close()

	milvusManager, err := raggo.NewVectorDBManager("milvus",
		raggo.WithAddress("localhost:19530"),
		raggo.WithDimension(1536),
		raggo.WithTopK(5),
	)
	if err != nil {
		log.Fatalf("Failed to create Milvus VectorDBManager: %v", err)
	}
	defer milvusManager.Close()

	// Use both databases
	useVectorDB(memoryManager, "memory_collection", allEmbeddedChunks)
	useVectorDB(milvusManager, "milvus_collection", allEmbeddedChunks)
}

func useVectorDB(manager *raggo.VectorDBManager, collectionName string, embeddedChunks []raggo.EmbeddedChunk) {
	// Ensure the collection exists (create if not)
	if err := manager.EnsureCollection(collectionName); err != nil {
		log.Fatalf("Failed to ensure collection: %v", err)
	}

	// Prepare vectors for insertion
	vectors := make([][]float32, len(embeddedChunks))
	for i, chunk := range embeddedChunks {
		vectors[i] = convertToFloat32(chunk.Embeddings["default"])
	}

	// Insert vectors
	if err := manager.InsertVectors(collectionName, vectors, nil); err != nil {
		log.Fatalf("Failed to insert: %v", err)
	}

	if len(vectors) == 0 {
		fmt.Printf("No vectors to search in %s\n", collectionName)
		return
	}

	// Perform a regular search
	queryVector := vectors[0] // Use the first vector as a query

	fmt.Printf("\nPerforming regular search on %s\n", collectionName)
	results, err := manager.Search(collectionName, queryVector)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}

	for _, result := range results {
		fmt.Printf("Regular Search - ID: %d, Score: %f\n", result.ID, result.Score)
	}

	// Perform a hybrid search only if we have at least two vectors
	if len(vectors) >= 2 {
		queryVector1 := vectors[len(vectors)-2]
		queryVector2 := vectors[len(vectors)-1]

		fmt.Printf("\nPerforming hybrid search on %s\n", collectionName)
		hybridResults, err := manager.HybridSearch(collectionName, [][]float32{queryVector1, queryVector2})
		if err != nil {
			log.Fatalf("Failed to perform hybrid search: %v", err)
		}

		for _, result := range hybridResults {
			fmt.Printf("Hybrid Search - ID: %d, Score: %f\n", result.ID, result.Score)
		}
	} else {
		fmt.Printf("\nNot enough vectors for hybrid search in %s\n", collectionName)
	}

	// Delete the collection
	if err := manager.DeleteCollection(collectionName); err != nil {
		log.Fatalf("Failed to delete collection: %v", err)
	}
}

// Helper function to convert []float64 to []float32
func convertToFloat32(input []float64) []float32 {
	output := make([]float32, len(input))
	for i, v := range input {
		output[i] = float32(v)
	}
	return output
}

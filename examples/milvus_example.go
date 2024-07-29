package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/teilomillet/raggo"
)

func main() {
	// Test with MilvusDB
	testVectorDB("milvus", raggo.WithAddress("localhost:19530"), raggo.WithDimension(128), raggo.WithTopK(5))

	// Test with InMemoryDB
	testVectorDB("memory", raggo.WithDimension(128), raggo.WithTopK(5))
}

func testVectorDB(dbType string, opts ...raggo.ManagerOption) {
	fmt.Printf("Testing %s database\n", dbType)

	manager, err := raggo.NewVectorDBManager(dbType, opts...)
	if err != nil {
		log.Fatalf("Failed to create VectorDBManager: %v", err)
	}
	defer manager.Close()

	collectionName := "test_collection"

	// Ensure the collection exists (create if not)
	err = manager.EnsureCollection(collectionName)
	if err != nil {
		log.Fatalf("Failed to ensure collection: %v", err)
	}

	// Create 10000 random vectors
	vectors := make([][]float32, 10000)
	for i := range vectors {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = rand.Float32()
		}
		vectors[i] = vector
	}

	// Insert vectors
	err = manager.InsertVectors(collectionName, vectors, nil)
	if err != nil {
		log.Fatalf("Failed to insert vectors: %v", err)
	}

	// Perform a regular search
	queryVector := make([]float32, 128)
	for i := 0; i < 128; i++ {
		queryVector[i] = float32(i)
	}

	fmt.Println("Performing regular search")
	results, err := manager.Search(collectionName, queryVector)
	if err != nil {
		log.Fatalf("Failed to search: %v", err)
	}

	for _, result := range results {
		fmt.Printf("Regular Search - ID: %d, Score: %f\n", result.ID, result.Score)
	}

	// Perform a hybrid search
	queryVector1 := vectors[len(vectors)-2]
	queryVector2 := vectors[len(vectors)-1]

	fmt.Println("Performing hybrid search")
	hybridResults, err := manager.HybridSearch(collectionName, [][]float32{queryVector1, queryVector2})
	if err != nil {
		log.Fatalf("Failed to perform hybrid search: %v", err)
	}

	for _, result := range hybridResults {
		fmt.Printf("Hybrid Search - ID: %d, Score: %f\n", result.ID, result.Score)
		fmt.Printf("  Vector1: %v\n", result.Metadata["vector1"])
		fmt.Printf("  Vector2: %v\n", result.Metadata["vector2"])
	}

	// Delete the collection
	err = manager.DeleteCollection(collectionName)
	if err != nil {
		log.Fatalf("Failed to delete collection: %v", err)
	}

	fmt.Printf("Finished testing %s database\n\n", dbType)
}


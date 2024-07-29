package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/teilomillet/raggo"
)

func main() {
	// Set log level to Debug for more detailed output
	raggo.SetLogLevel(raggo.LogLevelDebug)

	textStorage := make(map[int64]string)

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

	// Create Milvus VectorDB instance
	milvusDB, err := raggo.NewVectorDB(
		raggo.WithType("milvus"),
		raggo.WithAddress("localhost:19530"),
		raggo.WithMaxPoolSize(10),
		raggo.WithTimeout(30*time.Second),
	)
	if err != nil {
		log.Fatalf("Failed to create Milvus vector database: %v", err)
	}
	defer milvusDB.Close()

	// Connect to the database
	err = milvusDB.Connect(context.Background())
	if err != nil {
		log.Fatalf("Failed to connect to Milvus database: %v", err)
	}

	// Create collection
	collectionName := "test_collection_hybrid"
	schema := raggo.Schema{
		Name: collectionName,
		Fields: []raggo.Field{
			{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: false},
			{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
		},
	}
	log.Printf("Created collection %s with schema: %+v", collectionName, schema)

	exists, err := milvusDB.HasCollection(context.Background(), collectionName)
	if err != nil {
		log.Fatalf("Failed to check if collection exists: %v", err)
	}
	if exists {
		err = milvusDB.DropCollection(context.Background(), collectionName)
		if err != nil {
			log.Fatalf("Failed to drop existing collection: %v", err)
		}
		log.Printf("Dropped existing collection %s\n", collectionName)
	}

	err = milvusDB.CreateCollection(context.Background(), collectionName, schema)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}

	// Insert embeddings
	var records []raggo.Record
	for i, chunk := range embeddedChunks {
		id := int64(i)
		records = append(records, raggo.Record{
			Fields: map[string]interface{}{
				"ID":        id,
				"Embedding": chunk.Embeddings["default"],
			},
		})
		textStorage[id] = chunk.Text
		log.Printf("Stored text for ID %d: %s", id, truncateString(chunk.Text, 50))
	}

	// Insert the records into Milvus
	err = milvusDB.Insert(context.Background(), collectionName, records)
	if err != nil {
		log.Fatalf("Failed to insert records: %v", err)
	}

	// for i, record := range records {
	// 	log.Printf("Record %d: %+v", i, record)
	// }

	// Flush the inserted data
	err = milvusDB.Flush(context.Background(), collectionName)
	if err != nil {
		log.Fatalf("Failed to flush data: %v", err)
	}

	// Create index
	index := raggo.Index{
		Type:   "HNSW",
		Metric: "L2",
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 256,
		},
	}
	// Create index on Embedding field
	err = milvusDB.CreateIndex(context.Background(), collectionName, "Embedding", index)
	if err != nil {
		log.Fatalf("Failed to create index on Embedding: %v", err)
	}

	// Load collection
	err = milvusDB.LoadCollection(context.Background(), collectionName)
	if err != nil {
		log.Fatalf("Failed to load collection: %v", err)
	}

	log.Printf("Performing search with %d embedded chunks", len(embeddedChunks))
	log.Printf("Text storage contents:")
	for id, text := range textStorage {
		log.Printf("ID %d: %s", id, truncateString(text, 50))
	}

	// Perform a hybrid search
	var hybridResults []raggo.SearchResult

	if len(embeddedChunks) > 1 {
		query1 := embeddedChunks[0].Embeddings["default"]
		query2 := embeddedChunks[1].Embeddings["default"]
		topK := 5

		hybridResults, err = milvusDB.HybridSearch(context.Background(), collectionName, []raggo.Vector{query1, query2}, topK)
	} else if len(embeddedChunks) == 1 {
		query := embeddedChunks[0].Embeddings["default"]
		topK := 5

		hybridResults, err = milvusDB.Search(context.Background(), collectionName, query, topK)
	} else {
		log.Fatalf("No embedded chunks to search with")
	}

	if err != nil {
		log.Fatalf("Failed to perform search: %v", err)
	}

	log.Printf("Search returned %d results", len(hybridResults))

	// Print search results
	fmt.Println("\nMilvus Search Results:")
	if len(hybridResults) == 0 {
		fmt.Println("No results found.")
	} else {
		for i, result := range hybridResults {
			text, ok := textStorage[result.ID]
			if !ok {
				log.Printf("Text not found for ID %d", result.ID)
			}
			log.Printf("Result %d: ID=%d, Score=%f, Text found: %v", i, result.ID, result.Score, ok)
			fmt.Printf("%d. Score: %f, Text: %s\n", i+1, result.Score, truncateString(text, 50))
		}
	}
}

func truncateString(s string, length int) string {
	if len(s) <= length {
		return s
	}
	return s[:length-3] + "..."
}


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

	log.Println("About to create Milvus VectorDB instance")

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
	log.Println("Successfully created Milvus VectorDB instance")

	defer func() {
		log.Println("Attempting to close Milvus connection")
		err := milvusDB.Close()
		if err != nil {
			log.Printf("Error closing Milvus connection: %v", err)
		} else {
			log.Println("Successfully closed Milvus connection")
		}
	}()

	log.Println("Attempting to connect to Milvus database")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = milvusDB.Connect(ctx)
	if err != nil {
		log.Fatalf("Failed to connect to Milvus database: %v", err)
	}
	log.Println("Successfully connected to Milvus database")

	// Create collection
	collectionName := "test_collection_hybrid"
	schema := raggo.Schema{
		Name: collectionName,
		Fields: []raggo.Field{
			{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: false},
			{Name: "Embedding", DataType: "float_vector", Dimension: 1536},
			{Name: "Text", DataType: "varchar", MaxLength: 65535},
		},
	}
	log.Printf("Attempting to create collection %s with schema: %+v", collectionName, schema)

	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	exists, err := milvusDB.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("Failed to check if collection exists: %v", err)
	}
	log.Printf("Collection %s exists: %v", collectionName, exists)

	if exists {
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		err = milvusDB.DropCollection(ctx, collectionName)
		if err != nil {
			log.Fatalf("Failed to drop existing collection: %v", err)
		}
		log.Printf("Dropped existing collection %s\n", collectionName)
	}

	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	err = milvusDB.CreateCollection(ctx, collectionName, schema)
	if err != nil {
		log.Fatalf("Failed to create collection: %v", err)
	}
	log.Printf("Successfully created collection %s", collectionName)

	// Insert embeddings
	var records []raggo.Record
	for i, chunk := range embeddedChunks {
		id := int64(i)
		records = append(records, raggo.Record{
			Fields: map[string]interface{}{
				"ID":        id,
				"Embedding": chunk.Embeddings["default"],
				"Text":      chunk.Text,
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

	// Set the column names before searching
	milvusDB.SetColumnNames([]string{"ID", "Text"})

	// Perform a regular search
	var searchResults []raggo.SearchResult

	if len(embeddedChunks) > 0 {
		query := map[string]raggo.Vector{"Embedding": embeddedChunks[0].Embeddings["default"]}
		topK := 5
		searchParams := map[string]interface{}{
			"type": "HNSW",
			"ef":   64, // You might need to adjust this value
		}

		searchResults, err = milvusDB.Search(context.Background(), collectionName, query, topK, "L2", searchParams)
		if err != nil {
			log.Fatalf("Failed to perform regular search: %v", err)
		}

		log.Printf("Regular Search returned %d results", len(searchResults))

		fmt.Println("\nMilvus Regular Search Results:")
		if len(searchResults) == 0 {
			fmt.Println("No results found.")
		} else {
			for i, result := range searchResults {
				text, ok := result.Fields["Text"].(string)
				if !ok {
					log.Printf("Text not found for ID %d", result.ID)
					continue
				}
				fmt.Printf("%d. Score: %f, Text: %s\n", i+1, result.Score, truncateString(text, 50))
			}
		}

		// Perform a hybrid search
		var hybridResults []raggo.SearchResult

		if len(embeddedChunks) > 1 {
			query := map[string]raggo.Vector{
				"Embedding1": embeddedChunks[0].Embeddings["default"],
				"Embedding2": embeddedChunks[1].Embeddings["default"],
			}
			topK := 5

			hybridResults, err = milvusDB.HybridSearch(context.Background(), collectionName, query, topK, "L2", nil, nil)
			if err != nil {
				log.Fatalf("Failed to perform hybrid search: %v", err)
			}

			log.Printf("Hybrid Search returned %d results", len(hybridResults))

			fmt.Println("\nMilvus Hybrid Search Results:")
			if len(hybridResults) == 0 {
				fmt.Println("No results found.")
			} else {
				for i, result := range hybridResults {
					text, ok := result.Fields["Text"].(string)
					if !ok {
						log.Printf("Text not found for ID %d", result.ID)
						continue
					}
					fmt.Printf("%d. Score: %f, Text: %s\n", i+1, result.Score, truncateString(text, 50))
				}
			}
		} else {
			fmt.Println("Not enough embedded chunks for hybrid search.")
		}
	} else {
		log.Fatalf("No embedded chunks to search with")
	}
}

func truncateString(s string, length int) string {
	if len(s) <= length {
		return s
	}
	return s[:length-3] + "..."
}


// File: chromem.go

package rag

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/philippgille/chromem-go"
)

type ChromemDB struct {
	db          *chromem.DB
	collections map[string]*chromem.Collection
	mu          sync.RWMutex
	columnNames []string
	dimension   int
}

func newChromemDB(cfg *Config) (*ChromemDB, error) {
	log.Printf("Creating new ChromemDB with config: %+v", cfg)

	// Get dimension from config parameters
	dimension, ok := cfg.Parameters["dimension"].(int)
	if !ok {
		log.Printf("No dimension found in config parameters, using default 1536")
		dimension = 1536
	}
	log.Printf("Using dimension: %d", dimension)

	// Create DB
	var db *chromem.DB
	var err error
	if cfg.Address != "" {
		// Ensure directory exists
		dir := filepath.Dir(cfg.Address)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory for ChromemDB: %w", err)
		}

		log.Printf("Creating persistent ChromemDB at %s", cfg.Address)
		db, err = chromem.NewPersistentDB(cfg.Address, false) // Don't truncate existing DB
		if err != nil {
			log.Printf("Failed to create persistent ChromemDB: %v", err)
			return nil, fmt.Errorf("failed to create persistent ChromemDB: %w", err)
		}

		// Verify database file exists
		if _, err := os.Stat(cfg.Address); os.IsNotExist(err) {
			log.Printf("Warning: ChromemDB file %s does not exist after creation", cfg.Address)
			return nil, fmt.Errorf("ChromemDB file %s does not exist after creation", cfg.Address)
		}
	} else {
		log.Printf("Creating in-memory ChromemDB")
		db = chromem.NewDB()
	}

	if db == nil {
		log.Printf("ChromemDB is nil after creation")
		return nil, fmt.Errorf("ChromemDB is nil after creation")
	}

	// Test database by creating and removing a test collection
	testCol := "test_collection"
	log.Printf("Testing database by creating test collection %s", testCol)
	
	// Create test collection
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return nil, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}
	embeddingFunc := chromem.NewEmbeddingFuncOpenAI(apiKey, "text-embedding-3-small")
	
	col, err := db.CreateCollection(testCol, map[string]string{}, embeddingFunc)
	if err != nil {
		log.Printf("Failed to create test collection: %v", err)
		return nil, fmt.Errorf("failed to create test collection: %w", err)
	}

	if col == nil {
		log.Printf("Test collection is nil after creation")
		return nil, fmt.Errorf("test collection is nil after creation")
	}

	// Get collection to verify it exists
	if col = db.GetCollection(testCol, embeddingFunc); col == nil {
		log.Printf("Test collection not found after creation")
		return nil, fmt.Errorf("test collection not found after creation")
	}

	// Drop test collection by creating a new one with truncate=true
	col, err = db.CreateCollection(testCol, map[string]string{}, embeddingFunc)
	if err != nil {
		log.Printf("Failed to drop test collection: %v", err)
		return nil, fmt.Errorf("failed to drop test collection: %w", err)
	}

	log.Printf("Successfully created and tested ChromemDB")

	return &ChromemDB{
		db:          db,
		collections: make(map[string]*chromem.Collection),
		dimension:   dimension,
	}, nil
}

func (c *ChromemDB) Connect(ctx context.Context) error {
	log.Printf("Connecting to ChromemDB")
	// No explicit connect needed for chromem
	log.Printf("ChromemDB connected (no-op)")
	return nil
}

func (c *ChromemDB) Close() error {
	// No explicit close in chromem
	return nil
}

func (c *ChromemDB) HasCollection(ctx context.Context, name string) (bool, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	log.Printf("Checking if collection %s exists", name)

	// First check our local map
	if _, exists := c.collections[name]; exists {
		log.Printf("Collection %s found in local map", name)
		return true, nil
	}

	// Get OpenAI API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return false, fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	// Create embedding function using OpenAI's text-embedding-3-small
	embeddingFunc := chromem.NewEmbeddingFuncOpenAI(apiKey, "text-embedding-3-small")

	// Try to get the collection
	col := c.db.GetCollection(name, embeddingFunc)
	exists := col != nil

	if exists {
		log.Printf("Collection %s found in database", name)
		// Cache the collection in our map
		c.collections[name] = col
	} else {
		log.Printf("Collection %s not found in database", name)
	}

	return exists, nil
}

func (c *ChromemDB) DropCollection(ctx context.Context, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	delete(c.collections, name)
	return nil
}

func (c *ChromemDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Creating collection: %s (ignoring schema as Chromem doesn't use it)", name)

	// Check if collection already exists in our map
	if _, exists := c.collections[name]; exists {
		log.Printf("Collection %s already exists in our map", name)
		return nil
	}

	// Get OpenAI API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	// Create embedding function using OpenAI's text-embedding-3-small
	embeddingFunc := chromem.NewEmbeddingFuncOpenAI(apiKey, "text-embedding-3-small")

	// Create collection in ChromemDB with empty metadata
	col, err := c.db.CreateCollection(name, map[string]string{}, embeddingFunc)
	if err != nil {
		log.Printf("Failed to create collection %s: %v", name, err)
		return fmt.Errorf("failed to create collection %s: %w", name, err)
	}

	// Store collection in our map
	c.collections[name] = col
	log.Printf("Successfully created collection %s with dimension %d and embedding function %T", name, c.dimension, embeddingFunc)

	// Verify collection was created
	verifyCol := c.db.GetCollection(name, embeddingFunc)
	if verifyCol == nil {
		log.Printf("Warning: Collection %s was not properly created", name)
		return fmt.Errorf("collection %s was not properly created", name)
	}

	return nil
}

func (c *ChromemDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Inserting %d records into collection %s", len(data), collectionName)

	// Get collection from our map
	col, exists := c.collections[collectionName]
	if !exists {
		return fmt.Errorf("collection %s does not exist in collections map", collectionName)
	}

	// Convert records to chromem documents
	docs := make([]chromem.Document, len(data))
	validCount := 0

	for i, record := range data {
		// Extract content and metadata
		content, ok := record.Fields["Text"].(string)
		if !ok {
			log.Printf("Warning: Record %d has no 'Text' field or it's not a string, skipping", i)
			continue
		}

		metadata := make(map[string]string)
		if metaField, ok := record.Fields["Metadata"]; ok {
			if meta, ok := metaField.(map[string]interface{}); ok {
				for k, v := range meta {
					if str, ok := v.(string); ok {
						metadata[k] = str
					}
				}
			}
		}

		// Get embedding and convert to []float32 if needed
		var embedding []float32
		if embField, ok := record.Fields["Embedding"]; ok {
			switch e := embField.(type) {
			case []float32:
				embedding = e
			case Vector:
				embedding = toFloat32Slice(e)
			case []float64:
				embedding = make([]float32, len(e))
				for j, v := range e {
					embedding[j] = float32(v)
				}
			default:
				log.Printf("Warning: Record %d has invalid embedding type %T, skipping", i, embField)
				continue
			}
		} else {
			log.Printf("Warning: Record %d has no 'Embedding' field, skipping", i)
			continue
		}

		// Create document
		docs[validCount] = chromem.Document{
			ID:        fmt.Sprintf("%d", i),
			Content:   content,
			Metadata:  metadata,
			Embedding: embedding,
		}
		validCount++
	}

	// Trim docs to valid count
	docs = docs[:validCount]

	if validCount == 0 {
		log.Printf("Warning: No valid documents to insert into collection %s", collectionName)
		return nil
	}

	log.Printf("Converted %d/%d records to valid documents for collection %s", validCount, len(data), collectionName)

	// Insert documents in batches to avoid memory issues
	batchSize := 100
	for i := 0; i < len(docs); i += batchSize {
		end := i + batchSize
		if end > len(docs) {
			end = len(docs)
		}
		batch := docs[i:end]

		log.Printf("Inserting batch of %d documents (batch %d/%d) into collection %s", len(batch), (i/batchSize)+1, (len(docs)+batchSize-1)/batchSize, collectionName)
		for _, doc := range batch {
			err := col.AddDocument(ctx, doc)
			if err != nil {
				return fmt.Errorf("failed to insert document: %w", err)
			}
		}
	}

	log.Printf("Successfully inserted %d documents into collection %s", validCount, collectionName)

	return nil
}

func (c *ChromemDB) Flush(ctx context.Context, collectionName string) error {
	// No explicit flush in chromem
	return nil
}

func (c *ChromemDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	// No explicit index creation in chromem
	return nil
}

func (c *ChromemDB) LoadCollection(ctx context.Context, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	log.Printf("Loading collection: %s", name)

	// Get OpenAI API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("OPENAI_API_KEY environment variable not set")
	}

	// Create embedding function using OpenAI's text-embedding-3-small
	embeddingFunc := chromem.NewEmbeddingFuncOpenAI(apiKey, "text-embedding-3-small")

	// Get collection from ChromemDB
	col := c.db.GetCollection(name, embeddingFunc)
	if col == nil {
		log.Printf("Collection %s not found", name)
		return fmt.Errorf("collection %s not found", name)
	}

	// Store collection in our map
	c.collections[name] = col
	log.Printf("Successfully loaded collection %s", name)

	return nil
}

func (c *ChromemDB) Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error) {
	c.mu.RLock()

	// First check if collection exists in our map
	col, exists := c.collections[collectionName]
	c.mu.RUnlock()

	if !exists {
		// Try to load the collection
		err := c.LoadCollection(ctx, collectionName)
		if err != nil {
			return nil, fmt.Errorf("failed to load collection: %w", err)
		}

		// Get the collection again
		c.mu.RLock()
		col = c.collections[collectionName]
		c.mu.RUnlock()
	}

	// We only support single vector search for now
	if len(vectors) != 1 {
		return nil, fmt.Errorf("chromem only supports single vector search")
	}

	// Get the first vector
	var queryVector Vector
	for _, v := range vectors {
		queryVector = v
		break
	}

	// Convert query vector to float32
	query := toFloat32Slice(queryVector)

	log.Printf("Searching collection %s with query vector of length %d", collectionName, len(query))

	// Search documents using empty filters for where and whereDocument
	results, err := col.QueryEmbedding(ctx, query, topK, make(map[string]string), make(map[string]string))
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %w", err)
	}

	log.Printf("Found %d results (requested topK=%d)", len(results), topK)

	if len(results) == 0 {
		log.Printf("Warning: No results found in collection %s. This could indicate that either: (1) the collection is empty, (2) no similar documents were found, or (3) the collection was not properly loaded.", collectionName)
		return []SearchResult{}, nil
	}

	// Convert results
	searchResults := make([]SearchResult, len(results))
	for i, result := range results {
		fields := make(map[string]interface{})
		fields["Text"] = result.Content
		if len(result.Metadata) > 0 {
			fields["Metadata"] = result.Metadata
		}

		searchResults[i] = SearchResult{
			ID:     int64(i),
			Score:  float64(result.Similarity),
			Fields: fields,
		}
		log.Printf("Result %d: score=%f, content=%s", i, result.Similarity, result.Content)
	}

	return searchResults, nil
}

func (c *ChromemDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	// Not implemented for chromem
	return nil, fmt.Errorf("hybrid search not implemented for chromem")
}

func (c *ChromemDB) SetColumnNames(names []string) {
	c.columnNames = names
}

// Helper function to convert Vector to []float32
func toFloat32Slice(v Vector) []float32 {
	result := make([]float32, len(v))
	for i, val := range v {
		result[i] = float32(val)
	}
	return result
}

// File: milvus_example.go

package main

import (
	"context"
	"log"
	"math/rand"
	"time"

	"github.com/teilomillet/raggo"
)

const (
	nEntities, dim                              = 10000, 128
	collectionName                              = "hello_multi_vectors"
	idCol, keyCol, embeddingCol1, embeddingCol2 = "ID", "key", "vector1", "vector2"
	topK                                        = 3
)

func main() {
	ctx := context.Background()

	log.Println("start connecting to Milvus")
	db, err := raggo.NewVectorDB(
		raggo.WithType("milvus"),
		raggo.WithAddress("localhost:19530"),
	)
	if err != nil {
		log.Fatalf("failed to create Milvus client, err: %v", err)
	}
	if err := db.Connect(ctx); err != nil {
		log.Fatalf("failed to connect to Milvus, err: %v", err)
	}
	defer db.Close()

	// delete collection if exists
	has, err := db.HasCollection(ctx, collectionName)
	if err != nil {
		log.Fatalf("failed to check collection exists, err: %v", err)
	}
	if has {
		db.DropCollection(ctx, collectionName)
	}

	// create collection
	log.Printf("create collection `%s`\n", collectionName)
	schema := raggo.Schema{
		Name:        collectionName,
		Description: "hello_multi_vectors is a demo collection with multiple vector fields",
		Fields: []raggo.Field{
			{Name: idCol, DataType: "int64", PrimaryKey: true, AutoID: true},
			{Name: keyCol, DataType: "int64"},
			{Name: embeddingCol1, DataType: "float_vector", Dimension: dim},
			{Name: embeddingCol2, DataType: "float_vector", Dimension: dim},
		},
	}

	if err := db.CreateCollection(ctx, collectionName, schema); err != nil {
		log.Fatalf("create collection failed, err: %v", err)
	}

	// Generate and insert data
	var records []raggo.Record
	for i := 0; i < nEntities; i++ {
		vec1 := make([]float64, dim)
		vec2 := make([]float64, dim)
		for j := 0; j < dim; j++ {
			vec1[j] = rand.Float64()
			vec2[j] = rand.Float64()
		}
		records = append(records, raggo.Record{
			Fields: map[string]interface{}{
				keyCol:        rand.Int63() % 512,
				embeddingCol1: vec1,
				embeddingCol2: vec2,
			},
		})
	}

	log.Println("start to insert data into collection")
	if err := db.Insert(ctx, collectionName, records); err != nil {
		log.Fatalf("failed to insert random data into `%s`, err: %v", collectionName, err)
	}

	log.Println("insert data done, start to flush")
	if err := db.Flush(ctx, collectionName); err != nil {
		log.Fatalf("failed to flush data, err: %v", err)
	}
	log.Println("flush data done")

	// build index
	log.Println("start creating index HNSW")
	index := raggo.Index{
		Type:   "HNSW",
		Metric: "L2",
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 256,
		},
	}
	if err := db.CreateIndex(ctx, collectionName, embeddingCol1, index); err != nil {
		log.Fatalf("failed to create index for %s, err: %v", embeddingCol1, err)
	}
	if err := db.CreateIndex(ctx, collectionName, embeddingCol2, index); err != nil {
		log.Fatalf("failed to create index for %s, err: %v", embeddingCol2, err)
	}

	log.Printf("build HNSW index done for collection `%s`\n", collectionName)
	log.Printf("start to load collection `%s`\n", collectionName)

	// load collection
	if err := db.LoadCollection(ctx, collectionName); err != nil {
		log.Fatalf("failed to load collection, err: %v", err)
	}

	log.Println("load collection done")
	// Prepare vectors for hybrid search
	vec2search1 := records[len(records)-2].Fields[embeddingCol1].([]float64)
	vec2search2 := records[len(records)-1].Fields[embeddingCol2].([]float64)

	begin := time.Now()
	log.Println("start to execute hybrid search")
	searchVectors := map[string]raggo.Vector{
		embeddingCol1: vec2search1,
		embeddingCol2: vec2search2,
	}
	result, err := db.HybridSearch(ctx, collectionName, searchVectors, topK, "L2", map[string]interface{}{
		"type": "HNSW",
		"ef":   100,
	}, nil)
	if err != nil {
		log.Fatalf("failed to perform hybrid search, err: %v", err)
	}
	log.Printf("hybrid search `%s` done, latency %v\n", collectionName, time.Since(begin))
	for _, rs := range result {
		log.Printf("ID: %d, score %f, embedding1: %v, embedding2: %v\n",
			rs.ID, rs.Score,
			rs.Fields[embeddingCol1], rs.Fields[embeddingCol2])
	}

	db.DropCollection(ctx, collectionName)
}


package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/teilomillet/gofh"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo"
)

var (
	llm       gollm.LLM
	retriever *raggo.Retriever
)

func main() {
	// Initialize LLM
	var err error
	llm, err = gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Get the docs directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	// Fix the path construction - remove the duplicate "examples/chat"
	docsDir := filepath.Join(wd, "docs")

	// Print the path for debugging
	log.Printf("Loading documents from: %s", docsDir)

	// Verify directory exists
	if _, err := os.Stat(docsDir); os.IsNotExist(err) {
		log.Fatalf("Documents directory does not exist: %s", docsDir)
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// First, let's clean up any existing collection
	vectorDB, err := raggo.NewVectorDB(
		raggo.WithType("milvus"),
		raggo.WithAddress("localhost:19530"),
		raggo.WithTimeout(5*time.Minute),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Connect to the database
	err = vectorDB.Connect(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer vectorDB.Close()

	// Check and drop existing collection
	exists, err := vectorDB.HasCollection(ctx, "chat_docs")
	if err != nil {
		log.Fatal(err)
	}
	if exists {
		log.Println("Dropping existing collection")
		err = vectorDB.DropCollection(ctx, "chat_docs")
		if err != nil {
			log.Fatal(err)
		}
	}

	// Register documents with explicit chunking and debug output
	log.Println("Registering documents with debug settings...")
	err = raggo.Register(ctx, docsDir,
		raggo.WithVectorDB("milvus", map[string]string{"address": "localhost:19530"}),
		raggo.WithCollection("chat_docs", true),
		raggo.WithChunking(200, 50), // Adjusted chunk size and overlap
		raggo.WithEmbedding(
			"openai",
			"text-embedding-3-small",
			os.Getenv("OPENAI_API_KEY"),
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Create and load index
	log.Println("Creating and loading index...")
	err = vectorDB.CreateIndex(ctx, "chat_docs", "Embedding", raggo.Index{
		Type:   "HNSW",
		Metric: "L2",
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 256,
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Load the collection
	err = vectorDB.LoadCollection(ctx, "chat_docs")
	if err != nil {
		log.Fatal(err)
	}

	// Initialize retriever with debug settings
	log.Println("Initializing retriever with debug settings...")
	retriever, err = raggo.NewRetriever(
		raggo.WithRetrieveDB("milvus", "localhost:19530"),
		raggo.WithRetrieveCollection("chat_docs"),
		raggo.WithTopK(10),      // More results
		raggo.WithMinScore(0.1), // Lower threshold
		raggo.WithHybrid(false), // Simple search first
		raggo.WithRetrieveEmbedding(
			"openai",
			"text-embedding-3-small",
			os.Getenv("OPENAI_API_KEY"),
		),
	)
	if err != nil {
		log.Fatal(err)
	}
	defer retriever.Close()

	// Create web app
	app := gofh.New()

	// Main page
	app.Get("/").Handle(func(c *gofh.Context) gofh.Element {
		return gofh.Div(
			gofh.El("style", `
				body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: system-ui; }
				#chat { margin-bottom: 20px; }
				.message { padding: 10px; margin: 5px 0; border-radius: 5px; }
				.user { background: #e3f2fd; }
				.ai { background: #f5f5f5; }
				form { display: flex; gap: 10px; }
				input { flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
				button { padding: 8px 16px; background: #1976d2; color: white; border: none; border-radius: 4px; cursor: pointer; }
				.sources { font-size: 0.8em; color: #666; margin-top: 5px; }
			`),
			gofh.H1("RAG Chat Demo"),
			gofh.Div().ID("chat"),
			gofh.Form(
				gofh.Input("text", "msg").
					Attr("placeholder", "Type your question...").
					Attr("class", "w-full"),
				gofh.Button("Send"),
			).
				Attr("hx-post", "/chat").
				Attr("hx-target", "#chat").
				Attr("hx-swap", "beforeend"),
		)
	})

	// Chat endpoint
	app.Post("/chat").Handle(func(c *gofh.Context) gofh.Element {
		msg := c.GetFormValue("msg")

		log.Printf("\n=== New Search Query ===")
		log.Printf("Query: %q", msg)

		results, err := retriever.Retrieve(c.Request.Context(), msg)
		if err != nil {
			log.Printf("❌ Retrieval error: %v", err)
		}

		log.Printf("\n=== Search Results ===")
		log.Printf("Found %d results", len(results))

		var contexts []string
		var sources []string

		for i, result := range results {
			log.Printf("\nResult %d:", i+1)
			log.Printf("Score: %.4f", result.Score)
			log.Printf("Source: %s", result.Source)
			log.Printf("Content length: %d", len(result.Content))
			log.Printf("Content: %s", truncateString(result.Content, 200))
			if result.Metadata != nil {
				log.Printf("Metadata: %+v", result.Metadata)
			}

			contexts = append(contexts, result.Content)
			if result.Source != "" {
				shortPath := filepath.Base(result.Source)
				sources = append(sources, fmt.Sprintf("%s (%.2f)", shortPath, result.Score))
			}
		}

		// Generate response with improved prompt
		prompt := fmt.Sprintf(`Here are some relevant sections from our documentation:

%s

Based on this information, please answer the following question: %s

If the information isn't found in the provided context, please say so clearly.`,
			strings.Join(contexts, "\n\n---\n\n"),
			msg,
		)

		resp, err := llm.Generate(c.Request.Context(), gollm.NewPrompt(prompt))
		if err != nil {
			resp = "Error: " + err.Error()
		}

		userMsg := gofh.Div(
			gofh.P("You: "+msg),
		).Attr("class", "message user")

		aiMsg := gofh.Div(
			gofh.P("AI: "+resp),
			gofh.P("Sources: "+strings.Join(sources, ", ")).Attr("class", "sources"),
		).Attr("class", "message ai")

		return gofh.Div(userMsg, aiMsg)
	})

	log.Println("Chat server starting on http://localhost:8080")
	log.Fatal(app.Serve())
}

func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

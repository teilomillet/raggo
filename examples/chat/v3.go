package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/teilomillet/gofh"
	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo"
)

type ChatApp struct {
	rag *raggo.RAG
	llm gollm.LLM
}

func NewChatApp() (*ChatApp, error) {
	// Initialize LLM
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize LLM: %w", err)
	}

	// Initialize RAG with context processing
	rag, err := raggo.NewRAG(
		raggo.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
		raggo.WithMilvus("chat_docs"),
		func(c *raggo.RAGConfig) {
			c.ChunkSize = 100   // Increased from 200
			c.ChunkOverlap = 50 // Increased from 50
			c.BatchSize = 10
			c.TopK = 5       // Increased from 5
			c.MinScore = 0.1 // Decreased from 0.1
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RAG: %w", err)
	}

	return &ChatApp{
		rag: rag,
		llm: llm,
	}, nil
}

func (app *ChatApp) LoadDocuments(ctx context.Context, docsDir string) error {
	log.Printf("Loading documents from: %s", docsDir)
	return app.rag.ProcessWithContext(ctx, docsDir, "gpt-4o-mini")
}

func (app *ChatApp) Close() error {
	return app.rag.Close()
}

func main() {
	// Enable debug logging
	raggo.SetLogLevel(raggo.LogLevelDebug)

	// Create new chat app
	app, err := NewChatApp()
	if err != nil {
		log.Fatal(err)
	}
	defer app.Close()

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	// Construct the path to docs directory
	docsDir := filepath.Join(wd, "docs")

	// Debug: Print the path we're trying to use
	log.Printf("Looking for docs in: %s", docsDir)

	// Verify directory exists
	if _, err := os.Stat(docsDir); os.IsNotExist(err) {
		log.Fatalf("Documents directory does not exist: %s", docsDir)
	}

	// Process each file in the directory
	files, err := os.ReadDir(docsDir)
	if err != nil {
		log.Fatalf("Failed to read directory: %v", err)
	}

	for _, file := range files {
		if file.IsDir() {
			continue
		}
		filePath := filepath.Join(docsDir, file.Name())
		log.Printf("Processing file: %s", filePath)
		err = app.LoadDocuments(context.Background(), filePath)
		if err != nil {
			log.Printf("Error processing file %s: %v", filePath, err)
			// Continue with the next file instead of stopping
			continue
		}
	}

	// Create web app
	webApp := gofh.New()

	// Main page
	webApp.Get("/").Handle(func(c *gofh.Context) gofh.Element {
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
			gofh.H1("Context-Enhanced RAG Chat Demo"),
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
	webApp.Post("/chat").Handle(func(c *gofh.Context) gofh.Element {
		msg := c.GetFormValue("msg")

		log.Printf("\n=== New Search Query ===")
		log.Printf("Query: %q", msg)

		results, err := app.rag.Query(c.Request.Context(), msg)
		if err != nil {
			log.Printf("❌ Retrieval error: %v", err)
			return renderError(err)
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
			log.Printf("Content: %s", truncateString(result.Content, 500)) // Increased from 200
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

		resp, err := app.llm.Generate(c.Request.Context(), gollm.NewPrompt(prompt))
		if err != nil {
			return renderError(err)
		}

		return renderChat(msg, resp, sources)
	})

	log.Println("Context-enhanced chat server starting on http://localhost:8080")
	log.Fatal(webApp.Serve())
}

// UI helpers
func renderChat(query, response string, sources []string) gofh.Element {
	return gofh.Div(
		gofh.Div(
			gofh.P("You: "+query),
		).Attr("class", "message user"),
		gofh.Div(
			gofh.P("AI: "+response),
			gofh.P("Sources: "+strings.Join(sources, ", ")).Attr("class", "sources"),
		).Attr("class", "message ai"),
	)
}

func renderError(err error) gofh.Element {
	return gofh.Div(
		gofh.P("Error: "+err.Error()),
	).Attr("class", "message error")
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

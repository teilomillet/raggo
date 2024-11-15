// examples/chat/v2.go
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

	// Initialize RAG with OpenAI and local Milvus
	rag, err := raggo.NewRAG(
		raggo.WithOpenAI(os.Getenv("OPENAI_API_KEY")),
		raggo.WithMilvus("chat_docs"),
		// Optional: Add custom settings
		func(c *raggo.RAGConfig) {
			c.ChunkSize = 200
			c.ChunkOverlap = 50
			c.TopK = 10
			c.MinScore = 0.1
			c.UseHybrid = false
			c.SearchParams = map[string]interface{}{ // Add search params
				"nprobe": 10,
				"ef":     64,
				"type":   "HNSW",
			}
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
	return app.rag.LoadDocuments(ctx, docsDir)
}

func (app *ChatApp) Close() error {
	return app.rag.Close()
}

func main() {
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

	// Construct the correct path to docs directory
	// Changed: Remove duplicate "examples/chat" from path
	docsDir := filepath.Join(wd, "docs")

	// Debug: Print the path we're trying to use
	log.Printf("Looking for docs in: %s", docsDir)

	// Verify directory exists
	if _, err := os.Stat(docsDir); os.IsNotExist(err) {
		log.Fatalf("Documents directory does not exist: %s", docsDir)
	}

	// Load documents
	err = app.LoadDocuments(context.Background(), docsDir)
	if err != nil {
		log.Fatal(err)
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
	webApp.Post("/chat").Handle(func(c *gofh.Context) gofh.Element {
		msg := c.GetFormValue("msg")
		if msg == "" {
			return renderError(fmt.Errorf("empty message"))
		}

		log.Printf("\n=== New Search Query ===\nQuery: %q", msg)

		// Get relevant documents
		results, err := app.rag.Query(c.Request.Context(), msg)
		if err != nil {
			log.Printf("❌ Retrieval error: %v", err)
			return renderError(err)
		}

		log.Printf("Found %d results", len(results))

		// Prepare context and sources
		var contexts []string
		var sources []string
		for _, result := range results {
			contexts = append(contexts, result.Content)
			shortPath := filepath.Base(result.Source)
			sources = append(sources, fmt.Sprintf("%s (%.2f)", shortPath, result.Score))
		}

		// Generate response
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

	log.Println("Chat server starting on http://localhost:8080")
	log.Fatal(webApp.Serve())
}

// UI helpers
const defaultStyles = `
	body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: system-ui; }
	#chat { margin-bottom: 20px; }
	.message { padding: 10px; margin: 5px 0; border-radius: 5px; }
	.user { background: #e3f2fd; }
	.ai { background: #f5f5f5; }
	.error { background: #ffebee; }
	form { display: flex; gap: 10px; }
	input { flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
	button { padding: 8px 16px; background: #1976d2; color: white; border: none; border-radius: 4px; cursor: pointer; }
	.sources { font-size: 0.8em; color: #666; margin-top: 5px; }
`

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

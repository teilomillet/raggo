package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/teilomillet/gollm"
	"github.com/teilomillet/raggo"
)

func main() {
	log.Println("DEBUG: Starting memory enhancer example")

	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}
	log.Println("DEBUG: API key found")

	// Initialize LLM
	log.Println("DEBUG: Initializing LLM...")
	llm, err := gollm.NewLLM(
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o-mini"),
		gollm.SetAPIKey(apiKey),
		gollm.SetLogLevel(gollm.LogLevelInfo),
	)
	if err != nil {
		log.Fatalf("Failed to initialize LLM: %v", err)
	}
	log.Println("DEBUG: LLM initialized successfully")

	// Create memory context
	log.Println("DEBUG: Creating memory context...")

	// Drop existing collection if it exists
	vectorDB, err := raggo.NewVectorDB(
		raggo.WithType("milvus"),
		raggo.WithAddress("localhost:19530"),
	)
	if err != nil {
		log.Fatalf("Failed to create vector database: %v", err)
	}
	if err := vectorDB.Connect(context.Background()); err != nil {
		log.Fatalf("Failed to connect to vector database: %v", err)
	}
	exists, err := vectorDB.HasCollection(context.Background(), "tech_docs")
	if err != nil {
		log.Fatalf("Failed to check collection: %v", err)
	}
	if exists {
		log.Println("Dropping existing collection")
		err = vectorDB.DropCollection(context.Background(), "tech_docs")
		if err != nil {
			log.Fatalf("Failed to drop collection: %v", err)
		}
	}

	// Create memory context with optimized settings
	memoryContext, err := raggo.NewMemoryContext(apiKey,
		raggo.MemoryCollection("tech_docs"),
		raggo.MemoryTopK(5),
		raggo.MemoryMinScore(0.01),
		raggo.MemoryStoreLastN(10),
		raggo.MemoryStoreRAGInfo(true),
	)
	if err != nil {
		log.Fatalf("Failed to create memory context: %v", err)
	}

	// Configure hybrid search
	retriever := memoryContext.GetRetriever()
	if err := retriever.GetVectorDB().LoadCollection(context.Background(), "tech_docs"); err != nil {
		log.Fatalf("Failed to load collection: %v", err)
	}

	// Load technical documentation
	ctx := context.Background()
	docsDir := filepath.Join("examples", "chat", "docs")
	docs := []string{
		filepath.Join(docsDir, "microservices.txt"),
		filepath.Join(docsDir, "vector_databases.txt"),
		filepath.Join(docsDir, "rag_systems.txt"),
		filepath.Join(docsDir, "golang_basics.txt"),
		filepath.Join(docsDir, "embeddings.txt"),
	}

	for _, doc := range docs {
		content, err := os.ReadFile(doc)
		if err != nil {
			log.Printf("Warning: Failed to read %s: %v", doc, err)
			continue
		}

		// Store document content as memory
		err = memoryContext.StoreMemory(ctx, []gollm.MemoryMessage{
			{
				Role:    "system",
				Content: string(content),
				Tokens:  0, // Let the system calculate tokens
			},
		})
		if err != nil {
			log.Printf("Warning: Failed to store memory from %s: %v", doc, err)
		}
	}

	// Chat loop
	fmt.Println("\nChat started! Type 'exit' to end the conversation.")
	fmt.Println("\nExample questions you can ask:")
	fmt.Println("1. What is MountainPass's PressureValve system and how did it help during Black Friday?")
	fmt.Println("2. What are the key features of the PressureValve architecture?")
	fmt.Println("3. How did MountainPass handle their e-commerce scaling challenge?")

	// Initialize chat memory
	var memory []gollm.MemoryMessage
	systemPrompt := "You are a technical expert helping explain the MountainPass case study and their PressureValve system. " +
		"Focus on the specific details from the case study, including the challenges they faced, their solution, and the results they achieved. " +
		"Always reference specific numbers, features, and outcomes from the documentation."

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nQ: ")
		scanner.Scan()
		query := scanner.Text()
		if query == "exit" {
			break
		}

		// Create base prompt with query
		prompt := gollm.NewPrompt(query)

		// Enhance prompt with relevant context
		enhancedPrompt, err := memoryContext.EnhancePrompt(ctx, prompt, memory)
		if err != nil {
			log.Printf("Failed to enhance prompt: %v", err)
			continue
		}

		// Build final prompt with system context
		var promptBuilder strings.Builder
		promptBuilder.WriteString(systemPrompt)
		promptBuilder.WriteString("\n\nBased on the following context:\n")
		promptBuilder.WriteString(enhancedPrompt.String())
		promptBuilder.WriteString("\n\nPlease answer this question: ")
		promptBuilder.WriteString(query)

		finalPrompt := gollm.NewPrompt(promptBuilder.String())

		// Generate response
		fmt.Print("\nA: ")
		response, err := llm.Generate(ctx, finalPrompt)
		if err != nil {
			log.Printf("Failed to generate response: %v", err)
			continue
		}
		fmt.Println(response)

		// Store the interaction in memory
		memory = append(memory, gollm.MemoryMessage{
			Role:    "user",
			Content: query,
			Tokens:  0,
		}, gollm.MemoryMessage{
			Role:    "assistant",
			Content: response,
			Tokens:  0,
		})

		// Store last N messages
		if err := memoryContext.StoreLastN(ctx, memory, memoryContext.GetOptions().StoreLastN); err != nil {
			log.Printf("Failed to store memory: %v", err)
		}
	}

	// Example of updating options
	fmt.Println("\nUpdating memory context configuration...")
	memoryContext.UpdateOptions(
		raggo.MemoryTopK(10),
		raggo.MemoryMinScore(0.01),
		raggo.MemoryStoreLastN(20),
	)

	options := memoryContext.GetOptions()
	fmt.Printf("Current options: TopK=%d, MinScore=%.2f, Collection=%s, StoreLastN=%d, StoreRAGInfo=%v\n",
		options.TopK, options.MinScore, options.Collection, options.StoreLastN, options.StoreRAGInfo)
}
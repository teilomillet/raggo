package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/teilomillet/goal"
	"github.com/teilomillet/raggo"
)

// ResumeInfo represents structured information extracted from a resume
type ResumeInfo struct {
	Name                string   `json:"name"`
	ContactDetails      string   `json:"contact_details"`
	ProfessionalSummary string   `json:"professional_summary"`
	Skills              []string `json:"skills"`
	WorkExperience      string   `json:"work_experience"`
	Education           string   `json:"education"`
	AdditionalInfo      string   `json:"additional_info"`
}

func main() {

	// Set log level to Debug for more detailed output
	raggo.SetLogLevel(raggo.LogLevelWarn)

	// Initialize the LLM
	llm, err := goal.NewLLM(
		goal.SetProvider("openai"),
		goal.SetModel("gpt-4o-mini"),
		goal.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		goal.SetMaxTokens(2048),
	)
	if err != nil {
		log.Fatalf("Failed to create LLM: %v", err)
	}

	// Initialize raggo components
	parser := raggo.NewParser()
	chunker, err := raggo.NewChunker(
		raggo.ChunkSize(500),
		raggo.ChunkOverlap(50),
	)
	if err != nil {
		log.Fatalf("Failed to create chunker: %v", err)
	}

	embedder, err := raggo.NewEmbedder(
		raggo.SetProvider("openai"),
		raggo.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.SetModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	embeddingService := raggo.NewEmbeddingService(embedder)

	vectorDB, err := raggo.NewVectorDB(
		raggo.SetType("memory"),
		raggo.SetDimension(1536),
	)
	if err != nil {
		log.Fatalf("Failed to create vector database: %v", err)
	}
	defer vectorDB.Close()

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	// Construct the path to the testdata directory
	resumeDir := filepath.Join(wd, "testdata")
	log.Printf("Processing resumes from directory: %s", resumeDir)

	// Process resumes
	err = processResumes(llm, parser, chunker, embeddingService, vectorDB, resumeDir)
	if err != nil {
		log.Fatalf("Failed to process resumes: %v", err)
	}

	// Search for candidates based on a job offer
	jobOffer := "We are looking for a software engineer with 5+ years of experience in Go, distributed systems, and cloud technologies. The ideal candidate should have strong problem-solving skills and experience with Docker and Kubernetes."

	candidates, err := searchCandidates(context.Background(), llm, embeddingService, vectorDB, jobOffer)
	if err != nil {
		log.Fatalf("Failed to search candidates: %v", err)
	}

	fmt.Println("Top candidates for the job offer:")
	for i, candidate := range candidates {
		fmt.Printf("%d. %s (Score: %.4f)\n", i+1, candidate.Text, candidate.Score)
	}
}

func processResumes(llm goal.LLM, parser raggo.Parser, chunker raggo.Chunker, embeddingService *raggo.EmbeddingService, vectorDB raggo.VectorDB, resumeDir string) error {
	files, err := filepath.Glob(filepath.Join(resumeDir, "*.pdf"))
	if err != nil {
		return fmt.Errorf("failed to list resume files: %w", err)
	}
	if len(files) == 0 {
		return fmt.Errorf("no PDF files found in directory: %s", resumeDir)
	}
	log.Printf("Found %d resume files to process", len(files))

	for _, file := range files {
		log.Printf("Processing resume: %s", file)

		// Parse the resume
		doc, err := parser.Parse(file)
		if err != nil {
			log.Printf("Failed to parse resume %s: %v", file, err)
			continue
		}
		log.Printf("Successfully parsed resume: %s", file)

		// Step 1: Create a standardized summary
		summarizePrompt := goal.NewPrompt(
			"Summarize the given resume in the following standardized format:\n" +
				"Personal Information: [Name, contact details]\n" +
				"Professional Summary: [2-3 sentences summarizing key qualifications and career objectives]\n" +
				"Skills: [List of key skills, separated by commas]\n" +
				"Work Experience: [For each position: Job Title, Company, Dates, 1-2 key responsibilities or achievements]\n" +
				"Education: [Degree, Institution, Year]\n" +
				"Additional Information: [Any other relevant details like certifications, languages, etc.]\n" +
				"Ensure that the summary is concise and captures the most important information from the resume.\n\n" +
				"Resume Content:\n" + doc.Content,
		)

		standardizedSummary, err := llm.Generate(context.Background(), summarizePrompt)
		if err != nil {
			log.Printf("Failed to create standardized summary for resume %s: %v", file, err)
			continue
		}
		log.Printf("Successfully created standardized summary for resume: %s", file)

		// Step 2: Extract structured information from the standardized summary
		resumeInfo, err := goal.ExtractStructuredData[ResumeInfo](context.Background(), llm, standardizedSummary)
		if err != nil {
			log.Printf("Failed to extract structured data from standardized summary %s: %v", file, err)
			continue
		}
		log.Printf("Successfully extracted structured data from standardized summary: %s", file)

		// Combine structured info and full summary
		fullContent := fmt.Sprintf("Name: %s\nContact Details: %s\nProfessional Summary: %s\nSkills: %v\nWork Experience: %s\nEducation: %s\nAdditional Information: %s\n\nDetailed Summary:\n%s",
			resumeInfo.Name,
			resumeInfo.ContactDetails,
			resumeInfo.ProfessionalSummary,
			resumeInfo.Skills,
			resumeInfo.WorkExperience,
			resumeInfo.Education,
			resumeInfo.AdditionalInfo,
			standardizedSummary)

		// Chunk the content
		chunks := chunker.Chunk(fullContent)
		log.Printf("Created %d chunks for resume: %s", len(chunks), file)

		// Embed and save chunks
		embeddedChunks, err := embeddingService.EmbedChunks(context.Background(), chunks)
		if err != nil {
			log.Printf("Failed to embed chunks for resume %s: %v", file, err)
			continue
		}
		log.Printf("Successfully embedded %d chunks for resume: %s", len(embeddedChunks), file)

		log.Printf("Saving embeddings for resume: %s", file)
		err = vectorDB.SaveEmbeddings(context.Background(), "resumes", embeddedChunks)
		if err != nil {
			log.Printf("Failed to save embeddings for resume %s: %v", file, err)
			continue
		}
		log.Printf("Successfully processed and saved resume: %s", file)
	}

	// Add a final check (if possible with your VectorDB implementation)
	if counter, ok := vectorDB.(interface{ Count(string) (int, error) }); ok {
		count, err := counter.Count("resumes")
		if err != nil {
			log.Printf("Failed to get collection item count: %v", err)
		} else {
			log.Printf("Total items in 'resumes' collection: %d", count)
		}
	}

	return nil
}

func getCollectionItemCount(vectorDB raggo.VectorDB, collectionName string) (int, error) {
	// This is a placeholder. You'll need to implement this based on your VectorDB interface
	// It should return the number of items in the specified collection
	return 0, fmt.Errorf("getCollectionItemCount not implemented")
}

func searchCandidates(ctx context.Context, llm goal.LLM, embeddingService *raggo.EmbeddingService, vectorDB raggo.VectorDB, jobOffer string) ([]raggo.SearchResult, error) {
	// Summarize the job offer
	jobSummary, err := goal.Summarize(ctx, llm, jobOffer)
	if err != nil {
		return nil, fmt.Errorf("failed to summarize job offer: %w", err)
	}

	// Create a single chunk for the job summary
	jobChunk := raggo.Chunk{
		Text:          jobSummary,
		TokenSize:     len(strings.Split(jobSummary, " ")), // Simple word count as a proxy for token size
		StartSentence: 0,
		EndSentence:   1,
	}

	// Embed the job summary
	jobEmbeddings, err := embeddingService.EmbedChunks(ctx, []raggo.Chunk{jobChunk})
	if err != nil {
		return nil, fmt.Errorf("failed to embed job summary: %w", err)
	}

	if len(jobEmbeddings) == 0 {
		return nil, fmt.Errorf("no embeddings generated for job summary")
	}

	// Search for matching candidates
	results, err := vectorDB.Search(ctx, "resumes", jobEmbeddings[0].Embedding, 5, raggo.NewDefaultSearchParam())
	if err != nil {
		return nil, fmt.Errorf("failed to search for candidates: %w", err)
	}

	return results, nil
}

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"

	"github.com/danaugrs/go-tsne/tsne"
	"github.com/teilomillet/raggo"
	"gonum.org/v1/gonum/mat"
)

type ResumeEmbedding struct {
	Name      string    `json:"name"`
	Embedding []float64 `json:"embedding"`
	IsJob     bool      `json:"isJob"`
}

func main() {
	// Initialize components
	parser := raggo.NewParser()
	embedder, err := raggo.NewEmbedder(
		raggo.SetProvider("openai"),
		raggo.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		raggo.SetModel("text-embedding-3-small"),
	)
	if err != nil {
		log.Fatalf("Failed to create embedder: %v", err)
	}

	// Get the current working directory
	wd, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	// Construct the path to the CV directory
	cvDir := filepath.Join(wd, "testdata")

	// Process CVs
	embeddings, err := processCVs(parser, embedder, cvDir)
	if err != nil {
		log.Fatalf("Failed to process CVs: %v", err)
	}

	// Process job description
	jobDesc := `We are looking for a software engineer with 5+ years of experience in Go, distributed systems, and cloud technologies. The ideal candidate should have strong problem-solving skills and experience with Docker and Kubernetes.`
	jobEmbedding, err := embedder.Embed(context.Background(), jobDesc)
	if err != nil {
		log.Fatalf("Failed to embed job description: %v", err)
	}
	embeddings = append(embeddings, ResumeEmbedding{
		Name:      "Job Description",
		Embedding: jobEmbedding,
		IsJob:     true,
	})

	// Perform t-SNE
	tsneResults, err := performTSNE(embeddings)
	if err != nil {
		log.Fatalf("Failed to perform t-SNE: %v", err)
	}

	// Save t-SNE results
	err = saveTSNEResults(tsneResults, "tsne_results.json")
	if err != nil {
		log.Fatalf("Failed to save t-SNE results: %v", err)
	}

	// Start HTTP server for visualization
	go startHTTPServer()

	// Keep the main goroutine running
	select {}
}

func processCVs(parser raggo.Parser, embedder raggo.Embedder, cvDir string) ([]ResumeEmbedding, error) {
	files, err := filepath.Glob(filepath.Join(cvDir, "*.pdf"))
	if err != nil {
		return nil, fmt.Errorf("failed to list CV files: %w", err)
	}

	var embeddings []ResumeEmbedding

	for _, file := range files {
		doc, err := parser.Parse(file)
		if err != nil {
			log.Printf("Error parsing %s: %v", file, err)
			continue
		}

		embedding, err := embedder.Embed(context.Background(), doc.Content)
		if err != nil {
			log.Printf("Error embedding %s: %v", file, err)
			continue
		}

		name := filepath.Base(file)
		embeddings = append(embeddings, ResumeEmbedding{
			Name:      name,
			Embedding: embedding,
			IsJob:     false,
		})

		log.Printf("Processed and embedded %s", name)
	}

	return embeddings, nil
}

func performTSNE(embeddings []ResumeEmbedding) ([]ResumeEmbedding, error) {
	n := len(embeddings)
	d := len(embeddings[0].Embedding)
	X := mat.NewDense(n, d, nil)

	for i, emb := range embeddings {
		X.SetRow(i, emb.Embedding)
	}

	t := tsne.NewTSNE(2, 30, 100, 1000, true)
	Y := t.EmbedData(X, nil)

	tsneResults := make([]ResumeEmbedding, n)
	for i := 0; i < n; i++ {
		tsneResults[i] = ResumeEmbedding{
			Name:      embeddings[i].Name,
			Embedding: []float64{Y.At(i, 0), Y.At(i, 1)},
			IsJob:     embeddings[i].IsJob,
		}
	}

	return tsneResults, nil
}

func saveTSNEResults(embeddings []ResumeEmbedding, filename string) error {
	validEmbeddings := make([]ResumeEmbedding, 0, len(embeddings))
	for _, emb := range embeddings {
		if !math.IsNaN(emb.Embedding[0]) && !math.IsNaN(emb.Embedding[1]) {
			validEmbeddings = append(validEmbeddings, emb)
		} else {
			log.Printf("Warning: Skipping embedding for %s due to NaN values", emb.Name)
		}
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(validEmbeddings); err != nil {
		return fmt.Errorf("failed to encode embeddings: %w", err)
	}

	log.Printf("t-SNE results saved to %s (Saved %d out of %d embeddings)", filename, len(validEmbeddings), len(embeddings))
	return nil
}

func startHTTPServer() {
	http.HandleFunc("/", serveVisualization)
	http.HandleFunc("/tsne_results.json", serveTSNEResults)
	log.Println("Starting server on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func serveVisualization(w http.ResponseWriter, r *http.Request) {
	html := "<!DOCTYPE html>" +
		"<html>" +
		"<head>" +
		"<title>Interactive CV and Job Description Embeddings Visualization</title>" +
		"<script src=\"https://d3js.org/d3.v7.min.js\"></script>" +
		"<style>" +
		"body { font-family: Arial, sans-serif; }" +
		"#chart { width: 800px; height: 600px; margin: 0 auto; }" +
		".tooltip { " +
		"position: absolute; " +
		"background: white; " +
		"border: 1px solid #ddd; " +
		"padding: 10px; " +
		"pointer-events: none;" +
		"}" +
		".closest-cv { stroke: #ff9900; stroke-width: 2px; }" +
		"</style>" +
		"</head>" +
		"<body>" +
		"<h1>Interactive CV and Job Description Embeddings Visualization</h1>" +
		"<div id=\"chart\"></div>" +
		"<div id=\"info\" style=\"text-align: center; margin-top: 20px;\"></div>" +
		"<script>" +
		`
        d3.json('/tsne_results.json').then(data => {
            const width = 800;
            const height = 600;
            const margin = {top: 20, right: 20, bottom: 30, left: 40};

            const svg = d3.select('#chart')
                .append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom)
                .append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

            const x = d3.scaleLinear()
                .domain(d3.extent(data, d => d.embedding[0]))
                .range([0, width]);

            const y = d3.scaleLinear()
                .domain(d3.extent(data, d => d.embedding[1]))
                .range([height, 0]);

            const color = d => d.isJob ? 'red' : 'steelblue';

            // Find job description
            const jobDesc = data.find(d => d.isJob);

            // Calculate distances and find closest CV
            let closestCV = null;
            let minDistance = Infinity;
            data.forEach(d => {
                if (!d.isJob) {
                    d.distance = Math.sqrt(
                        Math.pow(d.embedding[0] - jobDesc.embedding[0], 2) +
                        Math.pow(d.embedding[1] - jobDesc.embedding[1], 2)
                    );
                    if (d.distance < minDistance) {
                        minDistance = d.distance;
                        closestCV = d;
                    }
                }
            });

            const tooltip = d3.select("body").append("div")
                .attr("class", "tooltip")
                .style("opacity", 0);

            svg.selectAll('circle')
                .data(data)
                .enter()
                .append('circle')
                .attr('cx', d => x(d.embedding[0]))
                .attr('cy', d => y(d.embedding[1]))
                .attr('r', d => d.isJob ? 7 : 5)
                .attr('fill', color)
                .attr('class', d => d === closestCV ? 'closest-cv' : '')
                .on("mouseover", (event, d) => {
                    tooltip.transition()
                        .duration(200)
                        .style("opacity", .9);
                    tooltip.html(` + "`" + `Name: ${d.name}<br/>
                                  ${d.isJob ? 'Job Description' : ` + "`" + `Distance: ${d.distance.toFixed(4)}` + "`" + `}` + "`" + `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 28) + "px");
                })
                .on("mouseout", () => {
                    tooltip.transition()
                        .duration(500)
                        .style("opacity", 0);
                });

            svg.append('g')
                .attr('transform', 'translate(0,' + height + ')')
                .call(d3.axisBottom(x));

            svg.append('g')
                .call(d3.axisLeft(y));

            // Add legend
            const legend = svg.append('g')
                .attr('class', 'legend')
                .attr('transform', 'translate(' + (width - 120) + ',' + 20 + ')');

            legend.append('circle')
                .attr('cx', 0)
                .attr('cy', 0)
                .attr('r', 5)
                .style('fill', 'steelblue');

            legend.append('circle')
                .attr('cx', 0)
                .attr('cy', 20)
                .attr('r', 7)
                .style('fill', 'red');

            legend.append('text')
                .attr('x', 10)
                .attr('y', 5)
                .text('CVs');

            legend.append('text')
                .attr('x', 10)
                .attr('y', 25)
                .text('Job Description');

            // Display info about closest CV
            d3.select('#info').html(` + "`" + `
                Closest CV: <strong>${closestCV.name}</strong><br>
                Distance: ${closestCV.distance.toFixed(4)}
            ` + "`" + `);
        });
        ` +
		"</script>" +
		"</body>" +
		"</html>"

	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(html))
}

func serveTSNEResults(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "tsne_results.json")
}

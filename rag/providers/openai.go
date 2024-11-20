// Package providers implements embedding service providers for the Raggo framework.
// The OpenAI provider offers high-quality text embeddings through OpenAI's API,
// supporting models like text-embedding-3-small and text-embedding-3-large.
package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

func init() {
	// Register the OpenAI provider when the package is initialized
	RegisterEmbedder("openai", NewOpenAIEmbedder)
}

// Default settings for the OpenAI embedder
const (
	// defaultEmbeddingAPI is the endpoint for OpenAI's embedding service
	defaultEmbeddingAPI = "https://api.openai.com/v1/embeddings"
	// defaultModelName is the recommended model for most use cases
	defaultModelName    = "text-embedding-3-small"
)

// OpenAIEmbedder implements the Embedder interface using OpenAI's API.
// It supports various embedding models and handles API communication,
// rate limiting, and error recovery. The embedder is designed to be
// thread-safe and can be used concurrently.
type OpenAIEmbedder struct {
	apiKey    string        // API key for authentication
	client    *http.Client  // HTTP client with timeout
	apiURL    string        // API endpoint URL
	modelName string        // Selected embedding model
}

// NewOpenAIEmbedder creates a new OpenAI embedding provider with the given
// configuration. The provider requires an API key and optionally accepts:
// - model: The embedding model to use (defaults to text-embedding-3-small)
// - api_url: Custom API endpoint URL
// - timeout: Custom timeout duration
//
// Example config:
//
//	config := map[string]interface{}{
//	    "api_key": "your-api-key",
//	    "model": "text-embedding-3-small",
//	    "timeout": 30 * time.Second,
//	}
func NewOpenAIEmbedder(config map[string]interface{}) (Embedder, error) {
	apiKey, ok := config["api_key"].(string)
	if !ok || apiKey == "" {
		return nil, fmt.Errorf("API key is required for OpenAI embedder")
	}

	e := &OpenAIEmbedder{
		apiKey:    apiKey,
		client:    &http.Client{Timeout: 30 * time.Second},
		apiURL:    defaultEmbeddingAPI,
		modelName: defaultModelName,
	}

	if model, ok := config["model"].(string); ok && model != "" {
		e.modelName = model
	}

	if apiURL, ok := config["api_url"].(string); ok && apiURL != "" {
		e.apiURL = apiURL
	}

	if timeout, ok := config["timeout"].(time.Duration); ok {
		e.client.Timeout = timeout
	}

	return e, nil
}

// embeddingRequest represents the JSON structure for API requests
type embeddingRequest struct {
	Input string `json:"input"` // Text to embed
	Model string `json:"model"` // Model to use
}

// embeddingResponse represents the JSON structure for API responses
type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"` // Vector representation
	} `json:"data"`
}

// Embed converts the input text into a vector representation using the
// configured OpenAI model. The method handles:
// - Request preparation and validation
// - API communication with retry logic
// - Response parsing and error handling
//
// The resulting vector captures the semantic meaning of the input text
// and can be used for similarity search operations.
func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	reqBody, err := json.Marshal(embeddingRequest{
		Input: text,
		Model: e.modelName,
	})
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.apiURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.apiKey)

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, resp.Status)
	}

	var embeddingResp embeddingResponse
	err = json.Unmarshal(body, &embeddingResp)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling response: %w", err)
	}

	if len(embeddingResp.Data) == 0 {
		return nil, fmt.Errorf("no embedding data in response")
	}

	return embeddingResp.Data[0].Embedding, nil
}

// GetDimension returns the output dimension for the current embedding model.
// Each model produces vectors of a fixed size:
// - text-embedding-3-small: 1536 dimensions
// - text-embedding-3-large: 3072 dimensions
// - text-embedding-ada-002: 1536 dimensions
//
// This information is crucial for configuring vector databases and ensuring
// compatibility across the system.
func (e *OpenAIEmbedder) GetDimension() (int, error) {
	switch e.modelName {
	case "text-embedding-3-small":
		return 1536, nil
	case "text-embedding-3-large":
		return 3072, nil
	case "text-embedding-ada-002":
		return 1536, nil
	default:
		return 0, fmt.Errorf("unknown model: %s", e.modelName)
	}
}

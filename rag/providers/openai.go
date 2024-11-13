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
	RegisterEmbedder("openai", NewOpenAIEmbedder)
}

const (
	defaultEmbeddingAPI = "https://api.openai.com/v1/embeddings"
	defaultModelName    = "text-embedding-3-small"
)

type OpenAIEmbedder struct {
	apiKey    string
	client    *http.Client
	apiURL    string
	modelName string
}

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

type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

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

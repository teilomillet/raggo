// Package rag provides retrieval-augmented generation capabilities.
package rag

import (
	"context"
	"sort"
)

// RRFReranker implements Reciprocal Rank Fusion (RRF) for combining and reranking search results.
// RRF is a robust rank fusion method that effectively combines results from different retrieval systems
// without requiring score normalization. It uses the formula: RRF(d) = Σ 1/(k + r(d))
// where d is a document, k is a constant, and r(d) is the rank of document d in each result list.
type RRFReranker struct {
	k float64 // k is a constant that prevents division by zero and controls the influence of high-ranked items
}

// NewRRFReranker creates a new RRF reranker with the specified k parameter.
// The k parameter controls ranking influence - higher values of k decrease the
// influence of high-ranked items. If k <= 0, it defaults to 60 (from the original RRF paper).
//
// Typical k values:
// - k = 60: Standard value from RRF literature, good general-purpose setting
// - k < 60: Increases influence of top-ranked items
// - k > 60: More weight to lower-ranked items, smoother ranking distribution
func NewRRFReranker(k float64) *RRFReranker {
	if k <= 0 {
		k = 60 // Default value from RRF paper
	}
	return &RRFReranker{k: k}
}

// Rerank combines and reranks results using Reciprocal Rank Fusion.
// It takes results from dense (semantic) and sparse (lexical) search and combines
// them using weighted RRF scores. The method handles cases where documents appear
// in both result sets by combining their weighted scores.
//
// Parameters:
//   - ctx: Context for potential future extensions (e.g., timeouts, cancellation)
//   - query: The original search query (reserved for future extensions)
//   - denseResults: Results from dense/semantic search
//   - sparseResults: Results from sparse/lexical search
//   - denseWeight: Weight for dense search results (normalized internally)
//   - sparseWeight: Weight for sparse search results (normalized internally)
//
// Returns:
//   - []SearchResult: Reranked results sorted by combined score
//   - error: Currently always nil, reserved for future extensions
//
// The reranking process:
// 1. Normalizes weights to sum to 1.0
// 2. Calculates RRF scores for each result based on rank
// 3. Applies weights to scores based on result source (dense/sparse)
// 4. Combines scores for documents appearing in both result sets
// 5. Sorts final results by combined score
func (r *RRFReranker) Rerank(
	ctx context.Context,
	query string,
	denseResults, sparseResults []SearchResult,
	denseWeight, sparseWeight float64,
) ([]SearchResult, error) {
	// Normalize weights
	totalWeight := denseWeight + sparseWeight
	if totalWeight > 0 {
		denseWeight /= totalWeight
		sparseWeight /= totalWeight
	} else {
		denseWeight = 0.5
		sparseWeight = 0.5
	}

	// Create maps to store combined scores
	scores := make(map[int64]float64)
	docMap := make(map[int64]SearchResult)

	// Process dense results
	for rank, result := range denseResults {
		rrf := 1.0 / (float64(rank+1) + r.k)
		scores[result.ID] = rrf * denseWeight
		docMap[result.ID] = result
	}

	// Process sparse results
	for rank, result := range sparseResults {
		rrf := 1.0 / (float64(rank+1) + r.k)
		if score, exists := scores[result.ID]; exists {
			scores[result.ID] = score + rrf*sparseWeight
		} else {
			scores[result.ID] = rrf * sparseWeight
			docMap[result.ID] = result
		}
	}

	// Convert scores back to results
	results := make([]SearchResult, 0, len(scores))
	for id, score := range scores {
		result := docMap[id]
		result.Score = score
		results = append(results, result)
	}

	// Sort by combined score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return all reranked results
	return results, nil
}

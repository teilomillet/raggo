package rag

import (
	"context"
	"sort"
)

// RRFReranker implements Reciprocal Rank Fusion for result reranking
type RRFReranker struct {
	k float64 // Constant to prevent division by zero and control ranking influence
}

// NewRRFReranker creates a new RRF reranker with the given k parameter
func NewRRFReranker(k float64) *RRFReranker {
	if k <= 0 {
		k = 60 // Default value from RRF paper
	}
	return &RRFReranker{k: k}
}

// Rerank combines and reranks results using Reciprocal Rank Fusion
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

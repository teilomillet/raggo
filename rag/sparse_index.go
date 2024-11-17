package rag

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
)

// BM25Parameters holds the parameters for BM25 scoring
type BM25Parameters struct {
	K1 float64 // Term saturation parameter (typically 1.2-2.0)
	B  float64 // Length normalization parameter (typically 0.75)
}

// DefaultBM25Parameters returns default BM25 parameters
func DefaultBM25Parameters() BM25Parameters {
	return BM25Parameters{
		K1: 1.5,
		B:  0.75,
	}
}

// BM25Index implements a sparse index using BM25 scoring
type BM25Index struct {
	mu            sync.RWMutex
	docs          map[int64]string                    // Document content by ID
	metadata      map[int64]map[string]interface{}    // Document metadata by ID
	termFreq      map[int64]map[string]int           // Term frequency per document
	docFreq       map[string]int                      // Document frequency per term
	docLength     map[int64]int                       // Length of each document
	avgDocLength  float64                             // Average document length
	totalDocs     int                                 // Total number of documents
	params        BM25Parameters                      // BM25 parameters
	preprocessor  func(string) []string               // Text preprocessing function
}

// NewBM25Index creates a new BM25 index with default parameters
func NewBM25Index() *BM25Index {
	return &BM25Index{
		docs:         make(map[int64]string),
		metadata:     make(map[int64]map[string]interface{}),
		termFreq:     make(map[int64]map[string]int),
		docFreq:      make(map[string]int),
		docLength:    make(map[int64]int),
		params:       DefaultBM25Parameters(),
		preprocessor: defaultPreprocessor,
	}
}

// defaultPreprocessor implements basic text preprocessing
func defaultPreprocessor(text string) []string {
	// Convert to lowercase and split into words
	words := strings.Fields(strings.ToLower(text))
	return words
}

// Add adds a document to the index
func (idx *BM25Index) Add(ctx context.Context, id int64, content string, metadata map[string]interface{}) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Store document and metadata
	idx.docs[id] = content
	idx.metadata[id] = metadata

	// Process terms
	terms := idx.preprocessor(content)
	termFreq := make(map[string]int)
	for _, term := range terms {
		termFreq[term]++
	}

	// Update index statistics
	idx.termFreq[id] = termFreq
	idx.docLength[id] = len(terms)
	for term := range termFreq {
		idx.docFreq[term]++
	}

	// Update collection statistics
	idx.totalDocs++
	var totalLength int
	for _, length := range idx.docLength {
		totalLength += length
	}
	idx.avgDocLength = float64(totalLength) / float64(idx.totalDocs)

	return nil
}

// Remove removes a document from the index
func (idx *BM25Index) Remove(ctx context.Context, id int64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	// Update document frequencies
	if termFreq, exists := idx.termFreq[id]; exists {
		for term := range termFreq {
			idx.docFreq[term]--
			if idx.docFreq[term] == 0 {
				delete(idx.docFreq, term)
			}
		}
	}

	// Remove document data
	delete(idx.docs, id)
	delete(idx.metadata, id)
	delete(idx.termFreq, id)
	delete(idx.docLength, id)

	// Update collection statistics
	idx.totalDocs--
	if idx.totalDocs > 0 {
		var totalLength int
		for _, length := range idx.docLength {
			totalLength += length
		}
		idx.avgDocLength = float64(totalLength) / float64(idx.totalDocs)
	} else {
		idx.avgDocLength = 0
	}

	return nil
}

// Search performs BM25 search on the index
func (idx *BM25Index) Search(ctx context.Context, query string, topK int) ([]SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	// Process query terms
	queryTerms := idx.preprocessor(query)
	scores := make(map[int64]float64)

	// Calculate BM25 scores
	for _, term := range queryTerms {
		if df, exists := idx.docFreq[term]; exists {
			idf := math.Log(1 + (float64(idx.totalDocs)-float64(df)+0.5)/(float64(df)+0.5))

			for docID, docTerms := range idx.termFreq {
				if tf, exists := docTerms[term]; exists {
					docLen := float64(idx.docLength[docID])
					numerator := float64(tf) * (idx.params.K1 + 1)
					denominator := float64(tf) + idx.params.K1*(1-idx.params.B+idx.params.B*docLen/idx.avgDocLength)
					scores[docID] += idf * numerator / denominator
				}
			}
		}
	}

	// Convert scores to results
	results := make([]SearchResult, 0, len(scores))
	for docID, score := range scores {
		results = append(results, SearchResult{
			ID:     docID,
			Score:  score,
			Fields: map[string]interface{}{
				"Text":     idx.docs[docID],
				"Metadata": idx.metadata[docID],
			},
		})
	}

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top K results
	if len(results) > topK {
		results = results[:topK]
	}

	return results, nil
}

// SetParameters updates the BM25 parameters
func (idx *BM25Index) SetParameters(params BM25Parameters) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.params = params
}

// SetPreprocessor sets a custom text preprocessing function
func (idx *BM25Index) SetPreprocessor(preprocessor func(string) []string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.preprocessor = preprocessor
}

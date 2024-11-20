// Package rag provides retrieval-augmented generation capabilities.
package rag

import (
	"context"
	"math"
	"sort"
	"strings"
	"sync"
)

// BM25Parameters holds the parameters for BM25 scoring algorithm.
// BM25 (Best Match 25) is a probabilistic ranking function that estimates
// the relevance of documents to a given search query based on term frequency,
// inverse document frequency, and document length normalization.
type BM25Parameters struct {
	K1 float64 // K1 controls term frequency saturation (1.2-2.0 typical)
	B  float64 // B controls document length normalization (0.75 typical)
}

// DefaultBM25Parameters returns recommended BM25 parameters based on
// empirical research. These values work well for most general-purpose
// text search applications:
// - K1 = 1.5: Balanced term frequency saturation
// - B = 0.75: Standard length normalization
func DefaultBM25Parameters() BM25Parameters {
	return BM25Parameters{
		K1: 1.5,
		B:  0.75,
	}
}

// BM25Index implements a sparse retrieval index using the BM25 ranking algorithm.
// It provides thread-safe document indexing and retrieval with the following features:
// - Efficient term-based document scoring
// - Document length normalization
// - Configurable text preprocessing
// - Metadata storage and retrieval
// - Thread-safe operations
type BM25Index struct {
	mu            sync.RWMutex                        // Protects concurrent access to index
	docs          map[int64]string                    // Stores original document content
	metadata      map[int64]map[string]interface{}    // Stores document metadata
	termFreq      map[int64]map[string]int           // Term frequency per document
	docFreq       map[string]int                      // Document frequency per term
	docLength     map[int64]int                       // Length of each document
	avgDocLength  float64                             // Average document length
	totalDocs     int                                 // Total number of documents
	params        BM25Parameters                      // BM25 scoring parameters
	preprocessor  func(string) []string               // Text preprocessing function
}

// NewBM25Index creates a new BM25 index with default parameters.
// The index is initialized with:
// - Default BM25 parameters (K1=1.5, B=0.75)
// - Basic preprocessor (lowercase, whitespace tokenization)
// - Empty document store and statistics
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

// defaultPreprocessor implements basic text preprocessing by:
// 1. Converting text to lowercase
// 2. Splitting on whitespace
// Users can replace this with custom preprocessing via SetPreprocessor
func defaultPreprocessor(text string) []string {
	// Convert to lowercase and split into words
	words := strings.Fields(strings.ToLower(text))
	return words
}

// Add indexes a new document with the given ID, content, and metadata.
// This operation is thread-safe and automatically updates all relevant
// index statistics including term frequencies, document lengths, and
// collection-wide averages.
//
// Parameters:
//   - ctx: Context for potential future extensions
//   - id: Unique document identifier
//   - content: Document text content
//   - metadata: Optional document metadata
//
// Returns error if the operation fails (currently always nil).
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

// Remove deletes a document from the index and updates all relevant statistics.
// This operation is thread-safe and maintains index consistency by:
// - Updating document frequencies
// - Removing document data
// - Recalculating collection statistics
//
// Parameters:
//   - ctx: Context for potential future extensions
//   - id: ID of document to remove
//
// Returns error if the operation fails (currently always nil).
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

// Search performs BM25-based retrieval on the index.
// The BM25 score for a document D and query Q is calculated as:
// score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
//
// Parameters:
//   - ctx: Context for potential future extensions
//   - query: Search query text
//   - topK: Maximum number of results to return
//
// Returns:
//   - []SearchResult: Sorted results by BM25 score
//   - error: Error if search fails (currently always nil)
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

// SetParameters updates the BM25 scoring parameters.
// This operation is thread-safe and affects all subsequent searches.
// Typical values:
// - K1: 1.2-2.0 (higher values increase term frequency influence)
// - B: 0.75 (lower values reduce length normalization effect)
func (idx *BM25Index) SetParameters(params BM25Parameters) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.params = params
}

// SetPreprocessor sets a custom text preprocessing function.
// The preprocessor converts raw text into terms for indexing and searching.
// Custom preprocessors can implement:
// - Stopword removal
// - Stemming/lemmatization
// - N-gram generation
// - Special character handling
func (idx *BM25Index) SetPreprocessor(preprocessor func(string) []string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.preprocessor = preprocessor
}

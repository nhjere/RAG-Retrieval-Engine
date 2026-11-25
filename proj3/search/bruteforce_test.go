package search

import (
	"math"
	"math/rand"
	"testing"

	"proj3-redesigned/dataset"
)

// randomVectors returns n random vectors of length dim, each scaled to
// length 1 like real embeddings, so a vector's best match is itself.
func randomVectors(r *rand.Rand, n, dim int) *dataset.Vectors {
	data := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		row := data[i*dim : (i+1)*dim]
		var norm float64
		for j := range row {
			row[j] = float32(r.NormFloat64())
			norm += float64(row[j]) * float64(row[j])
		}
		scale := float32(1 / math.Sqrt(norm))
		for j := range row {
			row[j] *= scale
		}
	}
	return &dataset.Vectors{Data: data, N: n, Dim: dim}
}

func TestBruteForceFindsSelf(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	vecs := randomVectors(r, 100, 16)

	// search the passages with themselves as queries
	results := BruteForce(vecs, vecs, 5)

	if len(results) != vecs.N {
		t.Fatalf("got %d result lists, want %d", len(results), vecs.N)
	}
	for q, res := range results {
		if len(res) != 5 {
			t.Fatalf("query %d: got %d results, want 5", q, len(res))
		}
		if res[0].ID != q {
			t.Errorf("query %d: top match is %d, want itself", q, res[0].ID)
		}
		for i := 1; i < len(res); i++ {
			if res[i].Score > res[i-1].Score {
				t.Errorf("query %d: results not sorted best first at rank %d", q, i)
			}
		}
	}
}

func TestBruteForceMatchesFullSort(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	passages := randomVectors(r, 500, 16)
	queries := randomVectors(r, 20, 16)

	results := BruteForce(passages, queries, 10)

	// compare each query against scoring and sorting every passage
	for q := 0; q < queries.N; q++ {
		all := make([]Result, passages.N)
		for p := 0; p < passages.N; p++ {
			all[p] = Result{ID: p, Score: Dot(queries.Row(q), passages.Row(p))}
		}
		sameResults(t, results[q], sortedBest(all, 10))
	}
}

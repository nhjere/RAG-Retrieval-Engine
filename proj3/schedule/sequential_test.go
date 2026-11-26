package schedule

import (
	"math"
	"math/rand"
	"testing"

	"proj3-redesigned/dataset"
	"proj3-redesigned/index"
	"proj3-redesigned/search"
)

const (
	testDim   = 16
	testNlist = 30
)

// randomVectors returns n random vectors of length dim, each scaled to length 1
func randomVectors(r *rand.Rand, n, dim int) *dataset.Vectors {
	data := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		row := data[i*dim : (i+1)*dim]
		var sum float64
		for j := range row {
			row[j] = float32(r.NormFloat64())
			sum += float64(row[j]) * float64(row[j])
		}
		scale := float32(1 / math.Sqrt(sum))
		for j := range row {
			row[j] *= scale
		}
	}
	return &dataset.Vectors{Data: data, N: n, Dim: dim}
}

// testSetup builds an index over random passages plus a set of queries;
// later tests for the parallel modes can reuse it
func testSetup() (*dataset.Vectors, *index.IVF, *dataset.Vectors) {
	r := rand.New(rand.NewSource(3))
	passages := randomVectors(r, 3000, testDim)
	queries := randomVectors(r, 50, testDim)
	idx := index.Build(passages, testNlist, 10, 1500, 42)
	return passages, idx, queries
}

func TestSequentialAllClustersMatchesBruteForce(t *testing.T) {
	passages, idx, queries := testSetup()

	// searching every cluster must give exactly the brute-force answer
	got := Sequential(idx, queries, 10, testNlist)
	want := search.BruteForce(passages, queries, 10)

	for q := range want {
		if len(got[q]) != len(want[q]) {
			t.Fatalf("query %d: got %d results, want %d", q, len(got[q]), len(want[q]))
		}
		for rank := range want[q] {
			if got[q][rank] != want[q][rank] {
				t.Fatalf("query %d rank %d: got %+v, want %+v", q, rank, got[q][rank], want[q][rank])
			}
		}
	}
}

func TestSequentialOnlyReturnsProbedClusters(t *testing.T) {
	_, idx, queries := testSetup()
	const nprobe = 2

	results := Sequential(idx, queries, 10, nprobe)

	for q, res := range results {
		// collect every passage in this query's probed clusters
		allowed := map[int]bool{}
		for _, c := range idx.Probe(queries.Row(q), nprobe) {
			_, ids := idx.Cluster(c)
			for _, id := range ids {
				allowed[id] = true
			}
		}
		for _, r := range res {
			if !allowed[r.ID] {
				t.Fatalf("query %d returned passage %d from a cluster it did not probe", q, r.ID)
			}
		}
	}
}

func TestSequentialResultsSortedBestFirst(t *testing.T) {
	_, idx, queries := testSetup()

	for q, res := range Sequential(idx, queries, 10, 5) {
		for i := 1; i < len(res); i++ {
			if res[i].Score > res[i-1].Score {
				t.Fatalf("query %d: rank %d scores higher than rank %d", q, i, i-1)
			}
		}
	}
}

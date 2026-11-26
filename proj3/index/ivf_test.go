package index

import (
	"math/rand"
	"testing"
)

const (
	testN     = 2000
	testDim   = 16
	testNlist = 20
)

func buildTestIndex(t *testing.T) (*IVF, *rand.Rand) {
	t.Helper()
	r := rand.New(rand.NewSource(7))
	passages := randomVectors(r, testN, testDim)
	return Build(passages, testNlist, 10, 1000, 42), r
}

func TestBuildEveryPassageOnce(t *testing.T) {
	idx, _ := buildTestIndex(t)

	seen := make([]int, testN)
	for _, id := range idx.IDs {
		seen[id]++
	}
	for p, count := range seen {
		if count != 1 {
			t.Fatalf("passage %d appears %d times, want 1", p, count)
		}
	}
}

func TestBuildRowsMatchPassages(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	passages := randomVectors(r, testN, testDim)
	idx := Build(passages, testNlist, 10, 1000, 42)

	// row i of Vecs must hold the vector of passage IDs[i]
	for row, id := range idx.IDs {
		want := passages.Row(id)
		got := idx.Vecs[row*testDim : (row+1)*testDim]
		for d := range want {
			if got[d] != want[d] {
				t.Fatalf("row %d does not hold passage %d", row, id)
			}
		}
	}
}

func TestOffsetsCoverAllRows(t *testing.T) {
	idx, _ := buildTestIndex(t)

	if len(idx.Offsets) != testNlist+1 {
		t.Fatalf("len(Offsets) = %d, want %d", len(idx.Offsets), testNlist+1)
	}
	if idx.Offsets[0] != 0 || idx.Offsets[testNlist] != testN {
		t.Fatalf("Offsets run %d..%d, want 0..%d", idx.Offsets[0], idx.Offsets[testNlist], testN)
	}
	for c := 0; c < testNlist; c++ {
		if idx.Offsets[c+1] < idx.Offsets[c] {
			t.Fatalf("Offsets decrease at cluster %d", c)
		}
	}
}

func TestClusterMatchesOffsets(t *testing.T) {
	idx, _ := buildTestIndex(t)

	for c := 0; c < testNlist; c++ {
		vecs, ids := idx.Cluster(c)
		size := idx.Offsets[c+1] - idx.Offsets[c]
		if len(ids) != size || len(vecs) != size*testDim {
			t.Fatalf("cluster %d: got %d ids and %d values, want %d and %d",
				c, len(ids), len(vecs), size, size*testDim)
		}
	}
}

func TestProbeFindsOwnCentroid(t *testing.T) {
	idx, _ := buildTestIndex(t)

	// a centroid used as the query must pick itself first
	for c := 0; c < testNlist; c++ {
		got := idx.Probe(idx.Centroids.Row(c), 1)
		if len(got) != 1 || got[0] != c {
			t.Fatalf("Probe(centroid %d, 1) = %v, want [%d]", c, got, c)
		}
	}
}

func TestProbeAllReturnsEveryCluster(t *testing.T) {
	idx, r := buildTestIndex(t)
	query := randomVectors(r, 1, testDim).Row(0)

	got := idx.Probe(query, testNlist)
	seen := map[int]bool{}
	for _, c := range got {
		seen[c] = true
	}
	if len(got) != testNlist || len(seen) != testNlist {
		t.Fatalf("Probe with nprobe=nlist returned %d clusters (%d distinct), want %d",
			len(got), len(seen), testNlist)
	}
}

func TestStats(t *testing.T) {
	idx, _ := buildTestIndex(t)

	minSize, maxSize, mean := idx.Stats()
	if want := float64(testN) / testNlist; mean != want {
		t.Errorf("mean = %v, want %v", mean, want)
	}
	if float64(minSize) > mean || float64(maxSize) < mean {
		t.Errorf("got min %d, mean %v, max %d; want min <= mean <= max", minSize, mean, maxSize)
	}
}

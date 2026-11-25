package search

import (
	"math/rand"
	"sort"
	"testing"
)

// sortedBest is the slow reference: sort everything best first with the same
// tie rule as TopK, then keep the first k.
func sortedBest(all []Result, k int) []Result {
	cp := make([]Result, len(all))
	copy(cp, all)
	sort.Slice(cp, func(i, j int) bool { return worse(cp[j], cp[i]) })
	if k > len(cp) {
		k = len(cp)
	}
	return cp[:k]
}

func sameResults(t *testing.T, got, want []Result) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("rank %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestTopKMatchesSort(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	all := make([]Result, 1000)
	top := NewTopK(10)
	for i := range all {
		all[i] = Result{ID: i, Score: r.Float32()}
		top.Push(all[i].ID, all[i].Score)
	}
	sameResults(t, top.Results(), sortedBest(all, 10))
}

func TestTopKTiesBreakByLowerID(t *testing.T) {
	// every score equal: the winners must be IDs 0..4 in order,
	// no matter what order they were pushed in
	top := NewTopK(5)
	for _, id := range []int{9, 3, 7, 0, 5, 1, 8, 2, 6, 4} {
		top.Push(id, 0.5)
	}
	got := top.Results()
	for i, res := range got {
		if res.ID != i {
			t.Errorf("rank %d: got ID %d, want %d", i, res.ID, i)
		}
	}
}

func TestTopKFewerThanK(t *testing.T) {
	top := NewTopK(10)
	all := []Result{{ID: 0, Score: 0.3}, {ID: 1, Score: 0.9}, {ID: 2, Score: 0.1}}
	for _, res := range all {
		top.Push(res.ID, res.Score)
	}
	sameResults(t, top.Results(), sortedBest(all, 10))
}

func TestTopKZeroK(t *testing.T) {
	top := NewTopK(0)
	top.Push(1, 0.9)
	if got := top.Results(); len(got) != 0 {
		t.Fatalf("k=0 returned %d results, want 0", len(got))
	}
}

func TestResultsDoesNotEmptyTopK(t *testing.T) {
	top := NewTopK(3)
	for i := 0; i < 10; i++ {
		top.Push(i, float32(i))
	}
	first := top.Results()
	second := top.Results()
	sameResults(t, second, first)
}

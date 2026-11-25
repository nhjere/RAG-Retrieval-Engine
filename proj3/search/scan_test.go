package search

import "testing"

func TestDot(t *testing.T) {
	got := Dot([]float32{1, 2, 3}, []float32{4, 5, 6})
	if got != 32 {
		t.Fatalf("Dot = %v, want 32", got)
	}
}

func TestScanBlockUsesIDs(t *testing.T) {
	// three 2-number vectors whose IDs are not their positions,
	// like one IVF cluster holding scattered passages
	query := []float32{1, 0}
	vecs := []float32{0.9, 0.1, 0.2, 0.8, 0.7, 0.3}
	ids := []int{42, 7, 99}

	top := NewTopK(2)
	ScanBlock(query, vecs, ids, 2, top)
	got := top.Results()

	want := []int{42, 99}
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i] {
			t.Errorf("rank %d: got ID %d, want %d", i, got[i].ID, want[i])
		}
	}
}

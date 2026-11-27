package eval

import (
	"math"
	"path/filepath"
	"testing"

	"proj3-redesigned/search"
)

func res(ids ...int) []search.Result {
	out := make([]search.Result, len(ids))
	for i, id := range ids {
		out[i] = search.Result{ID: id, Score: float32(len(ids) - i)}
	}
	return out
}

func TestRecallPerfect(t *testing.T) {
	exact := [][]search.Result{res(1, 2, 3), res(4, 5, 6)}
	if got := Recall(exact, exact, 3); got != 1 {
		t.Fatalf("identical results: recall = %v, want 1", got)
	}
}

func TestRecallNone(t *testing.T) {
	exact := [][]search.Result{res(1, 2, 3)}
	approx := [][]search.Result{res(7, 8, 9)}
	if got := Recall(approx, exact, 3); got != 0 {
		t.Fatalf("no overlap: recall = %v, want 0", got)
	}
}

func TestRecallPartialAndOrderFree(t *testing.T) {
	// query 0 finds 2 of 3 (in a different order), query 1 finds 3 of 3
	exact := [][]search.Result{res(1, 2, 3), res(4, 5, 6)}
	approx := [][]search.Result{res(3, 9, 1), res(6, 5, 4)}
	want := (2.0/3.0 + 1.0) / 2
	if got := Recall(approx, exact, 3); math.Abs(got-want) > 1e-9 {
		t.Fatalf("recall = %v, want %v", got, want)
	}
}

func TestGroundTruthRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "groundtruth.bin")
	saved := [][]search.Result{res(10, 20, 30), res(40, 50, 60)}
	if err := SaveGroundTruth(path, saved); err != nil {
		t.Fatalf("save: %v", err)
	}

	loaded, err := LoadGroundTruth(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(loaded) != len(saved) {
		t.Fatalf("loaded %d queries, want %d", len(loaded), len(saved))
	}
	for q := range saved {
		for i := range saved[q] {
			if loaded[q][i].ID != saved[q][i].ID {
				t.Fatalf("query %d rank %d: got ID %d, want %d", q, i, loaded[q][i].ID, saved[q][i].ID)
			}
		}
	}
	if Recall(saved, loaded, 3) != 1 {
		t.Fatal("recall of saved results against loaded answer key should be 1")
	}
}

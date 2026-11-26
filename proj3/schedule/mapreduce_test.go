package schedule

import (
	"strconv"
	"testing"

	"proj3-redesigned/search"
)

// sameAll fails if two result sets differ anywhere
func sameAll(t *testing.T, label string, got, want [][]search.Result) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d result lists, want %d", label, len(got), len(want))
	}
	for q := range want {
		if len(got[q]) != len(want[q]) {
			t.Fatalf("%s: query %d has %d results, want %d", label, q, len(got[q]), len(want[q]))
		}
		for rank := range want[q] {
			if got[q][rank] != want[q][rank] {
				t.Fatalf("%s: query %d rank %d: got %+v, want %+v",
					label, q, rank, got[q][rank], want[q][rank])
			}
		}
	}
}

func TestMapReduceMatchesSequential(t *testing.T) {
	_, idx, queries := testSetup()
	want := Sequential(idx, queries, 10, 5)

	for _, threads := range []int{1, 2, 3, 4, 7, 8, 12} {
		got := MapReduce(idx, queries, 10, 5, threads)
		sameAll(t, "threads="+strconv.Itoa(threads), got, want)
	}
}

func TestMapReduceAllClustersMatchesBruteForce(t *testing.T) {
	passages, idx, queries := testSetup()
	want := search.BruteForce(passages, queries, 10)
	got := MapReduce(idx, queries, 10, testNlist, 4)
	sameAll(t, "nprobe=nlist", got, want)
}

func TestMapReduceMoreThreadsThanClusters(t *testing.T) {
	// some threads own no clusters at all and must still reach both barriers
	_, idx, queries := testSetup()
	want := Sequential(idx, queries, 10, 5)
	got := MapReduce(idx, queries, 10, 5, testNlist+10)
	sameAll(t, "threads > nlist", got, want)
}

func TestSplitCoversEverythingOnce(t *testing.T) {
	for _, tc := range []struct{ n, parts int }{{10, 3}, {10, 10}, {3, 8}, {223, 12}, {0, 4}} {
		next := 0
		for i := 0; i < tc.parts; i++ {
			start, end := split(tc.n, tc.parts, i)
			if start != next || end < start {
				t.Fatalf("split(%d, %d, %d) = [%d, %d), want start %d",
					tc.n, tc.parts, i, start, end, next)
			}
			next = end
		}
		if next != tc.n {
			t.Fatalf("split(%d, %d, ...) covered %d items, want %d", tc.n, tc.parts, next, tc.n)
		}
	}
}

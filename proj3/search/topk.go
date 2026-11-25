package search

import "container/heap"

type Result struct {
	ID    int
	Score float32
}

type TopK struct {
	k int
	h MinHeap
}

type MinHeap []Result

// Create TopK structure
func NewTopK(k int) *TopK {
	return &TopK{k: k, h: make(MinHeap, 0, k)}
}

// reports whether a ranks below b: lower score loses and on a tie the higher ID loses
func worse(a, b Result) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.ID > b.ID
}

// return length of min heap
func (h MinHeap) Len() int {
	return len(h)
}

// return true if element i < j
func (h MinHeap) Less(i, j int) bool {
	return worse(h[i], h[j])
}

// swaps elements  at i and j in heap
func (h MinHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

// push new results into heap
func (h *MinHeap) Push(x any) {
	*h = append(*h, x.(Result))
}

// swap min element with last, then remove and return last
func (h *MinHeap) Pop() any {
	old := *h
	n := len(old)
	last := old[n-1]
	*h = old[:n-1]
	return last
}

// offers one candidate score but most are rejected with a single comparison
func (t *TopK) Push(id int, score float32) {
	// no room to return results so return
	if t.k <= 0 {
		return
	}
	// create candidate
	r := Result{ID: id, Score: score}
	if t.h.Len() < t.k {
		// heap not full, push score
		heap.Push(&t.h, r)
	} else if worse(t.h[0], r) {
		// kick out weakest score
		t.h[0] = r
		// restore order to re-establish new weakest
		heap.Fix(&t.h, 0)
	}
}

// Results returns the kept results best first, without emptying the TopK
func (t *TopK) Results() []Result {

	// copy top k contents to temp holder
	temp := make(MinHeap, len(t.h))
	copy(temp, t.h)

	// rebuild the slice with descending scores (best first)
	out := make([]Result, len(temp))
	for i := len(out) - 1; i >= 0; i-- {
		// pop from tmp so topK is preserved
		out[i] = heap.Pop(&temp).(Result)
	}
	return out
}

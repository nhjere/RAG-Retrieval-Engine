package search

import (
	"proj3-redesigned/dataset"
)

func BruteForce(passages, queries *dataset.Vectors, k int) [][]Result {

	// build list of id positions to be reused by every query
	ids := make([]int, passages.N)
	for i := range ids {
		ids[i] = i
	}

	// one result list per query
	results := make([][]Result, queries.N)

	// iterate through each query
	for q := 0; q < queries.N; q++ {
		// fresh top-k for this query
		top := NewTopK(k)
		// treat the whole passage set as one block and scan it
		ScanBlock(queries.Row(q), passages.Data, ids, passages.Dim, top)
		// store this query's best k (best first)
		results[q] = top.Results()
	}

	return results
}

package schedule

import (
	"proj3-redesigned/dataset"
	"proj3-redesigned/index"
	"proj3-redesigned/search"
)

// Sequential runs IVF search for every query on one thread
func Sequential(idx *index.IVF, queries *dataset.Vectors, k, nprobe int) [][]search.Result {
	results := make([][]search.Result, queries.N)

	// loop through each query in memory
	for q := 0; q < queries.N; q++ {
		query := queries.Row(q)
		top := search.NewTopK(k)

		// scan only the nprobe closest clusters to the query
		for _, c := range idx.Probe(query, nprobe) {
			vecs, ids := idx.Cluster(c)
			search.ScanBlock(query, vecs, ids, idx.Dim, top)
		}

		results[q] = top.Results()
	}
	return results
}

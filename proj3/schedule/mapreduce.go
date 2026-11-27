package schedule

import (
	"proj3-redesigned/concurrent"
	"proj3-redesigned/dataset"
	"proj3-redesigned/index"
	"proj3-redesigned/search"
	"sync"
	"time"
)

//	Basic Parallel Implementation - Map Reduce

// runs IVF search with threads goroutines
func MapReduce(idx *index.IVF, queries *dataset.Vectors, k, nprobe, threads int) [][]search.Result {
	results, _ := MapReduceTimed(idx, queries, k, nprobe, threads)
	return results
}

// same as MapReduce, also returns each thread's time between the two barriers
func MapReduceTimed(idx *index.IVF, queries *dataset.Vectors, k, nprobe, threads int) ([][]search.Result, []time.Duration) {

	nlist := idx.Centroids.N

	// shared results allocated for thread which will be populate concurrently
	partial := make([][]*search.TopK, threads)
	for t := range partial {
		partial[t] = make([]*search.TopK, queries.N)
	}

	// each query's list of clusters to search
	probes := make([][]int, queries.N)

	// final results after scores merge
	results := make([][]search.Result, queries.N)

	// each thread's build + map time
	mapTime := make([]time.Duration, threads)

	// define barriers (2) - one after assignments, one after scanning
	bar := concurrent.NewBarrier(threads)

	var wg sync.WaitGroup
	for t := 0; t < threads; t++ {
		wg.Add(1)
		go func(t int) {
			defer wg.Done()
			qStart, qEnd := split(queries.N, threads, t)
			cStart, cEnd := split(nlist, threads, t)

			// route: find each of my queries' closest clusters
			for q := qStart; q < qEnd; q++ {
				probes[q] = idx.Probe(queries.Row(q), nprobe)
			}

			// r1: every thread waits until all goRoutine's have probed their queries
			bar.Wait()
			mapStart := time.Now()

			// build: which queries probed each of my clusters
			byCluster := make([][]int, nlist)
			for q, list := range probes {
				for _, c := range list {
					if c >= cStart && c < cEnd {
						byCluster[c] = append(byCluster[c], q)
					}
				}
			}

			// map: scan my clusters into my row of partial
			for c := cStart; c < cEnd; c++ {
				vecs, ids := idx.Cluster(c)
				for _, q := range byCluster[c] {
					if partial[t][q] == nil {
						partial[t][q] = search.NewTopK(k)
					}
					search.ScanBlock(queries.Row(q), vecs, ids, idx.Dim, partial[t][q])
				}
			}
			mapTime[t] = time.Since(mapStart)

			// r2: every thread waits until all goRoutine's have scanned theur queries
			bar.Wait()

			// reduce: merge every thread's results for my queries
			for q := qStart; q < qEnd; q++ {
				top := search.NewTopK(k)
				for u := 0; u < threads; u++ {
					if p := partial[u][q]; p != nil {
						for _, r := range p.Results() {
							top.Push(r.ID, r.Score)
						}
					}
				}
				results[q] = top.Results()
			}

		}(t)
	}

	wg.Wait()
	return results, mapTime

}

// returns the range [start, end) of part i when n items are divided into parts pieces
func split(num, parts, i int) (start, end int) {
	size := num / parts
	start = i * size
	end = start + size
	if i == parts-1 {
		end = num
	}
	return start, end
}

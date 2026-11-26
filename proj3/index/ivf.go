package index

import (
	"proj3-redesigned/dataset"
	"proj3-redesigned/search"
)

type IVF struct {
	Centroids *dataset.Vectors // nlist centroids
	Vecs      []float32        // every passage vector in a contiguous array
	IDs       []int            // IDs[row] = original passage ID of that row in Vecs
	Offsets   []int            // cluster c is rows Offsets[c] up to Offsets[c+1]
	Dim       int
}

// trains centroids, assigns every passage, and groups passages by cluster
func Build(passages *dataset.Vectors, nlist, iters, sampleSize int, seed int64) *IVF {
	centroids := Train(passages, nlist, iters, sampleSize, seed)
	assign := Assign(passages, centroids)
	dim := passages.Dim

	// count cluster sizes
	counts := make([]int, nlist)
	for _, c := range assign {
		counts[c]++
	}

	// turn counts into starting rows
	offsets := make([]int, nlist+1)
	for c := 0; c < nlist; c++ {
		offsets[c+1] = offsets[c] + counts[c]
	}

	// copy each passage into its cluster's next free row
	vecs := make([]float32, passages.N*dim)
	ids := make([]int, passages.N)
	next := make([]int, nlist)
	copy(next, offsets[:nlist])
	for p, c := range assign {
		row := next[c]
		copy(vecs[row*dim:(row+1)*dim], passages.Row(p))
		ids[row] = p
		next[c]++
	}

	return &IVF{Centroids: centroids, Vecs: vecs, IDs: ids, Offsets: offsets, Dim: dim}
}

// get one cluster vector and id arrays
func (idx *IVF) Cluster(c int) (vecs []float32, ids []int) {
	start := idx.Offsets[c]
	end := idx.Offsets[c+1]
	cluster := idx.Vecs[start*idx.Dim : end*idx.Dim]
	clusterIds := idx.IDs[start:end]
	return cluster, clusterIds
}

// find a queries best clusters
func (idx *IVF) Probe(query []float32, nprobe int) []int {

	// score every centroid keeping the best nprobe
	top := search.NewTopK(nprobe)
	for i := 0; i < idx.Centroids.N; i++ {
		top.Push(i, search.Dot(query, idx.Centroids.Row(i)))
	}

	// keep just the cluster numbers (return best first)
	results := top.Results()
	clusters := make([]int, len(results))
	for i, r := range results {
		clusters[i] = r.ID
	}
	return clusters
}

// returns the smallest, largest, and average cluster size which informs variance for work stealing benefit
func (idx *IVF) Stats() (minSize, maxSize int, mean float64) {
	nlist := idx.Centroids.N
	if nlist <= 0 {
		return 0, 0, 0
	}

	// start with cluster 0 as both smallest and largest
	minSize = idx.Offsets[1] - idx.Offsets[0]
	maxSize = minSize
	for c := 1; c < nlist; c++ {
		size := idx.Offsets[c+1] - idx.Offsets[c]
		if size < minSize {
			minSize = size
		}
		if size > maxSize {
			maxSize = size
		}
	}

	// the last offset is the total passage count
	mean = float64(idx.Offsets[nlist]) / float64(nlist)
	return minSize, maxSize, mean
}

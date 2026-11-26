package index

import (
	"math"
	"math/rand"
	"proj3-redesigned/dataset"
	"proj3-redesigned/search"
)

// index of the centroids with highest dot product
func nearest(v []float32, centroids *dataset.Vectors) int {
	best := 0
	bestScore := search.Dot(v, centroids.Row(0))
	for c := 0; c < centroids.N; c++ {
		score := search.Dot(v, centroids.Row(c))
		if score > bestScore {
			best = c
			bestScore = score
		}
	}
	return best
}

// returns the nearest centroid index for every vector
func Assign(vecs, centroids *dataset.Vectors) []int {
	assignments := make([]int, vecs.N)
	for i := 0; i < vecs.N; i++ {
		assignments[i] = nearest(vecs.Row(i), centroids)
	}
	return assignments
}

// scale to length 1
func normalize(v []float32) {
	var sum float32
	for i := range v {
		sum += v[i] * v[i]
	}
	length := float32(math.Sqrt(float64(sum)))
	if length == 0 {
		return
	}
	for j := range v {
		v[j] = v[j] / length
	}
}

// random subset of n vectors
func sample(vecs *dataset.Vectors, n int, rng *rand.Rand) *dataset.Vectors {
	// nothing to sample
	if n >= vecs.N {
		return vecs
	}
	// random order of all indices
	order := rng.Perm(vecs.N)
	data := make([]float32, n*vecs.Dim)
	for i := 0; i < vecs.N; i++ {
		copy(data[i*vecs.Dim:(i+1)*vecs.Dim], vecs.Row(order[i]))
	}
	return &dataset.Vectors{Data: data, N: n, Dim: vecs.Dim}
}

// average each cluster's members into its centroid
func update(vecs *dataset.Vectors, assign []int, centroids *dataset.Vectors, rng *rand.Rand) {
	dim := vecs.Dim
	sums := make([]float32, centroids.N*dim)
	counts := make([]int, centroids.N)

	// add each vector to its cluster's sum
	for i := 0; i < vecs.N; i++ {
		c := assign[i]
		sum := sums[c*dim : (c+1)*dim]
		for d, x := range vecs.Row(i) {
			sum[d] += x
		}
		counts[c]++
	}

	for c := 0; c < centroids.N; c++ {
		row := centroids.Row(c)
		if counts[c] == 0 {
			// empty cluster means restart at a random vector
			copy(row, vecs.Row(rng.Intn(vecs.N)))
			continue
		}
		// average of members, then back to length 1
		sum := sums[c*dim : (c+1)*dim]
		n := float32(counts[c])
		for d := range row {
			row[d] = sum[d] / n
		}
		normalize(row)
	}

}

// Train learns nlist centroids from a random sample of vecs
func Train(vecs *dataset.Vectors, nlist, iters, sampleSize int, seed int64) *dataset.Vectors {
	// fixed seed results in the same index every run
	rng := rand.New(rand.NewSource(seed))
	s := sample(vecs, sampleSize, rng)
	// starting centroids nlist random distinct vectors
	centroids := sample(s, nlist, rng)
	// repeatedly update and reassign centroids until they settle in vector space
	for i := 0; i < iters; i++ {
		// assign every sample vector to a cluster
		assign := Assign(s, centroids)
		// move each centroid to its cluster's center
		update(s, assign, centroids, rng)
	}

	return centroids
}

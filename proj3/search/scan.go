package search

// Dot product of two vectors
func Dot(a, b []float32) float32 {
	// compute element-wise dot product
	sum := float32(0)
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// scores one query against a group of passage vectors and keeps the best matches
func ScanBlock(query []float32, vecs []float32, ids []int, dim int, top *TopK) {
	// iterate through each vector id
	for j := 0; j < len(ids); j++ {
		// single out vector from block
		v := vecs[dim*j : dim*(j+1)]
		// calculate score
		score := Dot(query, v)
		// push to topk struct
		top.Push(ids[j], score)
	}
}

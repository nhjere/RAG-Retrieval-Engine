package index

import (
	"math"
	"math/rand"
	"testing"

	"proj3-redesigned/dataset"
)

// randomVectors returns n random vectors of length dim, each scaled to length 1
func randomVectors(r *rand.Rand, n, dim int) *dataset.Vectors {
	v := &dataset.Vectors{Data: make([]float32, n*dim), N: n, Dim: dim}
	for i := 0; i < n; i++ {
		row := v.Row(i)
		for j := range row {
			row[j] = float32(r.NormFloat64())
		}
		normalize(row)
	}
	return v
}

// threeGroups builds 3 tight, well-separated groups of points, one around
// each of the first three axes, and returns the vectors and each point's group
func threeGroups(r *rand.Rand, perGroup, dim int) (*dataset.Vectors, []int) {
	n := 3 * perGroup
	v := &dataset.Vectors{Data: make([]float32, n*dim), N: n, Dim: dim}
	groups := make([]int, n)
	for i := 0; i < n; i++ {
		g := i / perGroup
		groups[i] = g
		row := v.Row(i)
		for j := range row {
			row[j] = float32(r.NormFloat64() * 0.05)
		}
		row[g] += 1
		normalize(row)
	}
	return v, groups
}

func TestTrainFindsObviousClusters(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	vecs, groups := threeGroups(r, 50, 8)

	centroids := Train(vecs, 3, 10, vecs.N, 42)
	assign := Assign(vecs, centroids)

	// every point in a group must share one label, and each group gets its own
	labelOf := map[int]int{}
	used := map[int]bool{}
	for i, g := range groups {
		label, seen := labelOf[g]
		if !seen {
			if used[assign[i]] {
				t.Fatalf("two groups were merged into cluster %d", assign[i])
			}
			labelOf[g] = assign[i]
			used[assign[i]] = true
			continue
		}
		if assign[i] != label {
			t.Fatalf("point %d in group %d got cluster %d, want %d", i, g, assign[i], label)
		}
	}
}

func TestTrainIsDeterministic(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	vecs := randomVectors(r, 500, 16)

	a := Train(vecs, 10, 5, 300, 42)
	b := Train(vecs, 10, 5, 300, 42)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("same seed gave different centroids at value %d", i)
		}
	}
}

func TestCentroidsHaveLengthOne(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	vecs := randomVectors(r, 500, 16)

	centroids := Train(vecs, 10, 5, 300, 42)
	for c := 0; c < centroids.N; c++ {
		var sum float64
		for _, x := range centroids.Row(c) {
			sum += float64(x) * float64(x)
		}
		if math.Abs(math.Sqrt(sum)-1) > 1e-4 {
			t.Errorf("centroid %d has length %v, want 1", c, math.Sqrt(sum))
		}
	}
}

func TestSampleCopies(t *testing.T) {
	r := rand.New(rand.NewSource(4))
	vecs := randomVectors(r, 100, 8)
	original := make([]float32, len(vecs.Data))
	copy(original, vecs.Data)

	s := sample(vecs, 10, rand.New(rand.NewSource(5)))
	for i := range s.Data {
		s.Data[i] = 99
	}
	for i := range vecs.Data {
		if vecs.Data[i] != original[i] {
			t.Fatal("changing the sample changed the original vectors")
		}
	}
}

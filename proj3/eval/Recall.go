package eval

import "proj3-redesigned/search"

// Recall returns the average fraction of each query's true top k
// (from brute force) that the approximate results also found
func Recall(approx, exact [][]search.Result, k int) float64 {
	n := len(approx)
	if len(exact) < n {
		n = len(exact)
	}
	if n == 0 || k <= 0 {
		return 0
	}

	total := 0.0
	for q := 0; q < n; q++ {
		// the true top k for this query
		truth := map[int]bool{}
		for i := 0; i < k && i < len(exact[q]); i++ {
			truth[exact[q][i].ID] = true
		}
		if len(truth) == 0 {
			continue
		}

		// how many of them the approximate search found
		hits := 0
		for i := 0; i < k && i < len(approx[q]); i++ {
			if truth[approx[q][i].ID] {
				hits++
			}
		}
		total += float64(hits) / float64(len(truth))
	}
	return total / float64(n)
}

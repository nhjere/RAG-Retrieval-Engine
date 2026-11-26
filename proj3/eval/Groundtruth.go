package eval

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"

	"proj3-redesigned/search"
)

// SaveGroundTruth writes brute-force results as
// [int32 numQueries][int32 k][numQueries*k int32 passage IDs]
func SaveGroundTruth(path string, results [][]search.Result) error {
	k := 0
	if len(results) > 0 {
		k = len(results[0])
	}

	ids := make([]int32, 0, len(results)*k)
	for q, res := range results {
		if len(res) != k {
			return fmt.Errorf("query %d has %d results, want %d", q, len(res), k)
		}
		for _, r := range res {
			ids = append(ids, int32(r.ID))
		}
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriter(f)
	for _, x := range []interface{}{int32(len(results)), int32(k), ids} {
		if err := binary.Write(w, binary.LittleEndian, x); err != nil {
			f.Close()
			return err
		}
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// LoadGroundTruth reads a file written by SaveGroundTruth.
// Only IDs are stored, so every Score is 0; Recall only compares IDs.
func LoadGroundTruth(path string) ([][]search.Result, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReader(f)

	var nq, k int32
	if err := binary.Read(r, binary.LittleEndian, &nq); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &k); err != nil {
		return nil, err
	}
	if nq < 0 || k < 0 {
		return nil, fmt.Errorf("%s: bad header nq=%d k=%d", path, nq, k)
	}

	ids := make([]int32, int(nq)*int(k))
	if err := binary.Read(r, binary.LittleEndian, ids); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}

	results := make([][]search.Result, nq)
	for q := range results {
		results[q] = make([]search.Result, k)
		for i := range results[q] {
			results[q][i].ID = int(ids[q*int(k)+i])
		}
	}
	return results, nil
}

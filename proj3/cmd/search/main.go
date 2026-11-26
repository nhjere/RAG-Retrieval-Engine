package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"proj3-redesigned/dataset"
	"proj3-redesigned/search"
	"time"
)

func main() {

	// read command line flags
	dataDir := flag.String("data", "data/small", "dataset folder")
	k := flag.Int("k", 10, "results per query")
	nq := flag.Int("nq", 0, "number of queries to run (0 = all)")
	show := flag.Int("show", 3, "queries to print results for")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: search -data DIR [-k K] [-nq N] [-show S]\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// load passage and query vectors
	passages, err := dataset.LoadFbin(filepath.Join(*dataDir, "passages.fbin"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	queries, err := dataset.LoadFbin(filepath.Join(*dataDir, "queries.fbin"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// both must come from the same model
	if passages.Dim != queries.Dim {
		fmt.Fprintf(os.Stderr, "dim mismatch: passages %d, queries %d\n", passages.Dim, queries.Dim)
		os.Exit(1)
	}

	// keep only the first nq queries
	if *nq > 0 && *nq < queries.N {
		queries.Data = queries.Data[:*nq*queries.Dim]
		queries.N = *nq
	}

	// time the search only by mode
	start := time.Now()
	results := search.BruteForce(passages, queries, *k)
	// results := search.sequential(passages, )
	elapsed := time.Since(start)

	fmt.Printf("passages=%d queries=%d k=%d\n", passages.N, queries.N, *k)
	fmt.Printf("search %.2fs\n", elapsed.Seconds())

	if *show > 0 {
		printResults(*dataDir, results, *show)
	}
}

// printResults shows the text of the first few queries and their matches
func printResults(dataDir string, results [][]search.Result, show int) {
	queryTexts, err := loadTexts(filepath.Join(dataDir, "queries.jsonl"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	passageTexts, err := loadTexts(filepath.Join(dataDir, "passages.jsonl"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	for q := 0; q < show && q < len(results); q++ {
		fmt.Printf("\nquery %d: %s\n", q, queryTexts[q])
		for rank, r := range results[q] {
			fmt.Printf("  %d. (%.3f) %s\n", rank+1, r.Score, passageTexts[r.ID])
		}
	}
}

func loadTexts(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var texts []string
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Bytes()
		var text string
		if err := json.Unmarshal(line, &text); err != nil {
			var record map[string]json.RawMessage
			if err := json.Unmarshal(line, &record); err != nil {
				return nil, fmt.Errorf("decode %s: %w", path, err)
			}
			for _, key := range []string{"text", "query", "contents", "passage"} {
				if raw, ok := record[key]; ok && json.Unmarshal(raw, &text) == nil {
					break
				}
			}
		}
		texts = append(texts, text)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return texts, nil
}

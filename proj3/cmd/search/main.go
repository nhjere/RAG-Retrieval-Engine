package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"proj3-redesigned/dataset"
	"proj3-redesigned/eval"
	"proj3-redesigned/index"
	"proj3-redesigned/schedule"
	"proj3-redesigned/search"
)

const usage = `usage: search -data DIR -mode MODE [options]

modes:
  brute   exact search over every passage; saves DIR/groundtruth.bin (run once per dataset)
  seq     IVF search on one thread; prints recall if groundtruth.bin exists

options:
`

func main() {

	// read command line flags
	dataDir := flag.String("data", "data/small", "dataset folder")
	mode := flag.String("mode", "seq", "brute or seq")
	k := flag.Int("k", 10, "results per query")
	nq := flag.Int("nq", 0, "number of queries to run (0 = all)")
	nlist := flag.Int("nlist", 0, "number of clusters (0 = sqrt of passage count)")
	nprobe := flag.Int("nprobe", 10, "clusters searched per query")
	iters := flag.Int("iters", 10, "k-means rounds")
	sampleSize := flag.Int("sample", 0, "k-means training sample (0 = 64 x nlist)")
	seed := flag.Int64("seed", 42, "random seed for k-means")
	show := flag.Int("show", 0, "queries to print results for")
	stats := flag.Bool("stats", false, "print cluster size stats")
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usage)
		flag.PrintDefaults()
	}
	flag.Parse()

	// reject a bad mode before spending time loading data
	if *mode != "brute" && *mode != "seq" {
		fmt.Fprintf(os.Stderr, "unknown mode %q\n\n", *mode)
		flag.Usage()
		os.Exit(2)
	}

	passages, queries := loadVectors(*dataDir, *nq)
	gtPath := filepath.Join(*dataDir, "groundtruth.bin")

	var results [][]search.Result
	switch *mode {
	case "brute":
		start := time.Now()
		results = search.BruteForce(passages, queries, *k)
		fmt.Printf("brute force: passages=%d queries=%d k=%d\n", passages.N, queries.N, *k)
		fmt.Printf("search %.2fs\n", time.Since(start).Seconds())
		if err := eval.SaveGroundTruth(gtPath, results); err != nil {
			fail(err)
		}
		fmt.Printf("saved answer key to %s\n", gtPath)

	case "seq":
		idx := buildIndex(passages, *nlist, *iters, *sampleSize, *seed, *stats)
		start := time.Now()
		results = schedule.Sequential(idx, queries, *k, *nprobe)
		elapsed := time.Since(start)
		fmt.Printf("mode=%s queries=%d k=%d nprobe=%d\n", *mode, queries.N, *k, *nprobe)
		fmt.Printf("search %.3fs\n", elapsed.Seconds())
		printRecall(gtPath, results, *k)
	}

	if *show > 0 {
		printResults(*dataDir, results, *show)
	}
}

// loadVectors reads passages and queries, checks they match, and keeps the first nq queries
func loadVectors(dataDir string, nq int) (*dataset.Vectors, *dataset.Vectors) {
	passages, err := dataset.LoadFbin(filepath.Join(dataDir, "passages.fbin"))
	if err != nil {
		fail(err)
	}
	queries, err := dataset.LoadFbin(filepath.Join(dataDir, "queries.fbin"))
	if err != nil {
		fail(err)
	}
	if passages.Dim != queries.Dim {
		fail(fmt.Errorf("dim mismatch: passages %d, queries %d", passages.Dim, queries.Dim))
	}
	if nq > 0 && nq < queries.N {
		queries.Data = queries.Data[:nq*queries.Dim]
		queries.N = nq
	}
	return passages, queries
}

// buildIndex fills in default settings, builds the IVF index, and reports the build time
func buildIndex(passages *dataset.Vectors, nlist, iters, sampleSize int, seed int64, stats bool) *index.IVF {
	if nlist <= 0 {
		nlist = int(math.Sqrt(float64(passages.N)))
	}
	if sampleSize <= 0 {
		sampleSize = 64 * nlist
	}
	if nlist >= sampleSize || nlist > passages.N {
		fail(fmt.Errorf("nlist %d must be smaller than the sample (%d) and passage count (%d)",
			nlist, sampleSize, passages.N))
	}

	start := time.Now()
	idx := index.Build(passages, nlist, iters, sampleSize, seed)
	fmt.Printf("built index: passages=%d nlist=%d sample=%d iters=%d in %.2fs\n",
		passages.N, nlist, sampleSize, iters, time.Since(start).Seconds())

	if stats {
		minSize, maxSize, mean := idx.Stats()
		fmt.Printf("cluster sizes: min=%d max=%d mean=%.1f (max/mean=%.2f)\n",
			minSize, maxSize, mean, float64(maxSize)/mean)
	}
	return idx
}

// printRecall compares results to the saved answer key, if there is one
func printRecall(gtPath string, results [][]search.Result, k int) {
	exact, err := eval.LoadGroundTruth(gtPath)
	if err != nil {
		fmt.Printf("recall: no answer key (run -mode=brute first)\n")
		return
	}
	if len(exact) < len(results) {
		fmt.Printf("recall: answer key has %d queries, need %d (rerun -mode=brute)\n",
			len(exact), len(results))
		return
	}
	fmt.Printf("recall@%d = %.4f\n", k, eval.Recall(results, exact, k))
}

// printResults shows the text of the first few queries and their matches
func printResults(dataDir string, results [][]search.Result, show int) {
	queryTexts, err := loadTexts(filepath.Join(dataDir, "queries.jsonl"))
	if err != nil {
		fail(err)
	}
	passageTexts, err := loadTexts(filepath.Join(dataDir, "passages.jsonl"))
	if err != nil {
		fail(err)
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
	for scanner.Scan() {
		var record struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &record); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		texts = append(texts, record.Text)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return texts, nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

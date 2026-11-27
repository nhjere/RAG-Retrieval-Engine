package main

import (
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
  brute      exact search over every passage; saves DIR/groundtruth.bin (run once per dataset)
  seq        IVF search on one thread; prints recall if groundtruth.bin exists
  mapreduce  IVF search with -threads goroutines (map-reduce with barriers)

options:
`

func main() {

	// read command line flags
	dataDir := flag.String("data", "data/small", "dataset folder")
	mode := flag.String("mode", "seq", "brute, seq, or mapreduce")
	k := flag.Int("k", 10, "results per query")
	nq := flag.Int("nq", 0, "number of queries to run (0 = all)")
	nlist := flag.Int("nlist", 0, "number of clusters (0 = sqrt of passage count)")
	nprobe := flag.Int("nprobe", 10, "clusters searched per query")
	threads := flag.Int("threads", 4, "goroutines for parallel modes")
	iters := flag.Int("iters", 10, "k-means rounds")
	sampleSize := flag.Int("sample", 0, "k-means training sample (0 = 64 x nlist)")
	seed := flag.Int64("seed", 42, "random seed for k-means")
	show := flag.Int("show", 0, "queries to print results for")
	stats := flag.Bool("stats", false, "print cluster size stats")
	timing := flag.Bool("timing", false, "print per-thread map time (mapreduce)")
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usage)
		flag.PrintDefaults()
	}
	flag.Parse()

	// reject a bad mode before spending time loading data
	if *mode != "brute" && *mode != "seq" && *mode != "mapreduce" {
		fmt.Fprintf(os.Stderr, "unknown mode %q\n\n", *mode)
		flag.Usage()
		os.Exit(2)
	}

	passages, queries := loadVectors(*dataDir, *nq)
	gtPath := filepath.Join(*dataDir, "groundtruth.bin")

	var results [][]search.Result
	var mapTime []time.Duration
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

	case "seq", "mapreduce":
		idx := buildIndex(passages, *nlist, *iters, *sampleSize, *seed, *stats)
		start := time.Now()
		if *mode == "seq" {
			results = schedule.Sequential(idx, queries, *k, *nprobe)
		} else {
			results, mapTime = schedule.MapReduceTimed(idx, queries, *k, *nprobe, *threads)
		}
		elapsed := time.Since(start)
		fmt.Printf("mode=%s threads=%d queries=%d k=%d nprobe=%d\n",
			*mode, threadCount(*mode, *threads), queries.N, *k, *nprobe)
		fmt.Printf("search %.3fs\n", elapsed.Seconds())
		printRecall(gtPath, results, *k)
		if *timing && mapTime != nil {
			printImbalance(mapTime)
		}
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
	queryTexts, err := dataset.LoadTexts(filepath.Join(dataDir, "queries.jsonl"))
	if err != nil {
		fail(err)
	}
	passageTexts, err := dataset.LoadTexts(filepath.Join(dataDir, "passages.jsonl"))
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

// printImbalance summarizes how unevenly the map work was spread across threads
func printImbalance(mapTime []time.Duration) {
	minT, maxT, sum := mapTime[0], mapTime[0], time.Duration(0)
	for _, d := range mapTime {
		if d < minT {
			minT = d
		}
		if d > maxT {
			maxT = d
		}
		sum += d
	}
	avg := sum / time.Duration(len(mapTime))
	fmt.Printf("map time per thread: min %.3fs avg %.3fs max %.3fs (max/avg = %.2f)\n",
		minT.Seconds(), avg.Seconds(), maxT.Seconds(), maxT.Seconds()/avg.Seconds())
}

// threadCount reports how many goroutines a mode actually uses
func threadCount(mode string, threads int) int {
	if mode == "seq" {
		return 1
	}
	return threads
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}

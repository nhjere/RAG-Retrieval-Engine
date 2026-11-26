package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// LoadTexts reads a .jsonl file where line i is {"id": i, "text": ...}
// and returns the texts so that texts[i] belongs to vector i
func LoadTexts(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	// allow long lines; the default limit is 64 KB
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var texts []string
	for scanner.Scan() {
		var rec struct {
			ID   int    `json:"id"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			return nil, fmt.Errorf("%s line %d: %w", path, len(texts)+1, err)
		}
		// ids must match line order or texts would pair with the wrong vectors
		if rec.ID != len(texts) {
			return nil, fmt.Errorf("%s: line %d has id %d", path, len(texts)+1, rec.ID)
		}
		texts = append(texts, rec.Text)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return texts, nil
}

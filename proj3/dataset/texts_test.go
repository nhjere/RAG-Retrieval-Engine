package dataset

import (
	"os"
	"path/filepath"
	"testing"
)

func writeLines(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return path
}

func TestLoadTexts(t *testing.T) {
	path := writeLines(t, `{"id": 0, "doc_id": "a", "text": "first"}
{"id": 1, "doc_id": "b", "text": "second"}
`)
	texts, err := LoadTexts(path)
	if err != nil {
		t.Fatalf("LoadTexts: %v", err)
	}
	if len(texts) != 2 || texts[0] != "first" || texts[1] != "second" {
		t.Fatalf("got %q, want [first second]", texts)
	}
}

func TestLoadTextsRejectsOutOfOrderIDs(t *testing.T) {
	path := writeLines(t, `{"id": 0, "text": "first"}
{"id": 5, "text": "wrong"}
`)
	if _, err := LoadTexts(path); err == nil {
		t.Fatal("expected error for id that doesn't match its line, got nil")
	}
}

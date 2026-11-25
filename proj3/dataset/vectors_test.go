package dataset

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeFbin writes a header and values in the .fbin layout. The header is
// passed separately so tests can write files whose header lies about the size.
func writeFbin(t *testing.T, n, dim int32, values []float32) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.fbin")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer f.Close()
	for _, x := range []interface{}{n, dim, values} {
		if err := binary.Write(f, binary.LittleEndian, x); err != nil {
			t.Fatalf("write: %v", err)
		}
	}
	return path
}

func TestLoadFbinReadsValues(t *testing.T) {
	values := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	path := writeFbin(t, 2, 3, values)

	v, err := LoadFbin(path)
	if err != nil {
		t.Fatalf("LoadFbin: %v", err)
	}
	if v.N != 2 || v.Dim != 3 {
		t.Fatalf("got N=%d Dim=%d, want N=2 Dim=3", v.N, v.Dim)
	}
	for i, want := range values {
		if v.Data[i] != want {
			t.Errorf("Data[%d] = %v, want %v", i, v.Data[i], want)
		}
	}
}

func TestRowReturnsCorrectVector(t *testing.T) {
	v := &Vectors{Data: []float32{1, 2, 3, 4, 5, 6}, N: 2, Dim: 3}
	row := v.Row(1)
	want := []float32{4, 5, 6}
	if len(row) != len(want) {
		t.Fatalf("len(Row(1)) = %d, want %d", len(row), len(want))
	}
	for i := range want {
		if row[i] != want[i] {
			t.Errorf("Row(1)[%d] = %v, want %v", i, row[i], want[i])
		}
	}
}

func TestLoadFbinRejectsWrongSize(t *testing.T) {
	// header promises 2 vectors of 3 values, but only 5 values follow
	path := writeFbin(t, 2, 3, []float32{1, 2, 3, 4, 5})
	if _, err := LoadFbin(path); err == nil {
		t.Fatal("expected error for truncated file, got nil")
	}
}

func TestLoadFbinRejectsBadHeader(t *testing.T) {
	path := writeFbin(t, 2, 0, nil)
	if _, err := LoadFbin(path); err == nil {
		t.Fatal("expected error for dim=0, got nil")
	}
}

func TestLoadFbinMissingFile(t *testing.T) {
	if _, err := LoadFbin(filepath.Join(t.TempDir(), "missing.fbin")); err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

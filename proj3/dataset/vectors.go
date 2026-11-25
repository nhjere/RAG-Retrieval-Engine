package dataset

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

// Vectors stores N vectors of length Dim in one flat slice.
// Vector i is Data[i*Dim : (i+1)*Dim].
type Vectors struct {
	Data []float32
	N    int
	Dim  int
}

// Row returns vector i without copying.
func (v *Vectors) Row(i int) []float32 {
	return v.Data[i*v.Dim : (i+1)*v.Dim]
}

// LoadFbin reads a vector file from disk into one large block of memory
func LoadFbin(path string) (*Vectors, error) {

	// open files
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}

	// close on return
	defer f.Close()

	// create buffer reader
	reader := bufio.NewReader(f)

	// read header and dim of Vectors
	var n, dim int32
	if err := binary.Read(reader, binary.LittleEndian, &n); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, binary.LittleEndian, &dim); err != nil {
		return nil, err
	}

	// check file size
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}

	// catches float64 data or a truncated file.
	want := int64(8) + int64(n)*int64(dim)*4
	if info.Size() != want {
		return nil, fmt.Errorf("%s: size %d, want %d (float64 data or truncated?)",
			path, info.Size(), want)
	}

	// bad Header so return
	if n < 0 || dim <= 0 {
		return nil, fmt.Errorf("%s: bad header n=%d dim=%d", path, n, dim)
	}

	// create slice of data of size n x dim
	data := make([]float32, int(n)*int(dim))

	// read all values from reader into data
	if err := binary.Read(reader, binary.LittleEndian, data); err != nil {
		return nil, err
	}

	return &Vectors{Data: data, N: int(n), Dim: int(dim)}, nil

}

package concurrent

import (
	"sync"
)

// define new barrier struct
type Barrier struct {
	mu    sync.Mutex
	cond  *sync.Cond
	n     int // goroutines that must arrive
	count int // arrived so far this round
	round int // which round we're on
}

// create new barrier
func NewBarrier(n int) *Barrier {
	if n < 1 {
		panic("barrier needs at least 1 goroutine")
	}
	b := &Barrier{n: n}
	b.cond = sync.NewCond(&b.mu)
	return b
}

// wait until all n threads enter then proceed
func (b *Barrier) Wait() {
	b.mu.Lock()
	defer b.mu.Unlock()

	r := b.round
	b.count++

	// last one to arrive
	if b.count == b.n {
		// reset for the next round
		b.count = 0
		// open the gate
		b.round++
		// wake everyone
		b.cond.Broadcast()
		return
	}

	// sleep until the round changes
	for r == b.round {
		// releases mu while asleep and re-locks on wake
		b.cond.Wait()
	}
}

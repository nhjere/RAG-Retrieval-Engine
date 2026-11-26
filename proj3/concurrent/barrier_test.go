package concurrent

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// finishWithin fails the test if f does not return in time, which is how
// a deadlock shows up instead of hanging the whole test run
func finishWithin(t *testing.T, d time.Duration, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		f()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatal("timed out: possible deadlock")
	}
}

func TestBarrierNobodyPassesEarly(t *testing.T) {
	const n, rounds = 8, 1000
	b := NewBarrier(n)
	var arrived [rounds]atomic.Int32

	finishWithin(t, 10*time.Second, func() {
		var wg sync.WaitGroup
		for g := 0; g < n; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for r := 0; r < rounds; r++ {
					arrived[r].Add(1)
					b.Wait()
					// everyone must have arrived before anyone gets past
					if got := arrived[r].Load(); got != n {
						t.Errorf("round %d: passed with only %d of %d arrived", r, got, n)
						return
					}
				}
			}()
		}
		wg.Wait()
	})
}

func TestBarrierSingleGoroutine(t *testing.T) {
	b := NewBarrier(1)
	finishWithin(t, time.Second, func() {
		for i := 0; i < 5; i++ {
			b.Wait() // with n = 1 the caller is always the last to arrive
		}
	})
}

func TestBarrierTwoRoundsBackToBack(t *testing.T) {
	// the map-reduce pattern: work, wait, work, wait, with uneven work
	const n = 4
	b := NewBarrier(n)
	var phase1, phase2 atomic.Int32

	finishWithin(t, 5*time.Second, func() {
		var wg sync.WaitGroup
		for g := 0; g < n; g++ {
			wg.Add(1)
			go func(g int) {
				defer wg.Done()
				time.Sleep(time.Duration(g) * time.Millisecond)
				phase1.Add(1)
				b.Wait()
				if phase1.Load() != n {
					t.Errorf("goroutine %d left round 1 before all arrived", g)
				}
				time.Sleep(time.Duration(n-g) * time.Millisecond)
				phase2.Add(1)
				b.Wait()
				if phase2.Load() != n {
					t.Errorf("goroutine %d left round 2 before all arrived", g)
				}
			}(g)
		}
		wg.Wait()
	})
}

func TestNewBarrierRejectsZero(t *testing.T) {

	defer func() {

		if recover() == nil {
			t.Fatal("NewBarrier(0) should panic")
		}
	}()
	NewBarrier(0)
}

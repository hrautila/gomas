
package main

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "flag"
    "time"
    "fmt"
)

var N int
var count int
var verbose bool

func init() {
    flag.IntVar(&N, "N", 611, "Matrix order")
    flag.IntVar(&count, "C", 5, "Test count")
    flag.BoolVar(&verbose, "v", false, "Verbosity")
}

var nullData []float64

func flushCache() {
    if len(nullData) == 0 {
        nullData = make([]float64, 1500*1500)
    }
    // write all
    for i, _ := range nullData {
        nullData[i] = 1e-10
    }
    zero := 0.0
    // read all
    for i, _ := range nullData {
        zero += nullData[i]
    }
    // don't allow optimizing out
    if zero < 10 {
        zero = 0.0
    }
}

func gflops(N int, secs float64) float64 {
    return 2.0*float64(int64(N)*int64(N)*int64(N))/secs * 1e-9
}

func main() {
    flag.Parse()

    C := cmat.NewMatrix(N, N)
    A := cmat.NewMatrix(N, N)
    B := cmat.NewMatrix(N, N)
    
    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    B.SetFrom(zeromean)

    cumtime := 0.0
    mintime := 0.0
    maxtime := 0.0
    for i := 0; i < count; i++ {
        flushCache()

        t1 := time.Now()
        // ----------------------------------------------

        blasd.Mult(C, A, B, 1.0, 0.0, gomas.NONE)

        // ----------------------------------------------
        t2 := time.Now()
        tm := t2.Sub(t1)

        if mintime == 0.0 || tm.Seconds() < mintime {
            mintime = tm.Seconds()
        }
        if maxtime == 0.0 || tm.Seconds() > maxtime {
            maxtime = tm.Seconds()
        }
        cumtime += tm.Seconds()
        if verbose {
            fmt.Printf("%3d  %12.4f msec, %9.4f gflops\n",
                i, 1e+3*tm.Seconds(), gflops(N, tm.Seconds()))
        }
    }
    cumtime /= float64(count)
    minflops := gflops(N, maxtime)
    avgflops := gflops(N, cumtime)
    maxflops := gflops(N, mintime)
    fmt.Printf("%5d %9.4f %9.4f %9.4f Gflops\n", N, minflops, avgflops, maxflops)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

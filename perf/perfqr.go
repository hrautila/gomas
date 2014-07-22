
package main

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/lapackd"
    "flag"
    "time"
    "fmt"
)

var N int
var count int
var verbose bool

func init() {
    flag.IntVar(&N, "N", 611, "Matrix cols")
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

func gflops(M, N int, secs float64) float64 {
    EM := float64(M)
    EN := float64(N)
    // from LAPACK dopla.f
    mults := EN*( ( ( 23.0/6.0 ) + EM + EN/2.0 ) + EN*(EM-EN/3.0) )
    adds  := EN*( (5.0/6.0) + EN*( 1.0/2.0 + (EM - EN/3.0) ) )
    return (mults + adds)/secs * 1e-9
}

func main() {
    flag.Parse()

    M := N + N/10

    conf := gomas.CurrentConf()

    A := cmat.NewMatrix(M, N)
    A0 := cmat.NewCopy(A)
    tau := cmat.NewMatrix(N, 1)
    W := lapackd.Workspace(lapackd.QRFactorWork(A, conf))
    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)

    cumtime := 0.0
    mintime := 0.0
    maxtime := 0.0
    for i := 0; i < count; i++ {
        flushCache()

        t1 := time.Now()
        // ----------------------------------------------

        lapackd.QRFactor(A, tau, W, conf)

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
                i, 1e+3*tm.Seconds(), gflops(M, N, tm.Seconds()))
        }
        blasd.Copy(A, A0)
    }
    cumtime /= float64(count)
    minflops := gflops(M, N, maxtime)
    avgflops := gflops(M, N, cumtime)
    maxflops := gflops(M, N, mintime)
    fmt.Printf("%5d %5d %3d %9.4f %9.4f %9.4f Gflops\n", M, N, conf.LB, minflops, avgflops, maxflops)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

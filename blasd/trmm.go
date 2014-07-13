
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package blasd

// #cgo CFLAGS: -O3 -march=native -fomit-frame-pointer -ffast-math -Iinc -I.
// #cgo LDFLAGS: -lm
// #include "inc/interfaces.h"
import  "C"
import "unsafe"
import "github.com/hrautila/cmat"
import "github.com/hrautila/gomas"

func trmm(Bc, Ac *cmat.FloatMatrix, alpha float64, bits, N, S, E int, conf *gomas.Config) error {
    var Am, Bm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Bm.md = (*C.double)(unsafe.Pointer(&Bc.Data()[0]))
    Bm.step = C.int(Bc.Stride())

    C.__d_trmm_blk(
        (*C.mdata_t)(unsafe.Pointer(&Bm)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        C.double(alpha), C.int(bits),
        C.int(N), C.int(S), C.int(E),
        C.int(conf.KB), C.int(conf.NB), C.int(conf.MB))

    return nil
}


/*
 * Triangular matrix multiplication.
 */
func MultTrm(B, A *cmat.FloatMatrix, alpha float64, bits int, confs... *gomas.Config) *gomas.Error {
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }

    if B.Len() == 0 || A.Len() == 0 {
        return nil
    }

    ok := true
    ar, ac := A.Size()
    br, bc := B.Size()
    P := ac
    E := bc
    switch {
    case bits & gomas.RIGHT != 0:
        ok = bc == ar && ar == ac
        E = br
    case bits & gomas.LEFT != 0:
        fallthrough
    default:
        ok = ac == br && ar == ac
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "MultTrm")
    }

    // single threaded
    if conf.NProc == 1 || conf.WB <= 0 || E < conf.WB/2 {
        trmm(B, A, alpha, bits, P, 0, E,  conf)
        return nil
    }
    
    // parallelized
    wait := make(chan int, 4)
    _, nN := blocking(0, E, conf.WB/2)
    nT := 0
    for j := 0; j < nN; j++ {
        jS := blockIndex(j, nN, conf.WB/2, E)
        jL := blockIndex(j+1, nN, conf.WB/2, E)
        task := func(q chan int) {
            trmm(B, A, alpha, bits, P, jS, jL, conf)
            q <- 1
        }
        conf.Sched.Schedule(gomas.NewTask(task, wait))
        nT += 1
    }
    for nT > 0 {
        <- wait
        nT -= 1
    }
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

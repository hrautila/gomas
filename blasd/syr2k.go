
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

func syr2k(Cc, Ac, Bc *cmat.FloatMatrix, alpha, beta float64, bits, P, S, E int, conf *gomas.Config)  {
    var Am, Cm, Bm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Bm.md = (*C.double)(unsafe.Pointer(&Bc.Data()[0]))
    Bm.step = C.int(Bc.Stride())
    Cm.md = (*C.double)(unsafe.Pointer(&Cc.Data()[0]))
    Cm.step = C.int(Cc.Stride())

    C.__d_rank2_blk(
        (*C.mdata_t)(unsafe.Pointer(&Cm)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta), C.int(bits),
        C.int(P), C.int(S), C.int(E),
        C.int(conf.KB), C.int(conf.NB), C.int(conf.MB))
}


func Update2Sym(Cc, A, B *cmat.FloatMatrix, alpha, beta float64, bits int, confs... *gomas.Config)  *gomas.Error {

    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }

    ok := true
    cr, cc := Cc.Size()
    ar, ac := A.Size()
    br, bc := B.Size()

    if cr*cc == 0 {
        return nil
    }
    P := ac
    E := cr
    if bits & gomas.TRANS != 0 && bits & gomas.TRANSA == 0 {
        bits |= gomas.TRANSA
    }
    switch {
    case bits & gomas.TRANSA != 0:
        ok = cr == cc && cr == ac && bc == ac && br == ar
        P = ar
    default:
        ok = cr == cc && cr == ar && br == ar && bc == ac
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "Update2Sym")
    }
    if conf.NProc == 1 || conf.WB <= 0 || E <= conf.WB {
        syr2k(Cc, A, B, alpha, beta, bits, P, 0, E, conf)
        return nil
    }
    // parallelized
    wait := make(chan int, 4)
    _, nN := blocking(0, E, conf.WB)
    nT := 0
    for j := 0; j < nN; j++ {
        jS := blockIndex(j, nN, conf.WB, E)
        jE := blockIndex(j+1, nN, conf.WB, E)
        task := func(q chan int) {
            syr2k(Cc, A, B, alpha, beta, bits, P, jS, jE, conf)
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

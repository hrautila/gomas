
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

func syrk(Cc, Ac *cmat.FloatMatrix, alpha, beta float64, bits, P, S, E int, conf *gomas.Config) error {
    var Am, Cm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Cm.md = (*C.double)(unsafe.Pointer(&Cc.Data()[0]))
    Cm.step = C.int(Cc.Stride())

    C.__d_rank_blk(
        (*C.mdata_t)(unsafe.Pointer(&Cm)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        C.double(alpha), C.double(beta), C.int(bits),
        C.int(P), C.int(S), C.int(E),
        C.int(conf.KB), C.int(conf.NB), C.int(conf.MB))
    return nil
}

/*
 * UpdateSym performs symmetric rank-k update C = beta*C + alpha*A*A.T or
 * C = beta*C + alpha*A.T*A if gomas.TRANS bit is set.
 */
func UpdateSym(c, a *cmat.FloatMatrix, alpha, beta float64, bits int, confs... *gomas.Config) *gomas.Error {

    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }

    ok := true
    cr, cc := c.Size()
    ar, ac := a.Size()
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
        ok = cr == cc && cr == ac 
        P = ar
    default:
        ok = cr == cc && cr == ar 
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "UpdateSym")
    } 
    if conf.NProc == 1 || conf.WB <= 0 || E <= conf.WB {
        syrk(c, a, alpha, beta, bits, P, 0, E, conf)
        return nil
    }

    // parallelized
    var sbits int = 0
    wait := make(chan int, 4)
    nM, nN := blocking(E, E, conf.WB)
    nT := 0
    if bits & gomas.TRANS != 0 {
        sbits |= gomas.TRANSA
    } else {
        sbits |= gomas.TRANSB
    }
    if bits & gomas.LOWER != 0 {
        sbits |= gomas.LOWER
        for j := 0; j < nN; j++ {
            jS := blockIndex(j, nN, conf.WB, E)
            jL := blockIndex(j+1, nN, conf.WB, E)
            // update lower trapezoidal/triangular blocks
            task := func(q chan int) {
                updtrm(c, a, a, alpha, beta, sbits, P, jS, jL, jS, E, conf)
                //syrk(c, a, alpha, beta, bits, P, jS, jL, conf)
                q <- 1
            }
            conf.Sched.Schedule(gomas.NewTask(task, wait))
            nT += 1
        }
    } else {
        sbits |= gomas.UPPER
        for j := 0; j < nM; j++ {
            jS := blockIndex(j, nM, conf.WB, E)
            jL := blockIndex(j+1, nM, conf.WB, E)
            // update upper trapezoidal/triangular blocks
            task := func(q chan int) {
                updtrm(c, a, a, alpha, beta, sbits, P, jS, E, jS, jL, conf)
                //syrk(c, a, alpha, beta, bits, P, jS, jL, conf)
                q <- 1
            }
            conf.Sched.Schedule(gomas.NewTask(task, wait))
            nT += 1
        }
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

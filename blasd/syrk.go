
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
    syrk(c, a, alpha, beta, bits, P, 0, E, conf)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

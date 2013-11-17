
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


func updtrm(Cc, Ac, Bc *cmat.FloatMatrix, alpha, beta float64, bits, P, S, L, R, E int, conf *gomas.Config) error {
    var Am, Cm, Bm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Bm.md = (*C.double)(unsafe.Pointer(&Bc.Data()[0]))
    Bm.step = C.int(Bc.Stride())
    Cm.md = (*C.double)(unsafe.Pointer(&Cc.Data()[0]))
    Cm.step = C.int(Cc.Stride())

    C.__d_update_trm_blk(
        (*C.mdata_t)(unsafe.Pointer(&Cm)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta), C.int(bits),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(conf.KB), C.int(conf.NB), C.int(conf.MB))
    return nil
}


func UpdateTrm(Cc, A, B *cmat.FloatMatrix, alpha, beta float64, bits int, confs... *gomas.Config) *gomas.Error {

    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    ok := true
    cr, cc := Cc.Size()
    ar, ac := A.Size()
    br, bc := B.Size()
    P := ac
    L := cc
    E := cr
    switch bits & (gomas.TRANSA|gomas.TRANSB) {
    case gomas.TRANSA|gomas.TRANSB:
        ok = cr == ac && cc == br && ar == bc
        P = ar
    case gomas.TRANSA:
        ok = cr == ac && cc == bc && ar == br
        P = ar
    case gomas.TRANSB:
        ok = cr == ar && cc == br && ac == bc
    default:
        ok = cr == ar && cc == bc && ac == br
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "UpdateTrm")
    }
    updtrm(Cc, A, B, alpha, beta, bits, P, 0, L, 0, E, conf)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

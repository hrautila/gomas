
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


func trsm(Bc, Ac *cmat.FloatMatrix, alpha float64, bits, N, S, E int, conf *gomas.Config) error {
    var Am, Bm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Bm.md = (*C.double)(unsafe.Pointer(&Bc.Data()[0]))
    Bm.step = C.int(Bc.Stride())

    C.__d_solve_blocked(
        (*C.mdata_t)(unsafe.Pointer(&Bm)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)),
        C.double(alpha), C.int(bits),
        C.int(N), C.int(S), C.int(E),
        C.int(conf.KB), C.int(conf.NB), C.int(conf.MB))

    return nil
}


func SolveTrm(B, A *cmat.FloatMatrix, alpha float64, bits int, confs... *gomas.Config) *gomas.Error {
    
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
    E := bc
    switch  {
    case bits & gomas.RIGHT != 0:
        ok = bc == ar && ar == ac
        E = br
    case bits & gomas.LEFT != 0:
        fallthrough
    default:
        ok = ac == br && ar == ac
        E = bc
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "SolveTrm")
    }
    trsm(B, A, alpha, bits, ac, 0, E, conf)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

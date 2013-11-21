
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


func plus(Ac, Bc *cmat.FloatMatrix, alpha, beta float64, bits, S, L, R, E int)  {
    var Am, Bm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Bm.md = (*C.double)(unsafe.Pointer(&Bc.Data()[0]))
    Bm.step = C.int(Bc.Stride())

    C.__d_scale_plus(
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta), C.int(bits),
        C.int(S), C.int(L), C.int(R), C.int(E))
}

/*
 * Compute A := alpha*A + beta*op(B) 
 */
func Plus(A, B *cmat.FloatMatrix, alpha, beta float64, bits int, confs... *gomas.Config) *gomas.Error {
    ok := true
    if A.Len() == 0 || B.Len() == 0 {
        return nil
    }
    ar, ac := A.Size()
    br, bc := B.Size()
    L := ac
    E := ar
    switch bits & (gomas.TRANSB|gomas.TRANS) {
    case gomas.TRANSB, gomas.TRANS:
        ok = ac == br && ar == bc
    default:
        ok = ar == br && ac == bc
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "Plus")
    }
    plus(A, B, alpha, beta, bits, 0, L, 0, E)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

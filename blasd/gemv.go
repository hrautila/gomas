
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

func gemv(Y, A, X *cmat.FloatMatrix, alpha, beta float64, bits, S, L, R, E int)  {
    var Am C.mdata_t
    var Xm, Ym C.mvec_t
    
    xr, _ := X.Size()
    yr, _ := Y.Size()
    Am.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    Am.step = C.int(A.Stride())
    Xm.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    Ym.md = (*C.double)(unsafe.Pointer(&Y.Data()[0]))
    Ym.inc = C.int(1)
    Xm.inc = C.int(1)

    // if row vectors, change increment
    if xr == 1 {
        Xm.inc = C.int(X.Stride())
    }
    if yr == 1 {
        Ym.inc = C.int(Y.Stride())
    }

    C.__d_gemv_unb(
        (*C.mvec_t)(unsafe.Pointer(&Ym)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        (*C.mvec_t)(unsafe.Pointer(&Xm)),
        C.double(alpha),
        /*C.double(beta),*/
        C.int(bits),
        C.int(S), C.int(L), C.int(R), C.int(E))
}


func MVMult(Y, A, X *cmat.FloatMatrix, alpha, beta float64, bits int, confs... *gomas.Config) *gomas.Error {
    ok := true
    yr, yc := Y.Size()
    ar, ac := A.Size()
    xr, xc := X.Size()

    if ar*ac == 0 {
        return nil
    }
    if yr != 1 && yc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "MVMult")
    }
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "MVMult")
    }
    nx := X.Len()
    ny := Y.Len()

    if bits & gomas.TRANSA != 0 {
        bits |= gomas.TRANS
    }
    if bits & gomas.TRANS != 0 {
        ok = ny == ac && nx == ar
    } else {
        ok = ny == ar && nx == ac
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "MVMult")
    }
    if beta != 1.0 {
        vscal(Y, beta, ny)
    }
    gemv(Y, A, X, alpha, beta, bits, 0, nx, 0, ny)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

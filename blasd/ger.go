
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

func ger(A, X, Y *cmat.FloatMatrix, alpha float64, N, M int) error {
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

    C.__d_update_ger_unb(
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        (*C.mvec_t)(unsafe.Pointer(&Xm)),
        (*C.mvec_t)(unsafe.Pointer(&Ym)), 
        C.double(alpha),  C.int(N), C.int(M))
    return nil
}


func MVUpdate(A, X, Y *cmat.FloatMatrix, alpha float64, confs... *gomas.Config) *gomas.Error {
    ar, ac := A.Size()
    yr, yc := Y.Size()
    xr, xc := X.Size()

    if ar*ac == 0 {
        return nil
    }
    if yr != 1 && yc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "MVUpdate")
    }
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "MVUpdate")
    }
    nx := X.Len()
    ny := Y.Len()
    if  ac != ny || ar != nx {
        return gomas.NewError(gomas.ESIZE, "MVUpdate")
    }
    ger(A, X, Y, alpha, ny, nx)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

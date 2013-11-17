
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

func vswap(X, Y *cmat.FloatMatrix, N int) {
    var x, y C.mvec_t

    xr, _ := X.Size()
    x.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    x.inc = C.int(1)
    if xr == 1 {
        x.inc = C.int(X.Stride())
    }
    yr, _ := Y.Size()
    y.md = (*C.double)(unsafe.Pointer(&Y.Data()[0]))
    y.inc = C.int(1)
    if yr == 1 {
        y.inc = C.int(Y.Stride())
    }
    C.__d_vec_swap(
        (*C.mvec_t)(unsafe.Pointer(&x)),
        (*C.mvec_t)(unsafe.Pointer(&y)),  C.int(N))
    return
}


func Swap(X, Y *cmat.FloatMatrix, confs ...*gomas.Config) *gomas.Error {
    xr, xc := X.Size()
    yr, yc := Y.Size()
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "Swap")
    }
    if yr != 1 && yc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "Swap")
    }
    if X.Len() != Y.Len() {
        return gomas.NewError(gomas.ESIZE, "Swap")
    }
    vswap(X, Y, X.Len())
    return nil
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

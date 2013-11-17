
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

func dot(X, Y *cmat.FloatMatrix, N int) float64 {
    var x, y C.mvec_t
    var dc C.double

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
    dc = C.__d_vec_dot_recursive(
        (*C.mvec_t)(unsafe.Pointer(&x)),
        (*C.mvec_t)(unsafe.Pointer(&y)),  C.int(N))
    return float64(dc)
}

func Dot(X, Y *cmat.FloatMatrix, confs... *gomas.Config) float64 {
    if X.Len() == 0 || Y.Len() == 0 {
        return 0.0
    }
    xr, xc := X.Size()
    if xr != 1 && xc != 1 {
        return 0.0
    }
    yr, yc := Y.Size()
    if yr != 1 && yc != 1 {
        return 0.0
    }
    nx := X.Len()
    if nx != Y.Len() {
        return 0.0
    }
    return dot(X, Y, nx)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

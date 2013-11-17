
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

func axpby(Y, X *cmat.FloatMatrix, alpha, beta float64, N int)  {
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
    if beta == 1.0 {
        C.__d_vec_axpy(
            (*C.mvec_t)(unsafe.Pointer(&y)),
            (*C.mvec_t)(unsafe.Pointer(&x)),
            C.double(alpha), C.int(N))
    } else {
        C.__d_vec_axpby(
            (*C.mvec_t)(unsafe.Pointer(&y)),
            (*C.mvec_t)(unsafe.Pointer(&x)),
            C.double(alpha), C.double(beta), C.int(N))
    }
    return
}

func Axpy(Y, X *cmat.FloatMatrix, alpha float64, confs ...*gomas.Config) *gomas.Error {
    if X.Len() == 0 || Y.Len() == 0 {
        return nil
    }
    xr, xc := X.Size()
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "Axpy")
    }
    yr, yc := Y.Size()
    if yr != 1 && yc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "Axpy")
    }
    nx := X.Len()
    if nx != Y.Len() {
        return gomas.NewError(gomas.ESIZE, "Axpy")
    }
    axpby(Y, X, alpha, 1.0, nx)
    return nil
}



func Axpby(Y, X *cmat.FloatMatrix, alpha, beta float64, confs ...*gomas.Config) *gomas.Error {
    if X.Len() == 0 || Y.Len() == 0 {
        return nil
    }
    xr, xc := X.Size()
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "Axpby")
    }
    yr, yc := Y.Size()
    if yr != 1 && yc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "Axpby")
    }
    nx := X.Len()
    if nx != Y.Len() {
        return gomas.NewError(gomas.ESIZE, "Axpby")
    }
    axpby(Y, X, alpha, beta, nx)
    return nil
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

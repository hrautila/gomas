
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

func iamax(X *cmat.FloatMatrix, N int) int {
    var x C.mvec_t
    var ix C.int

    xr, _ := X.Size()
    x.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    x.inc = C.int(1)
    if xr == 1 {
        x.inc = C.int(X.Stride())
    }
    ix = C.__d_vec_iamax(
        (*C.mvec_t)(unsafe.Pointer(&x)), C.int(N))
    return int(ix)
}


func IAmax(X *cmat.FloatMatrix, confs ...*gomas.Config) int {
    if X.Len() == 0 {
        return -1
    }
    xr, xc := X.Size()
    if xr != 1 && xc != 1 {
        return -1
    }
    return iamax(X, X.Len())
}

func Amax(X *cmat.FloatMatrix, confs ...*gomas.Config) float64 {
    switch (X.Len()) {
    case 0:
        return 0.0;
    case 1:
        return X.Get(0, 0);
    }
    ix := IAmax(X, confs...)
    if ix == -1 {
        return 0.0
    }
    return X.Get(ix, 0)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


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

func trmv(X, A *cmat.FloatMatrix, alpha float64, bits, N int) error {
    var Am C.mdata_t
    var Xm C.mvec_t
    
    xr, _ := X.Size()
    Am.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    Am.step = C.int(A.Stride())
    Xm.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    Xm.inc = C.int(1)
    // if row vectors, change increment
    if xr == 1 {
        Xm.inc = C.int(X.Stride())
    }
    C.__d_trmv_unb(
        (*C.mvec_t)(unsafe.Pointer(&Xm)),
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        C.double(alpha), C.int(bits), C.int(N))
    return nil
}


func MVMultTrm(X, A *cmat.FloatMatrix, alpha float64, bits int, confs... *gomas.Config) *gomas.Error {

    ar, ac := A.Size()
    xr, xc := X.Size()

    if ar*ac == 0 {
        return nil
    }
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "MVMultTrm")
    }
    nx := X.Len()
    if  ac != nx || ar != ac {
        return gomas.NewError(gomas.ESIZE, "MVMultTrm")
    }
    if ar == 1 {
        vscal(X, alpha*A.Get(0, 0), nx)
    } else {
        trmv(X, A, alpha, bits, nx)
    }
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

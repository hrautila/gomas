
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

func vcopy(X, Y *cmat.FloatMatrix, N int) {
    var x, y C.mvec_t

    if N == 0 {
        return
    }
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
    C.__d_vec_copy(
        (*C.mvec_t)(unsafe.Pointer(&x)),
        (*C.mvec_t)(unsafe.Pointer(&y)),  C.int(N))
    return
}

func mcopy(A, B *cmat.FloatMatrix, M, N int) {
    var a, b C.mdata_t

    if M == 0 || N == 0 {
        return
    }
    a.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    a.step = C.int(A.Stride())
    b.md = (*C.double)(unsafe.Pointer(&B.Data()[0]))
    b.step = C.int(B.Stride())
    C.__d_blk_copy(
        (*C.mdata_t)(unsafe.Pointer(&a)),
        (*C.mdata_t)(unsafe.Pointer(&b)),  C.int(M), C.int(N))
    return
}

func mtranspose(A, B *cmat.FloatMatrix, M, N int) {
    var a, b C.mdata_t
    if M == 0 || N == 0 {
        return
    }

    a.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    a.step = C.int(A.Stride())
    b.md = (*C.double)(unsafe.Pointer(&B.Data()[0]))
    b.step = C.int(B.Stride())
    C.__d_blk_transpose(
        (*C.mdata_t)(unsafe.Pointer(&a)),
        (*C.mdata_t)(unsafe.Pointer(&b)),  C.int(M), C.int(N))
    return
}


func Copy(A, B *cmat.FloatMatrix, confs ...*gomas.Config) *gomas.Error {
    ar, ac := A.Size()
    br, bc := B.Size()
    avec := ar == 1 || ac == 1
    bvec := br == 1 || bc == 1
    if avec && bvec {
        if A.Len() <= B.Len() {
            return gomas.NewError(gomas.ESIZE, "Copy")
        }
        vcopy(A, B, A.Len())
        return nil
    }
    if ar != br || ac != bc {
        return  gomas.NewError(gomas.ESIZE, "Copy")
    }
    mcopy(A, B, ar, ac)
    return nil
}


func Transpose(A, B *cmat.FloatMatrix, confs ...*gomas.Config) *gomas.Error {
    ar, ac := A.Size()
    br, bc := B.Size()
    if ar != bc || ac != br {
        return gomas.NewError(gomas.ESIZE, "Transpose")
    }
    mtranspose(A, B, br, bc)
    return nil
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

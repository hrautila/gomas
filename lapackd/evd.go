
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    //"math"
)


func EigenSym(D, A, W *cmat.FloatMatrix, bits int, confs... *gomas.Config) (err *gomas.Error) {

    var sD, sE, E, tau, Wred cmat.FloatMatrix
    var vv *cmat.FloatMatrix

    err = nil
    vv = nil
    conf := gomas.CurrentConf(confs...)
    
    if m(A) != n(A) || D.Len() != m(A) {
        err = gomas.NewError(gomas.ESIZE, "EigenSym")
        return
    }
    if bits & gomas.WANTV != 0 && W.Len() < 3*n(A) {
        err = gomas.NewError(gomas.EWORK, "EigenSym")
        return
    }

    if bits & (gomas.LOWER|gomas.UPPER) == 0 {
        bits = bits | gomas.LOWER
    }
    ioff := 1
    if bits & gomas.LOWER != 0 {
        ioff = -1
    }
    E.SetBuf(n(A)-1, 1, n(A)-1, W.Data())
    tau.SetBuf(n(A), 1, n(A), W.Data()[n(A)-1:])
    wrl := W.Len() - 2*n(A) - 1
    Wred.SetBuf(wrl, 1, wrl, W.Data()[2*n(A)-1:])

    // reduce to tridiagonal
    if err = TRDReduce(A, &tau, &Wred, bits, conf); err != nil {
        err.Update("EigenSym")
        return
    }
    sD.Diag(A)
    sE.Diag(A, ioff)
    blasd.Copy(D, &sD)
    blasd.Copy(&E, &sE)

    if bits & gomas.WANTV != 0 {
        if err = TRDBuild(A, &tau, &Wred, n(A), bits, conf); err != nil {
            err.Update("EigenSym")
            return
        }
        vv = A
    }

    // resize workspace
    wrl = W.Len() - n(A) - 1
    Wred.SetBuf(wrl, 1, wrl, W.Data()[n(A)-1:])

    if err = TRDEigen(D, &E, vv, &Wred, bits, conf); err != nil {
        err.Update("EigenSym")
        return
    }
    return
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

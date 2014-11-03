
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package test

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/lapackd"
    "testing"
)

func test1(N int, beta float64, t *testing.T) {
    var sI cmat.FloatMatrix

    if N & 0x1 != 0 {
        N = N + 1
    }
    D := cmat.NewMatrix(N, 1)
    Z := cmat.NewMatrix(N, 1)
    Y := cmat.NewMatrix(N, 1)
    V := cmat.NewMatrix(N, 1)
    Q := cmat.NewMatrix(N, N)
    I := cmat.NewMatrix(N, N)

    D.SetAt(0, 1.0)
    Z.SetAt(0, 2.0)
    for i := 1; i < N-1; i++ {
        if i < N/2 {
            D.SetAt(i, 2.0 - float64(N/2-i)*beta)
        } else {
            D.SetAt(i, 2.0 + float64(i+1-N/2)*beta)
        }
        Z.SetAt(i, beta)
    }
    D.SetAt(N-1, 10.0/3.0)
    Z.SetAt(N-1, 2.0)
    w := blasd.Nrm2(Z)
    blasd.InvScale(Z, w)
    rho := 1.0/(w*w)

    lapackd.TRDSecularSolveAll(Y, V, Q, D, Z, rho)
    lapackd.TRDSecularEigen(Q, V, nil)
    blasd.Mult(I, Q, Q, 1.0, 0.0, gomas.TRANSA)
    sI.Diag(I)
    sI.Add(-1.0)
    nrm := lapackd.NormP(I, lapackd.NORM_ONE)
    t.Logf("N=%d, beta=%e ||I - Q.T*Q||_1: %e\n", N, beta, nrm)
}

func TestSecularSolve(t *testing.T) {
    N := 314
    beta := 1e-3
    test1(N, beta, t)
    beta = 1e-6
    test1(N, beta, t)
    beta = 1e-9
    test1(N, beta, t)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

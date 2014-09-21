
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package main

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/lapackd"
    "testing"
    "math"
)


func asRow(d, s *cmat.FloatMatrix) *cmat.FloatMatrix {
    d.SetBuf(1, s.Len(), 1, s.Data())
    return d
}

func setDiagonals(A *cmat.FloatMatrix, kind int) string {
    var sD, sEu, sEl cmat.FloatMatrix
    var desc string
    
    switch kind {
    case 2:
        desc = "middle"
    case 1:
        desc = "top   "
    default:
        desc = "bottom"
    }

    sD.Diag(A, 0)
    sEu.Diag(A, 1)
    sEl.Diag(A, -1)
    N := sD.Len()
    for k := 0; k < N; k++ {
        if k < N-1 {
            sEu.SetAt(k, 1.0)
            sEl.SetAt(k, 1.0)
        }
        switch kind {
        case 2:  // midheavy
            if k < N/2 {
                sD.SetAt(k, float64(k+1))
            } else {
                sD.SetAt(k, float64(N-k))
            }
            break
        case 1:   // top heavy
            sD.SetAt(N-1-k, float64(k+1))
            break
        default:  // bottom heavy
            sD.SetAt(k, float64(k+1))
            break
        }
    }
    return desc
}

// d = |d| - |s|
func absMinus(d, s *cmat.FloatMatrix) *cmat.FloatMatrix {
    for k := 0; k < d.Len(); k++ {
        tmp := math.Abs(d.GetAt(k))
        d.SetAt(k, math.Abs(s.GetAt(k))-tmp)
    }
    return d
}

func test_trdevd(N, flags, kind int, verbose bool, t *testing.T) {
    var At, sD, sE, tmp cmat.FloatMatrix

    A0 := cmat.NewMatrix(N, N)
    desc := setDiagonals(A0, kind)
    At.SubMatrix(A0, 0, 0, N, N)
    sD.Diag(A0, 0)
    sE.Diag(A0, 1)
    D := cmat.NewCopy(&sD)
    E := cmat.NewCopy(&sE)

    V := cmat.NewMatrix(N, N)
    sD.Diag(V, 0)
    sD.Add(1.0)

    W := cmat.NewMatrix(4*N, 1)
    C := cmat.NewMatrix(N, N)

    if verbose && N < 10 {
        t.Logf("A0:\n%v\n", A0.ToString("%6.3f"))
        t.Logf("V.pre:\n%v\n", V.ToString("%6.3f"))
    }
    lapackd.TRDEigen(D, E, V, W, flags|gomas.WANTV)
    for k := 0; k < N-1; k++ {
        if E.GetAt(k) != 0.0 {
            t.Logf("E[%d] != 0.0 (%e)\n", k, E.GetAt(k))
        }
    }

    blasd.Mult(C, V, V, 1.0, 0.0, gomas.TRANSB)
    sD.Diag(C)
    sD.Add(-1.0)
    nrmv := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Mult(C, V, A0, 1.0, 0.0, gomas.TRANSA)
    blasd.Mult(&At, C, V, 1.0, 0.0, gomas.NONE)
    if verbose && N < 10 {
        t.Logf("D:\n%v\n", asRow(&tmp, D).ToString("%6.3f"))
        t.Logf("V:\n%v\n", V.ToString("%6.3f"))
        t.Logf("V.T*A*V\n%v\n", At.ToString("%6.3f"))
    }
    sD.Diag(&At)
    blasd.Axpy(&sD, D, -1.0)
    nrma := lapackd.NormP(&At, lapackd.NORM_ONE)
    
    t.Logf("N=%d [%s] ||V.T*A*V - eigen(A)||_1: %e\n", N, desc, nrma)
    t.Logf("  ||I - V.T*V||_1: %e\n", nrmv)
}

func TestBottomHeavy(t *testing.T) {
    N := 211
    test_trdevd(N, gomas.UPPER, 0, true, t)
}

func TestTopHeavy(t *testing.T) {
    N := 211
    test_trdevd(N, gomas.UPPER, 1, true, t)
}

func TestMiddleHeavy(t *testing.T) {
    N := 211
    test_trdevd(N, gomas.UPPER, 2, true, t)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

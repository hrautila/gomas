
// Copyright (c) Harri Rautila, 2013

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
    "math"
)

func setDiagonals(A *cmat.FloatMatrix, offdiag, kind int) string {
    var sD, sE cmat.FloatMatrix
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
    sE.Diag(A, offdiag)
    N := sD.Len()
    for k := 0; k < N; k++ {
        if k < N-1 {
            sE.SetAt(k, 1.0)
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

func asRow(d, s *cmat.FloatMatrix) *cmat.FloatMatrix {
    d.SetBuf(1, s.Len(), 1, s.Data())
    return d
}

// d = |d| - |s|
func absMinus(d, s *cmat.FloatMatrix) *cmat.FloatMatrix {
    for k := 0; k < d.Len(); k++ {
        tmp := math.Abs(d.GetAt(k))
        d.SetAt(k, math.Abs(s.GetAt(k))-tmp)
    }
    return d
}

func test_bdsvd(N, flags, kind int, verbose bool, t *testing.T) {
    var At, sD, sE, tmp cmat.FloatMatrix

    uplo := "upper"
    offdiag := 1
    if flags & gomas.LOWER != 0 {
        offdiag = -1
        uplo = "lower"
    }
    A0 := cmat.NewMatrix(N, N)
    desc := setDiagonals(A0, offdiag, kind)
    At.SubMatrix(A0, 0, 0, N, N)
    sD.Diag(A0, 0)
    sE.Diag(A0, offdiag)
    D := cmat.NewCopy(&sD)
    E := cmat.NewCopy(&sE)

    // unit singular vectors
    U := cmat.NewMatrix(N, N)
    sD.Diag(U, 0)
    sD.Add(1.0)

    V := cmat.NewMatrix(N, N)
    sD.Diag(V, 0)
    sD.Add(1.0)

    W := cmat.NewMatrix(4*N, 1)
    C := cmat.NewMatrix(N, N)

    lapackd.BDSvd(D, E, U, V, W, flags|gomas.WANTU|gomas.WANTV)

    blasd.Mult(C, U, U, 1.0, 0.0, gomas.TRANSA)
    sD.Diag(C)
    sD.Add(-1.0)
    nrmu := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Mult(C, V, V, 1.0, 0.0, gomas.TRANSB)
    sD.Add(-1.0)
    nrmv := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Mult(C, U, A0, 1.0, 0.0, gomas.TRANSA)
    blasd.Mult(&At, C, V, 1.0, 0.0, gomas.TRANSB)
    if verbose && N < 10 {
        t.Logf("D:\n%v\n", asRow(&tmp, D))
        t.Logf("U:\n%v\n", U)
        t.Logf("V:\n%v\n", V)
        t.Logf("U.T*A*V\n%v\n", &At)
    }
    sD.Diag(&At)
    blasd.Axpy(&sD, D, -1.0)
    nrma := lapackd.NormP(&At, lapackd.NORM_ONE)
    
    t.Logf("N=%d [%s,%s] ||U.T*A*V - bdsvd(A)||_1: %e\n", N, uplo, desc, nrma)
    t.Logf("  ||I - U.T*U||_1: %e\n", nrmu)
    t.Logf("  ||I - V.T*V||_1: %e\n", nrmv)
}

func TestBDsvdBottomHeavy(t *testing.T) {
    N := 113
    test_bdsvd(N, gomas.UPPER, 0, true, t)
    test_bdsvd(N, gomas.LOWER, 0, true, t)
}


func TestBDsvdTopHeavy(t *testing.T) {
    N := 113
    test_bdsvd(N, gomas.UPPER, 1, true, t)
    test_bdsvd(N, gomas.LOWER, 1, true, t)
}

func TestBDsvdMiddleHeavy(t *testing.T) {
    N := 113
    test_bdsvd(N, gomas.UPPER, 2, true, t)
    test_bdsvd(N, gomas.LOWER, 2, true, t)
}

func __TestSweep(t *testing.T) {
    
    N := 5
    D0 := cmat.NewMatrix(1, N)
    E0 := cmat.NewMatrix(1, N-1)
    D1 := cmat.NewMatrix(1, N)
    E1 := cmat.NewMatrix(1, N-1)
    
    for k := 0; k < N; k++ {
        D0.SetAt(k, float64(k+1))
        D1.SetAt(N-1-k, float64(k+1))
        if k < N-1 {
            E0.SetAt(k, 1.0)
            E1.SetAt(k, 1.0)
        }
    }

    f := 0.5; g := -0.45
    lapackd.BDQRsweep(D0, E0, f, g)
    lapackd.BDQLsweep(D1, E1, f, g)
    t.Logf("D0:%v\n", D0);
    t.Logf("D1:%v\n", D1);
    t.Logf("E0:%v\n", E0);
    t.Logf("E1:%v\n", E1);
}

func __TestZeroQRSweep(t *testing.T) {
    
    N := 5
    D0 := cmat.NewMatrix(1, N)
    E0 := cmat.NewMatrix(1, N-1)
    D1 := cmat.NewMatrix(1, N)
    E1 := cmat.NewMatrix(1, N-1)
    
    for k := 0; k < N; k++ {
        D0.SetAt(k, float64(k+1))
        D1.SetAt(k, float64(k+1))
        if k < N-1 {
            E0.SetAt(k, 1.0)
            E1.SetAt(k, 1.0)
        }
    }

    f := D0.GetAt(0); g := E0.GetAt(0)
    lapackd.BDQRsweep(D0, E0, f, g)
    lapackd.BDQRzero(D1, E1)
    t.Logf("D0.std :%v\n", D0);
    t.Logf("D1.zero:%v\n", D1);
    t.Logf("E0.std :%v\n", E0);
    t.Logf("E1:zero %v\n", E1);
}

func __TestZeroQLSweep(t *testing.T) {
    
    N := 5
    D0 := cmat.NewMatrix(1, N)
    E0 := cmat.NewMatrix(1, N-1)
    D1 := cmat.NewMatrix(1, N)
    E1 := cmat.NewMatrix(1, N-1)
    
    for k := 0; k < N; k++ {
        D0.SetAt(N-k-1, float64(k+1))
        D1.SetAt(N-k-1, float64(k+1))
        if k < N-1 {
            E0.SetAt(k, 1.0)
            E1.SetAt(k, 1.0)
        }
    }

    f := D0.GetAt(4); g := E0.GetAt(3)
    lapackd.BDQLsweep(D0, E0, f, g)
    lapackd.BDQLzero(D1, E1)
    t.Logf("D0.std :%v\n", D0);
    t.Logf("D1.zero:%v\n", D1);
    t.Logf("E0.std :%v\n", E0);
    t.Logf("E1.zero:%v\n", E1);
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

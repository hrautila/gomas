
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package test

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/lapackd"
    "testing"
)


func errorLoc(D, C *cmat.FloatMatrix) (r, c int) {
    var dmat, cmat cmat.FloatMatrix
    dr, dc := D.Size()
    for i := 0; i < dr; i++ {
        dmat.SubMatrix(D, i, 0, 1, dc)
        cmat.SubMatrix(C, i, 0, 1, dc)
        if ! dmat.AllClose(&cmat) {
            r = i
            break;
        }
    }
    for i := 0; i < dc; i++ {
        dmat.SubMatrix(D, 0, i, dr, 1)
        cmat.SubMatrix(C, 0, i, dr, 1)
        if ! dmat.AllClose(&cmat) {
            c = i
            break;
        }
    }
    return
}

func TestBKLowerBig(t *testing.T) {
    N := 411
    normsrc := cmat.NewFloatNormSource(5.0, 10.0)
    A := cmat.NewMatrix(N, N)
    A.SetFrom(normsrc, cmat.LOWER)
    A0 := cmat.NewCopy(A)

    ipiv := lapackd.NewPivots(N)
    ipiv0 := lapackd.NewPivots(N)

    conf := gomas.NewConf()
    conf.LB = 0

    // unblocked
    W := lapackd.Workspace(lapackd.BKFactorWork(A, conf))
    err := lapackd.BKFactor(A, W, ipiv, gomas.LOWER, conf)
    if err != nil {
        t.Logf("unblk.err: %v\n", err)
    }

    // blocked
    conf.LB = 8
    W = lapackd.Workspace(lapackd.BKFactorWork(A0, conf))
    err = lapackd.BKFactor(A0, W, ipiv0, gomas.LOWER, conf)
    if err != nil {
        t.Logf("blk.err: %v\n", err)
    }
    ok := A.AllClose(A0)
    t.Logf("N=%d unblk.A == blk.A : %v\n", N, ok)

    if ! ok {
        r, c := errorLoc(A, A0)
        t.Logf("unblk.A != blk.A at: %d, %d\n", r, c)
        for k, _ := range ipiv {
            t.Logf("%3d  %3d  %3d\n", k, ipiv[k], ipiv0[k])
        }
    }
}

func TestSolveBKLowerBig(t *testing.T) {
    N := 427
    normsrc := cmat.NewFloatNormSource(5.0, 10.0)
    A := cmat.NewMatrix(N, N)
    A.SetFrom(normsrc, cmat.LOWER)

    X := cmat.NewMatrix(N, 2)
    X.SetFrom(normsrc)
    B := cmat.NewCopy(X)
    blasd.MultSym(B, A, X, 1.0, 0.0, gomas.LOWER|gomas.LEFT)
    
    ipiv := lapackd.NewPivots(N)

    conf := gomas.NewConf()
    conf.LB = 16
    W := lapackd.Workspace(lapackd.BKFactorWork(A, conf))
    lapackd.BKFactor(A, W, ipiv, gomas.LOWER, conf)

    lapackd.BKSolve(B, A, ipiv, gomas.LOWER, conf)
    ok := B.AllClose(X)
    t.Logf("N=%d unblk.BK(X) == A.-1*B : %v\n", N, ok)
    blasd.Plus(B, X, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("  ||X - A.-1*B||_1: %.4e\n", nrm)

}

func TestBKUpperBig(t *testing.T) {
    N := 513
    normsrc := cmat.NewFloatNormSource(5.0, 10.0)
    A := cmat.NewMatrix(N, N)
    A.SetFrom(normsrc, cmat.UPPER)
    A0 := cmat.NewCopy(A)

    ipiv := lapackd.NewPivots(N)
    ipiv0 := lapackd.NewPivots(N)

    conf := gomas.NewConf()
    conf.LB = 0

    // unblocked
    W := lapackd.Workspace(lapackd.BKFactorWork(A, conf))
    err := lapackd.BKFactor(A, W, ipiv, gomas.UPPER, conf)
    if err != nil {
        t.Logf("unblk.err: %v\n", err)
    }

    // blocked
    conf.LB = 16
    W = lapackd.Workspace(lapackd.BKFactorWork(A0, conf))
    err = lapackd.BKFactor(A0, W, ipiv0, gomas.UPPER, conf)
    if err != nil {
        t.Logf("blk.err: %v\n", err)
    }
    ok0 := A.AllClose(A0)
    t.Logf("N=%d unblk.A == blk.A : %v\n", N, ok0)

    if ! ok0 {
        r, c := errorLoc(A, A0)
        t.Logf("unblk.A != blk.A at: %d, %d\n", r, c)
        for k, _ := range ipiv {
            t.Logf("%3d  %3d  %3d\n", k, ipiv[k], ipiv0[k])
        }
    }
}

func TestSolveBKUpperBig(t *testing.T) {
    N := 511
    normsrc := cmat.NewFloatNormSource(5.0, 10.0)
    A := cmat.NewMatrix(N, N)
    A.SetFrom(normsrc, cmat.LOWER)

    X := cmat.NewMatrix(N, 2)
    X.SetFrom(normsrc)
    B := cmat.NewCopy(X)
    blasd.MultSym(B, A, X, 1.0, 0.0, gomas.UPPER|gomas.LEFT)
    
    ipiv := lapackd.NewPivots(N)

    conf := gomas.NewConf()
    conf.LB = 16
    W := lapackd.Workspace(lapackd.BKFactorWork(A, conf))
    lapackd.BKFactor(A, W, ipiv, gomas.UPPER, conf)

    lapackd.BKSolve(B, A, ipiv, gomas.UPPER, conf)
    ok := B.AllClose(X)
    t.Logf("N=%d unblk.BKSolve.X == A.-1*B : %v\n", N, ok)
    blasd.Plus(B, X, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("  ||X - A.-1*B||_1: %.4e\n", nrm)

}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

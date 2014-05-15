
// Copyright (c) Harri Rautila, 2013,2014

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

// test: min || B - A*X ||
func TestLeastSquaresQR(t *testing.T) {
    M := 811
    N := 723
    K := 311
    nb := 32
    conf := gomas.NewConf()
    conf.LB = nb

    tau := cmat.NewMatrix(N, 1)
    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    B0 := cmat.NewMatrix(N, K)
    B0.SetFrom(src)
    B := cmat.NewMatrix(M, K)

    // B = A*B0
    blasd.Mult(B, A, B0, 1.0, 0.0, gomas.NONE, conf)

    W := lapackd.Workspace(lapackd.QRFactorWork(A, conf))
    err := lapackd.QRFactor(A, tau, W, conf)
    if err != nil {
        t.Logf("DecomposeQR: %v\n", err)
    }

    // B' = A.-1*B
    err = lapackd.QRSolve(B, A, tau, W, gomas.NONE, conf)
    if err != nil {
        t.Logf("SolveQR: %v\n", err)
    }

    // expect B[0:N,0:K] == B0[0:N,0:K], B[N:M,0:K] == 0
    var X cmat.FloatMatrix

    X.SubMatrix(B, 0, 0, N, K)
    blasd.Plus(&X, B0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(&X, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d  ||B0 - min( ||A*X - B0|| ) ||_1: %e\n", M, N, nrm)
}

// test: min ||X|| s.t A.T*X = B
func TestSolveQR(t *testing.T) {
    M := 799
    N := 711
    K := 241
    nb := 32
    conf := gomas.NewConf()
    conf.LB = nb

    tau := cmat.NewMatrix(N, 1)
    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    A0 := cmat.NewCopy(A)
    B0 := cmat.NewMatrix(M, K)
    B0.SetFrom(src)
    B := cmat.NewCopy(B0)

    W := lapackd.Workspace(lapackd.QRFactorWork(A, conf))
    lapackd.QRFactor(A, tau, W, conf)

    lapackd.QRSolve(B, A, tau, W, gomas.TRANS, conf)

    var Bmin cmat.FloatMatrix
    Bmin.SubMatrix(B0, 0, 0, N, K)
    blasd.Mult(&Bmin, A0, B, 1.0, -1.0, gomas.TRANSA, conf)

    nrm := lapackd.NormP(&Bmin, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||B - A.T*X||_1: %e\n", M, N, nrm)
}

// test: min || B - A.T*X ||
func TestLeastSquaresLQ(t *testing.T) {
    M := 723
    N := 811
    K := 273
    nb := 32
    conf := gomas.NewConf()
    conf.LB = nb

    tau := cmat.NewMatrix(M, 1)
    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    B0 := cmat.NewMatrix(M, K)
    B0.SetFrom(src)
    B := cmat.NewMatrix(N, K)

    // B = A.T*B0
    blasd.Mult(B, A, B0, 1.0, 0.0, gomas.TRANSA, conf)

    W := lapackd.Workspace(lapackd.LQFactorWork(A, conf))
    lapackd.LQFactor(A, tau, W, conf)

    // B' = A.-1*B
    lapackd.LQSolve(B, A, tau, W, gomas.TRANS, conf)

    // expect B[0:M,0:K] == B0[0:M,0:K], B[M:N,0:K] == 0
    var X cmat.FloatMatrix

    X.SubMatrix(B, 0, 0, M, K)
    blasd.Plus(&X, B0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(&X, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d  ||B0 - min( ||A.T*X - B0|| ) ||_1: %e\n", M, N, nrm)
}


// test: min ||X|| s.t. A*X = B
func TestSolveLQ(t *testing.T) {
    M := 743
    N := 809
    K := 281
    nb := 32
    conf := gomas.NewConf()
    conf.LB = nb

    tau := cmat.NewMatrix(N, 1)
    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    A0 := cmat.NewCopy(A)
    B0 := cmat.NewMatrix(N, K)
    B0.SetFrom(src)
    B := cmat.NewCopy(B0)

    W := lapackd.Workspace(lapackd.LQFactorWork(A, conf))
    lapackd.LQFactor(A, tau, W, conf)

    lapackd.LQSolve(B, A, tau, W, gomas.NONE, conf)

    var Bmin cmat.FloatMatrix
    Bmin.SubMatrix(B0, 0, 0, M, K)
    blasd.Mult(&Bmin, A0, B, 1.0, -1.0, gomas.NONE, conf)

    nrm := lapackd.NormP(&Bmin, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||B - A*X||_1: %e\n", M, N, nrm)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

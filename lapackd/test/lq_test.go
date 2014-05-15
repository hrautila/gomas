
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

func TestLQFactor(t *testing.T) {
    M := 611
    N := 715
    nb := 32
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(M, 1)

    A1 := cmat.NewCopy(A)
    tau1 := cmat.NewCopy(tau)

    conf.LB = 0
    W := cmat.NewMatrix(M+N, 1)
    lapackd.LQFactor(A, tau, W, conf)

    conf.LB = nb
    W1 := lapackd.Workspace(lapackd.LQFactorWork(A1, conf))
    lapackd.LQFactor(A1, tau1, W1, conf)
    
    blasd.Plus(A1, A, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||blk.LQ(A) - unblk.LQ(A)||_1: %e\n", M, N, nrm)
}


// test: ||A - A*Q*Q.T||_1 ~= 0
func TestLQMultRight(t *testing.T) {
    M := 511
    N := 627
    nb := 24
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(M, 1)
    W := cmat.NewMatrix(M+N, 1)

    A0 := cmat.NewCopy(A)
    A1 := cmat.NewCopy(A)
    A2 := cmat.NewCopy(A)

    conf.LB = 0
    lapackd.LQFactor(A, tau, W, conf)

    // unblocked A1 := A1*Q*Q.T
    conf.LB = 0
    lapackd.LQMult(A1, A, tau, W, gomas.RIGHT, conf)
    lapackd.LQMult(A1, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)

    // blocked A2 := A2*Q*Q.T
    conf.LB = nb
    W = lapackd.Workspace(lapackd.LQMultWork(A2, gomas.RIGHT, conf))
    lapackd.LQMult(A2, A, tau, W, gomas.RIGHT, conf)
    lapackd.LQMult(A2, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)

    // A1 - A0 == 0
    blasd.Plus(A1, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d, unblk.||A - A*Q*Q.T||_1: %e\n", M, N, nrm)

    // A2 - A0 == 0
    blasd.Plus(A2, A0, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(A2, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d, nb=%d blk.||A - A*Q*Q.T||_1: %e\n", M, N, nb, nrm)
}

// test: ||C - Q*Q.T*C||_1 ~= 0;
//   multipling from left requires: m(C) == n(A) [n(Q)]
func TestLQMultLeft(t *testing.T) {
    M := 771
    N := 813
    nb := 16
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    // C0 := A
    C0 := cmat.NewCopy(A)
    C1t := cmat.NewMatrix(N, M)
    blasd.Transpose(C1t, C0)
    C2t := cmat.NewCopy(C1t)

    tau := cmat.NewMatrix(M, 1)
    W := cmat.NewMatrix(M+N, 1)

    conf.LB = 0
    lapackd.LQFactor(A, tau, W, conf)

    // A0 := Q.T*A0
    conf.LB = 0
    lapackd.LQMult(C2t, A, tau, W, gomas.LEFT, conf)
    lapackd.LQMult(C2t, A, tau, W, gomas.LEFT|gomas.TRANS, conf)

    // A0 := Q.T*A0
    conf.LB = nb
    W1 := lapackd.Workspace(lapackd.LQMultWork(C1t, gomas.LEFT, conf))
    lapackd.LQMult(C1t, A, tau, W1, gomas.LEFT, conf)
    lapackd.LQMult(C1t, A, tau, W1, gomas.LEFT|gomas.TRANS, conf)

    blasd.Plus(C0, C1t, 1.0, -1.0, gomas.TRANSB)
    nrm := lapackd.NormP(C0, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d, ||C - Q*Q*T*C||_1: %e\n", M, N, nrm)

    blasd.Plus(C1t, C2t, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(C1t, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d, ||unblk(Q*Q.T*C) - blk(Q*Q*T*C)||_1: %e\n", M, N, nrm)
}

func TestLQBuild(t *testing.T) {
    var dc cmat.FloatMatrix

    M := 877
    N := 913
    K := 831
    lb := 48
    conf := gomas.NewConf()
    _ = lb

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(M, 1)
    W := cmat.NewMatrix(M, 1)
    C := cmat.NewMatrix(M, M)
    dc.Diag(C)

    conf.LB = lb
    lapackd.LQFactor(A, tau, W, conf)
    A1 := cmat.NewCopy(A)

    conf.LB = 0
    lapackd.LQBuild(A, tau, W, K, conf)
    if N < 10 {
        t.Logf("unblk.LQBuild Q:\n%v\n", A)
    }
    blasd.Mult(C, A, A, 1.0, 0.0, gomas.TRANSB, conf)
    blasd.Add(&dc, -1.0)
    n0 := lapackd.NormP(C, lapackd.NORM_ONE)

    conf.LB = lb
    W2 := lapackd.Workspace(lapackd.LQBuildWork(A, conf))
    lapackd.LQBuild(A1, tau, W2, K, conf)
    if N < 10 {
        t.Logf("blk.LQBuild Q:\n%v\n", A1)
    }
    blasd.Mult(C, A1, A1, 1.0, 0.0, gomas.TRANSB, conf)
    blasd.Add(&dc, -1.0)
    n1 := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Plus(A, A1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(A, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d, K=%d ||unblk.LQBuild(A) - blk.LQBuild(A)||_1 :%e\n", M, N, K, n2)
    t.Logf("unblk M=%d, N=%d, K=%d ||I - Q*Q.T||_1 : %e\n", M, N, K, n0)
    t.Logf("  blk M=%d, N=%d, K=%d ||I - Q*Q.T||_1 : %e\n", M, N, K, n1)
}



// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

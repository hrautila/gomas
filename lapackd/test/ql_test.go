
// Copyright (c) Harri Rautila, 2013,2014

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


// test: unblk.DecomposeQL == blk.DecomposeQL
func TestDecomposeQL(t *testing.T) {
    var t0 cmat.FloatMatrix
    M := 911
    N := 835
    nb := 32
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(M, 1)
    W := cmat.NewMatrix(M+N, 1)

    A1 := cmat.NewCopy(A)
    tau1 := cmat.NewCopy(tau)

    conf.LB = 0
    lapackd.DecomposeQL(A, tau, W, conf)

    conf.LB = nb
    W1 := lapackd.Workspace(lapackd.WorksizeQL(A1, conf))
    lapackd.DecomposeQL(A1, tau1, W1, conf)

    if N < 10 {
        t.Logf("unblkQL(A):\n%v\n", A)
        t0.SetBuf(1, tau.Len(), 1, tau.Data())
        t.Logf("unblkQL.tau:\n%v\n", &t0)
        t.Logf("blkQL(A):\n%v\n", A1)
        t0.SetBuf(1, tau1.Len(), 1, tau1.Data())
        t.Logf("blkQL.tau:\n%v\n", &t0)
    }

    blasd.Plus(A1, A, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||blkQL(A) - unblkQL(A)||_1: %e\n", M, N, nrm)

    blasd.Axpy(tau1, tau, -1.0)
    nrm = blasd.Nrm2(tau1)
    t.Logf("     ||blkQL.tau - unblkQL.tau||_1: %e\n", nrm)

}

func TestMultQLLeft(t *testing.T) {
    var d, di0, di1 cmat.FloatMatrix
    M := 901
    N := 887
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    
    C0 := cmat.NewMatrix(M, N)
    d.Diag(C0, N-M)
    ones := cmat.NewFloatConstSource(1.0)
    d.SetFrom(ones)
    C1 := cmat.NewCopy(C0)

    I0 := cmat.NewMatrix(N, N)
    I1 := cmat.NewCopy(I0)
    di0.Diag(I0)
    di1.Diag(I1)

    tau := cmat.NewMatrix(N, 1)
    W := cmat.NewMatrix(lb*(M+N), 1)

    conf.LB = lb
    lapackd.DecomposeQL(A, tau, W, conf)

    lapackd.MultQL(C0, A, tau, W, gomas.LEFT, conf)
    // I = Q*Q.T - I
    blasd.Mult(I0, C0, C0, 1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&di0, -1.0)
    n0 := lapackd.NormP(I0, lapackd.NORM_ONE)
    
    conf.LB = lb
    lapackd.MultQL(C1, A, tau, W, gomas.LEFT, conf)
    // I = Q*Q.T - I
    blasd.Mult(I1, C1, C1, 1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&di1, -1.0)
    n1 := lapackd.NormP(I1, lapackd.NORM_ONE)

    if N < 10 {
        t.Logf("unblk Q*C0:\n%v\n", C0)
        t.Logf("blk   Q*C2:\n%v\n", C1)
    }
    blasd.Plus(C0, C1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(C0, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d ||unblk.MultQL(C) - blk.MultQL(C)||_1: %e\n", M, N, n2)
    t.Logf("unblk M=%d, N=%d ||I - Q.T*Q||_1: %e\n", M, N, n0)
    t.Logf("blk   M=%d, N=%d ||I - Q.T*Q||_1: %e\n", M, N, n1)
}

func TestMultQLLeftTrans(t *testing.T) {
    var d, di0, di1 cmat.FloatMatrix
    M := 901
    N := 887
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    
    C0 := cmat.NewMatrix(M, N)
    d.Diag(C0, N-M)
    ones := cmat.NewFloatConstSource(1.0)
    d.SetFrom(ones)
    C1 := cmat.NewCopy(C0)

    I0 := cmat.NewMatrix(N, N)
    I1 := cmat.NewCopy(I0)
    di0.Diag(I0)
    di1.Diag(I1)

    tau := cmat.NewMatrix(N, 1)
    W := cmat.NewMatrix(lb*(M+N), 1)

    conf.LB = lb
    lapackd.DecomposeQL(A, tau, W, conf)

    lapackd.MultQL(C0, A, tau, W, gomas.LEFT|gomas.TRANS, conf)
    // I = Q*Q.T - I
    blasd.Mult(I0, C0, C0, 1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&di0, -1.0)
    n0 := lapackd.NormP(I0, lapackd.NORM_ONE)
    
    conf.LB = lb
    lapackd.MultQL(C1, A, tau, W, gomas.LEFT|gomas.TRANS, conf)
    // I = Q*Q.T - I
    blasd.Mult(I1, C1, C1, 1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&di1, -1.0)
    n1 := lapackd.NormP(I1, lapackd.NORM_ONE)

    if N < 10 {
        t.Logf("unblk Q*C0:\n%v\n", C0)
        t.Logf("blk   Q*C2:\n%v\n", C1)
    }
    blasd.Plus(C0, C1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(C0, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d ||unblk.MultQL(C) - blk.MultQL(C)||_1: %e\n", M, N, n2)
    t.Logf("unblk M=%d, N=%d ||I - Q*Q.T||_1: %e\n", M, N, n0)
    t.Logf("blk   M=%d, N=%d ||I - Q*Q.T||_1: %e\n", M, N, n1)
}


func TestMultQLRight(t *testing.T) {
    var d, di0, di1 cmat.FloatMatrix
    M := 891
    N := 853
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    
    C0 := cmat.NewMatrix(N, M)
    d.Diag(C0, M-N)
    ones := cmat.NewFloatConstSource(1.0)
    d.SetFrom(ones)
    C1 := cmat.NewCopy(C0)

    I0 := cmat.NewMatrix(N, N)
    I1 := cmat.NewCopy(I0)
    di0.Diag(I0)
    di1.Diag(I1)

    tau := cmat.NewMatrix(N, 1)
    W := cmat.NewMatrix(lb*(M+N), 1)

    conf.LB = lb
    lapackd.DecomposeQL(A, tau, W, conf)

    conf.LB = 0
    lapackd.MultQL(C0, A, tau, W, gomas.RIGHT, conf)
    // I = Q*Q.T - I
    blasd.Mult(I0, C0, C0, 1.0, 0.0, gomas.TRANSB, conf)
    blasd.Add(&di0, -1.0)
    n0 := lapackd.NormP(I0, lapackd.NORM_ONE)
    
    conf.LB = lb
    lapackd.MultQL(C1, A, tau, W, gomas.RIGHT, conf)
    // I = Q*Q.T - I
    blasd.Mult(I1, C1, C1, 1.0, 0.0, gomas.TRANSB, conf)
    blasd.Add(&di1, -1.0)
    n1 := lapackd.NormP(I1, lapackd.NORM_ONE)

    if N < 10 {
        t.Logf("unblk C0*Q:\n%v\n", C0)
        t.Logf("blk. C2*Q:\n%v\n", C1)
    }
    blasd.Plus(C0, C1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(C0, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d ||unblk.MultQL(C) - blk.MultQL(C)||_1: %e\n", M, N, n2)
    t.Logf("unblk M=%d, N=%d ||I - Q*Q.T||_1: %e\n", M, N, n0)
    t.Logf("blk   M=%d, N=%d ||I - Q*Q.T||_1: %e\n", M, N, n1)
}

// test: C = C*Q.T 
func TestMultQLRightTrans(t *testing.T) {
    var d, di0, di1 cmat.FloatMatrix
    M := 891
    N := 853
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    
    C0 := cmat.NewMatrix(N, M)
    d.Diag(C0, M-N)
    ones := cmat.NewFloatConstSource(1.0)
    d.SetFrom(ones)
    C1 := cmat.NewCopy(C0)

    I0 := cmat.NewMatrix(N, N)
    I1 := cmat.NewCopy(I0)
    di0.Diag(I0)
    di1.Diag(I1)

    tau := cmat.NewMatrix(N, 1)
    W := cmat.NewMatrix(lb*(M+N), 1)

    conf.LB = lb
    lapackd.DecomposeQL(A, tau, W, conf)

    conf.LB = 0
    lapackd.MultQL(C0, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)
    // I = Q*Q.T - I
    blasd.Mult(I0, C0, C0, 1.0, 0.0, gomas.TRANSB, conf)
    blasd.Add(&di0, -1.0)
    n0 := lapackd.NormP(I0, lapackd.NORM_ONE)
    
    conf.LB = lb
    lapackd.MultQL(C1, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)
    // I = Q*Q.T - I
    blasd.Mult(I1, C1, C1, 1.0, 0.0, gomas.TRANSB, conf)
    blasd.Add(&di1, -1.0)
    n1 := lapackd.NormP(I1, lapackd.NORM_ONE)

    if N < 10 {
        t.Logf("unblk C0*Q:\n%v\n", C0)
        t.Logf("blk. C2*Q:\n%v\n", C1)
    }
    blasd.Plus(C0, C1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(C0, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d ||unblk.MultQL(C) - blk.MultQL(C)||_1: %e\n", M, N, n2)
    t.Logf("unblk M=%d, N=%d ||I - Q*Q.T||_1: %e\n", M, N, n0)
    t.Logf("blk   M=%d, N=%d ||I - Q*Q.T||_1: %e\n", M, N, n1)
}


func TestBuildQL(t *testing.T) {
    var dc cmat.FloatMatrix
    M := 711
    N := 707
    K := 707
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(N, 1)

    W := cmat.NewMatrix(M+N, 1)
    C := cmat.NewMatrix(N, N)

    conf.LB = lb
    lapackd.DecomposeQL(A, tau, W, conf)
    A1 := cmat.NewCopy(A)

    conf.LB = 0
    lapackd.BuildQL(A, tau, W, K, conf)
    blasd.Mult(C, A, A, 1.0, 0.0, gomas.TRANSA, conf)
    dc.Diag(C)
    blasd.Add(&dc, -1.0)
    if N < 10 {
        t.Logf("unblk.BuildQL Q:\n%v\n", A)
        t.Logf("unblk.BuildQL Q.T*Q:\n%v\n", C)
    }
    n0 := lapackd.NormP(C, lapackd.NORM_ONE)

    conf.LB = lb
    W1 := lapackd.Workspace(lapackd.WorksizeBuildQL(A1, conf))
    lapackd.BuildQL(A1, tau, W1, K, conf)
    if N < 10 {
        t.Logf("blk.BuildQL Q:\n%v\n", A1)
    }
    // compute: I - Q.T*Q
    blasd.Mult(C, A1, A1, 1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&dc, -1.0)
    n1 := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Plus(A, A1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(A, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d, K=N ||unblk.BuildQL(A) - blk.BuildQL(A)||_1 :%e\n", M, N, n2)
    t.Logf("  unblk M=%d, N=%d, K=N ||Q.T*Q - I||_1 : %e\n", M, N, n0)
    t.Logf("  blk   M=%d, N=%d, K=N ||Q.T*Q - I||_1 : %e\n", M, N, n1)
}


func TestBuildQLwithK(t *testing.T) {
    var dc cmat.FloatMatrix
    M := 711
    N := 707
    K := 691
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(N, 1)

    W := cmat.NewMatrix(M+N, 1)
    C := cmat.NewMatrix(N, N)

    conf.LB = lb
    lapackd.DecomposeQL(A, tau, W, conf)
    A1 := cmat.NewCopy(A)

    conf.LB = 0
    lapackd.BuildQL(A, tau, W, K, conf)
    blasd.Mult(C, A, A, 1.0, 0.0, gomas.TRANSA, conf)
    dc.Diag(C)
    blasd.Add(&dc, -1.0)
    if N < 10 {
        t.Logf("unblk.BuildQL Q:\n%v\n", A)
        t.Logf("unblk.BuildQL Q.T*Q:\n%v\n", C)
    }
    n0 := lapackd.NormP(C, lapackd.NORM_ONE)

    conf.LB = lb
    W1 := lapackd.Workspace(lapackd.WorksizeBuildQL(A1, conf))
    lapackd.BuildQL(A1, tau, W1, K, conf)
    if N < 10 {
        t.Logf("blk.BuildQL Q:\n%v\n", A1)
    }
    // compute: I - Q.T*Q
    blasd.Mult(C, A1, A1, 1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&dc, -1.0)
    n1 := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Plus(A, A1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(A, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d, K=%d ||unblk.BuildQL(A) - blk.BuildQL(A)||_1 :%e\n", M, N, K, n2)
    t.Logf("unblk M=%d, N=%d, K=%d ||Q.T*Q - I||_1 : %e\n", M, N, K, n0)
    t.Logf("blk   M=%d, N=%d, K=%d ||Q.T*Q - I||_1 : %e\n", M, N, K, n1)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

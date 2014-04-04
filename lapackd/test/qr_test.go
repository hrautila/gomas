
// Copyright (c) Harri Rautila, 2012,2013

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

// test that unblocked QR and QRT are equal
func TestDecomposeQR(t *testing.T) {
    M := 411
	N := 375
    nb := 16

    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(M, N)
    //W := cmat.NewMatrix(N, nb)
    tau := cmat.NewMatrix(N, 1)
    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)

    A0 := cmat.NewCopy(A)
    tau0 := cmat.NewCopy(tau)

    // blocked: QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)
    
    conf.LB = 0
    lapackd.DecomposeQR(A0, tau0, W, conf)

    ok := A.AllClose(A0)
    t.Logf("blk.DecomposeQR(A) == unblk.DecomposeQR(A): %v\n", ok)

    ok = tau0.AllClose(tau)
    t.Logf("blk QR.tau == unblk QR.tau: %v\n", ok)
}

// QR decompose A, then compute ||A - Q*R||_1, should be small
func TestUnblkMultQLeft(t *testing.T) {
    M := 711
	N := 593
    A := cmat.NewMatrix(M, N)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    conf := gomas.NewConf()
    conf.LB = 0

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)

    // C = TriU(QR) = R
    C := cmat.TriU(cmat.NewCopy(A), cmat.NONE)

    // C = Q*C
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.LEFT, conf))
    err := lapackd.MultQ(C, A, tau, W, gomas.LEFT, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }

    // A = A - QR
    blasd.Plus(A0, C, 1.0, -1.0, gomas.NONE)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - Q*R||_1: %e\n", M, N, nrm)
}

// QR decompose A, then compute ||A - (R.T*Q.T).T||_1, should be small
func TestUnblkMultQRight(t *testing.T) {
    M := 711
	N := 593
    A := cmat.NewMatrix(M, N)
    C := cmat.NewMatrix(N, M)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    conf := gomas.NewConf()
    conf.LB = 0

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)

    // C = transpose(TriU(QR)) = R.T
    C.Transpose(cmat.TriU(cmat.NewCopy(A), cmat.NONE))

    // C = C*Q.T = R.T*Q.T
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.RIGHT, conf))
    err := lapackd.MultQ(C, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }

    // A = A - QR
    blasd.Plus(A0, C, 1.0, -1.0, gomas.TRANSB)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - (R.T*Q.T).T||_1: %e\n", M, N, nrm)
}


// QR decompose A, then compute ||A - Q*R||_1, should be small
func TestBlockedMultQLeft(t *testing.T) {
    M := 713
	N := 645
    A := cmat.NewMatrix(M, N)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    conf := gomas.NewConf()
    conf.LB = 32

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)
    
    // C = TriU(QR) = R
    C := cmat.TriU(cmat.NewCopy(A), cmat.NONE)

    // C = Q*C
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.LEFT, conf))
    err := lapackd.MultQ(C, A, tau, W, gomas.LEFT, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }

    // A = A - QR
    blasd.Plus(A0, C, 1.0, -1.0, gomas.NONE)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - Q*R||_1: %e\n", M, N, nrm)
}


// QR decompose A, then compute ||A - (R.T*Q.T).T||_1, should be small
func TestBlockedMultQRight(t *testing.T) {
    M := 711
	N := 593
    A := cmat.NewMatrix(M, N)
    C := cmat.NewMatrix(N, M)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    conf := gomas.NewConf()
    conf.LB = 32

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)

    // C = transpose(TriU(QR)) = R.T
    C.Transpose(cmat.TriU(cmat.NewCopy(A), cmat.NONE))

    // C = C*Q.T = R.T*Q.T
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.RIGHT, conf))
    err := lapackd.MultQ(C, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }

    // A = A - QR
    blasd.Plus(A0, C, 1.0, -1.0, gomas.TRANSB)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - (R.T*Q.T).T||_1: %e\n", M, N, nrm)
}


// m > n: A[m,n], I[m,m] --> A == I*A == Q*Q.T*A
func TestUnblkMultQLeftIdent(t *testing.T) {
    M := 411
	N := 399
    A := cmat.NewMatrix(M, N)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)
    C  := cmat.NewCopy(A)
    conf := gomas.NewConf()
    conf.LB = 0

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)
    //t.Logf("T:\n%v\n", T)

    // C = Q.T*A
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.LEFT, conf))
    lapackd.MultQ(C, A, tau, W, gomas.LEFT|gomas.TRANS, conf)
    
    // C = Q*C == Q*Q.T*A
    lapackd.MultQ(C, A, tau, W, gomas.LEFT, conf)
    //t.Logf("A*Q*Q.T:\n%v\n", C)

    // A = A - Q*Q.T*A
    blasd.Plus(A0, C, 1.0, -1.0, gomas.NONE)
	// ||A - Q*Q.T*A||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - Q*Q.T*A||_1: %e\n", M, N, nrm)
}


// m > n: A[m,n], I[m,m] --> A == I*A == Q*Q.T*A
func TestBlkMultQLeftIdent(t *testing.T) {
    M := 411
	N := 399
    A := cmat.NewMatrix(M, N)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)
    C  := cmat.NewCopy(A)
    conf := gomas.NewConf()
    conf.LB = 32

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)
    //t.Logf("T:\n%v\n", T)

    // C = Q.T*A
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.LEFT, conf))
    lapackd.MultQ(C, A, tau, W, gomas.LEFT|gomas.TRANS, conf)
    
    // C = Q*C == Q*Q.T*A
    lapackd.MultQ(C, A, tau, W, gomas.LEFT, conf)
    //t.Logf("A*Q*Q.T:\n%v\n", C)

    // A = A - Q*Q.T*A
    blasd.Plus(A0, C, 1.0, -1.0, gomas.NONE)
	// ||A - Q*Q.T*A||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - Q*Q.T*A||_1: %e\n", M, N, nrm)
}

// m > n: A[m,n], I[m,m] --> A.T == A.T*I == A.T*Q*Q.T
func TestUnblkMultQRightIdent(t *testing.T) {
    M := 521
	N := 497
    A := cmat.NewMatrix(M, N)
    C := cmat.NewMatrix(N, M)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)
    C.Transpose(A)
    conf := gomas.NewConf()
    conf.LB = 0

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)

    // C = A.T*Q
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.RIGHT, conf))
    lapackd.MultQ(C, A, tau, W, gomas.RIGHT, conf)

    // C = C*Q.T == A.T*Q*Q.T
    lapackd.MultQ(C, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)

    // A = A - (A.T*Q*Q.T).T
    blasd.Plus(A0, C, 1.0, -1.0, gomas.TRANSB)
	// ||A - Q*Q.T*A||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - (A.T*Q*Q.T).T||_1: %e\n", M, N, nrm)
}

// m > n: A[m,n], I[m,m] --> A.T == A.T*I == A.T*Q*Q.T
func TestBlockedMultQRightIdent(t *testing.T) {
    M := 511
	N := 489
    A := cmat.NewMatrix(M, N)
    C := cmat.NewMatrix(N, M)
    tau := cmat.NewMatrix(N, 1)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)
    C.Transpose(A)
    conf := gomas.NewConf()
    conf.LB = 32

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.WorksizeQR(A, conf))
    lapackd.DecomposeQR(A, tau, W, conf)

    // C = A.T*Q
    W = lapackd.Workspace(lapackd.WorksizeMultQ(C, gomas.RIGHT, conf))
    lapackd.MultQ(C, A, tau, W, gomas.RIGHT, conf)
    
    // C = C*Q.T == A.T*Q*Q.T
    lapackd.MultQ(C, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)
    //t.Logf("A*Q*Q.T:\n%v\n", C)

    // A = A - (A.T*Q*Q.T).T
    blasd.Plus(A0, C, 1.0, -1.0, gomas.TRANSB)
	// ||A - (A.T*Q*Q.T).T||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - (A.T*Q*Q.T).T||_1: %e\n", M, N, nrm)
}

func TestBuildQR(t *testing.T) {
    var d cmat.FloatMatrix

    M := 911
    N := 899
    K := 873
    lb := 36
    conf := gomas.NewConf()

    A := cmat.NewMatrix(M, N)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    tau := cmat.NewMatrix(N, 1)
    W := cmat.NewMatrix(N+M, 1)

    C := cmat.NewMatrix(N, N)
    d.Diag(C)
    
    conf.LB = lb
    lapackd.DecomposeQR(A, tau, W, conf)
    A1 := cmat.NewCopy(A)

    conf.LB = 0
    lapackd.BuildQR(A, tau, W, K, conf)

    blasd.Mult(C, A, A,  1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&d, -1.0)
    n0 := lapackd.NormP(C, lapackd.NORM_ONE)

    conf.LB = lb
    W2 := lapackd.Workspace(lapackd.WorksizeBuildQR(A, conf))
    lapackd.BuildQR(A1, tau, W2, K, conf)

    blasd.Mult(C, A1, A1,  1.0, 0.0, gomas.TRANSA, conf)
    blasd.Add(&d, -1.0)
    n1 := lapackd.NormP(C, lapackd.NORM_ONE)

    blasd.Plus(A, A1, 1.0, -1.0, gomas.NONE)
    n2 := lapackd.NormP(A, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d, K=%d ||unblk.BuildQR(A) - blk.BuildQR(A)||_1 :%e\n", M, N, K, n2)
    t.Logf("unblk M=%d, N=%d, K=%d ||I - Q.T*Q||_1: %e\n", M, N, K, n0)
    t.Logf("  blk M=%d, N=%d, K=%d ||I - Q.T*Q||_1: %e\n", M, N, K, n1)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

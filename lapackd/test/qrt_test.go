
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

// test that unblocked and blocked QRT are equal
func TestDecomposeQRT(t *testing.T) {
    M := 615
	N := 591
    nb := 16

    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(M, N)
    T := cmat.NewMatrix(nb, N)
    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)

    A0 := cmat.NewCopy(A)
    T0 := cmat.NewMatrix(N, N)

    // blocked: QR = A = Q*R
    W := lapackd.Workspace(lapackd.QRTFactorWork(A, conf))
    lapackd.QRTFactor(A, T, W, conf)

    conf.LB = 0
    lapackd.QRTFactor(A0, T0, W, conf)

    ok := A.AllClose(A0)
    t.Logf("blk.DecomposeQRT(A) == unblk.DecomposeQRT(A): %v\n", ok)

}


// QR decompose A, then compute ||A - Q*R||_1, should be small
func TestMultQTLeft(t *testing.T) {
    M := 513
	N := 477
    nb := 16
    A := cmat.NewMatrix(M, N)
    T := cmat.NewMatrix(nb, N)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    conf := gomas.NewConf()
    conf.LB = nb
    //t.Logf("A0:\n%v\n", A0)

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.QRTFactorWork(A, conf))
    lapackd.QRTFactor(A, T, W, conf)
    //t.Logf("T:\n%v\n", T)

    // C = TriU(QR) = R
    C := cmat.TriU(cmat.NewCopy(A), cmat.NONE)
    //t.Logf("R:\n%v\n", C)

    // C = Q*C
    W = lapackd.Workspace(lapackd.QRTMultWork(C, T, gomas.LEFT, conf))
    err := lapackd.QRTMult(C, A, T, W, gomas.LEFT, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }

    // A = A - QR
    blasd.Plus(A0, C, 1.0, -1.0, gomas.NONE)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - Q*R||_1: %e\n", M, N, nrm)
}

// this is A = Q*R --> A.T == R.T*Q.T  --> ||A - (R.T*Q.T).T||_1
func TestQRTMultRight(t *testing.T) {
    M := 511
	N := 493
    nb := 16
    A  := cmat.NewMatrix(M, N)
    C  := cmat.NewMatrix(N, M)
    T  := cmat.NewMatrix(nb, N)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    conf := gomas.NewConf()
    conf.LB = nb

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.QRTFactorWork(A, conf))
    lapackd.QRTFactor(A, T, W, conf)

    // A.T = R.T*Q.T 
    // C =  transpose(TriU(QR)) = R.T
    C.Transpose(cmat.TriU(cmat.NewCopy(A), cmat.NONE))

    // A.T = C*Q.T = R.T*Q.T
    W = lapackd.Workspace(lapackd.QRTMultWork(C, T, gomas.RIGHT, conf))
    err := lapackd.QRTMult(C, A, T, W, gomas.RIGHT|gomas.TRANS, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }
        
    // A = A - (R.T*Q.T).T
    blasd.Plus(A0, C, 1.0, -1.0, gomas.TRANSB)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - (R.T*Q.T).T||_1: %e\n", M, N, nrm)
}

// m > n: A[m,n], I[m,m] --> A == I*A == Q*Q.T*A
func TestQRTMultLeftIdent(t *testing.T) {
    M := 411
	N := 399
    nb := 16
    A := cmat.NewMatrix(M, N)
    T := cmat.NewMatrix(nb, N)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)
    C  := cmat.NewCopy(A)
    conf := gomas.NewConf()
    conf.LB = nb
    //t.Logf("A0:\n%v\n", A0)

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.QRTFactorWork(A, conf))
    lapackd.QRTFactor(A, T, W, conf)
    //t.Logf("T:\n%v\n", T)

    // C = Q.T*A
    W = lapackd.Workspace(lapackd.QRTMultWork(C, T, gomas.LEFT, conf))
    lapackd.QRTMult(C, A, T, W, gomas.LEFT|gomas.TRANS, conf)
    
    // C = Q*C == Q*Q.T*A
    lapackd.QRTMult(C, A, T, W, gomas.LEFT, conf)
    //t.Logf("A*Q*Q.T:\n%v\n", C)

    // A = A - Q*Q.T*A
    blasd.Plus(A0, C, 1.0, -1.0, gomas.NONE)
	// ||A - Q*Q.T*A||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - Q*Q.T*A||_1: %e\n", M, N, nrm)
}

// m > n: A[m,n], I[m,m] --> A.T == A.T*I == A.T*Q*Q.T
func TestQRTMultRightIdent(t *testing.T) {
    M := 511
	N := 399
    nb := 16
    A := cmat.NewMatrix(M, N)
    C := cmat.NewMatrix(N, M)
    T := cmat.NewMatrix(nb, N)

    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)
    C.Transpose(A)

    conf := gomas.NewConf()
    conf.LB = nb

    // QR = A = Q*R
    W := lapackd.Workspace(lapackd.QRTFactorWork(A, conf))
    lapackd.QRTFactor(A, T, W, conf)

    // C = A*Q
    W = lapackd.Workspace(lapackd.QRTMultWork(C, T, gomas.RIGHT, conf))
    err := lapackd.QRTMult(C, A, T, W, gomas.RIGHT, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }
    // C = C*Q.T == A*Q*Q.T
    err = lapackd.QRTMult(C, A, T, W, gomas.RIGHT|gomas.TRANS, conf)
    if err != nil {
        t.Logf("err: %v\n", err)
    }

    // A = A - (A.T*Q*Q.T).T
    blasd.Plus(A0, C, 1.0, -1.0, gomas.TRANSB)
	// ||A - Q*R||_1
	nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
	t.Logf("M=%d,N=%d  ||A - (A.T*Q*Q.T).T||_1: %e\n", M, N, nrm)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

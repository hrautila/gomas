
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
func TestUnblockedMultQ(t *testing.T) {
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


// QR decompose A, then compute ||A - Q*R||_1, should be small
func TestBlockedMultQ(t *testing.T) {
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


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

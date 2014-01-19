
// Copyright (c) Harri Rautila, 2012-2014

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


// test: unblk.ReduceHess(A) == blk.ReduceHess(A)
func TestReduceHess(t *testing.T) {
	N := 375
    nb := 16

    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)

    A0 := cmat.NewCopy(A)
    tau0 := cmat.NewCopy(tau)

    // blocked reduction
    W := lapackd.Workspace(lapackd.WorksizeHess(A, conf))
    lapackd.ReduceHess(A, tau, W, conf)
    
    // unblocked reduction
    conf.LB = 0
    lapackd.ReduceHess(A0, tau0, W, conf)

    ok := A.AllClose(A0)
    t.Logf("blk.ReduceHess(A) == unblk.ReduceHess(A): %v\n", ok)

    ok = tau0.AllClose(tau)
    t.Logf("blk HessQ.tau == unblk HessQ.tau: %v\n", ok)

    // ||A - A0||_1
    blasd.Plus(A, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A, lapackd.NORM_ONE)
    t.Logf("||H - H0||_1: %e\n", nrm)
}

// test: A - Q*Hess(A)*Q.T  == 0
func TestMultHess(t *testing.T) {
	N := 377
    nb := 16

    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    zeromean := cmat.NewFloatNormSource()
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    // reduction
    W := lapackd.Workspace(lapackd.WorksizeHess(A, conf))
    lapackd.ReduceHess(A, tau, W, conf)

    var Hlow cmat.FloatMatrix
    H := cmat.NewCopy(A)
    
    // set triangular part below first subdiagonal to zeros
    zeros := cmat.NewFloatConstSource(0.0)
    Hlow.SubMatrix(H, 1, 0, N-1, N-1)
    Hlow.SetFrom(zeros, cmat.LOWER|cmat.UNIT)
    H1 := cmat.NewCopy(H)

    // H := Q*H*Q.T
    conf.LB = nb
    lapackd.MultQHess(H, A, tau, W, gomas.LEFT, conf)
    lapackd.MultQHess(H, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)

    // H := Q*H*Q.T
    conf.LB = 0
    lapackd.MultQHess(H1, A, tau, W, gomas.LEFT, conf)
    lapackd.MultQHess(H1, A, tau, W, gomas.RIGHT|gomas.TRANS, conf)

    // compute ||Q*Hess(A)*Q.T - A||_1
    blasd.Plus(H, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(H, lapackd.NORM_ONE)
    t.Logf("  blk.|| Q*Hess(A)*Q.T - A ||_1 : %e\n", nrm)

    blasd.Plus(H1, A0, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(H1, lapackd.NORM_ONE)
    t.Logf("unblk.|| Q*Hess(A)*Q.T - A ||_1 : %e\n", nrm)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

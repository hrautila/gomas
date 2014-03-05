
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



// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

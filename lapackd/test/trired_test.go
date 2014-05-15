
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

func TestTriRedLower(t *testing.T) {
    N := 713
    nb := 32
    conf := gomas.NewConf()
    conf.LB = 0

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src, cmat.LOWER)
    A1 := cmat.NewCopy(A)
    tau1 := cmat.NewCopy(tau)
    _ = A1

    W := lapackd.Workspace(N)
    W1 := lapackd.Workspace(N*nb)

    e := lapackd.TRDReduce(A, tau, W, gomas.LOWER, conf)
    if e != nil {
        t.Logf("unblk.e: %v\n", e)
    }

    conf.LB = nb
    e = lapackd.TRDReduce(A1, tau1, W1, gomas.LOWER, conf)
    if e != nil {
        t.Logf("blk.e: %v\n", e)
    }

    blasd.Plus(A, A1, -1.0, 1.0, gomas.NONE)
    nrm := lapackd.NormP(A, lapackd.NORM_ONE)
    t.Logf("N=%d, ||unblk.Trired(A) - blk.Trired(A)||_1: %e\n", N, nrm)

    blasd.Axpy(tau, tau1, -1.0)
    nrm = blasd.Nrm2(tau)
    t.Logf("   ||unblk.Trired(tau) - blk.Trired(tau)||_1: %e\n", nrm)
}

func TestTriRedUpper(t *testing.T) {
    N := 843
    nb := 48
    conf := gomas.NewConf()
    conf.LB = 0

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src, cmat.UPPER)
    A1 := cmat.NewCopy(A)
    tau1 := cmat.NewCopy(tau)
    _ = A1

    W := lapackd.Workspace(N)
    W1 := lapackd.Workspace(N*nb)

    lapackd.TRDReduce(A, tau, W, gomas.UPPER, conf)

    conf.LB = nb
    lapackd.TRDReduce(A1, tau1, W1, gomas.UPPER, conf)

    blasd.Plus(A, A1, -1.0, 1.0, gomas.NONE)
    nrm := lapackd.NormP(A, lapackd.NORM_ONE)
    t.Logf("N=%d, ||unblk.Trired(A) - blk.Trired(A)||_1: %e\n", N, nrm)
    blasd.Axpy(tau, tau1, -1.0)
    nrm = blasd.Nrm2(tau)
    t.Logf("   ||unblk.Trired(tau) - blk.Trired(tau)||_1: %e\n", nrm)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

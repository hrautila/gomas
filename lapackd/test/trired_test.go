
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
    //"fmt"
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

func TestTrdMultLower(t *testing.T) {
    var dt, et, da, ea cmat.FloatMatrix
    N := 843
    nb := 48
    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    src := cmat.NewFloatNormSource()
    // create symmetric matrix
    A.SetFrom(src, cmat.SYMM)
    A0 := cmat.NewCopy(A)

    W := lapackd.Workspace(lapackd.TRDReduceWork(A, conf))
    lapackd.TRDReduce(A, tau, W, gomas.LOWER, conf)

    // make tridiagonal matrix T
    T0 := cmat.NewMatrix(N, N)
    dt.Diag(T0)
    da.Diag(A)
    blasd.Copy(&dt, &da)
    
    ea.Diag(A, -1)
    et.Diag(T0, -1)
    blasd.Copy(&et, &ea)
    et.Diag(T0, 1)
    blasd.Copy(&et, &ea)
    T1 := cmat.NewCopy(T0)

    // compute Q*T*Q.T (unblocked)
    conf.LB = 0
    lapackd.TRDMult(T0, A, tau, W, gomas.LEFT|gomas.LOWER, conf)
    lapackd.TRDMult(T0, A, tau, W, gomas.RIGHT|gomas.TRANS|gomas.LOWER, conf)

    blasd.Plus(T0, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(T0, lapackd.NORM_ONE)
    t.Logf("N=%d, unblk.||A - Q*T*Q.T||_1: %e\n", N, nrm)
    
    // compute Q*T*Q.T (blocked)
    conf.LB = nb
    lapackd.TRDMult(T1, A, tau, W, gomas.LEFT|gomas.LOWER, conf)
    lapackd.TRDMult(T1, A, tau, W, gomas.RIGHT|gomas.TRANS|gomas.LOWER, conf)

    blasd.Plus(T1, A0, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(T1, lapackd.NORM_ONE)
    t.Logf("N=%d,   blk.||A - Q*T*Q.T||_1: %e\n", N, nrm)
}


func TestTrdMultUpper(t *testing.T) {
    var dt, et, da, ea cmat.FloatMatrix
    N := 843
    nb := 48
    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    src := cmat.NewFloatNormSource()
    // create symmetric matrix
    A.SetFrom(src, cmat.SYMM)
    A0 := cmat.NewCopy(A)

    W := lapackd.Workspace(lapackd.TRDReduceWork(A, conf))
    lapackd.TRDReduce(A, tau, W, gomas.UPPER, conf)
    
    // make tridiagonal matrix T
    T0 := cmat.NewMatrix(N, N)
    dt.Diag(T0)
    da.Diag(A)
    blasd.Copy(&dt, &da)
    
    ea.Diag(A, 1)
    et.Diag(T0, 1)
    blasd.Copy(&et, &ea)
    et.Diag(T0, -1)
    blasd.Copy(&et, &ea)
    T1 := cmat.NewCopy(T0)

    // compute Q*T*Q.T (unblocked)
    conf.LB = 0
    lapackd.TRDMult(T0, A, tau, W, gomas.LEFT|gomas.UPPER, conf)
    lapackd.TRDMult(T0, A, tau, W, gomas.RIGHT|gomas.TRANS|gomas.UPPER, conf)

    blasd.Plus(T0, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(T0, lapackd.NORM_ONE)
    t.Logf("N=%d, unblk.||A - Q*T*Q.T||_1: %e\n", N, nrm)
    
    // compute Q*T*Q.T (blocked)
    conf.LB = nb
    W = lapackd.Workspace(lapackd.TRDMultWork(A, gomas.LEFT|gomas.UPPER, conf))
    lapackd.TRDMult(T1, A, tau, W, gomas.LEFT|gomas.UPPER, conf)
    lapackd.TRDMult(T1, A, tau, W, gomas.RIGHT|gomas.TRANS|gomas.UPPER, conf)

    blasd.Plus(T1, A0, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(T1, lapackd.NORM_ONE)
    t.Logf("N=%d,   blk.||A - Q*T*Q.T||_1: %e\n", N, nrm)
}


// test: ||T - Q.T*A*Q|| == 0
func TestTrdMultALower(t *testing.T) {
    var dt, et, da, ea cmat.FloatMatrix
    N := 843
    nb := 48
    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    src := cmat.NewFloatNormSource()
    // create symmetric matrix
    A.SetFrom(src, cmat.SYMM)
    A0 := cmat.NewCopy(A)
    A1 := cmat.NewCopy(A)

    W := lapackd.Workspace(lapackd.TRDReduceWork(A, conf))
    lapackd.TRDReduce(A, tau, W, gomas.LOWER, conf)

    // make tridiagonal matrix T
    T0 := cmat.NewMatrix(N, N)
    dt.Diag(T0)
    da.Diag(A)
    blasd.Copy(&dt, &da)
    
    ea.Diag(A, -1)
    et.Diag(T0, -1)
    blasd.Copy(&et, &ea)
    et.Diag(T0, 1)
    blasd.Copy(&et, &ea)

    // compute Q.T*A*Q (unblocked)
    conf.LB = 0
    lapackd.TRDMult(A0, A, tau, W, gomas.LEFT|gomas.TRANS|gomas.LOWER, conf)
    lapackd.TRDMult(A0, A, tau, W, gomas.RIGHT|gomas.LOWER, conf)

    blasd.Plus(A0, T0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
    t.Logf("N=%d, unblk.||T - Q.T*A*Q||_1: %e\n", N, nrm)
    
    // compute Q.T*A*Q (blocked)
    conf.LB = nb
    lapackd.TRDMult(A1, A, tau, W, gomas.LEFT|gomas.TRANS|gomas.LOWER, conf)
    lapackd.TRDMult(A1, A, tau, W, gomas.RIGHT|gomas.LOWER, conf)

    blasd.Plus(A1, T0, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("N=%d,   blk.||T - Q.T*A*Q||_1: %e\n", N, nrm)
}

// test: ||T - Q.T*A*Q|| == 0
func TestTrdMultAUpper(t *testing.T) {
    var dt, et, da, ea cmat.FloatMatrix
    N := 843
    nb := 48
    conf := gomas.NewConf()
    conf.LB = nb

    A := cmat.NewMatrix(N, N)
    tau := cmat.NewMatrix(N, 1)
    src := cmat.NewFloatNormSource()
    // create symmetric matrix
    A.SetFrom(src, cmat.SYMM)
    A0 := cmat.NewCopy(A)
    A1 := cmat.NewCopy(A)

    W := lapackd.Workspace(lapackd.TRDReduceWork(A, conf))
    lapackd.TRDReduce(A, tau, W, gomas.UPPER, conf)

    // make tridiagonal matrix T
    T0 := cmat.NewMatrix(N, N)
    dt.Diag(T0)
    da.Diag(A)
    blasd.Copy(&dt, &da)
    
    ea.Diag(A, 1)
    et.Diag(T0, 1)
    blasd.Copy(&et, &ea)
    et.Diag(T0, -1)
    blasd.Copy(&et, &ea)

    // compute Q.T*A*Q (unblocked)
    conf.LB = 0
    lapackd.TRDMult(A0, A, tau, W, gomas.LEFT|gomas.TRANS|gomas.UPPER, conf)
    lapackd.TRDMult(A0, A, tau, W, gomas.RIGHT|gomas.UPPER, conf)

    blasd.Plus(A0, T0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
    t.Logf("N=%d, unblk.||T - Q.T*A*Q||_1: %e\n", N, nrm)
    
    // compute Q.T*A*Q (blocked)
    conf.LB = nb
    lapackd.TRDMult(A1, A, tau, W, gomas.LEFT|gomas.TRANS|gomas.UPPER, conf)
    lapackd.TRDMult(A1, A, tau, W, gomas.RIGHT|gomas.UPPER, conf)

    blasd.Plus(A1, T0, 1.0, -1.0, gomas.NONE)
    nrm = lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("N=%d,   blk.||T - Q.T*A*Q||_1: %e\n", N, nrm)
}




// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

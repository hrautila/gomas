
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

func TestBidiagReduceUnblocked(t *testing.T) {

    N := 217
    M := 269
    conf := gomas.NewConf()
    conf.LB = 0

    zeromean := cmat.NewFloatNormSource()
    A := cmat.NewMatrix(M, N)
    A.SetFrom(zeromean)
    tauq := cmat.NewMatrix(N, 1)
    taup := cmat.NewMatrix(N, 1)

    At := cmat.NewMatrix(N, M)
    blasd.Transpose(At, A)
    tauqt := cmat.NewMatrix(N, 1)
    taupt := cmat.NewMatrix(N, 1)

    W := lapackd.Workspace(M+N)

    lapackd.BDReduce(A, tauq, taup, W, conf)
    lapackd.BDReduce(At, tauqt, taupt, W, conf)

    // BiRed(A) == BiRed(A.T).T
    blasd.Plus(At, A, 1.0, -1.0, gomas.TRANSB)
    blasd.Axpy(tauqt, taup, -1.0)
    blasd.Axpy(taupt, tauq, -1.0)

    nrm := lapackd.NormP(At, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d || BiRed(A) - BiRed(A.T).T||_1 : %e\n", M, N, nrm)
    nrm = lapackd.NormP(taupt, lapackd.NORM_ONE)
    t.Logf("  || BiRed(A).tauq - BiRed(A.T).taup||_1 : %e\n", nrm)
    nrm = lapackd.NormP(tauqt, lapackd.NORM_ONE)
    t.Logf("  || BiRed(A).taup - BiRed(A.T).tauq||_1 : %e\n", nrm)
}

func TestBidiagReduceBlockedTall(t *testing.T) {

    N := 711
    M := 883
    nb := 64
    conf := gomas.NewConf()
    conf.LB = 0

    zeromean := cmat.NewFloatNormSource()
    A := cmat.NewMatrix(M, N)
    A.SetFrom(zeromean)
    tauq := cmat.NewMatrix(N, 1)
    taup := cmat.NewMatrix(N, 1)

    A1 := cmat.NewCopy(A)
    tauq1 := cmat.NewMatrix(N, 1)
    taup1 := cmat.NewMatrix(N, 1)

    W := lapackd.Workspace(M+N)
    W1 := lapackd.Workspace(nb*(M+N+1))

    lapackd.BDReduce(A, tauq, taup, W, conf)
    conf.LB = nb
    lapackd.BDReduce(A1, tauq1, taup1, W1, conf)

    // unblk.BiRed(A) == blk.BiRed(A)
    blasd.Plus(A1, A, 1.0, -1.0, gomas.NONE)
    blasd.Axpy(tauq1, tauq, -1.0)
    blasd.Axpy(taup1, taup, -1.0)

    nrm := lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d || unblk.BiRed(A) - blk.BiRed(A)||_1 : %e\n", M, N, nrm)
    nrm = lapackd.NormP(taup1, lapackd.NORM_ONE)
    t.Logf("  || unblk.BiRed(A).tauq - blk.BiRed(A).taup||_1 : %e\n", nrm)
    nrm = lapackd.NormP(tauq1, lapackd.NORM_ONE)
    t.Logf("  || unblk.BiRed(A).taup - blk.BiRed(A).tauq||_1 : %e\n", nrm)
}


func TestReduceBidiagBlkWide(t *testing.T) {

    N := 911
    M := 823
    nb := 48
    conf := gomas.NewConf()
    conf.LB = 0

    zeromean := cmat.NewFloatNormSource()
    A := cmat.NewMatrix(M, N)
    A.SetFrom(zeromean)
    tauq := cmat.NewMatrix(N, 1)
    taup := cmat.NewMatrix(N, 1)

    A1 := cmat.NewCopy(A)
    tauq1 := cmat.NewMatrix(N, 1)
    taup1 := cmat.NewMatrix(N, 1)

    W := lapackd.Workspace(M+N)

    lapackd.BDReduce(A, tauq, taup, W, conf)
    conf.LB = nb
    W1 := lapackd.Workspace(lapackd.BDReduceWork(A1, conf))
    lapackd.BDReduce(A1, tauq1, taup1, W1, conf)

    // BiRed(A) == BiRed(A.T).T
    blasd.Plus(A1, A, 1.0, -1.0, gomas.NONE)
    blasd.Axpy(tauq1, tauq, -1.0)
    blasd.Axpy(taup1, taup, -1.0)

    nrm := lapackd.NormP(A1, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d || BiRed(A) - blk.BiRed(A)||_1 : %e\n", M, N, nrm)
    nrm = lapackd.NormP(taup1, lapackd.NORM_ONE)
    t.Logf("  || BiRed(A).tauq - blk.BiRed(A).taup||_1 : %e\n", nrm)
    nrm = lapackd.NormP(tauq1, lapackd.NORM_ONE)
    t.Logf("  || BiRed(A).taup - blk.BiRed(A).tauq||_1 : %e\n", nrm)
}

func TestBiredTall(t *testing.T) {
    N := 643
    M := 743
    nb := 32
    conf := gomas.NewConf()
    conf.LB = 0
    ediag := 1

    zeromean := cmat.NewFloatNormSource()
    A := cmat.NewMatrix(M, N)
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    tauq := cmat.NewMatrix(N, 1)
    taup := cmat.NewMatrix(N, 1)

    W := lapackd.Workspace(M+N)
    lapackd.BDReduce(A, tauq, taup, W, conf)

    var D, E, Bd, Be cmat.FloatMatrix
    D.Diag(A)
    E.Diag(A, ediag)
    
    B := cmat.NewMatrix(M, N)
    Bd.Diag(B)
    Be.Diag(B, ediag)
    blasd.Copy(&Bd, &D)
    blasd.Copy(&Be, &E)

    Bt := cmat.NewMatrix(N, M)
    blasd.Transpose(Bt, B)

    conf.LB = nb
    W0 := lapackd.Workspace(lapackd.BDMultWork(B, conf))
    lapackd.BDMult(B, A, tauq, W0, gomas.MULTQ|gomas.LEFT, conf)
    lapackd.BDMult(B, A, taup, W0, gomas.MULTP|gomas.RIGHT|gomas.TRANS, conf)

    lapackd.BDMult(Bt, A, taup, W0, gomas.MULTP|gomas.LEFT, conf)
    lapackd.BDMult(Bt, A, tauq, W0, gomas.MULTQ|gomas.RIGHT|gomas.TRANS, conf)

    blasd.Plus(B, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||A - Q*B*P.T||_1   : %e\n", M, N, nrm)
    blasd.Plus(Bt, A0, 1.0, -1.0, gomas.TRANSB)
    nrm = lapackd.NormP(Bt, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||A.T - P*B.T*Q.T||_1 : %e\n", M, N, nrm)
}


func TestBiredWide(t *testing.T) {
    N := 811
    M := 693
    nb := 32
    conf := gomas.NewConf()
    conf.LB = 0
    ediag := -1

    zeromean := cmat.NewFloatNormSource()
    A := cmat.NewMatrix(M, N)
    A.SetFrom(zeromean)
    A0 := cmat.NewCopy(A)

    tauq := cmat.NewMatrix(N, 1)
    taup := cmat.NewMatrix(N, 1)

    W := lapackd.Workspace(M+N)
    lapackd.BDReduce(A, tauq, taup, W, conf)

    var D, E, Bd, Be cmat.FloatMatrix
    D.Diag(A)
    E.Diag(A, ediag)
    
    B := cmat.NewMatrix(M, N)
    Bd.Diag(B)
    Be.Diag(B, ediag)
    blasd.Copy(&Bd, &D)
    blasd.Copy(&Be, &E)

    Bt := cmat.NewMatrix(N, M)
    blasd.Transpose(Bt, B)

    conf.LB = nb
    W0 := lapackd.Workspace(lapackd.BDMultWork(B, conf))
    lapackd.BDMult(B,  A, tauq, W0, gomas.MULTQ|gomas.LEFT, conf)
    lapackd.BDMult(Bt, A, tauq, W0, gomas.MULTQ|gomas.RIGHT|gomas.TRANS, conf)

    lapackd.BDMult(B,  A, taup, W0, gomas.MULTP|gomas.RIGHT|gomas.TRANS, conf)
    lapackd.BDMult(Bt, A, taup, W0, gomas.MULTP|gomas.LEFT, conf)

    blasd.Plus(B, A0, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||A - Q*B*P.T||_1   : %e\n", M, N, nrm)
    blasd.Plus(Bt, A0, 1.0, -1.0, gomas.TRANSB)
    nrm = lapackd.NormP(Bt, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||A.T - P*B.T*Q.T||_1 : %e\n", M, N, nrm)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


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

func TestReduceBidiag(t *testing.T) {

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

    lapackd.ReduceBidiag(A, tauq, taup, W, conf)
    lapackd.ReduceBidiag(At, tauqt, taupt, W, conf)

    // BiRed(A) == BiRed(A.T).T
    blasd.Plus(At, A, 1.0, -1.0, gomas.TRANSB)
    blasd.Axpy(tauqt, taup, -1.0)
    blasd.Axpy(taupt, tauq, -1.0)

    nrm := lapackd.NormP(At, lapackd.NORM_ONE)
    t.Logf("|| BiRed(A) - BiRed(A.T).T||_1 : %e\n", nrm)
    nrm = lapackd.NormP(taupt, lapackd.NORM_ONE)
    t.Logf("|| BiRed(A).tauq - BiRed(A.T).taup||_1 : %e\n", nrm)
    nrm = lapackd.NormP(tauqt, lapackd.NORM_ONE)
    t.Logf("|| BiRed(A).taup - BiRed(A.T).tauq||_1 : %e\n", nrm)
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
    lapackd.ReduceBidiag(A, tauq, taup, W, conf)

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
    W0 := lapackd.Workspace(lapackd.WorksizeMultQBD(B, conf))
    lapackd.MultQBD(B, A, tauq, W0, gomas.MULTQ|gomas.LEFT, conf)
    lapackd.MultQBD(B, A, taup, W0, gomas.MULTP|gomas.RIGHT|gomas.TRANS, conf)

    lapackd.MultQBD(Bt, A, taup, W0, gomas.MULTP|gomas.LEFT, conf)
    lapackd.MultQBD(Bt, A, tauq, W0, gomas.MULTQ|gomas.RIGHT|gomas.TRANS, conf)

    blasd.Plus(B, A0, 1.0, -1.0, gomas.NONE)
    t.Logf("||A - Q*B*P.T||_1   : %e\n", lapackd.NormP(B, lapackd.NORM_ONE))
    blasd.Plus(Bt, A0, 1.0, -1.0, gomas.TRANSB)
    t.Logf("||A.T - P*B.T*Q.T||_1 : %e\n", lapackd.NormP(Bt, lapackd.NORM_ONE))
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
    lapackd.ReduceBidiag(A, tauq, taup, W, conf)

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
    W0 := lapackd.Workspace(lapackd.WorksizeMultQBD(B, conf))
    lapackd.MultQBD(B,  A, tauq, W0, gomas.MULTQ|gomas.LEFT, conf)
    lapackd.MultQBD(Bt, A, tauq, W0, gomas.MULTQ|gomas.RIGHT|gomas.TRANS, conf)

    lapackd.MultQBD(B,  A, taup, W0, gomas.MULTP|gomas.RIGHT|gomas.TRANS, conf)
    lapackd.MultQBD(Bt, A, taup, W0, gomas.MULTP|gomas.LEFT, conf)

    blasd.Plus(B, A0, 1.0, -1.0, gomas.NONE)
    t.Logf("||A - Q*B*P.T||_1   : %e\n", lapackd.NormP(B, lapackd.NORM_ONE))
    blasd.Plus(Bt, A0, 1.0, -1.0, gomas.TRANSB)
    t.Logf("||A.T - P*B.T*Q.T||_1 : %e\n", lapackd.NormP(Bt, lapackd.NORM_ONE))
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

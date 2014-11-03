
// Copyright (c) Harri Rautila, 2014

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


func testTall(M, N int, square bool, t *testing.T) {
    var A, A0, W, S, U, Uu, V, Vv *cmat.FloatMatrix
    var sD cmat.FloatMatrix
    var s string

    wsize := M*N
    if (wsize < 100) {
        wsize = 100
    }

    S  = cmat.NewMatrix(N, 1)
    A  = cmat.NewMatrix(M, N)
    V  = cmat.NewMatrix(N, N)
    Vv = cmat.NewMatrix(N, N)
    if square {
        U  = cmat.NewMatrix(M, M)
        Uu = cmat.NewMatrix(M, M)
    } else {
        U  = cmat.NewMatrix(M, N)
        Uu = cmat.NewMatrix(N, N)
    }

    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    A0 = cmat.NewCopy(A)
    W  = cmat.NewMatrix(wsize, 1)

    if err := lapackd.SVD(S, U, V, A, W, gomas.WANTU|gomas.WANTV); err != nil {
        t.Errorf("SVD error: %v\n", err)
        return
    }
    
    // ||I - U.T*U||
    sD.Diag(Uu)
    blasd.Mult(Uu, U, U, 1.0, 0.0, gomas.TRANSA)
    blasd.Add(&sD, -1.0)
    nrm0 := lapackd.NormP(Uu, lapackd.NORM_ONE)

    // ||I - V*V.T||
    sD.Diag(Vv)
    blasd.Mult(Vv, V, V, 1.0, 0.0, gomas.TRANSA)
    blasd.Add(&sD, -1.0)
    nrm1 := lapackd.NormP(Vv, lapackd.NORM_ONE)

    if square {
        // left vectors are M-by-M
        Sg := cmat.NewMatrix(M, N)
        A1 := cmat.NewMatrix(M, N)
        sD.Diag(Sg)
        blasd.Copy(&sD, S)
        blasd.Mult(A1, U, Sg, 1.0, 0.0, gomas.NONE)
        blasd.Mult(A0, A1, V, -1.0, 1.0, gomas.NONE)
        s = "U=[m,m], V=[n,n]"
    } else {
        // left vectors are M-by-N
        lapackd.MultDiag(U, S, gomas.RIGHT)
        blasd.Mult(A0, U, V, -1.0, 1.0, gomas.NONE)
        s = "U=[m,n], V=[n,n]"
    }
    nrm2 := lapackd.NormP(A0, lapackd.NORM_ONE)

    t.Logf("M=%d, N=%d, %s ||A - U*S*V.T||_1 :%e\n", M, N, s, nrm2)
    t.Logf("  ||I - U.T*U||_1 : %e\n", nrm0)
    t.Logf("  ||I - V*V.T||_1 : %e\n", nrm1)
}


// test: M < N, U=[m,m] and V=[m,n] or V=[n,n] (square)
func testWide(M, N int, square bool, t *testing.T) {
    var A, A0, W, S, U, Uu, V, Vv *cmat.FloatMatrix
    var sD cmat.FloatMatrix
    var s string

    wsize := M*N
    if (wsize < 100) {
        wsize = 100
    }

    S  = cmat.NewMatrix(M, 1)
    A  = cmat.NewMatrix(M, N)
    U  = cmat.NewMatrix(M, M)
    Uu = cmat.NewMatrix(M, M)
    if square {
        V  = cmat.NewMatrix(N, N)
        Vv = cmat.NewMatrix(N, N)
    } else {
        V  = cmat.NewMatrix(M, N)
        Vv = cmat.NewMatrix(M, M)
    }

    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    A0 = cmat.NewCopy(A)
    W  = cmat.NewMatrix(wsize, 1)

    if err := lapackd.SVD(S, U, V, A, W, gomas.WANTU|gomas.WANTV); err != nil {
        t.Errorf("SVD error: %v\n", err)
        return
    }

    // ||I - U.T*U||
    sD.Diag(Uu)
    blasd.Mult(Uu, U, U, 1.0, 0.0, gomas.TRANSA)
    blasd.Add(&sD, -1.0)
    nrm0 := lapackd.NormP(Uu, lapackd.NORM_ONE)

    // ||I - V*V.T||
    sD.Diag(Vv)
    blasd.Mult(Vv, V, V, 1.0, 0.0, gomas.TRANSB)
    blasd.Add(&sD, -1.0)
    nrm1 := lapackd.NormP(Vv, lapackd.NORM_ONE)

    if square {
        // right vectors are N-by-N
        Sg := cmat.NewMatrix(M, N)
        A1 := cmat.NewMatrix(M, N)
        sD.Diag(Sg)
        blasd.Copy(&sD, S)
        blasd.Mult(A1, Sg, V, 1.0, 0.0, gomas.NONE)
        blasd.Mult(A0, U, A1, -1.0, 1.0, gomas.NONE)
        s = "U=[m,m], V=[n,n]"
    } else {
        // right vectors are M-by-N
        lapackd.MultDiag(V, S, gomas.LEFT)
        blasd.Mult(A0, U, V, -1.0, 1.0, gomas.NONE)
        s = "U=[m,m], V=[m,n]"
    }
    nrm2 := lapackd.NormP(A0, lapackd.NORM_ONE)

    if N < 10 {
        t.Logf("A - U*S*V.T:\n%v\n", A0)
    }
    t.Logf("M=%d, N=%d, %s ||A - U*S*V.T||_1 :%e\n", M, N, s, nrm2)
    t.Logf("  ||I - U.T*U||_1 : %e\n", nrm0)
    t.Logf("  ||I - V*V.T||_1 : %e\n", nrm1)
}

func TestSVDTall(t *testing.T) {
    M := 245
    N := 213
    testTall(M, N, false, t);
    testTall(M, N, true, t);
}

func TestSVDWide(t *testing.T) {
    M := 245
    N := 213
    testWide(N, M, false, t);
    testWide(N, M, true, t);
}

func TestSVDTallSkinny(t *testing.T) {
    M := 313
    N := 101
    testTall(M, N, false, t);
    testTall(M, N, true, t);
}

func TestSVDWideSkinny(t *testing.T) {
    M := 313
    N := 101
    testWide(N, M, false, t);
    testWide(N, M, true, t);
}



// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

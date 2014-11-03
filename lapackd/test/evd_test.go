
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


func testEigen(N int, bits int, t *testing.T) {
    var A, A0, W, D, V *cmat.FloatMatrix
    var sD cmat.FloatMatrix
    var s string = "lower"

    if bits & gomas.UPPER != 0 {
        s = "upper"
    }

    wsize := N*N
    if (wsize < 100) {
        wsize = 100
    }

    D  = cmat.NewMatrix(N, 1)
    A  = cmat.NewMatrix(N, N)
    V  = cmat.NewMatrix(N, N)

    src := cmat.NewFloatNormSource()
    A.SetFrom(src, cmat.SYMM)
    A0 = cmat.NewCopy(A)
    W  = cmat.NewMatrix(wsize, 1)

    if err := lapackd.EigenSym(D, A, W, bits|gomas.WANTV); err != nil {
        t.Errorf("EigenSym error: %v\n", err)
        return
    }
    
    // ||I - V.T*V||
    sD.Diag(V)
    blasd.Mult(V, A, A, 1.0, 0.0, gomas.TRANSA)
    blasd.Add(&sD, -1.0)
    nrm1 := lapackd.NormP(V, lapackd.NORM_ONE)

    // left vectors are M-by-N
    V.Copy(A)
    lapackd.MultDiag(V, D, gomas.RIGHT)
    blasd.Mult(A0, V, A, -1.0, 1.0, gomas.TRANSB)
    nrm2 := lapackd.NormP(A0, lapackd.NORM_ONE)

    t.Logf("N=%d, [%s] ||A - V*D*V.T||_1 :%e\n", N, s, nrm2)
    t.Logf("  ||I - V.T*V||_1 : %e\n", nrm1)
}



func TestEigenLower(t *testing.T) {
    N := 191
    testEigen(N, gomas.LOWER, t);
}

func TestEigenUpper(t *testing.T) {
    N := 191
    testEigen(N, gomas.UPPER, t);
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

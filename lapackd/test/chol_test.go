
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
    //"math"
)

func TestUnblockedDecomposeCHOL(t *testing.T) {
    N := 119
    nb := 0

    conf := gomas.NewConf()
    conf.LB = nb

    Z  := cmat.NewMatrix(N, N)
	AL := cmat.NewMatrix(N, N)
	AU := cmat.NewMatrix(N, N)

    unitrand := cmat.NewFloatUniformSource()
    Z.SetFrom(unitrand)

    blasd.Mult(AL, Z, Z, 1.0, 0.0, gomas.TRANSB)
    AU.Copy(AL)

    eu := lapackd.CHOLFactor(AU, gomas.UPPER, conf)
    el := lapackd.CHOLFactor(AL, gomas.LOWER, conf)
    _, _ = eu, el

    Z.Transpose(AU)
    if N < 10 {
        t.Logf("AU.T:\n%v\n", Z)
        t.Logf("AL:\n%v\n", AL)
    }
    ok := AL.AllClose(Z)
    t.Logf("Decompose(AL) == Decompose(AU).T: %v\n", ok)
}

func TestBlockedDecomposeCHOL(t *testing.T) {
    N := 119
    nb := 16

    conf := gomas.NewConf()
    conf.LB = nb

    Z  := cmat.NewMatrix(N, N)
	AL := cmat.NewMatrix(N, N)
	AU := cmat.NewMatrix(N, N)

    unitrand := cmat.NewFloatUniformSource()
    Z.SetFrom(unitrand)

    blasd.Mult(AL, Z, Z, 1.0, 0.0, gomas.TRANSB)
    AU.Copy(AL)

    eu := lapackd.CHOLFactor(AU, gomas.UPPER, conf)
    el := lapackd.CHOLFactor(AL, gomas.LOWER, conf)
    _, _ = eu, el

    Z.Transpose(AU)
    if N < 10 {
        t.Logf("AU.T:\n%v\n", Z)
        t.Logf("AL:\n%v\n", AL)
    }
    ok := AL.AllClose(Z)
    t.Logf("Decompose(AL) == Decompose(AU).T: %v\n", ok)
}


func TestUpperCHOL(t *testing.T) {
    N := 311
    K := 43
    nb := 0

    conf := gomas.NewConf()
    conf.LB = nb

    Z  := cmat.NewMatrix(N, N)
	A  := cmat.NewMatrix(N, N)
	A0 := cmat.NewMatrix(N, N)
	B  := cmat.NewMatrix(N, K)
	X  := cmat.NewMatrix(N, K)

    unitrand := cmat.NewFloatUniformSource()
    Z.SetFrom(unitrand)

    blasd.Mult(A, Z, Z, 1.0, 0.0, gomas.TRANSB)
    A0.Copy(A)

    B.SetFrom(unitrand)
    X.Copy(B)

    // A = chol(A) = U.T*U
    t.Logf("Unblocked version: nb=%d\n", conf.LB)
    lapackd.CHOLFactor(A, gomas.UPPER, conf)
    // X = A.-1*B = U.-1*(U.-T*B)
    lapackd.CHOLSolve(X, A, gomas.UPPER)
    // B = B - A*X
    blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    // ||B - A*X||_1
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("N=%d:  ||B - A*X||_1: %e\n", N, nrm)

    // A = chol(A) = U.T*U
    A.Copy(A0)
    B.SetFrom(unitrand)
    X.Copy(B)
    conf.LB = 16
    t.Logf("Blocked version: nb=%d\n", conf.LB)
    lapackd.CHOLFactor(A, gomas.UPPER, conf)
    // X = A.-1*B = U.-1*(U.-T*B)
    lapackd.CHOLSolve(X, A, gomas.UPPER)
    // B = B - A*X
    blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    // ||B - A*X||_1
    nrm = lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("N=%d:  ||B - A*X||_1: %e\n", N, nrm)

}


func TestLowerCHOL(t *testing.T) {
    N := 311
	K := 43
    nb := 0

    conf := gomas.NewConf()
    conf.LB = nb

    Z  := cmat.NewMatrix(N, N)
	A  := cmat.NewMatrix(N, N)
	A0 := cmat.NewMatrix(N, N)
	B  := cmat.NewMatrix(N, K)
	X  := cmat.NewMatrix(N, K)

    unitrand := cmat.NewFloatUniformSource()
    Z.SetFrom(unitrand)

    blasd.Mult(A, Z, Z, 1.0, 0.0, gomas.TRANSB)
    A0.Copy(A)

    B.SetFrom(unitrand)
	X.Copy(B)

    // R = chol(A) = L*L.T
    t.Logf("Unblocked version: nb=%d\n", conf.LB)
    lapackd.CHOLFactor(A, gomas.LOWER, conf)
    // X = A.-1*B = L.-T*(L.-1*B)
    lapackd.CHOLSolve(X, A, gomas.LOWER)
    // B = B - A*X
    blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("N=%d:  ||B - A*X||_1: %e\n", N, nrm)


    A.Copy(A0)
    B.SetFrom(unitrand)
	X.Copy(B)
    conf.LB = 16
    // R = chol(A) = L*L.T
    t.Logf("Bblocked version: nb=%d\n", conf.LB)
    lapackd.CHOLFactor(A, gomas.LOWER, conf)
    // X = A.-1*B = L.-T*(L.-1*B)
    lapackd.CHOLSolve(X, A, gomas.LOWER)
    // B = B - A*X
    blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    nrm = lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("N=%d:  ||B - A*X||_1: %e\n", N, nrm)

}



// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

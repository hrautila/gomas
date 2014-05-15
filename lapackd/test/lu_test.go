
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package test

import (
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/lapackd"
	"github.com/hrautila/gomas/blasd"
	"testing"
)

func TestLUnoPivoting(t *testing.T) {
	N := 91
	K := 17
	nb := 0

	A  := cmat.NewMatrix(N, N)
	A0 := cmat.NewMatrix(N, N)
	B  := cmat.NewMatrix(N, K)
    X  := cmat.NewMatrix(N, K)

    unitrand := cmat.NewFloatUniformSource()
    A.SetFrom(unitrand)
    A0.Copy(A)
    B.SetFrom(unitrand)
	X.Copy(B)

    conf := gomas.DefaultConf()
    conf.LB = nb

	// R = lu(A) = P*L*U
	lapackd.LUFactor(A, nil, conf)
	// X = A.-1*B = U.-1*(L.-1*B)
	lapackd.LUSolve(X, A, nil, gomas.NONE)
	// B = B - A*X
	blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("Unblocked version: nb=%d\n", conf.LB)
	t.Logf("N=%d  ||B - A*X||_1: %e\n", N, nrm)

    A.Copy(A0)
    B.SetFrom(unitrand)
	X.Copy(B)
    conf.LB = 16

	// R = lu(A) = P*L*U
	lapackd.LUFactor(A, nil, conf)
	// X = A.-1*B = U.-1*(L.-1*B)
	lapackd.LUSolve(X, A, nil, gomas.NONE)
	// B = B - A*X
	blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    nrm = lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("Blocked version: nb=%d\n", conf.LB)
	t.Logf("N=%d  ||B - A*X||_1: %e\n", N, nrm)
}

func TestLU(t *testing.T) {
	N := 119
	K := 41
	nb := 0

	A  := cmat.NewMatrix(N, N)
	A0 := cmat.NewMatrix(N, N)
	B  := cmat.NewMatrix(N, K)
    X  := cmat.NewMatrix(N, K)

    unitrand := cmat.NewFloatUniformSource()
    A.SetFrom(unitrand)
    A0.Copy(A)
    B.SetFrom(unitrand)
	X.Copy(B)
	piv := lapackd.NewPivots(N)

    conf := gomas.DefaultConf()
    conf.LB = nb

	// R = lu(A) = P*L*U
	lapackd.LUFactor(A, piv, conf)
	// X = A.-1*B = U.-1*(L.-1*B)
	lapackd.LUSolve(X, A, piv, gomas.NONE)
	// B = B - A*X
	blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    nrm := lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("Unblocked decomposition: nb=%d\n", conf.LB)
	t.Logf("N=%d  ||B - A*X||_1: %e\n", N, nrm)


    // blocked
    conf.LB = 16
    A.Copy(A0)
    B.SetFrom(unitrand)
    X.Copy(B)
	// lu(A) = P*L*U
	lapackd.LUFactor(A, piv, conf)
	// X = A.-1*B = U.-1*(L.-1*B)
	lapackd.LUSolve(X, A, piv, gomas.NONE)
	// B = B - A*X
	blasd.Mult(B, A0, X, -1.0, 1.0, gomas.NONE)
    nrm = lapackd.NormP(B, lapackd.NORM_ONE)
    t.Logf("Blocked decomposition: nb=%d\n", conf.LB)
	t.Logf("N=%d  ||B - A*X||_1: %e\n", N, nrm)
}

func TestRowPivot(t *testing.T) {
	N := 7
	K := 3
	B := cmat.NewMatrix(N, K)
    B0:= cmat.NewMatrix(N, K)
    ipv := lapackd.NewPivots(7)
    for k, _ := range ipv {
        ipv[k] = k+1
    }
    ipv[0] = 3
    ipv[3] = 7
    ipv[5] = 7
    t.Logf("pivots: %v\n", ipv)
    rowsetter := func(i, j int, v float64) float64 {
        return float64(i+1)
    }
    B.Map(&cmat.FloatEvaluator{rowsetter})
    B0.Copy(B)

    lapackd.ApplyRowPivots(B, ipv, lapackd.FORWARD)
    t.Logf("pivot forward ...\n")
    lapackd.ApplyRowPivots(B, ipv, lapackd.BACKWARD)
    t.Logf("pivot backward ...\n")
    ok := B.AllClose(B0)
    t.Logf("result is original: %v\n", ok)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/matops package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package util

import (
    "github.com/hrautila/cmat"
    "testing"
    //"fmt"
)

func m(A *cmat.FloatMatrix) int {
    r, _ := A.Size()
    return r
}

func n(A *cmat.FloatMatrix) int {
    _, c := A.Size()
    return c
}

func addConst(A *cmat.FloatMatrix, c float64) {
    adder := func(a float64) float64 {
        return a+c
    }
    A.Map(&cmat.FloatFunction{adder})
}

func TestPartition1H(t *testing.T) {
    var AL, AR, A0, a1, A2 cmat.FloatMatrix
    A := cmat.NewMatrix(1, 6)
    Partition1x2(&AL, &AR, A, 0, PLEFT)
    t.Logf("n(AL)=%d, n(AR)=%d\n", n(&AL), n(&AR))
    for n(&AL) < n(A) {
        addConst(&AR, 1.0)
        t.Logf("n(AR)=%d; %v\n", n(&AR), &AR)
        Repartition1x2to1x3(&AL, &A0, &a1, &A2, A, 1, PRIGHT)
        t.Logf("n(A0)=%d, n(A2)=%d, a1=%.1f\n", n(&A0), n(&A2), a1.Get(0, 0))
        Continue1x3to1x2(&AL, &AR, &A0, &a1, A, PRIGHT)
    }
    t.Logf("A:%v\n", A)
}

func TestPartition2D(t *testing.T) {
    var ATL, ATR, ABL, ABR, As cmat.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 cmat.FloatMatrix

    csource := cmat.NewFloatConstSource(1.0)
    A := cmat.NewMatrix(6, 6)
    As.SubMatrix(A, 1, 1, 4, 4)
    As.SetFrom(csource)

    Partition2x2(&ATL, &ATR, &ABL, &ABR, &As, 0, 0, PTOPLEFT)
    t.Logf("ATL:\n%v\n", &ATL)

    t.Logf("n(ATL)=%d, n(As)=%d\n", n(&ATL), n(&As))
    k := 0
    for n(&ATL) < n(&As) && k < n(&As) {
        Repartition2x2to3x3(&ATL, 
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, &As, 1, PBOTTOMRIGHT)
        t.Logf("n(A00)=%d, n(a01)=%d, n(A02)=%d\n", n(&A00), n(&a01), n(&A02))
        t.Logf("n(a10)=%d, n(a11)=%d, n(a12)=%d\n", n(&a10), n(&a11), n(&a12))
        t.Logf("n(A20)=%d, n(a21)=%d, n(A22)=%d\n", n(&A20), n(&a21), n(&A22))
        //t.Logf("n(a12)=%d [%d], n(a11)=%d\n", n(&a12), a12.Len(), a11.Len())
        a11.Set(0, 0, a11.Get(0, 0)+1.0)
        addConst(&a21, -2.0)

        Continue3x3to2x2(&ATL, &ATR, &ABL, &ABR, &A00, &a11, &A22, &As, PBOTTOMRIGHT)
        t.Logf("n(ATL)=%d, n(As)=%d\n", n(&ATL), n(&As))
        k += 1
    }
    t.Logf("A:\n%v\n", A)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

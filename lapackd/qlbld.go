
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/util"
    //"fmt"
)

/*
 * Unblocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * Parameter nk is last nk elementary reflectors that are not used in computing
 * the matrix Q. Parameter mk length of the first unused elementary reflectors
 * First nk columns are zeroed and subdiagonal mk-nk is set to unit.
 *
 * Compatible with lapack.DORG2L subroutine.
 */
func unblkBuildQL(A, Tvec, W *cmat.FloatMatrix, mk, nk int, mayClear bool) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, a10, a11, a21, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w12, D cmat.FloatMatrix

    // (mk, nk) = (rows, columns) of upper left partition
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, mk, nk, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, nk, util.PTOP)

    // zero the left side
    if nk > 0 && mayClear {
        blasd.Scale(&ABL, 0.0)
        blasd.Scale(&ATL, 0.0)
        D.Diag(&ATL, nk-mk)
        blasd.Add(&D, 1.0)
    }

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, nil,
            &a10, &a11, nil,
            nil,  &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     Tvec, 1, util.PBOTTOM)
        // ------------------------------------------------------
        w12.SubMatrix(W, 0, 0, a10.Len(), 1)
        applyHouseholder2x1(&tau1, &a01, &a10, &A00, &w12, gomas.LEFT)

        blasd.Scale(&a01, -tau1.Get(0, 0))
        a11.Set(0, 0, 1.0 - tau1.Get(0, 0))

        // zero bottom elements
        blasd.Scale(&a21, 0.0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   Tvec, util.PBOTTOM)
    }
}


/*
 * Blocked code for generating M by N matrix Q with orthogonal columns which
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * If the number K of elementary reflectors is not multiple of the blocking
 * factor lb, then unblocked code is used first to generate the upper left corner
 * of the matrix Q. 
 *
 * Compatible with lapack.DORGQL subroutine.
 */
func blkBuildQL(A, Tvec, Twork, W *cmat.FloatMatrix, K, lb int, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR, AL cmat.FloatMatrix
    var A00, A01, A10, A11, A21, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau, t2, Wrk, D, T cmat.FloatMatrix

    nk := n(A) - K
    mk := m(A) - K
    uk := K % lb
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, mk+uk, nk+uk, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, nk+uk, util.PTOP)

    // zero the left side
    if nk+uk > 0 {
        blasd.Scale(&ABL, 0.0)
        if uk > 0 {
            // number of reflectors is not multiple of blocking factor
            // do the first part with unblocked code.
            unblkBuildQL(&ATL, &tT, W, m(&ATL)-uk, n(&ATL)-uk, true)
        } else {
            // blocking factor is multiple of K
            blasd.Scale(&ATL, 0.0)
            D.Diag(&ATL)
            blasd.Add(&D, 1.0)
        }
    }

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            &A10, &A11, nil,
            nil,  &A21, &A22,   A, lb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau,
            &t2,     Tvec, lb, util.PBOTTOM)
        // ------------------------------------------------------
        util.Merge2x1(&AL, &A01, &A11)

        // build block reflector
        T.SubMatrix(Twork, 0, 0, n(&A11), n(&A11))
        unblkQLBlockReflector(&T, &AL, &tau)

        // update left side i.e. A10 and A00 with (I - Y*T*Y.T)
        ar, ac := A10.Size()
        Wrk.SubMatrix(W, 0, 0, ac, ar)
        updateQLLeft(&A10, &A00, &A11, &A01, &T, &Wrk, false, conf)
        
        // update current block
        unblkBuildQL(&AL, &tau, W, m(&A01), 0, false)

        // zero bottom rows
        blasd.Scale(&A21, 0.0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau,   Tvec, util.PBOTTOM)
    }

}

/*
 * Generate the M by N matrix Q with orthogonal columns which
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * Arguments
 *   A     On entry, the elementary reflectors as returned by DecomposeQR().
 *         stored below diagonal of the M by N matrix A.
 *         On exit, the orthogonal matrix Q
 *
 *   tau   Scalar coefficents of elementary reflectors
 *
 *   W     Workspace
 *
 *   K     The number of elementary reflector whose product define the matrix Q
 *
 * Compatible with lapackd.ORGQL.
 */
func QLBuild(A, tau, W *cmat.FloatMatrix, K int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    if K <= 0 || K > n(A) {
        return gomas.NewError(gomas.EVALUE, "BuildQL", K)
    }
    wsz := wsBuildQL(A, 0)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "BuildQL", wsz)
    }

    // adjust blocking factor for workspace size
    lb := estimateLB(A, W.Len(), wsBuildQL)
    lb = imin(lb, conf.LB)

    if lb == 0 || n(A) <= lb {
        unblkBuildQL(A, tau, W, m(A)-K, n(A)-K, true)
    } else {
        var Twork, Wrk cmat.FloatMatrix
        Twork.SetBuf(lb, lb, lb, W.Data())
        Wrk.SetBuf(n(A)-lb, lb, n(A)-lb, W.Data()[Twork.Len():])
        blkBuildQL(A, tau, &Twork, &Wrk, K, lb, conf)
    }
    return err
}


func wsBuildQL(A *cmat.FloatMatrix, lb int) int {
    if lb > 0 {
        return lb*n(A)
    }
    return n(A)
}

func QLBuildWork(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsBuildQL(A, conf.LB)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

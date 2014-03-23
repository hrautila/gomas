
// Copyright (c) Harri Rautila, 2013

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
 * are defined as the first N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(k)H(k-1)...H(1)  , 0 < k <= M, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(k)*H(k-1)...*H(1)*I ie. from bottom to top.
 *
 * If k < M rows k+1:M are cleared and diagonal entries [k+1:M,k+1:M] are
 * set to unit. Then the matrix Q is generated by right multiplying elements below
 * of i'th elementary reflector H(i).
 * 
 * Compatible to lapack.xORG2L subroutine.
 */
func unblkBuildLQ(A, Tvec, W *cmat.FloatMatrix, mk, nk int, mayClear bool) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, a12, a21, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w12, D cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, mk, nk, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, mk, util.PBOTTOM)

    // zero the bottom part
    if mk > 0 && mayClear {
        blasd.Scale(&ABL, 0.0)
        blasd.Scale(&ABR, 0.0)
        D.Diag(&ABR)
        blasd.Add(&D, 1.0)
    }

    for m(&ATL) > 0 && n(&ATL) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, &a12,
            nil,  &a21, &A22,   A, 1, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     Tvec, 1, util.PTOP)
        // ------------------------------------------------------

        w12.SubMatrix(W, 0, 0, a21.Len(), 1)
        applyHouseholder2x1(&tau1, &a12, &a21, &A22, &w12, gomas.RIGHT)

        blasd.Scale(&a12, -tau1.Get(0, 0))
        a11.Set(0, 0, 1.0 - tau1.Get(0, 0))

        // zero 
        blasd.Scale(&a10, 0.0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   Tvec, util.PTOP)
    }
}

func blkBuildLQ(A, Tvec, Twork, W *cmat.FloatMatrix, K, lb int, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR, AL cmat.FloatMatrix
    var A00, A10, A11, A12, A21, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau, t2, Wrk, D, T cmat.FloatMatrix

    nk := n(A) - K
    mk := m(A) - K
    uk := K % lb
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, mk+uk, nk+uk, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, mk+uk, util.PBOTTOM)

    // zero the bottom part
    if nk+uk > 0 {
        blasd.Scale(&ABL, 0.0)
        if uk > 0 {
            // number of reflectors is not multiple of blocking factor
            // do the first part with unblocked code.
            unblkBuildLQ(&ABR, &tB, W, m(&ABR)-uk, n(&ABR)-uk, true)
        } else {
            // blocking factor is multiple of K
            blasd.Scale(&ABR, 0.0)
            D.Diag(&ABR)
            blasd.Add(&D, 1.0)
        }
    }

    for m(&ATL) > 0 && n(&ATL) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, &A12,
            nil,  &A21, &A22,   A, lb, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau,
            &t2,     Tvec, lb, util.PTOP)
        // ------------------------------------------------------
        util.Merge1x2(&AL, &A11, &A12)

        // build block reflector
        T.SubMatrix(Twork, 0, 0, n(&A11), n(&A11))
        unblkBlockReflectorLQ(&T, &AL, &tau)

        // update A21 and A22 with (I - Y*T*Y.T) from right
        ar, ac := A21.Size()
        Wrk.SubMatrix(W, 0, 0, ar, ac)
        updateRightLQ(&A21, &A22, &A11, &A12, &T, &Wrk, false, conf)
        
        // update current block
        unblkBuildLQ(&AL, &tau, W, 0, n(&A12), false)

        // zero top rows
        blasd.Scale(&A10, 0.0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau,   Tvec, util.PTOP)
    }
}

/*
 * Generate the M by N matrix Q with orthogonal rows which
 * are defined as the first M rows of the product of K first elementary
 * reflectors.
 *
 * Arguments
 *   A     On entry, the elementary reflectors as returned by DecomposeLQ().
 *         stored right of diagonal of the M by N matrix A.
 *         On exit, the orthogonal matrix Q
 *
 *   tau   Scalar coefficents of elementary reflectors
 *
 *   W     Workspace
 *
 *   K     The number of elementary reflector whose product define the matrix Q
 *
 *   conf  Optional blocking configuration.
 *
 * Compatible with lapackd.ORGLQ.
 */
func BuildLQ(A, tau, W *cmat.FloatMatrix, K int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    if K <= 0 || K > n(A) {
        return gomas.NewError(gomas.EVALUE, "BuildLQ", K)
    }
    wsz := wsBuildLQ(A, 0)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "BuildLQ", wsz)
    }

    // adjust blocking factor for workspace size
    lb := estimateLB(A, W.Len(), wsBuildLQ)
    //lb = imin(lb, conf.LB)
    lb = conf.LB
    if lb == 0 || m(A) <= lb {
        unblkBuildLQ(A, tau, W, m(A)-K, n(A)-K, true)
    } else {
        var Twork, Wrk cmat.FloatMatrix
        Twork.SetBuf(lb, lb, lb, W.Data())
        Wrk.SetBuf(m(A)-lb, lb, m(A)-lb, W.Data()[Twork.Len():])
        blkBuildLQ(A, tau, &Twork, &Wrk, K, lb, conf)
    }
    return err
}

func wsBuildLQ(A *cmat.FloatMatrix, lb int) int {
    if lb > 0 {
        return lb*m(A)
    }
    return m(A)
}

func WorksizeBuildLQ(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsBuildLQ(A, conf.LB)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

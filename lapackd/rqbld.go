
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
 * are defined as the last N columns of the product of K first elementary
 * reflectors.
 *
 * Parameters nk = n(A)-K, mk = m(A)-K define the initial partitioning of
 * matrix A.
 *
 *  Q = H(0)H(2)...H(k-1)  , 0 < k < M, where H(i) = I - tau*v*v.T
 *
 * Computation is ordered as H(0)*H(1)...*H(k-1)*I ie. from top to bottom.
 *
 * Compatible to lapack.xORG2R subroutine.
 */
func unblkBuildRQ(A, Tvec, W *cmat.FloatMatrix, mk, nk int, mayClear bool) {
    var ATL, ATR, ABR cmat.FloatMatrix
    var A00, a01, a10, a11, a12, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w12, D cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        nil,  &ABR,   /**/ A, mk, nk, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,    /**/ Tvec, mk, util.PTOP)

    // zero the top part
    if mk > 0 && mayClear {
        blasd.Scale(&ATL, 0.0)
        blasd.Scale(&ATR, 0.0)
        D.Diag(&ATL, n(&ATL)-mk)
        blasd.Add(&D, 1.0)
    }

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, nil,
            &a10, &a11, &a12,
            nil,  nil,  &A22,   /**/ A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     /**/ Tvec, 1, util.PBOTTOM)
        // ------------------------------------------------------

        w12.SubMatrix(W, 0, 0, a01.Len(), 1)
        applyHouseholder2x1(&tau1, &a10, &a01, &A00, &w12, gomas.RIGHT)

        blasd.Scale(&a10, -tau1.Get(0, 0))
        a11.Set(0, 0, 1.0 - tau1.Get(0, 0))

        // zero 
        blasd.Scale(&a12, 0.0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            nil,  &ABR,  /**/ &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,  /**/ &t0, &tau1,   Tvec, util.PBOTTOM)
    }
}

func blkBuildRQ(A, Tvec, Twork, W *cmat.FloatMatrix, K, lb int, conf *gomas.Config) {
    var ATL, ATR, ABR, AL cmat.FloatMatrix
    var A00, A01, A10, A11, A12, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau, t2, Wrk, D, T cmat.FloatMatrix

    nk := n(A) - K
    mk := m(A) - K
    uk := K % lb
    util.Partition2x2(
        &ATL, &ATR,
        nil,  &ABR,   /**/ A, mk+uk, nk+uk, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,    /**/ Tvec, mk+uk, util.PTOP)

    // zero the bottom part __CHECK HERE: nk? or mk?
    if nk+uk > 0 {
        blasd.Scale(&ATR, 0.0)
        if uk > 0 {
            // number of reflectors is not multiple of blocking factor
            // do the first part with unblocked code.
            unblkBuildRQ(&ATL, &tT, W, m(&ATL)-uk, n(&ATL)-uk, true)
        } else {
            // blocking factor is multiple of K
            blasd.Scale(&ATL, 0.0)
            D.Diag(&ATL, n(&ATL)-m(&ATL))
            blasd.Add(&D, 1.0)
        }
    }

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            &A10, &A11, &A12,
            nil,  nil,  &A22,   /**/ A, lb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau,
            &t2,    /**/ Tvec, n(&A11), util.PBOTTOM)
        // ------------------------------------------------------
        util.Merge1x2(&AL, &A10, &A11)

        // build block reflector
        T.SubMatrix(Twork, 0, 0, n(&A11), n(&A11))
        unblkBlockReflectorRQ(&T, &AL, &tau)

        // update A00 and A01 with (I - Y*T*Y.T) from right
        ar, ac := A01.Size()
        Wrk.SubMatrix(W, 0, 0, ar, ac)
        updateRightRQ(&A01, &A00, &A11, &A10, &T, &Wrk, true, conf)
        
        // update current block
        unblkBuildRQ(&AL, &tau, W, 0, n(&A10), false)

        // zero top rows
        blasd.Scale(&A12, 0.0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            nil,  &ABR,  /**/ &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   /**/ &t0, &tau,   Tvec, util.PBOTTOM)
    }
}

/*
 * Generate the M by N matrix Q with orthogonal rows which
 * are defined as the last M rows of the product of K last elementary
 * reflectors.
 *
 * Arguments
 *   A     On entry, the elementary reflectors as returned by RQFactor().
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
 * Compatible with lapackd.ORGRQ.
 */
func RQBuild(A, tau, W *cmat.FloatMatrix, K int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    if K <= 0 || K > n(A) {
        return gomas.NewError(gomas.EVALUE, "RQBuild", K)
    }
    wsz := wsBuildRQ(A, 0)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "RQBuild", wsz)
    }

    // adjust blocking factor for workspace size
    lb := estimateLB(A, W.Len(), wsBuildRQ)
    //lb = imin(lb, conf.LB)
    lb = conf.LB
    if lb == 0 || m(A) <= lb {
        unblkBuildRQ(A, tau, W, m(A)-K, n(A)-K, true)
    } else {
        var Twork, Wrk cmat.FloatMatrix
        Twork.SetBuf(lb, lb, lb, W.Data())
        Wrk.SetBuf(m(A)-lb, lb, m(A)-lb, W.Data()[Twork.Len():])
        blkBuildRQ(A, tau, &Twork, &Wrk, K, lb, conf)
    }
    return err
}

func wsBuildRQ(A *cmat.FloatMatrix, lb int) int {
    if lb > 0 {
        return lb*m(A)
    }
    return m(A)
}

func RQBuildWork(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsBuildRQ(A, conf.LB)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

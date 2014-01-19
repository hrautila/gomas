
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
 * Unblocked QR decomposition with block reflector T.
 */
func unblockedQRT(A, T, W *cmat.FloatMatrix) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, a12, A20, a21, A22 cmat.FloatMatrix
    var TTL, TTR, TBL, TBR cmat.FloatMatrix
    var T00, t01, T02, t11, t12, T22, w12 cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &TTL, &TTR,
        &TBL, &TBR, T, 0, 0, util.PTOPLEFT)

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, &a12,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&TTL,
            &T00, &t01, &T02,
            nil,  &t11, &t12,
            nil,  nil,  &T22,   T, 1, util.PBOTTOMRIGHT)

        // ------------------------------------------------------

        computeHouseholder(&a11, &a21, &t11)

        // H*[a12 A22].T
        w12.SubMatrix(W, 0, 0, a12.Len(), 1)
        applyHouseholder2x1(&t11, &a21, &a12, &A22, &w12, gomas.LEFT)
        
        // update T
        tauval := t11.Get(0, 0)
        if tauval != 0.0 {
            // t01 := -tauval*(a10.T + &A20.T*a21)
            //a10.CopyTo(&t01)
            blasd.Axpby(&t01, &a10, 1.0, 0.0)
            blasd.MVMult(&t01, &A20, &a21, -tauval, -tauval, gomas.TRANSA)
            // t01 := T00*t01
            blasd.MVMultTrm(&t01, &T00, 1.0, gomas.UPPER)
        }

        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &TTL, &TTR,
            &TBL, &TBR,   &T00, &t11, &T22,   T, util.PBOTTOMRIGHT)
    }
    return err
}


func blockedQRT(A, T, W *cmat.FloatMatrix, conf *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR, AL, AR cmat.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 cmat.FloatMatrix
    var TL, TR, W2 cmat.FloatMatrix
    var T00, T01, T02 cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition1x2(
        &TL, &TR,  T, 0, util.PLEFT)

    nb := conf.LB
    for m(&ABR)-nb > 0 && n(&ABR)-nb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22,   A, nb, util.PBOTTOMRIGHT)
        util.Repartition1x2to1x3(&TL,
            &T00, &T01, &T02,   T, nb, util.PRIGHT)
        util.Partition1x2(
            &AL, &AR,    &ABR, nb, util.PLEFT)
        // --------------------------------------------------------
        // decompose left side AL == /A11\ 
        //                           \A21/
        unblockedQRT(&AL, &T01, W)

        // update A'tail i.e. A12 and A22 with (I - Y*T*Y.T).T * A'tail
        // compute: Q*T.C == C - Y*(C.T*Y*T).T
        ar, ac := A12.Size()
        W2.SubMatrix(W, 0, 0, ac, ar)
        updateWithQTLeft(&A12, &A22, &A11, &A21, &T01, &W2, true, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue1x3to1x2(
            &TL, &TR, &T00, &T01, T, util.PRIGHT)
    }
    if m(&ABR) > 0 && n(&ABR) > 0 {
        T01.SubMatrix(&TR, 0, 0, n(&ABR), n(&ABR))
        unblockedQRT(&ABR, &T01, W)
    }
    return err
}

// update T: T = -T1*Y1.T*Y2*T2 
//  Y1 = /Y10\   Y2 = /Y11\
//       \Y20/        \Y21/
//
//  T = -T1 * [Y10.T*Y11 + Y20.T*Y21]*T2
//  
//  T1 is K*K triangular upper matrix
//  T2 is nb*nb triangular upper matrix
//  T  is K*nb block matrix
//  Y10 is nb*K block matrix
//  Y20 is M-K-nb*K block matrix
//  Y11 is nb*nb triangular lower unit diagonal matrix
//  Y21 is M-K-nb*nb block matrix
//  
func updateQRTReflector(T, Y10, Y20, Y11, Y21, T1, T2 *cmat.FloatMatrix, conf *gomas.Config ) {
    // T = Y10.T
    if n(Y10) == 0 {
        return
    }
    // T = Y10.T
    blasd.Plus(T, Y10, 0.0, 1.0, gomas.TRANSB)
    // T = Y10.T*Y11
    blasd.MultTrm(T, Y11, 1.0, gomas.LOWER|gomas.UNIT|gomas.RIGHT, conf)
    // T = T + Y20.T*Y21
    blasd.Mult(T, Y20, Y21, 1.0, 1.0, gomas.TRANSA, conf)
    // -- here: T == Y1.T*Y2

    // T = -T1*T
    blasd.MultTrm(T, T1, -1.0, gomas.UPPER, conf)
    // T = T*T2
    blasd.MultTrm(T, T2, 1.0, gomas.UPPER|gomas.RIGHT, conf)
}

/*
 * Build full block reflect T for nc columns from sequence of reflector stored in S.
 * Reflectors in S are the diagonal of T, off-diagonal values of reflector are computed
 * from elementary reflector store in lower triangular part of A. 
 */
func buildQRTReflector(T, A, S *cmat.FloatMatrix, nc int, conf *gomas.Config) *gomas.Error {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A10, A11, A20, A21, A22 cmat.FloatMatrix
    var TTL, TTR , TBL, TBR cmat.FloatMatrix
    var T00, T01, T02, T11, T12, T22 cmat.FloatMatrix
    var SL, SR cmat.FloatMatrix
    var S00, S01, S02 cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &TTL, &TTR,
        &TBL, &TBR, T, 0, 0, util.PTOPLEFT)
    util.Partition1x2(
        &SL, &SR,  S, 0, util.PLEFT)

    nb := conf.LB
    for m(&ABR)-nb > 0 && n(&ABR)-nb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&TTL,
            &T00, &T01, &T02,
            nil,  &T11, &T12,
            nil,  nil,  &T22,   T, nb, util.PBOTTOMRIGHT)
        util.Repartition1x2to1x3(&SL,
            &S00, &S01, &S02,   S, nb, util.PRIGHT)
        // --------------------------------------------------------
        // update T01: T01 = -T00*Y1.T*Y2*T11 
        //  Y1 = /A10\   Y2 = /A11\
        //       \A20/        \A21/
        //
        T11.Copy(&S01)
        updateQRTReflector(&T01, &A10, &A20, &A11, &A21, &T00, &S01, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &TTL, &TTR,
            &TBL, &TBR,   &T00, &T11, &T22,   T, util.PBOTTOMRIGHT)
        util.Continue1x3to1x2(
            &SL, &SR, &S00, &S01, S, util.PRIGHT)
    }
    if m(&ABR) > 0 && n(&ABR) > 0 {
    }
    return nil
}


/*
 * Compute QR factorization of a M-by-N matrix A using compact WY transformation: A = Q * R,
 * where Q = I - Y*T*Y.T, T is block reflector and Y holds elementary reflectors as lower
 * trapezoidal matrix saved below diagonal elements of the matrix A.
 *
 * Arguments:
 *  A    On entry, the M-by-N matrix A. On exit, the elements on and above
 *       the diagonal contain the min(M,N)-by-N upper trapezoidal matrix R.
 *       The elements below the diagonal with the matrix 'T', represent
 *       the ortogonal matrix Q as product of elementary reflectors.
 *
 * T     On exit, the K block reflectors which, together with trilu(A) represent
 *       the ortogonal matrix Q as Q = I - Y*T*Y.T where Y = trilu(A).
 *       K is ceiling(N/LB) where LB is blocking size from used blocking configuration.
 *       The matrix T is LB*N augmented matrix of K block reflectors, T = [T(0) T(1) .. T(K-1)].
 *       Block reflector T(n) is LB*LB matrix, expect reflector T(K-1) that is IB*IB matrix
 *       where IB = min(LB, K % LB)
 *
 * W     Workspace, required size returned by WorksizeQRT().
 *
 * conf  Optional blocking configuration. If not provided then default configuration
 *       is used.
 *
 * Returns:
 *      Error indicator.
 *
 * DecomposeQRT is compatible with lapack.DGEQRT
 */
func DecomposeQRT(A, T, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf  := gomas.CurrentConf(confs...)
    ok    := false
    rsize := 0
    wsz   := WorksizeQRT(A, conf)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "DecomposeQRT", wsz)
    }

    tr, tc := T.Size()
    if conf.LB == 0 || conf.LB > n(A) {
        ok = tr == tc && tr == n(A)
        rsize = n(A)*n(A)
    } else {
        ok = tr == conf.LB && tc == n(A)
        rsize = conf.LB*n(A)
    }
    if !ok {
        return gomas.NewError(gomas.ESMALL, "DecomposeQRT", rsize)
    }

    if conf.LB == 0 || n(A) <= conf.LB {
        err = unblockedQRT(A, T, W)
    } else {
        Wrk := cmat.MakeMatrix(n(A), conf.LB, W.Data())
        err = blockedQRT(A, T, Wrk, conf)
    }
    return err
}

/*
 * Calculate required workspace to decompose matrix A using compact WY transformation.
 * If blocking configuration is not provided then default configuation will be used.
 *
 * Returns size of workspace as number of elements.
 */
func WorksizeQRT(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    sz := n(A)
    if conf.LB > 0 && n(A) > conf.LB {
        sz *= conf.LB
    }
    return sz
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

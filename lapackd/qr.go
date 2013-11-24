
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
 * Unblocked QR decomposition. As implemented
 * in lapack.xGEQR2 subroutine.
 */
func unblockedQR(A, Tvec, W *cmat.FloatMatrix) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w12  cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, 0, util.PTOP)

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     Tvec, 1, util.PBOTTOM)
        // ------------------------------------------------------
        computeHouseholder(&a11, &a21, &tau1, gomas.LEFT)

        w12.SubMatrix(W, 0, 0, a12.Len(), 1)
        applyHouseholder2x1(&tau1, &a21, &a12, &A22, &w12, gomas.LEFT)
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
 * Blocked QR decomposition with compact WY transform. As implemented
 * in lapack.xGEQRF subroutine.
 */
func blockedQR(A, Tvec, Twork, W *cmat.FloatMatrix, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR, AL, AR cmat.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 cmat.FloatMatrix
    var TT, TB cmat.FloatMatrix
    var t0, tau, t2 cmat.FloatMatrix
    var Wrk, w1 cmat.FloatMatrix
    
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &TT,
        &TB,  Tvec, 0, util.PTOP)

    nb := conf.LB
    for m(&ABR)-nb > 0 && n(&ABR)-nb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22,   A, nb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&TT,
            &t0,
            &tau,
            &t2,     Tvec, nb, util.PBOTTOM)
        util.Partition1x2(
            &AL, &AR,    &ABR, nb, util.PLEFT)

        // current block size
        cb, rb := A11.Size()
        if rb < cb {
            cb = rb
        }
        // --------------------------------------------------------
        // decompose left side AL == /A11\ 
        //                           \A21/
        w1.SubMatrix(W, 0, 0, cb, 1)
        unblockedQR(&AL, &tau, &w1)

        // build block reflector
        unblkQRBlockReflector(Twork, &AL, &tau)

        // update A'tail i.e. A12 and A22 with (I - Y*T*Y.T).T * A'tail
        // compute: C - Y*(C.T*Y*T).T
        ar, ac := A12.Size()
        Wrk.SubMatrix(W, 0, 0, ac, ar)
        updateWithQTLeft(&A12, &A22, &A11, &A21, Twork, &Wrk, cb, true, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &TT,
            &TB,   &t0, &tau,   Tvec, util.PBOTTOM)
    }

    // last block with unblocked
    if m(&ABR) > 0 && n(&ABR) > 0 {
        w1.SubMatrix(W, 0, 0, n(&ABR), 1)
        unblockedQR(&ABR, &t2, &w1)
    }
}

// compute:
//      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T 
// or
//      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T 
//
//
// where  C = /C1\   Y = /Y1\
//            \C2/       \Y2/
//
// C1 is nb*K, C2 is P*K, Y1 is nb*nb trilu, Y2 is P*nb, T is nb*nb
// W = K*nb
func updateWithQTLeft(C1, C2, Y1, Y2, T, W *cmat.FloatMatrix, nb int, transpose bool, conf *gomas.Config) {
    // W = C1.T
    blasd.Plus(W, C1, 0.0, 1.0, gomas.TRANSB)
    // W = C1.T*Y1
    blasd.MultTrm(W, Y1, 1.0, gomas.LOWER|gomas.UNIT|gomas.RIGHT, conf)
    // W = W + C2.T*Y2
    blasd.Mult(W, C2, Y2, 1.0, 1.0, gomas.TRANSA, conf)
    // --- here: W == C.T*Y ---

    tflags := gomas.UPPER|gomas.RIGHT 
    if ! transpose {
        tflags |= gomas.TRANSA
    }
    // W = W*T or W*T.T
    blasd.MultTrm(W, T, 1.0, tflags, conf)
    // --- here: W == C.T*Y*T or C.T*Y*T.T ---

    // C2 = C2 - Y2*W.T
    blasd.Mult(C2, Y2, W, -1.0, 1.0, gomas.TRANSB, conf)
    //  W = Y1*W.T ==> W.T = W*Y1.T
    blasd.MultTrm(W, Y1, 1.0, gomas.LOWER|gomas.UNIT|gomas.TRANSA|gomas.RIGHT, conf)
    
    // C1 = C1 - W.T
    blasd.Plus(C1, W, 1.0, -1.0, gomas.TRANSB)
    // --- here: C = (I - Y*T*Y.T).T * C ---
}

// compute:
//      C*Q.T = C*(I -Y*T*Y.T).T ==  C - C*Y*T.T*Y.T
// or
//      C*Q   = (I -Y*T*Y.T)*C   ==  C - C*Y*T*Y.T
//
//
// where  C = ( C1 C2 )   Y = ( Y1 )
//                            ( Y2 )
//
// C1 is K*nb, C2 is K*P, Y1 is nb*nb trilu, Y2 is P*nb, T is nb*nb
// W = K*nb
func updateWithQTRight(C1, C2, Y1, Y2, T, W *cmat.FloatMatrix, nb int, transpose bool, conf *gomas.Config) {
    // -- compute: W = C*Y = C1*Y1 + C2*Y2

    // W = C1
    blasd.Plus(W, C1, 0.0, 1.0, gomas.NONE)
    // W = C1*Y1
    blasd.MultTrm(W, Y1, 1.0, gomas.LOWER|gomas.UNIT|gomas.RIGHT, conf)
    // W = W + C2*Y2
    blasd.Mult(W, C2, Y2, 1.0, 1.0, gomas.NONE, conf)
    // --- here: W == C*Y ---

    tflags := gomas.UPPER|gomas.RIGHT 
    if transpose {
        tflags |= gomas.TRANSA
    }
    // W = W*T or W*T.T
    blasd.MultTrm(W, T, 1.0, tflags, conf)
    // --- here: W == C*Y*T or C*Y*T.T ---

    // C2 = C2 - W*Y2.T
    blasd.Mult(C2, W, Y2, -1.0, 1.0, gomas.TRANSB, conf)
    // C1 = C1 - W*Y1.T
    //  W = W*Y1 
    blasd.MultTrm(W, Y1, 1.0, gomas.LOWER|gomas.UNIT|gomas.RIGHT|gomas.TRANSA, conf)
    // C1 = C1 - W
    blasd.Plus(C1, W, 1.0, -1.0, gomas.NONE)
    // --- here: C = (I - Y*T*Y.T).T * C ---
}

 /*
  * Compute QR factorization of a M-by-N matrix A: A = Q * R.
  *
  * Arguments:
  *  A   On entry, the M-by-N matrix A. On exit, the elements on and above
  *      the diagonal contain the min(M,N)-by-N upper trapezoidal matrix R.
  *      The elements below the diagonal with the column vector 'tau', represent
  *      the ortogonal matrix Q as product of elementary reflectors.
  *
  * tau  On exit, the scalar factors of the elemenentary reflectors.
  *
  * W    Workspace, N-by-nb matrix used for work space in blocked invocations. 
  *
  * conf The blocking configuration. If nil then default blocking configuration
  *      is used. Member conf.LB defines blocking size of blocked algorithms.
  *      If it is zero then unblocked algorithm is used.
  *
  * Returns:
  *      Error indicator.
  *
  * DecomposeQR is compatible with lapack.DGEQRF
  */
func DecomposeQR(A, tau, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    wsz := WorksizeQR(A, conf)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "DecomposeQR", wsz)
    }

    if conf.LB == 0 || n(A) <= conf.LB {
        unblockedQR(A, tau, W)
    } else {
        var Twork, Wrk cmat.FloatMatrix
        // block reflector T in first LB*LB elements in workspace
        // the rest, n(A)-LB*LB, is workspace for intermediate matrix operands
        Twork.SetBuf(conf.LB, conf.LB, -1, W.Data())
        Wrk.SetBuf(n(A)-conf.LB, conf.LB, -1, W.Data()[Twork.Len():])
        blockedQR(A, tau, &Twork, &Wrk, conf)
    }
    return err
}

/*
 * Allocate a workspace with of given size.
 */
func Workspace(sz int) *cmat.FloatMatrix {
    return cmat.NewMatrix(sz, 1)
}

/*
 * Calculate required workspace to decompose matrix A with current blocking configuration.
 * If blocking configuration is not provided then default configuation will be used.
 *
 * Returns size of workspace as number of elements.
 */
func WorksizeQR(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    sz := n(A)
    if conf.LB > 0 && n(A) > conf.LB {
        sz *= conf.LB
    }
    return sz
}

/*
 * like LAPACK/dlafrt.f
 *
 * Build block reflector T from HH reflector stored in TriLU(A) and coefficients
 * in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  z |   z = -tau*T*Y.T*v
 *     | 0  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 */
func unblkQRBlockReflector(T, A, tau *cmat.FloatMatrix) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, A20, a21, A22 cmat.FloatMatrix
    var TTL, TTR, TBL, TBR cmat.FloatMatrix
    var T00, t01, T02, t11, t12, T22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &TTL, &TTR,
        &TBL, &TBR, T, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,  tau, 0, util.PTOP)

    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&TTL,
            &T00, &t01, &T02,
            nil,  &t11, &t12,
            nil,  nil,  &T22,   T, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, util.PBOTTOM)
        // --------------------------------------------------

        // t11 := tau
        tauval := tau1.Get(0, 0)
        if tauval != 0.0 {
            t11.Set(0, 0, tauval)

            // t01 := -tauval*(a10.T + &A20.T*a21)
            blasd.Axpby(&t01, &a10, 1.0, 0.0)
            blasd.MVMult(&t01, &A20, &a21, -tauval, -tauval, gomas.TRANSA)
            // t01 := T00*t01
            blasd.MVMultTrm(&t01, &T00, 1.0, gomas.UPPER)
        }

        // --------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &TTL, &TTR,
            &TBL, &TBR,   &T00, &t11, &T22,   T, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   tau, util.PBOTTOM)
    }
}


/*
 * Build block reflector T from Householder elementary reflectors stored in TriLU(A)
 * and scalar factors in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  z |   z = -tau*T*Y.T*v
 *     | 0  c |   c = tau
 *
 * Compatible with lapack.DLAFRT
 */
func BuildT(T, A, tau *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {

    if n(T) < n(A) || m(T) < n(A) {
        return gomas.NewError(gomas.ESIZE, "BuildT")
    }

    unblkQRBlockReflector(T, A, tau)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

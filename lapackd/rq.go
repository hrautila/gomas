
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/util"
    //"fmt"
)


/*
 * Unblocked RQ decomposition. As implemented
 * in lapack.DGERQ2 subroutine.
 */
func unblockedRQ(A, Tvec, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a11, a01, a10, A22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w12  cmat.FloatMatrix

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, 0, util.PBOTTOM)

    for m(&ATL) > 0 && n(&ATL) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, nil,
            &a10, &a11, nil, 
            nil,  nil,  &A22,   A, 1, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     Tvec, 1, util.PTOP)
        // ------------------------------------------------------
        computeHouseholder(&a11, &a10, &tau1)

        w12.SubMatrix(W, 0, 0, a01.Len(), 1)
        applyHouseholder2x1(&tau1, &a10, &a01, &A00, &w12, gomas.RIGHT)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   Tvec, util.PTOP)
    }
}

/*
 * Blocked RQ decomposition with compact WY transform. As implemented
 * in lapack.DGERQF subroutine.
 */
func blockedRQ(A, Tvec, Twork, W *cmat.FloatMatrix, lb int, conf *gomas.Config) {
    var ATL, ABR, AL cmat.FloatMatrix
    var A00, A01, A10, A11, A22 cmat.FloatMatrix
    var TT, TB cmat.FloatMatrix
    var t0, tau, t2 cmat.FloatMatrix
    var Wrk, w1 cmat.FloatMatrix
    
    util.Partition2x2(
        &ATL, nil,
        nil, &ABR,   /**/ A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &TT,
        &TB,  /**/ Tvec, 0, util.PBOTTOM)

    for m(&ATL)-lb > 0 && n(&ATL)-lb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            &A10, &A11, nil,
            nil,  nil,  &A22,  /**/  A, lb, util.PTOPLEFT)
        util.Repartition2x1to3x1(&TT,
            &t0,
            &tau,
            &t2,   /**/  Tvec, n(&A11), util.PTOP)

        // current block size
        cb, rb := A11.Size()
        if rb < cb {
            cb = rb
        }
        // --------------------------------------------------------
        // decompose left side AL == ( A10 A11 ) 
        w1.SubMatrix(W, 0, 0, cb, 1)
        util.Merge1x2(&AL, &A10, &A11)
        unblockedRQ(&AL, &tau, &w1)

        // build block reflector
        unblkBlockReflectorRQ(Twork, &AL, &tau)

        // compute: (A00 A01)(I - Y*T*Y.T)
        ar, ac := A01.Size()
        Wrk.SubMatrix(W, 0, 0, ar, ac)
        updateRightRQ(&A01, &A00, &A11, &A10, Twork, &Wrk, false, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &A11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &TT,
            &TB,   &t0, &tau,   Tvec, util.PTOP)
    }

    // last block with unblocked
    if m(&ATL) > 0 && n(&ATL) > 0 {
        w1.SubMatrix(W, 0, 0, m(&ATL), 1)
        unblockedRQ(&ATL, &TT, &w1)
    }
}

// compute:
//      Q.T*C = (I -Y*T*Y.T).T*C ==  C - Y*(C.T*Y*T).T 
// or
//      Q*C   = (I -Y*T*Y.T)*C   ==  C - Y*(C.T*Y*T.T).T 
//
//
// where  C = ( C2 )   Y = ( Y2 Y1 )
//            ( C1 )      
//
// C1 is nb*K, C2 is P*K, Y1 is nb*nb triuu, Y2 is nb*P, T is nb*nb
// W = K*nb
func updateLeftRQ(C1, C2, Y1t, Y2t, T, W *cmat.FloatMatrix, transpose bool, conf *gomas.Config) {

    // W = C1.T
    blasd.Plus(W, C1, 0.0, 1.0, gomas.TRANSB)
    // W = C1.T*Y1.T
    blasd.MultTrm(W, Y1t, 1.0, gomas.RIGHT|gomas.LOWER|gomas.UNIT|gomas.TRANSA, conf)
    // W = W + C2.T*Y2.T
    blasd.Mult(W, C2, Y2t, 1.0, 1.0, gomas.TRANSA|gomas.TRANSB, conf)
    // --- here: W == C.T*Y == C1.T*Y1.T + C2.T*Y2.T ---

    tflags := gomas.RIGHT|gomas.LOWER
    if ! transpose {
        tflags |= gomas.TRANSA
    }
    // W = W*T or W*T.T
    blasd.MultTrm(W, T, 1.0, tflags, conf)
    // --- here: W == C.T*Y*T or C.T*Y*T.T ---

    // C2 = C2 - Y2*W.T
    blasd.Mult(C2, Y2t, W, -1.0, 1.0, gomas.TRANSA|gomas.TRANSB, conf)

    //  W = Y1*W.T ==> W.T = W*Y1
    blasd.MultTrm(W, Y1t, 1.0, gomas.RIGHT|gomas.LOWER|gomas.UNIT, conf)
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
// where  C = ( C2 C1 ), Y = ( Y2 Y1 )
//                     
// C1 is K*nb, C2 is K*P, Y1 is nb*nb triuu, Y2 is nb*P, T is nb*nb
// W = K*nb
func updateRightRQ(C1, C2, Y1t, Y2t, T, W *cmat.FloatMatrix, transpose bool, conf *gomas.Config) {
    // -- compute: W = C*Y = C1*Y1 + C2*Y2

    // W = C1
    blasd.Plus(W, C1, 0.0, 1.0, gomas.NONE)
    // W = C1*Y1t.T
    blasd.MultTrm(W, Y1t, 1.0, gomas.RIGHT|gomas.LOWER|gomas.UNIT|gomas.TRANSA, conf)
    // W = W + C2*Y2t.T
    blasd.Mult(W, C2, Y2t, 1.0, 1.0, gomas.TRANSB, conf)
    // --- here: W == C*Y ---

    tflags := gomas.RIGHT|gomas.LOWER
    if transpose {
        tflags |= gomas.TRANSA
    }

    // W = W*T or W*T.T
    blasd.MultTrm(W, T, 1.0, tflags, conf)
    // --- here: W == C*Y*T or C*Y*T.T ---

    // C2 = C2 - W*Y2t
    blasd.Mult(C2, W, Y2t, -1.0, 1.0, gomas.NONE, conf)
    // C1 = C1 - W*Y1t
    //  W = W*Y1 
    blasd.MultTrm(W, Y1t, 1.0, gomas.RIGHT|gomas.LOWER|gomas.UNIT, conf)
    // C1 = C1 - W
    blasd.Plus(C1, W, 1.0, -1.0, gomas.NONE)
    // --- here: C = (I - Y*T*Y.T).T * C ---
}

 /*
  * Compute RQ factorization of a M-by-N matrix A: A = R*Q 
  *
  * Arguments:
  *  A    On entry, the M-by-N matrix A, M <= N. On exit, upper triangular matrix R
  *       and the orthogonal matrix Q as product of elementary reflectors.
  *
  * tau  On exit, the scalar factors of the elementary reflectors.
  *
  * W    Workspace, M-by-nb matrix used for work space in blocked invocations. 
  *
  * conf The blocking configuration. If nil then default blocking configuration
  *      is used. Member conf.LB defines blocking size of blocked algorithms.
  *      If it is zero then unblocked algorithm is used.
  *
  * Returns:
  *      Error indicator.
  *
  * Additional information
  *
  *  Ortogonal matrix Q is product of elementary reflectors H(k)
  *
  *    Q = H(0)H(1),...,H(K-1), where K = min(M,N)
  *
  *  Elementary reflector H(k) is stored on row k of A right of the diagonal with
  *  implicit unit value on diagonal entry. The vector TAU holds scalar factors of
  *  the elementary reflectors.
  *
  *  Contents of matrix A after factorization is as follow:
  *
  *    ( v0 v0 r  r  r  r )  M=4, N=6
  *    ( v1 v1 v1 r  r  r )  
  *    ( v2 v2 v2 v2 r  r )  
  *    ( v3 v3 v3 v3 v3 r )  
  *
  *  where l is element of L, vk is element of H(k).
  *
  *  RQFactor is compatible with lapack.DGERQF
  */
func RQFactor(A, tau, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)

    // must have: M <= N
    if m(A) > n(A) {
        return gomas.NewError(gomas.ESIZE, "RQFactor")
    }

    wsmin := wsLQ(A, 0)
    if W == nil || W.Len() < wsmin {
        return gomas.NewError(gomas.EWORK, "RQFactor", wsmin)
    }
    lb := estimateLB(A, W.Len(), wsRQ)
    lb = imin(lb, conf.LB)
    if lb == 0 || m(A) <= lb {
        unblockedRQ(A, tau, W)
    } else {
        var Twork, Wrk cmat.FloatMatrix
        // block reflector T in first LB*LB elements in workspace
        // the rest, m(A)-LB*LB, is workspace for intermediate matrix operands
        Twork.SetBuf(lb, lb, lb, W.Data())
        Wrk.SetBuf(m(A)-lb, lb, m(A)-lb, W.Data()[Twork.Len():])
        blockedRQ(A, tau, &Twork, &Wrk, lb, conf)
    }
    return err
}

/*
 * Calculate required workspace to decompose matrix A with current blocking configuration.
 * If blocking configuration is not provided then default configuation will be used.
 *
 * Returns size of workspace as number of elements.
 */
func RQFactorWork(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsRQ(A, conf.LB)
}

func wsRQ(A *cmat.FloatMatrix, lb int) int {
    if lb > 0 {
        return lb*m(A)
    }
    return m(A)
}

/*
 * like LAPACK/dlafrt.f
 *
 * Build block reflector T from HH reflector stored in TriLU(A) and coefficients
 * in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  0 |   z = -tau*T*Y.T*v
 *     | z  c |   c = tau
 *
 * Q = H(1)H(2)...H(k) building forward here.
 */
func unblkBlockReflectorRQ(T, A, tau *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a10, A20, a11, a21, A22 cmat.FloatMatrix
    var TTL, TBR cmat.FloatMatrix
    var T00, t11, t21, T22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  /**/ A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x2(
        &TTL, nil,
        nil,  &TBR, /**/ T, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tT,
        &tB,  /**/ tau, 0, util.PBOTTOM)

    for m(&ATL) > 0 && n(&ATL) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00,  nil,  nil,
            &a10,  &a11, nil,
            &A20,  &a21, &A22,  /**/ A, 1, util.PTOPLEFT)
        util.Repartition2x2to3x3(&TTL,
            &T00, nil,  nil,
            nil,  &t11, nil,
            nil,  &t21, &T22,  /**/ T, 1, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,   /**/  tau, 1, util.PTOP)
        // --------------------------------------------------

        // t11 := tau
        tauval := tau1.Get(0, 0)
        if tauval != 0.0 {
            t11.Set(0, 0, tauval)

            // t21 := -tauval*(a21 + A20*a10)
            blasd.Axpby(&t21, &a21, 1.0, 0.0)
            blasd.MVMult(&t21, &A20, &a10, -tauval, -tauval, gomas.NONE)
            // t21 := T22*t21
            blasd.MVMultTrm(&t21, &T22, 1.0, gomas.LOWER)
        }

        // --------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,  /**/ &A00, &a11, &A22,   A, util.PTOPLEFT)
        util.Continue3x3to2x2(
            &TTL, nil,
            nil,  &TBR, /**/  &T00, &t11, &T22,   T, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tT,
            &tB,  /**/ &t0, &tau1,   tau, util.PTOP)
    }
}


/*
 * Build block reflector T from Householder elementary reflectors stored in A
 * and scalar factors in tau.
 *
 * Q = I - Y*T*Y.T; Householder H = I - tau*v*v.T
 *
 * T = | T  0 |   z = -tau*T*Yt*v
 *     | z  c |   c = tau
 *
 * Compatible with lapack.DLAFRT
 */
func RQReflector(T, A, tau *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {

    if n(T) < m(A) || m(T) < m(A) {
        return gomas.NewError(gomas.ESIZE, "BuildLQT")
    }

    unblkBlockReflectorRQ(T, A, tau)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

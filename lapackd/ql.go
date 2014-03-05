
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
 * Unblocked QL decomposition. As implemented
 * in lapack.xGEQL2 subroutine.
 */
func unblockedQL(A, Tvec, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a01, a10, a11, A22 cmat.FloatMatrix
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
        computeHouseholder(&a11, &a01, &tau1)

        w12.SubMatrix(W, 0, 0, a10.Len(), 1)
        applyHouseholder2x1(&tau1, &a01, &a10, &A00, &w12, gomas.LEFT)
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
 * Blocked QR decomposition with compact WY transform. As implemented
 * in lapack.xGEQRF subroutine.
 */
func blockedQL(A, Tvec, Twork, W *cmat.FloatMatrix, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR, AL cmat.FloatMatrix
    var A00, A01, A10, A11, A22 cmat.FloatMatrix
    var TT, TB cmat.FloatMatrix
    var t0, tau, t2 cmat.FloatMatrix
    var Wrk, w1 cmat.FloatMatrix
    
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &TT,
        &TB,  Tvec, 0, util.PBOTTOM)

    nb := conf.LB
    for m(&ATL)-nb > 0 && n(&ATL)-nb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            &A10, &A11, nil,
            nil,  nil,  &A22,   A, nb, util.PTOPLEFT)
        util.Repartition2x1to3x1(&TT,
            &t0,
            &tau,
            &t2,     Tvec, nb, util.PTOP)

        // current block size
        cb, rb := A11.Size()
        if rb < cb {
            cb = rb
        }
        // --------------------------------------------------------
        // decompose righ side AL == /A01\ 
        //                           \A11/
        w1.SubMatrix(W, 0, 0, cb, 1)
        util.Merge2x1(&AL, &A01, &A11)
        unblockedQL(&AL, &tau, &w1)

        // build block reflector
        unblkQLBlockReflector(Twork, &AL, &tau)

        // update A'tail i.e. A10 and A00 with (I - Y*T*Y.T).T * A'tail
        // compute: C - Y*(C.T*Y*T).T
        ar, ac := A10.Size()
        Wrk.SubMatrix(W, 0, 0, ac, ar)
        updateQLLeft(&A10, &A00, &A11, &A01, Twork, &Wrk, true, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &TT,
            &TB,   &t0, &tau,   Tvec, util.PTOP)
    }

    // last block with unblocked
    if m(&ATL) > 0 && n(&ATL) > 0 {
        w1.SubMatrix(W, 0, 0, n(&ATL), 1)
        unblockedQL(&ATL, &t0, &w1)
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
// C1 is nb*K, C2 is P*K, Y1 is nb*nb TriLU, Y2 is P*nb, T is nb*nb TriL
// W = K*nb
func updateQLLeft(C1, C2, Y1, Y2, T, W *cmat.FloatMatrix, transpose bool, conf *gomas.Config) {
    // W = C1.T
    blasd.Plus(W, C1, 0.0, 1.0, gomas.TRANSB)

    // W = C1.T*Y1
    blasd.MultTrm(W, Y1, 1.0, gomas.UPPER|gomas.UNIT|gomas.RIGHT, conf)
    // W = W + C2.T*Y2
    blasd.Mult(W, C2, Y2, 1.0, 1.0, gomas.TRANSA, conf)
    // --- here: W == C.T*Y ---

    tflags := gomas.LOWER|gomas.RIGHT 
    if ! transpose {
        tflags |= gomas.TRANSA
    }
    // W = W*T or W*T.T
    blasd.MultTrm(W, T, 1.0, tflags, conf)
    // --- here: W == C.T*Y*T or C.T*Y*T.T ---

    // C2 = C2 - Y2*W.T
    blasd.Mult(C2, Y2, W, -1.0, 1.0, gomas.TRANSB, conf)

    //  W = Y1*W.T ==> W.T = W*Y1.T
    blasd.MultTrm(W, Y1, 1.0, gomas.UPPER|gomas.UNIT|gomas.TRANSA|gomas.RIGHT, conf)
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
func updateQLRight(C1, C2, Y1, Y2, T, W *cmat.FloatMatrix, transpose bool, conf *gomas.Config) {
    // -- compute: W = C*Y = C1*Y1 + C2*Y2

    // W = C1
    blasd.Plus(W, C1, 0.0, 1.0, gomas.NONE)
    // W = C1*Y1
    blasd.MultTrm(W, Y1, 1.0, gomas.UPPER|gomas.UNIT|gomas.RIGHT, conf)
    // W = W + C2*Y2
    blasd.Mult(W, C2, Y2, 1.0, 1.0, gomas.NONE, conf)
    // --- here: W == C*Y ---

    tflags := gomas.LOWER|gomas.RIGHT 
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
  * Compute QL factorization of a M-by-N matrix A: A = Q * L.
  *
  * Arguments:
  *  A   On entry, the M-by-N matrix A. On exit, the elements on and below
  *      the diagonal contain the min(M,N)-by-N lower trapezoidal matrix L.
  *      The elements above the diagonal with the column vector 'tau', represent
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
  * Additional information
  *
  *  Ortogonal matrix Q is product of elementary reflectors H(k)
  *
  *    Q = H(k)...H(2)H(1), where K = min(M,N)
  *
  *  Elementary reflector H(k) is stored on column k of A above the diagonal with
  *  implicit unit value on diagonal entry. The vector TAU holds scalar factors
  *  of the elementary reflectors.
  *
  *  Contents of matrix A after factorization is as follow:
  *
  *    ( v1 v2 v3 v4 )   for M=6, N=4
  *    ( v1 v2 v3 v4 )
  *    ( l  v2 v3 v4 )
  *    ( l  l  v3 v4 )
  *    ( l  l  l  v4 )
  *    ( l  l  l  l  )
  *
  *  where l is element of L, vk is element of H(k).
  *
  * DecomposeQL is compatible with lapack.DGEQLF
  */
func DecomposeQL(A, tau, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var tauh cmat.FloatMatrix
    conf := gomas.CurrentConf(confs...)
    wsmin := wsQL(A, 0)
    if W == nil || W.Len() < wsmin {
        return gomas.NewError(gomas.EWORK, "DecomposeQL", wsmin)
    }
    if tau.Len() < imin(m(A), n(A)) {
        return gomas.NewError(gomas.ESIZE, "DecomposeQL")
    }
    tauh.SubMatrix(tau, 0, 0, imin(m(A), n(A)), 1)
    lb := conf.LB
    if lb == 0 || n(A) <= lb {
        unblockedQL(A, &tauh, W)
    } else {
        var Twork, Wrk cmat.FloatMatrix
        // block reflector T in first LB*LB elements in workspace
        // the rest, n(A)-LB*LB, is workspace for intermediate matrix operands
        Twork.SetBuf(conf.LB, conf.LB, -1, W.Data())
        Wrk.SetBuf(n(A)-conf.LB, conf.LB, -1, W.Data()[Twork.Len():])
        blockedQL(A, &tauh, &Twork, &Wrk, conf)
    }
    return err
}

func wsQL(A *cmat.FloatMatrix, lb int) int {
    sz := n(A)
    if lb > 0 && n(A) > lb {
        sz = (n(A)+lb)*lb
    }
    return sz
}


/*
 * Calculate required workspace to decompose matrix A with current blocking
 * configuration. If blocking configuration is not provided then default
 * configuation will be used.
 *
 * Returns size of workspace as number of elements.
 */
func WorksizeQL(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsQL(A, conf.LB)
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
func unblkQLBlockReflector(T, A, tau *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a01, a11, A02, a12, A22 cmat.FloatMatrix
    var TTL, TBR cmat.FloatMatrix
    var T00, t11, t21, T22 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x2(
        &TTL, nil,
        nil,  &TBR, T, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tT,
        &tB,  tau, 0, util.PBOTTOM)

    for m(&ATL) > 0 && n(&ATL) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            nil,  &a11, &a12,
            nil,  nil,  &A22,   A, 1, util.PTOPLEFT)
        util.Repartition2x2to3x3(&TTL,
            &T00, nil,  nil,
            nil,  &t11, nil,
            nil,  &t21, &T22,   T, 1, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, util.PTOP)
        // --------------------------------------------------

        // t11 := tau
        tauval := tau1.Get(0, 0)
        if tauval != 0.0 {
            t11.Set(0, 0, tauval)

            // t21 := -tauval*(a12.T + &A02.T*a12)
            blasd.Axpby(&t21, &a12, 1.0, 0.0)
            blasd.MVMult(&t21, &A02, &a01, -tauval, -tauval, gomas.TRANSA)
            // t21 := T22*t01
            blasd.MVMultTrm(&t21, &T22, 1.0, gomas.LOWER)
        }

        // --------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        util.Continue3x3to2x2(
            &TTL, nil,
            nil,  &TBR,   &T00, &t11, &T22,   T, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   tau, util.PTOP)
    }
}

func BuildQLT(T, A, tau *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var tauh cmat.FloatMatrix

    if n(T) < n(A) || m(T) < n(A) {
        return gomas.NewError(gomas.ESIZE, "BuildQLT")
    }

    tauh.SubMatrix(tau, 0, 0, imin(m(A), n(A)), 1)
    unblkQLBlockReflector(T, A, &tauh)
    return nil
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

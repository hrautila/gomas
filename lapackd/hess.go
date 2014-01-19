
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
 * (1) Quintana-Orti, van de Geijn:
 *     Improving the Performance of Reduction to Hessenberg form, 2006
 *
 * (2) Van Zee, van de Geijn, Quintana-Orti:
 *     Algorithms for Reducing a Matrix to Condensed Form, 2010, FLAME working notes #53
 *     
 *                              I(k)|  0  
 * Householder reflector H(k) = --------- 
 *                               0  | H(k)
 *
 * update: H(k)*A*H(k) =
 *
 *           I | 0 | 0    A00 | a01 | A02   I | 0 | 0      A00 | a01 |  A02*H
 *           ---------    ---------------   ---------      -------------------
 *           0 | 1 | 0 *  a10 | a11 | a12 * 0 | 1 | 0  ==  a10 | a11 |  a12*H
 *           ---------    ---------------   ---------     --------------------
 *           0 | 0 | H    A20 | a21 | A22   0 | 0 | H      A20 | a21 | H*A22*H
 *
 * For blocked version elementary reflectors are combined to block reflector H = I - Y*T*Y.T
 * Elementary reflectors are computed to block [A11; A21].T with unblocked algorithm and the
 * block A01 needs to be updated afterwards. (Not during the reflector computation as happens
 * in unblocked version)
 *
 * Approximate flops needed: (10/3)*N^3
 */



/*
 * Computes upper Hessenberg reduction of N-by-N matrix A using unblocked
 * algorithm as described in (1).
 *
 * Hessengerg reduction: A = Q.T*B*Q, Q unitary, B upper Hessenberg
 *  Q = H(0)*H(1)*...*H(k) where H(k) is k'th Householder reflector.
 *
 * Compatible with lapack.DGEHD2.
 */
func unblkHessGQvdG(A, Tvec, W *cmat.FloatMatrix, row int) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a11, a21, A22 cmat.FloatMatrix
    var AL, AR, A0, a1, A2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w12, v1 cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, row, 0, util.PTOPLEFT)
    util.Partition1x2(
        &AL, &AR,   A, 0, util.PLEFT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, 0, util.PTOP)

    v1.SubMatrix(W, 0, 0, m(A), 1)

    for m(&ABR) > 1 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &a11, nil,
            nil,  &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition1x2to1x3(&AL,
            &A0,  &a1,  &A2,    A, 1, util.PRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     Tvec, 1, util.PBOTTOM)

        // ------------------------------------------------------
        // a21 = [beta; H(k)].T
        computeHouseholderVec(&a21, &tau1)
        tauval := tau1.Get(0, 0)
        beta := a21.Get(0, 0)
        a21.Set(0, 0, 1.0)

        // v1 := A2*a21
        blasd.MVMult(&v1, &A2, &a21, 1.0, 0.0, gomas.NONE)
        
        // A2 := A2 - tau*v1*a21   (A2 := A2*H(k))
        blasd.MVUpdate(&A2, &v1, &a21, -tauval)
        
        w12.SubMatrix(W, 0, 0, n(&A22), 1)
        // w12 := a21.T*A22 = A22.T*a21
        blasd.MVMult(&w12, &A22, &a21, 1.0, 0.0, gomas.TRANS)
        // A22 := A22 - tau*a21*w12   (A22 := H(k)*A22)
        blasd.MVUpdate(&A22, &a21, &w12, -tauval)

        a21.Set(0, 0, beta)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue1x3to1x2(
            &AL,  &AR,    &A0,  &a1,   A, util.PRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   Tvec, util.PBOTTOM)
    }
}



/*
 * Update vector with compact WY Householder block
 *   (I - Y*T*Y.T)*v  = v - Y*T*Y.T*v 
 *
 * LEFT:
 *    1 | 0 * v0 = v0     = v0
 *    0 | Q   v1   Q*v1   = v1 - Y*T*Y.T*v1
 *
 *    1 | 0   * v0 = v0     = v0
 *    0 | Q.T   v1   Q.T*v1 = v1 - Y*T.T*Y.T*v1
 * 
 * RIGHT:
 *   v0 | v1 * 1 | 0  = v0 | v1*Q    = v0 | v1 - v1*Y*T*Y.T
 *             0 | Q
 *
 *   v0 | v1 * 1 | 0  = v0 | v1*Q.T  = v0 | v1 - v1*Y*T.T*Y.T
 *             0 | Q.T
 */
func updateVecLeftWY2(v, Y1, Y2, T, w *cmat.FloatMatrix, bits int) {
    var v1, v2 cmat.FloatMatrix
    var w0 cmat.FloatMatrix

    v1.SubMatrix(v, 1, 0, n(Y1), 1)
    v2.SubMatrix(v, n(Y1)+1, 0, m(Y2), 1)
    w0.SubMatrix(w, 0, 0, m(Y1), 1)

    // w0 := Y1.T*v1 + Y2.T*v2
    blasd.Copy(&w0, &v1)
    blasd.MVMultTrm(&w0, Y1, 1.0, gomas.LOWER|gomas.UNIT|gomas.TRANS)
    blasd.MVMult(&w0, Y2, &v2, 1.0, 1.0, gomas.TRANS)

    // w0 := op(T)*w0
    blasd.MVMultTrm(&w0, T, 1.0, bits|gomas.UPPER)

    // v2 := v2 - Y2*w0
    blasd.MVMult(&v2, Y2, &w0, -1.0, 1.0, gomas.NONE)

    // v1 := v1 - Y1*w0
    blasd.MVMultTrm(&w0, Y1, 1.0, gomas.LOWER|gomas.UNIT)
    blasd.Axpy(&v1, &w0, -1.0)
}

    
/*
 * 
 *  Building reduction block for blocked algorithm as described in (1).
 *
 *  A. update next column
 *    a10        [(U00)     (U00)  ]   [(a10)    (V00)            ]
 *    a11 :=  I -[(u10)*T00*(u10).T] * [(a11)  - (v01) * T00 * a10]
 *    a12        [(U20)     (U20)  ]   [(a12)    (V02)            ]
 *
 *  B. compute Householder reflector for updated column
 *    a21, t11 := Householder(a21)
 *
 *  C. update intermediate reductions
 *    v10      A02*a21
 *    v11  :=  a12*a21
 *    v12      A22*a21
 *
 *  D. update block reflector
 *    t01 :=  A20*a21
 *    t11 :=  t11
 */
func unblkBuildHessGQvdG(A, T, V, W *cmat.FloatMatrix) *gomas.Error {

    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, A20, a21, A22 cmat.FloatMatrix
    var AL, AR, A0, a1, A2 cmat.FloatMatrix
    var TTL, TTR, TBL, TBR cmat.FloatMatrix
    var T00, t01, t11, T22 cmat.FloatMatrix
    var VL, VR, V0, v1, V2, Y0 cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &TTL, &TTR,
        &TBL, &TBR, T, 0, 0, util.PTOPLEFT)
    util.Partition1x2(
        &AL,  &AR,   A, 0, util.PLEFT)
    util.Partition1x2(
        &VL,  &VR,   V, 0, util.PLEFT)

    var beta float64

    for n(&VR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&TTL,
            &T00, &t01, nil,
            nil,  &t11, nil,
            nil,  nil,  &T22,   T, 1, util.PBOTTOMRIGHT)
        util.Repartition1x2to1x3(&AL,
            &A0, &a1, &A2,      A, 1, util.PRIGHT)
        util.Repartition1x2to1x3(&VL,
            &V0, &v1, &V2,      V, 1, util.PRIGHT)

        // ------------------------------------------------------
        // Compute Hessenberg update for next column of A:
        if n(&V0) > 0 {
            // y10 := T00*a10  (use t01 as workspace?)
            blasd.Axpby(&t01, &a10, 1.0, 0.0)
            blasd.MVMultTrm(&t01, &T00, 1.0, gomas.UPPER)

            // a1 := a1 - V0*T00*a10
            blasd.MVMult(&a1, &V0, &t01, -1.0, 1.0, gomas.NONE)

            // update a1 := (I - Y*T*Y.T).T*a1 (here t01 as workspace)
            Y0.SubMatrix(A, 1, 0, n(&A00), n(&A00))
            updateVecLeftWY2(&a1, &Y0, &A20, &T00, &t01, gomas.TRANS)
            a10.Set(0, -1, beta)
        }

        // Compute Householder reflector
        computeHouseholderVec(&a21, &t11)
        beta = a21.Get(0, 0)
        a21.Set(0, 0, 1.0)

        // v1 := A2*a21
        blasd.MVMult(&v1, &A2, &a21, 1.0, 0.0, gomas.NONE)

        // update T
        tauval := t11.Get(0, 0)
        if tauval != 0.0 {
            // t01 := -tauval*A20.T*a21
            blasd.MVMult(&t01, &A20, &a21, -tauval, 0.0, gomas.TRANS) 
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
        util.Continue1x3to1x2(
            &AL,  &AR,    &A0,  &a1,     A, util.PRIGHT)
        util.Continue1x3to1x2(
            &VL,  &VR,    &V0,  &v1,     V, util.PRIGHT)
    }
    A.Set(n(V), n(V)-1, beta)
    return nil
}


// Compute: (I - Y*T*Y.T).T*C = C - Y*(C.T*Y*T)
func updateHessLeftWY(C, Y1, Y2, T, V *cmat.FloatMatrix, conf *gomas.Config) {
    var C1, C2 cmat.FloatMatrix

    if C.Len() == 0 {
        return
    }
    C1.SubMatrix(C, 1, 0, m(Y1), n(C))
    C2.SubMatrix(C, 1+n(Y1), 0, m(Y2), n(C))

    updateWithQTLeft(&C1, &C2, Y1, Y2, T, V, true, conf)
}

// Compute: C*(I - Y*T*Y.T) = C - C*Y*T*Y.T = C - V*T*Y.T
func updateHessRightWY(C, Y1, Y2, T, W *cmat.FloatMatrix, conf *gomas.Config) {
    var C1, C2 cmat.FloatMatrix

    if C.Len() == 0 {
        return
    }
    C1.SubMatrix(C, 0, 1, m(C), n(Y1))
    C2.SubMatrix(C, 0, 1+n(Y1), m(C), m(Y2))

    updateWithQTRight(&C1, &C2, Y1, Y2, T, W, false, conf)
}

/*
 * Blocked version of Hessenberg reduction algorithm as presented in (1). This
 * version uses compact-WY transformation.
 *
 * Some notes:
 *
 * Elementary reflectors stored in [A11; A21].T are not on diagonal of A11. Update of
 * a block aligned with A11; A21 is as follow
 *
 * 1. Update from left Q(k)*C:
 *                                         c0   0                            c0
 * (I - Y*T*Y.T).T*C = C - Y*(C.T*Y)*T.T = C1 - Y1 * (C1.T.Y1+C2.T*Y2)*T.T = C1-Y1*W
 *                                         C2   Y2                           C2-Y2*W
 *
 * where W = (C1.T*Y1+C2.T*Y2)*T.T and first row of C is not affected by update
 * 
 * 2. Update from right C*Q(k):
 *                                       0
 * C - C*Y*T*Y.T = c0;C1;C2 - c0;C1;C2 * Y1 *T*(0;Y1;Y2) = c0; C1-W*Y1; C2-W*Y2
 *                                       Y2
 * where  W = (C1*Y1 + C2*Y2)*T and first column of C is not affected
 *
 */
func blkHessGQvdG(A, Tvec, W *cmat.FloatMatrix, nb int, conf *gomas.Config) *gomas.Error {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A11, A12, A21, A22, A2 cmat.FloatMatrix
    var tT, tB, td cmat.FloatMatrix
    var t0, t1, t2, T cmat.FloatMatrix
    var V, VT, VB, /*V0, V1, V2,*/ Y1, Y2, W0 cmat.FloatMatrix
    
    //fmt.Printf("blkHessGQvdG...\n")
    T.SubMatrix(W, 0, 0, conf.LB, conf.LB)
    V.SubMatrix(W, conf.LB, 0, m(A), conf.LB)
    td.Diag(&T)

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tT,
        &tB,  Tvec, 0, util.PTOP)

    for m(&ABR) > nb+1 && n(&ABR) > nb {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &A11, &A12,
            nil,  &A21, &A22,   A, nb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &t1,
            &t2,     Tvec, nb, util.PBOTTOM)

        util.Partition2x1(
            &VT,
            &VB,  &V, m(&ATL), util.PTOP)
        // ------------------------------------------------------

        unblkBuildHessGQvdG(&ABR, &T, &VB, nil)
        blasd.Copy(&t1, &td)

        // m(Y) == m(ABR)-1, n(Y) == n(A11)
        Y1.SubMatrix(&ABR, 1, 0, n(&A11), n(&A11))
        Y2.SubMatrix(&ABR, 1+n(&A11), 0, m(&A21)-1, n(&A11))

        // [A01; A02] == ATR := ATR*(I - Y*T*Y.T)
        updateHessRightWY(&ATR, &Y1, &Y2, &T, &VT, conf)

        // A2 = [A12; A22].T
        util.Merge2x1(&A2, &A12, &A22)

        // A2 := A2 - VB*T*A21.T
        be := A21.Get(0, -1)
        A21.Set(0, -1, 1.0)
        blasd.MultTrm(&VB, &T, 1.0, gomas.UPPER|gomas.RIGHT)
        blasd.Mult(&A2, &VB, &A21, -1.0, 1.0, gomas.TRANSB, conf)
        A21.Set(0, -1, be)

        // A2 := (I - Y*T*Y.T).T * A2
        W0.SubMatrix(&V, 0, 0, n(&A2), n(&Y2))
        updateHessLeftWY(&A2, &Y1, &Y2, &T, &W0, conf)

        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &t1,   Tvec, util.PBOTTOM)
    }
    
    if m(&ABR) > 1 {
        // do the rest with unblocked
        util.Merge2x1(&A2, &ATR, &ABR)
        W0.SetBuf(m(A), 1, m(A), W.Data())
        unblkHessGQvdG(&A2, &tB, &W0, m(&ATR))
    }
    return nil
}


/*
 * Reduce general matrix A to upper Hessenberg form H by similiarity
 * transformation H = Q.T*A*Q.
 *
 * Arguments:
 *  A    On entry, the general matrix A. On exit, the elements on and
 *       above the first subdiagonal contain the reduced matrix H.
 *       The elements below the first subdiagonal with the vector tau
 *       represent the ortogonal matrix A as product of elementary reflectors.
 *
 *  tau  On exit, the scalar factors of the elementary reflectors.
 *
 *  W    Workspace, as defined by WorksizeHess()
 *
 *  conf The blocking configration. 
 * 
 * ReduceHess is compatible with lapack.DGEHRD.
 */
func ReduceHess(A, tau, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)

    wmin := m(A)
    wopt := WorksizeHess(A, conf)
    wsz  := W.Len()
    if wsz < wmin {
        return gomas.NewError(gomas.EWORK, "ReduceHess", wmin)
    }
    // use blocked version if workspace big enough for blocksize 4
    lb := conf.LB
    if wsz < wopt {
        lb = estimateLB(A, wsz, wsHess)
    }
    if lb == 0 || n(A) <= lb {
        unblkHessGQvdG(A, tau, W, 0)
    } else {
        // blocked version
        var W0 cmat.FloatMatrix
        // shape workspace for blocked algorithm
        W0.SetBuf(m(A)+lb, lb, m(A)+lb, W.Data())
        blkHessGQvdG(A, tau, &W0, lb, conf)
    }
    return err
}

/*
 * Multiply and replace C with product of C and Q where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 * as returned by ReduceHess().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     Hessenberg reduction as returned by ReduceHess() where the lower trapezoidal
 *        part, on and below first subdiagonal, holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors. A column vector.
 *
 *  W     Workspace matrix,  required size is returned by WorksizeMultHess().
 *
 *  bits  Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block sized. If it is zero
 *        unblocked invocation is assumed.
 *
 *        flags        result
 *        -------------------------------------
 *        LEFT         C = Q*C     n(A) == m(C)
 *        RIGHT        C = C*Q     n(C) == m(A)
 *        TRANS|LEFT   C = Q.T*C   n(A) == m(C)
 *        TRANS|RIGHT  C = C*Q.T   n(C) == m(A)
 *
 */
func MultQHess(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var Qh, Ch, tauh cmat.FloatMatrix

    Qh.SubMatrix(A, 1, 0, m(A)-1, n(A)-1)
    tauh.SubMatrix(tau, 0, 0, m(tau)-1, 1)
    if flags & gomas.RIGHT != 0 {
        Ch.SubMatrix(C, 0, 1, m(C), n(C)-1)
    } else {
        Ch.SubMatrix(C, 1, 0, m(C)-1, n(C))
    }
    err := MultQ(&Ch, &Qh, &tauh, W, flags, confs...)
    if err != nil {
        err.Update("MultQHess")
    }
    return err
}

func wsHess(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 || lb > n(A) {
        return m(A)
    }
    // need space for block reflector T and intermediate results
    return lb*(m(A)+lb)
}

/*
 * Compute worksize needed for Hessenberg reduction of matrix A with
 * a blocking configuration.
 */
func WorksizeHess(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsHess(A, conf.LB)
}

func WorksizeMultQHess(A *cmat.FloatMatrix, flags int, confs... *gomas.Config) int {
    return WorksizeMultQ(A, flags, confs...)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

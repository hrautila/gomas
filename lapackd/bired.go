
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/util"
    "github.com/hrautila/gomas/blasd"
    //"fmt"
)

/*
 * G.Van Zee, R. van de Geijn
 *   Algorithms for Reducing a Matrix to Condensed Form
 *   2010, Flame working note #53 
 */

/*
 * Basic bidiagonal reduction for one column and row computing
 * from left to right when M >= N.
 *
 *   A  = ( 1 - tauq*v*v.T ) * A * ( 1 - taup*u*u.T )
 *
 *      = ( 1 - tauq*( 1 )*( 1 v.T )) * A * (1 - taup* ( 0 )*( 0 u.T ))
 *                   ( v )                             ( u )
 *
 *  1. Compute first Householder reflector [1 v].T,  len(v) == len(a21)
 *
 *     a12'  = (1 - tauq*( 1 )(1 v.T))( a12 ) 
 *     A22'              ( v )        ( A22 ) 
 *      
 *           = ( a12 -  tauq * (a12 + v.T*A22) )  
 *             ( A22 -  tauq*v*(a12 + v.T*A22) ) 
 *
 *           = ( a12 - tauq * y21 )    [y21 = a12 + A22.T*v]
 *             ( A22 - tauq*v*y21 )
 *
 *  2. Compute second Householder reflector [u], len(u) == len(a12)
 *
 *      A''  = ( a21; A22' )(1 - taup*( 0 )*( 0 u.T )) 
 *                                    ( u )              
 *
 *           = ( a21; A22' ) - taup * ( 0;  A22'*u*u.T )
 *
 *     A22   = A22' - taup*A22'*u*u.T
 *           = A22  - tauq*v*y21 - taup*(A22 - tauq*v*y21)*u*u.T
 *           = A22  - tauq*v*y21 - taup*(A22*u - tauq*v*y21*u)*u.T
 *
 *     y21   = a12 + A22.T*v
 *     z21   = A22*u - tauq*v*y21*u 
 *
 * The unblocked algorithm
 * -----------------------
 *  1.  v, tauq := HOUSEV(a11, a21)
 *  2.  y21 := a12 + A22.T*v
 *  3.  a12 := a12 - tauq*v
 *  4.  u, taup := HOUSEV(a12)
 *  5.  beta := DOT(u, y21)
 *  6.  z21  := A22*u - tauq*beta*v
 *  7.  A22  := A22 - tauq*v*y21
 *  8.  A22  := A22 - taup*z21*u
 */

/*
 * Compute unblocked bidiagonal reduction for A when M >= N
 *
 * Diagonal and first super/sub diagonal are overwritten with the 
 * upper/lower bidiagonal matrix B.
 *
 * This computing (1-tauq*v*v.T)*A*(1-taup*u.u.T) from left to right. 
 */
func unblkReduceBidiagLeft(A, tauq, taup, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a11, a12t, a21, A22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var tpT, tpB, tp0, taup1, tp2 cmat.FloatMatrix
    var y21, z21 cmat.FloatMatrix
    var v0 float64
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
    util.Partition2x1(
        &tpT,
        &tpB,  taup, 0, util.PTOP)

	
    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &a11, &a12t,
            nil,  &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PBOTTOM)
        util.Repartition2x1to3x1(&tpT,
            &tp0,
            &taup1,
            &tp2,     taup, 1, util.PBOTTOM)
		
        // set temp vectors for this round
        y21.SetBuf(n(&a12t), 1, n(&a12t), W.Data())
        z21.SetBuf(m(&a21),  1, m(&a21),  W.Data()[y21.Len():])
        // ------------------------------------------------------

        // Compute householder to zero subdiagonal entries
        computeHouseholder(&a11, &a21, &tauq1)
		
        // y21 := a12 + A22.T*a21
        blasd.Axpby(&y21, &a12t, 1.0, 0.0)
        blasd.MVMult(&y21, &A22, &a21, 1.0, 1.0, gomas.TRANSA)
        
        // a12t := a12t - tauq*y21
        tauqv := tauq1.Get(0, 0)
        blasd.Axpy(&a12t, &y21, -tauqv)

        // Compute householder to zero elements above 1st superdiagonal
        computeHouseholderVec(&a12t, &taup1)
        v0 = a12t.Get(0, 0)
        a12t.Set(0, 0, 1.0)
        taupv := taup1.Get(0, 0)

        // [u == a12t, v == a21]
        beta  := blasd.Dot(&y21, &a12t)
        // z21 := tauq*beta*u
        blasd.Axpby(&z21, &a21, tauqv*beta, 0.0)
        // z21 := A22*u - z21
        blasd.MVMult(&z21, &A22, &a12t, 1.0, -1.0, gomas.NONE)
        // A22 := A22 - tauq*v*y21
        blasd.MVUpdate(&A22, &a21, &y21, -tauqv)
        // A22 := A22 - taup*z21*u
        blasd.MVUpdate(&A22, &z21, &a12t, -taupv)

        a12t.Set(0, 0, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tpT,
            &tpB,   &tp0, &taup1,   taup, util.PBOTTOM)
    }
}

/*
 *  Computing transformation from right to left.
 *
 *  1. Compute first Householder reflector [1 u], len(u) == len(a12)
 *
 *    A'  = ( a21 A22 )(1 - taup*( 1 )*( 1 u.T )) 
 *                               ( u )              
 *
 *        = ( a21 A22 ) - taup*(a21 + A22*u; a21*u.T + A22*u*u.T)
 *
 *        = ( a21 - taup*(a21 + A22*u); A22 - taup*(a21 + A22*u)*u.T)
 *         
 *          y21  = a21 + A22*u
 *          a21' = a21 - taup*y21
 *          A22' = A22 - taup*y21*u.T
 *
 *  2. Compute second Householder reflector [v],  len(v) == len(a21)
 * 
 *    a12'' = (1 - tauq*( 0 )(0 v.T))( a12' ) 
 *    A22''             ( v )        ( A22' ) 
 *
 *          = ( a12'                    )
 *            ( A22' -  tauq*v*v.T*A22' )
 *
 *    A22   = A22' - tauq*v*v.T*A22'
 *          = A22  - taup*y21*u.T - tauq*v*v.T*(A22 - taup*y21*u.T)
 *          = A22  - taup*y21*u.T - tauq*v*v.T*A22 + tauq*v*v.T*taup*y21*u.T
 *          = A22  - taup*y21*u.T - tauq*v*(A22.T*v - taup*v.T*y21*u.T)
 *
 *  The unblocked algorithm
 *  -----------------------
 *   1.  u, taup = HOUSEV(a11, a12)
 *   2.  y21  = a21 + A22*u
 *   3.  a21' = a21 - taup*y21
 *   4.  v, tauq = HOUSEV(a21')
 *   5.  beta = DOT(y21, v)
 *   6.  z21  = A22.T*v - taup*beta*u.T
 *   7.  A22  = A22 - taup*y21*u.T
 *   8.  A22  = A22 - tauq*v*z21
 */


/*
 * Compute unblocked bidiagonal reduction for A when M < N
 *
 * Diagonal and first sub diagonal are overwritten with the lower
 * bidiagonal matrix B.
 */
func unblkReduceBidiagRight(A, tauq, taup, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a11, a12t, a21, A22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var tpT, tpB, tp0, taup1, tp2 cmat.FloatMatrix
    var y21, z21 cmat.FloatMatrix
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
    util.Partition2x1(
        &tpT,
        &tpB,  taup, 0, util.PTOP)

	
    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &a11, &a12t,
            nil,  &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PBOTTOM)
        util.Repartition2x1to3x1(&tpT,
            &tp0,
            &taup1,
            &tp2,     taup, 1, util.PBOTTOM)
		
        // set temp vectors for this round
        y21.SetBuf(m(&a21),  1, m(&a21),  W.Data())
        z21.SetBuf(n(&a12t), 1, n(&a12t), W.Data()[y21.Len():])
        // ------------------------------------------------------
        // Compute householder to zero superdiagonal entries
        computeHouseholder(&a11, &a12t, &taup1)
		
        // y21 := a21 + A22.T*a12t  (len(y21) == len(a21))
        blasd.Axpby(&y21, &a21, 1.0, 0.0)
        blasd.MVMult(&y21, &A22, &a12t, 1.0, 1.0, gomas.NONE)
        
        // a21 := a21 - taup*y21
        taupv := taup1.Get(0, 0)
        blasd.Axpy(&a21, &y21, -taupv)

        // Compute householder to zero elements below 1st subdiagonal
        computeHouseholderVec(&a21, &tauq1)

        v0    := a21.Get(0, 0)
        tauqv := tauq1.Get(0, 0)
        a21.Set(0, 0, 1.0)

        // [u == a12t, v == a21, len(z21) == len(a12t), len(y21) == len(a21)]
        beta  := blasd.Dot(&y21, &a21)
        // z21 := tauq*beta*v
        blasd.Axpby(&z21, &a12t, taupv*beta, 0.0)
        // z21 := A22*u - z21
        blasd.MVMult(&z21, &A22, &a21, 1.0, -1.0, gomas.TRANSA)
        // A22 := A22 - taup*y21*u.T
        blasd.MVUpdate(&A22, &y21, &a12t, -taupv)
        // A22 := A22 - taup*z21*u
        blasd.MVUpdate(&A22, &a21, &z21, -tauqv)

        a21.Set(0, 0, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tpT,
            &tpB,   &tp0, &taup1,   taup, util.PBOTTOM)
    }
}


/*
 * Reduce a general M-by-N matrix A to upper or lower bidiagonal form B
 * by an ortogonal transformation A = Q*B*P.T,  B = Q.T*A*P
 *
 *
 * Arguments
 *   A     On entry, the real M-by-N matrix. On exit the upper/lower
 *         bidiagonal matrix and ortogonal matrices Q and P.
 *
 *   tauq  Scalar factors for elementary reflector forming the
 *         ortogonal matrix Q.
 *
 *   taup  Scalar factors for elementary reflector forming the
 *         ortogonal matrix P.
 *
 *   W     Workspace needed for reduction.
 *
 *   conf  Current blocking configuration. Optional.
 *
 *
 * Details
 *
 * Matrices Q and P are products of elementary reflectors H(k) and G(k)
 *
 * If M > N:
 *     Q = H(1)*H(2)*...*H(N)   and P = G(1)*G(2)*...*G(N-1)
 *
 * where H(k) = 1 - tauq*v*v.T and G(k) = 1 - taup*u*u.T
 *
 * Elementary reflector H(k) are stored on columns of A below the diagonal with
 * implicit unit value on diagonal entry. Vector TAUQ holds corresponding scalar
 * factors. Reflector G(k) are stored on rows of A right of first superdiagonal
 * with implicit unit value on superdiagonal. Corresponding scalar factors are
 * stored on vector TAUP.
 * 
 * If M < N:
 *   Q = H(1)*H(2)*...*H(N-1)   and P = G(1)*G(2)*...*G(N)
 *
 * where H(k) = 1 - tauq*v*v.T and G(k) = 1 - taup*u*u.T
 *
 * Elementary reflector H(k) are stored on columns of A below the first sub diagonal 
 * with implicit unit value on sub diagonal entry. Vector TAUQ holds corresponding 
 * scalar factors. Reflector G(k) are sotre on rows of A right of diagonal with
 * implicit unit value on superdiagonal. Corresponding scalar factors are stored
 * on vector TAUP.
 *
 * Contents of matrix A after reductions are as follows.
 *
 *    M = 6 and N = 5:                  M = 5 and N = 6:
 *
 *    (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
 *    (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
 *    (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
 *    (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
 *    (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
 *    (  v1  v2  v3  v4  v5 )
 */
func ReduceBidiag(A, tauq, taup, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    _ = conf

    wmin := wsBired(A, 0)
    wsz  := W.Len()
    if wsz < wmin {
        return gomas.NewError(gomas.EWORK, "ReduceBidiag", wmin)
    }
    if m(A) >= n(A) {
        unblkReduceBidiagLeft(A, tauq, taup, W)
    } else {
        unblkReduceBidiagRight(A, tauq, taup, W)
    }
    return err
}


/*
 * Multiply and replace C with product of C and Q or P where Q and P are real orthogonal matrices
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(1) H(2) . . . H(k)   and   P = G(1) G(2). . . G(k)
 *
 * as returned by ReduceBidiag().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     Bidiagonal reduction as returned by ReduceBidiag() where the lower trapezoidal
 *        part, on and below first subdiagonal, holds the product Q. The upper
 *        trapezoidal part holds the product P.
 *
 *  tau   The scalar factors of the elementary reflectors. If flag MULTQ is set then holds
 *        scalar factors for Q. If flag MULTP is set then holds scalar factors for P.
 *        Expected to be column vector is size min(M(A), N(A)).
 *
 *  bits  Indicators, valid bits LEFT, RIGHT, TRANS, MULTQ, MULTP
 *
 *        flags              result
 *        ------------------------------------------
 *        MULTQ,LEFT         C = Q*C     n(A) == m(C)
 *        MULTQ,RIGHT        C = C*Q     n(C) == m(A)
 *        MULTQ,TRANS,LEFT   C = Q.T*C   n(A) == m(C)
 *        MULTQ,TRANS,RIGHT  C = C*Q.T   n(C) == m(A)
 *        MULTP,LEFT         C = P*C     n(A) == m(C)
 *        MULTP,RIGHT        C = C*P     n(C) == m(A)
 *        MULTP,TRANS,LEFT   C = P.T*C   n(A) == m(C)
 *        MULTP,TRANS,RIGHT  C = C*P.T   n(C) == m(A)
 *
 */
func MultQBD(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var Qh, Ch, Ph, tauh cmat.FloatMatrix
    var err *gomas.Error = nil

    // if MULTP then flip TRANSPOSE-bit 
    if flags & gomas.MULTP != 0 {
        // LQ.P     = G(k)G(k-1)...G(1) and
        // Bidiag.P = G(1)G(2)...G(k)
        //  therefore flip the TRANSPOSE bit.
        if flags & gomas.TRANS != 0 {
            flags &= ^gomas.TRANS
        } else {
            flags |= gomas.TRANS
        }
    }

    if m(A) >= n(A) {
        switch flags & (gomas.MULTQ|gomas.MULTP) {
        case gomas.MULTQ:
            tauh.SubMatrix(tau, 0, 0, n(A), 1)
            err = MultQ(C, A, &tauh, W, flags, confs...)

        case gomas.MULTP:
            Ph.SubMatrix(A, 0, 1, n(A)-1, n(A)-1)
            tauh.SubMatrix(tau, 0, 0, n(A)-1, 1)
            if flags & gomas.RIGHT != 0 {
                Ch.SubMatrix(C, 0, 1, m(C), n(C)-1)
            } else {
                Ch.SubMatrix(C, 1, 0, m(C)-1, n(C))
            }
            err = MultLQ(&Ch, &Ph, &tauh, W, flags, confs...)
        }
    } else {
        switch flags & (gomas.MULTQ|gomas.MULTP) {
        case gomas.MULTQ:
            Qh.SubMatrix(A, 1, 0, m(A)-1, m(A)-1)
            tauh.SubMatrix(tau, 0, 0, m(A)-1, 1)
            if flags & gomas.RIGHT != 0 {
                Ch.SubMatrix(C, 0, 1, m(C), n(C)-1)
            } else {
                Ch.SubMatrix(C, 1, 0, m(C)-1, n(C))
            }
            err = MultQ(&Ch, &Qh, &tauh, W, flags, confs...)

        case gomas.MULTP: 
            tauh.SubMatrix(tau, 0, 0, m(A), 1)
            err = MultLQ(C, A, &tauh, W, flags, confs...)
        }
    }
    if err != nil {
        err.Update("MultQBD")
    }
    return err
}


func wsBired(A *cmat.FloatMatrix, lb int) int  {
    if lb == 0 {
        return m(A)+n(A)
    }
    return (lb + m(A) + n(A))*lb
}

func wsMultQbdLeft(A *cmat.FloatMatrix, lb int) int  {
    nq := wsMultQLeft(A, lb)
    np := wsMultLQLeft(A, lb)
    if np > nq {
        return np
    }
    return nq
}

func wsMultQbdRight(A *cmat.FloatMatrix, lb int) int  {
    nq := wsMultQRight(A, lb)
    np := wsMultLQRight(A, lb)
    if np > nq {
        return np
    }
    return nq
}


/*
 * Calculate worksize needed for bidiagonal reduction with a blocking
 * configuration.
 */
func WorksizeBidiag(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsBired(A, conf.LB)
}

func WorksizeMultQBD(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    nl := wsMultQbdLeft(A, conf.LB)
    nr := wsMultQbdRight(A, conf.LB)
    if nl > nr {
        return nl
    }
    return nr
    
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:



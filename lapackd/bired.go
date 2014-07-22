
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
 * (1) G.Van Zee, R. van de Geijn
 *       Algorithms for Reducing a Matrix to Condensed Form
 *       2010, Flame working note #53 
 */

/*
 * Basic bidiagonal reduction for one column and row computing
 * from left to right when M >= N.
 *
 *   A  = ( 1 - tauq*u*u.T ) * A * ( 1 - taup*v*v.T )
 *
 *      = ( 1 - tauq*( 1 )*( 1 u.T )) * A * (1 - taup* ( 0 )*( 0 v.T ))
 *                   ( u )                             ( v )
 *
 *  1. Compute first Householder reflector [1 u].T,  len(u) == len(a21)
 *
 *     a12'  = (1 - tauq*( 1 )(1 u.T))( a12 ) 
 *     A22'              ( u )        ( A22 ) 
 *      
 *           = ( a12 -  tauq * (a12 + u.T*A22) )  
 *             ( A22 -  tauq*v*(a12 + u.T*A22) ) 
 *
 *           = ( a12 - tauq * y21 )    [y21 = a12 + A22.T*u]
 *             ( A22 - tauq*v*y21 )
 *
 *  2. Compute second Householder reflector [v], len(v) == len(a12)
 *
 *      A''  = ( a21; A22' )(1 - taup*( 0 )*( 0 v.T )) 
 *                                    ( v )              
 *
 *           = ( a21; A22' ) - taup * ( 0;  A22'*v*v.T )
 *
 *     A22   = A22' - taup*A22'*v*v.T
 *           = A22  - tauq*u*y21 - taup*(A22 - tauq*u*y21)*v*v.T
 *           = A22  - tauq*u*y21 - taup*(A22*v - tauq*u*y21*v)*v.T
 *
 *     y21   = a12 + A22.T*u
 *     z21   = A22*v - tauq*u*y21*v 
 *
 * The unblocked algorithm
 * -----------------------
 *  1.  u, tauq := HOUSEV(a11, a21)
 *  2.  y21 := a12 + A22.T*u
 *  3.  a12 := a12 - tauq*u
 *  4.  v, taup := HOUSEV(a12)
 *  5.  beta := DOT(v, y21)
 *  6.  z21  := A22*v - tauq*beta*u
 *  7.  A22  := A22 - tauq*u*y21
 *  8.  A22  := A22 - taup*z21*v
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

        // [v == a12t, u == a21]
        beta  := blasd.Dot(&y21, &a12t)
        // z21 := tauq*beta*u
        blasd.Axpby(&z21, &a21, tauqv*beta, 0.0)
        // z21 := A22*v - z21
        blasd.MVMult(&z21, &A22, &a12t, 1.0, -1.0, gomas.NONE)
        // A22 := A22 - tauq*u*y21
        blasd.MVUpdate(&A22, &a21, &y21, -tauqv)
        // A22 := A22 - taup*z21*v
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
 * This is adaptation of BIRED_LAZY_UNB algorithm from (1).
 */
func unblkBuildBidiagLeft(A, tauq, taup, Y, Z *cmat.FloatMatrix) {
    var ATL, ATR, ABR cmat.FloatMatrix
    var A00, a01, A02, a10, a11, a12t, A20, a21, A22 cmat.FloatMatrix
    var YTL, YBR, ZTL, ZBR cmat.FloatMatrix
    var Y00, y10, Y20, y11, y21, Y22 cmat.FloatMatrix
    var Z00, z10, Z20, z11, z21, Z22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var tpT, tpB, tp0, taup1, tp2 cmat.FloatMatrix
    var w00 cmat.FloatMatrix
    var v0 float64

    // Y is workspace for building updates for first Householder.
    // And Z is space for build updates for second Householder
    // Y is n(A)-2,nb and Z is m(A)-1,nb  

    util.Partition2x2(
        &ATL, &ATR,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &YTL, nil,
        nil,  &YBR, Y, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &ZTL, nil,
        nil,  &ZBR, Z, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
    util.Partition2x1(
        &tpT,
        &tpB,  taup, 0, util.PTOP)

    k := 0
    for k < n(Y) {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12t,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&YTL,
            &Y00, nil,  nil,
            &y10, &y11, nil,
            &Y20, &y21, &Y22,   Y, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&ZTL,
            &Z00, nil,  nil,
            &z10, &z11, nil,
            &Z20, &z21, &Z22,   Z, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PBOTTOM)
        util.Repartition2x1to3x1(&tpT,
            &tp0,
            &taup1,
            &tp2,     taup, 1, util.PBOTTOM)
		
        // set temp vectors for this round
        w00.SubMatrix(Z, 0, n(Z)-1, n(&A20), 1)
        // ------------------------------------------------------
        // u10 == a10, U20 == A20, u21 == a21,
        // v10 == a01, V20 == A02, v21 == a12t
        if n(&Y20) > 0 {
            // a11 := a11 - u10t*y10 - z10*v10
            aa := blasd.Dot(&a10, &y10)
            aa += blasd.Dot(&z10, &a01)
            a11.Set(0, 0, a11.Get(0, 0) - aa)
            // a21 := a21 - U20*y10 - Z20*v10
            blasd.MVMult(&a21, &A20, &y10, -1.0, 1.0, gomas.NONE)
            blasd.MVMult(&a21, &Z20, &a01, -1.0, 1.0, gomas.NONE)
            // a12t := a12t - u10.T*Y20.T - z10.T*V20.T
            blasd.MVMult(&a12t, &Y20, &a10, -1.0, 1.0, gomas.NONE)
            blasd.MVMult(&a12t, &A02, &z10, -1.0, 1.0, gomas.TRANS)
            // here restore bidiagonal entry
            a01.Set(-1, 0, v0)
        }
        // Compute householder to zero subdiagonal entries
        computeHouseholder(&a11, &a21, &tauq1)
        tauqv := tauq1.Get(0, 0)
		
        // y21 := a12 + A22.T*u21 - Y20*U20.T*u21 - V20*Z20.T*u21
        blasd.Axpby(&y21, &a12t, 1.0, 0.0)
        blasd.MVMult(&y21, &A22, &a21, 1.0, 1.0, gomas.TRANSA)

        // w00 := U20.T*u21 [= A20.T*a21]
        blasd.MVMult(&w00, &A20, &a21, 1.0, 0.0, gomas.TRANS)
        // y21 := y21 - U20*w00 [U20 == A20]
        blasd.MVMult(&y21, &Y20, &w00, -1.0, 1.0, gomas.NONE)
        // w00 := Z20.T*u21
        blasd.MVMult(&w00, &Z20, &a21, 1.0, 0.0, gomas.TRANS)
        // y21 := y21 - V20*w00  [V20 == A02.T]
        blasd.MVMult(&y21, &A02, &w00, -1.0, 1.0, gomas.TRANS)

        // a12t := a12t - tauq*y21
        blasd.Scale(&y21, tauqv)
        blasd.Axpy(&a12t, &y21, -1.0)

        // Compute householder to zero elements above 1st superdiagonal
        computeHouseholderVec(&a12t, &taup1)
        v0 = a12t.Get(0, 0)
        a12t.Set(0, 0, 1.0)
        taupv := taup1.Get(0, 0)

        // z21 := taup*(A22*v - U20*Y20.T*v - Z20*V20.T*v - beta*u)
        // [v == a12t, u == a21]
        beta  := blasd.Dot(&y21, &a12t)
        // z21 := beta*u
        blasd.Axpby(&z21, &a21, beta, 0.0)
        // w00 = Y20.T*v
        blasd.MVMult(&w00, &Y20, &a12t, 1.0, 0.0, gomas.TRANS)
        // z21 = z21 + U20*w00
        blasd.MVMult(&z21, &A20, &w00, 1.0, 1.0, gomas.NONE)
        // w00 := V20.T*v  (V20.T == A02)
        blasd.MVMult(&w00, &A02, &a12t, 1.0, 0.0, gomas.NONE)
        // z21 := z21 + Z20*w00
        blasd.MVMult(&z21, &Z20, &w00, 1.0, 1.0, gomas.NONE)
        // z21 := -taup*z21 + taup*A22*v
        blasd.MVMult(&z21, &A22, &a12t, taupv, -taupv, gomas.NONE)

        k += 1
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &YTL, nil,
            nil,  &YBR,   &Y00, &y11, &Y22,   Y, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &ZTL, nil,
            nil,  &ZBR,   &Z00, &z11, &Z22,   Z, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tpT,
            &tpB,   &tp0, &taup1,   taup, util.PBOTTOM)
    }
    // restore 
    ATR.Set(-1, 0, v0)
}

/*
 * This is adaptation of BIRED_BLK algorithm from (1).
 */
func blkBidiagLeft(A, tauq, taup, W *cmat.FloatMatrix, lb int, conf *gomas.Config) {
    var ATL, ABR cmat.FloatMatrix
    var A00, A11, A12, A21, A22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var tpT, tpB, tp0, taup1, tp2 cmat.FloatMatrix
    var ZT, ZB, YT, YB cmat.FloatMatrix
    var Z0, Z1, Z2, Y0, Y1, Y2 cmat.FloatMatrix
    var Y, Z cmat.FloatMatrix
	
    // setup work buffers
    Z.SetBuf(m(A), lb, m(A),   W.Data())
    Y.SetBuf(n(A), lb, n(A), W.Data()[Z.Len():])

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
    util.Partition2x1(
        &tpT,
        &tpB,  taup, 0, util.PTOP)
    util.Partition2x1(
        &YT,
        &YB,  &Y, 0, util.PTOP)
    util.Partition2x1(
        &ZT,
        &ZB,  &Z, 0, util.PTOP)

    for m(&ABR) > lb && n(&ABR) > lb {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &A11, &A12,
            nil,  &A21, &A22,   A, lb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&tpT,
            &tp0,
            &taup1,
            &tp2,     taup, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&ZT,
            &Z0,
            &Z1,
            &Z2,     &Z, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&YT,
            &Y0,
            &Y1,
            &Y2,     &Y, lb, util.PBOTTOM)
		
        // ------------------------------------------------------

        //util.Merge2x1(&AL, &A11, &A21)
        unblkBuildBidiagLeft(&ABR, &tauq1, &taup1, &YB, &ZB)

        // set super-diagonal entry to one
        v0 := A12.Get(m(&A12)-1, 0)
        A12.Set(m(&A12)-1, 0, 1.0)

        // A22 := A22 - U2*Y2.T
        blasd.Mult(&A22, &A21, &Y2, -1.0, 1.0, gomas.TRANSB, conf)
        // A22 := A22 - Z2*V2.T
        blasd.Mult(&A22, &Z2, &A12, -1.0, 1.0, gomas.NONE, conf)

        // restore super-diagonal entry
        A12.Set(m(&A12)-1, 0, v0)

        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tpT,
            &tpB,   &tp0, &taup1,   taup, util.PBOTTOM)
        util.Continue3x1to2x1(
            &ZT,
            &ZB,   &Z0, &Z1,   &Z, util.PBOTTOM)
        util.Continue3x1to2x1(
            &YT,
            &YB,   &Y0, &Y1,   &Y, util.PBOTTOM)
    }
    
    if n(&ABR) > 0 {
        // do rest with unblocked 
        unblkReduceBidiagLeft(&ABR, &tqB, &tpB, W)
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
 * This is adaptation of BIRED_LAZY_UNB algorithm from (1).
 *
 * Z matrix accumulates updates of row transformations i.e. first
 * Householder that zeros off diagonal entries on row. Vector z21
 * is updates for current round, Z20 are already accumulated updates.
 * Vector z21 updates a12 before next transformation.
 *
 * Y matrix accumulates updates on column tranformations ie Householder
 * that zeros elements below sub-diagonal. Vector y21 is updates for current
 * round, Y20 are already accumulated updates.  Vector y21 updates
 * a21 befor next transformation.
 *
 * Z, Y matrices upper trigonal part is not needed, temporary vector
 * w00 that has maximum length of n(Y) is placed on the last column of
 * Z matrix on each iteration.
 */
func unblkBuildBidiagRight(A, tauq, taup, Y, Z *cmat.FloatMatrix) {
    var ATL, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a10, a11, a12t, A20, a21, A22 cmat.FloatMatrix
    var YTL, YBR, ZTL, ZBR cmat.FloatMatrix
    var Y00, y10, Y20, y11, y21, Y22 cmat.FloatMatrix
    var Z00, z10, Z20, z11, z21, Z22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var tpT, tpB, tp0, taup1, tp2 cmat.FloatMatrix
    var w00 cmat.FloatMatrix
    var v0 float64

    // Y is workspace for building updates for first Householder.
    // And Z is space for build updates for second Householder
    // Y is n(A)-2,nb and Z is m(A)-1,nb  

    util.Partition2x2(
        &ATL, nil,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &YTL, nil,
        nil,  &YBR, Y, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &ZTL, nil,
        nil,  &ZBR, Z, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
    util.Partition2x1(
        &tpT,
        &tpB,  taup, 0, util.PTOP)

    k := 0
    for k < n(Y) {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12t,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&YTL,
            &Y00, nil,  nil,
            &y10, &y11, nil,
            &Y20, &y21, &Y22,   Y, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&ZTL,
            &Z00, nil,  nil,
            &z10, &z11, nil,
            &Z20, &z21, &Z22,   Z, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PBOTTOM)
        util.Repartition2x1to3x1(&tpT,
            &tp0,
            &taup1,
            &tp2,     taup, 1, util.PBOTTOM)
		
        // set temp vectors for this round,
        w00.SubMatrix(Z, 0, n(Z)-1, m(&A02), 1)
        // ------------------------------------------------------
        // u10 == a10, U20 == A20, u21 == a21,
        // v10 == a01, V20 == A02, v21 == a12t
        if n(&Y20) > 0 {
            // a11 := a11 - u10t*z10 - y10*v10
            aa := blasd.Dot(&a10, &z10)
            aa += blasd.Dot(&y10, &a01)
            a11.Set(0, 0, a11.Get(0, 0) - aa)
            // a12t := a12t - V20*z10 - Z20*u10
            blasd.MVMult(&a12t, &A02, &y10, -1.0, 1.0, gomas.TRANS)
            blasd.MVMult(&a12t, &Z20, &a10, -1.0, 1.0, gomas.NONE)
            // a21 := a21 - Y20*v10 - U20*z10
            blasd.MVMult(&a21, &Y20, &a01, -1.0, 1.0, gomas.NONE)
            blasd.MVMult(&a21, &A20, &z10, -1.0, 1.0, gomas.NONE)
            // here restore bidiagonal entry
            a10.Set(0, -1, v0)
        }
        // Compute householder to zero superdiagonal entries
        computeHouseholder(&a11, &a12t, &taup1)
        taupv := taup1.Get(0, 0)
		
        // y21 := a21 + A22*v21 - Y20*U20.T*v21 - V20*Z20.T*v21
        blasd.Axpby(&y21, &a21, 1.0, 0.0)
        blasd.MVMult(&y21, &A22, &a12t, 1.0, 1.0, gomas.NONE)

        // w00 := U20.T*v21 [= A02*a12t]
        blasd.MVMult(&w00, &A02, &a12t, 1.0, 0.0, gomas.NONE)
        // y21 := y21 - U20*w00 [U20 == A20]
        blasd.MVMult(&y21, &Y20, &w00, -1.0, 1.0, gomas.NONE)
        // w00 := Z20.T*v21
        blasd.MVMult(&w00, &Z20, &a12t, 1.0, 0.0, gomas.TRANS)
        // y21 := y21 - V20*w00  [V20 == A02.T]
        blasd.MVMult(&y21, &A20, &w00, -1.0, 1.0, gomas.NONE)

        // a21 := a21 - taup*y21
        blasd.Scale(&y21, taupv)
        blasd.Axpy(&a21, &y21, -1.0)

        // Compute householder to zero elements below 1st subdiagonal
        computeHouseholderVec(&a21, &tauq1)
        v0 = a21.Get(0, 0)
        a21.Set(0, 0, 1.0)
        tauqv := tauq1.Get(0, 0)

        // z21 := tauq*(A22*y - V20*Y20.T*u - Z20*U20.T*u - beta*v)
        // [v == a12t, u == a21]
        beta  := blasd.Dot(&y21, &a21)
        // z21 := beta*v
        blasd.Axpby(&z21, &a12t, beta, 0.0)
        // w00 = Y20.T*u
        blasd.MVMult(&w00, &Y20, &a21, 1.0, 0.0, gomas.TRANS)
        // z21 = z21 + V20*w00 == A02.T*w00
        blasd.MVMult(&z21, &A02, &w00, 1.0, 1.0, gomas.TRANS)
        // w00 := U20.T*u  (U20.T == A20.T)
        blasd.MVMult(&w00, &A20, &a21, 1.0, 0.0, gomas.TRANS)
        // z21 := z21 + Z20*w00
        blasd.MVMult(&z21, &Z20, &w00, 1.0, 1.0, gomas.NONE)
        // z21 := -tauq*z21 + tauq*A22*v
        blasd.MVMult(&z21, &A22, &a21, tauqv, -tauqv, gomas.TRANS)
        // ------------------------------------------------------
        k += 1
        util.Continue3x3to2x2(
            &ATL, nil, 
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &YTL, nil,
            nil,  &YBR,   &Y00, &y11, &Y22,   Y, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &ZTL, nil,
            nil,  &ZBR,   &Z00, &z11, &Z22,   Z, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tpT,
            &tpB,   &tp0, &taup1,   taup, util.PBOTTOM)
    }
    // restore 
    ABL.Set(0, -1, v0)
}

func blkBidiagRight(A, tauq, taup, W *cmat.FloatMatrix, lb int, conf *gomas.Config) {
    var ATL, ABR cmat.FloatMatrix
    var A00, A11, A12, A21, A22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var tpT, tpB, tp0, taup1, tp2 cmat.FloatMatrix
    var ZT, ZB, YT, YB cmat.FloatMatrix
    var Z0, Z1, Z2, Y0, Y1, Y2 cmat.FloatMatrix
    var Y, Z cmat.FloatMatrix
	
    // setup work buffers
    Z.SetBuf(n(A), lb, n(A), W.Data())
    Y.SetBuf(m(A), lb, m(A), W.Data()[Z.Len():])

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
    util.Partition2x1(
        &tpT,
        &tpB,  taup, 0, util.PTOP)
    util.Partition2x1(
        &YT,
        &YB,  &Y, 0, util.PTOP)
    util.Partition2x1(
        &ZT,
        &ZB,  &Z, 0, util.PTOP)

    for m(&ABR) > lb && n(&ABR) > lb {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &A11, &A12,
            nil,  &A21, &A22,   A, lb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&tpT,
            &tp0,
            &taup1,
            &tp2,     taup, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&ZT,
            &Z0,
            &Z1,
            &Z2,     &Z, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&YT,
            &Y0,
            &Y1,
            &Y2,     &Y, lb, util.PBOTTOM)
        // ------------------------------------------------------
        unblkBuildBidiagRight(&ABR, &tauq1, &taup1, &YB, &ZB)

        // set sub-diagonal entry to one
        v0 := A21.Get(0, n(&A21)-1)
        A21.Set(0, n(&A21)-1, 1.0)

        // A22 := A22 - U2*Z2.T
        blasd.Mult(&A22, &A21, &Z2, -1.0, 1.0, gomas.TRANSB, conf)
        // A22 := A22 - Y2*V2.T
        blasd.Mult(&A22, &Y2, &A12, -1.0, 1.0, gomas.NONE, conf)

        // restore sub-diagonal entry
        A21.Set(0, n(&A21)-1, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tpT,
            &tpB,   &tp0, &taup1,   taup, util.PBOTTOM)
        util.Continue3x1to2x1(
            &ZT,
            &ZB,   &Z0, &Z1,   &Z, util.PBOTTOM)
        util.Continue3x1to2x1(
            &YT,
            &YB,   &Y0, &Y1,   &Y, util.PBOTTOM)
    }
    
    if n(&ABR) > 0 {
        // do rest with unblocked 
        unblkReduceBidiagRight(&ABR, &tqB, &tpB, W)
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
 * where H(k) = 1 - tauq*u*u.T and G(k) = 1 - taup*v*v.T
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
 * where H(k) = 1 - tauq*u*u.T and G(k) = 1 - taup*v*v.T
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
 *    (  d   e   v1  v1  v1 )           (  d   v1  v1  v1  v1  v1 )
 *    (  u1  d   e   v2  v2 )           (  e   d   v2  v2  v2  v2 )
 *    (  u1  u2  d   e   v3 )           (  u1  e   d   v3  v3  v3 )
 *    (  u1  u2  u3  d   e  )           (  u1  u2  e   d   v4  v4 )
 *    (  u1  u2  u3  u4  d  )           (  u1  u2  u3  e   d   v5 )
 *    (  u1  u2  u3  u4  u5 )
 */
func BDReduce(A, tauq, taup, W *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    _ = conf

    wmin := wsBired(A, 0)
    wsz  := W.Len()
    if wsz < wmin {
        return gomas.NewError(gomas.EWORK, "ReduceBidiag", wmin)
    }
    lb := conf.LB
    wneed := wsBired(A, lb)
    if wneed > wsz {
        lb = estimateLB(A, wsz, wsBired)
    }
    if m(A) >= n(A) {
        if lb > 0 && n(A) > lb {
            blkBidiagLeft(A, tauq, taup, W, lb, conf)
        } else {
            unblkReduceBidiagLeft(A, tauq, taup, W)
        }
    } else {
        if lb > 0 && m(A) > lb {
            blkBidiagRight(A, tauq, taup, W, lb, conf)
        } else {
            unblkReduceBidiagRight(A, tauq, taup, W)
        }
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
func BDMult(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
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
            err = QRMult(C, A, &tauh, W, flags, confs...)

        case gomas.MULTP:
            Ph.SubMatrix(A, 0, 1, n(A)-1, n(A)-1)
            tauh.SubMatrix(tau, 0, 0, n(A)-1, 1)
            if flags & gomas.RIGHT != 0 {
                Ch.SubMatrix(C, 0, 1, m(C), n(C)-1)
            } else {
                Ch.SubMatrix(C, 1, 0, m(C)-1, n(C))
            }
            err = LQMult(&Ch, &Ph, &tauh, W, flags, confs...)
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
            err = QRMult(&Ch, &Qh, &tauh, W, flags, confs...)

        case gomas.MULTP: 
            tauh.SubMatrix(tau, 0, 0, m(A), 1)
            err = LQMult(C, A, &tauh, W, flags, confs...)
        }
    }
    if err != nil {
        err.Update("BDMult")
    }
    return err
}

/*
 * Generate one of the orthogonal matrices Q or P.T determined by BDReduce() when
 * reducing a real matrix A to bidiagonal form. Q and P.T are defined as products
 * elementary reflectors H(i) or G(i) respectively.
 *
 * Orthogonal matrix Q is generated if flag WANTQ is set. And matrix P respectively
 * of flag WANTP is set.
 */
func BDBuild(A, tau, W *cmat.FloatMatrix, K, flags int, confs... *gomas.Config) *gomas.Error {
    var Qh, Ph, tauh, d, s cmat.FloatMatrix
    var err *gomas.Error = nil

    if m(A) == 0 || n(A) == 0 {
        return nil
    }

    if m(A) >= n(A) {
        switch flags & (gomas.WANTQ|gomas.WANTP) {
        case gomas.WANTQ:
            tauh.SubMatrix(tau, 0, 0, n(A), 1)
            err = QRBuild(A, &tauh, W, K, confs...)

        case gomas.WANTP:
            // Shift P matrix embedded in A down and fill first column and row
            // to unit vector
            for j := n(A)-1; j > 0; j-- {
                s.SubMatrix(A, j-1, j, 1, n(A)-j)
                d.SubMatrix(A, j,   j, 1, n(A)-j)
                blasd.Copy(&d, &s)
                A.Set(j, 0, 0.0)
            }
            // zero  first row and set first entry to one
            d.Row(A, 0)
            blasd.Scale(&d, 0.0)
            d.Set(0, 0, 1.0)

            Ph.SubMatrix(A, 1, 1, n(A)-1, n(A)-1)
            tauh.SubMatrix(tau, 0, 0, n(A)-1, 1)
            if K > n(A)-1 {
                K = n(A) - 1
            }
            err = LQBuild(&Ph, &tauh, W, K, confs...)
        }
    } else {
        switch flags & (gomas.WANTQ|gomas.WANTP) {
        case gomas.WANTQ: 
            // Shift Q matrix embedded in A right and fill first column and row
            // to unit vector
            for j := m(A)-1; j > 0; j-- {
                s.SubMatrix(A, j, j-1, m(A)-j, 1)
                d.SubMatrix(A, j, j,   m(A)-j, 1)
                blasd.Copy(&d, &s)
                A.Set(0, j, 0.0)
            }
            // zero first column and set first entry to one
            d.Column(A, 0)
            blasd.Scale(&d, 0.0)
            d.Set(0, 0, 1.0)

            Qh.SubMatrix(A, 1, 1, m(A)-1, m(A)-1)
            tauh.SubMatrix(tau, 0, 0, m(A)-1, 1)
            if K > m(A) - 1 {
                K = m(A) - 1
            }
            err = QRBuild(&Qh, &tauh, W, K, confs...)

        case gomas.WANTP:
            tauh.SubMatrix(tau, 0, 0, m(A), 1)
            err = LQBuild(A, &tauh, W, K, confs...)
        }
    }
    if err != nil {
        err.Update("BDBuild")
    }
    return err
}

func wsBired(A *cmat.FloatMatrix, lb int) int  {
    if lb == 0 {
        return m(A)+n(A)
    }
    return (m(A) + n(A))*lb
}

func wsMultBidiagLeft(A *cmat.FloatMatrix, lb int) int  {
    nq := wsMultQLeft(A, lb)
    np := wsMultLQLeft(A, lb)
    if np > nq {
        return np
    }
    return nq
}

func wsMultBidiagRight(A *cmat.FloatMatrix, lb int) int  {
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
func BDReduceWork(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsBired(A, conf.LB)
}

func BDMultWork(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    nl := wsMultBidiagLeft(A, conf.LB)
    nr := wsMultBidiagRight(A, conf.LB)
    if nl > nr {
        return nl
    }
    return nr
    
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:




// Copyright (c) Harri Rautila, 2013,2014

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
 * Tridiagonal reduction of LOWER triangular symmetric matrix, zero elements below 1st
 * subdiagonal:
 *
 *   A =  (1 - tau*u*u.t)*A*(1 - tau*u*u.T)
 *     =  (I - tau*( 0   0   )) (a11 a12) (I - tau*( 0  0   ))
 *        (        ( 0  u*u.t)) (a21 A22) (        ( 0 u*u.t))
 *
 *  a11, a12, a21 not affected
 *
 *  from LEFT:
 *    A22 = A22 - tau*u*u.T*A22
 *  from RIGHT:
 *    A22 = A22 - tau*A22*u.u.T
 *
 *  LEFT and RIGHT:
 *    A22   = A22 - tau*u*u.T*A22 - tau*(A22 - tau*u*u.T*A22)*u*u.T
 *          = A22 - tau*u*u.T*A22 - tau*A22*u*u.T + tau*tau*u*u.T*A22*u*u.T
 *    [x    = tau*A22*u (vector)]  (SYMV)
 *    A22   = A22 - u*x.T - x*u.T + tau*u*u.T*x*u.T
 *    [beta = tau*u.T*x (scalar)]  (DOT)
 *          = A22 - u*x.T - x*u.T + beta*u*u.T
 *          = A22 - u*(x - 0.5*beta*u).T - (x - 0.5*beta*u)*u.T
 *    [w    = x - 0.5*beta*u]      (AXPY)
 *          = A22 - u*w.T - w*u.T  (SYR2)
 *
 * Result of reduction for N = 5:
 *    ( d  .  .  . . )
 *    ( e  d  .  . . )
 *    ( v1 e  d  . . )
 *    ( v1 v2 e  d . )
 *    ( v1 v2 v3 e d )
 */
func unblkReduceTridiagLower(A, tauq, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a11, a21, A22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var y21 cmat.FloatMatrix
    var v0 float64
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
	
    for m(&ABR) > 0 && n(&ABR) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &a11, nil,
            nil,  &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PBOTTOM)
        // set temp vectors for this round
        y21.SetBuf(n(&A22),  1, n(&A22),  W.Data())
        // ------------------------------------------------------

        // Compute householder to zero subdiagonal entries
        computeHouseholderVec(&a21, &tauq1)
        tauqv := tauq1.Get(0, 0)
		
        // set subdiagonal to unit
        v0 = a21.Get(0, 0)
        a21.Set(0, 0, 1.0)
        
        // y21 := tauq*A22*a21
        blasd.MVMultSym(&y21, &A22, &a21, tauqv, 0.0, gomas.LOWER)
        // beta := tauq*a21.T*y21
        beta := tauqv*blasd.Dot(&a21, &y21)
        // y21  := y21 - 0.5*beta*a21
        blasd.Axpy(&y21, &a21, -0.5*beta)
        // A22 := A22 - a21*y21.T - y21*a21.T 
        blasd.MVUpdate2Sym(&A22, &a21, &y21, -1.0, gomas.LOWER)

        // restore subdiagonal
        a21.Set(0, 0, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
    }
}

/*
 * This is adaptation of TRIRED_LAZY_UNB algorithm from (1).
 */
func unblkBuildTridiagLower(A, tauq, Y, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a10, a11, A20, a21, A22 cmat.FloatMatrix
    var YTL, YBR cmat.FloatMatrix
    var Y00, y10, y11, Y20, y21, Y22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var w12 cmat.FloatMatrix
    var v0 float64
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x2(
        &YTL, nil,
        nil,  &YBR, Y, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
	
    k := 0
    for k < n(Y) {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10,  &a11, nil,
            &A20,  &a21, &A22,   A, 1, util.PBOTTOMRIGHT)
        util.Repartition2x2to3x3(&YTL,
            &Y00, nil,  nil,
            &y10,  &y11, nil,
            &Y20,  &y21, &Y22,  Y, 1, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PBOTTOM)
        // set temp vectors for this round
        //w12.SetBuf(y10.Len(), 1, y10.Len(), W.Data())
        w12.SubMatrix(Y, 0, 0, 1, n(&Y00))
        // ------------------------------------------------------

        if n(&Y00) > 0 {
            aa := blasd.Dot(&a10, &y10)
            aa += blasd.Dot(&y10, &a10)
            a11.Set(0, 0, a11.Get(0, 0) - aa)
            
            // a21 := a21 - A20*y10
            blasd.MVMult(&a21, &A20, &y10, -1.0, 1.0, gomas.NONE)
            // a21 := a21 - Y20*a10
            blasd.MVMult(&a21, &Y20, &a10, -1.0, 1.0, gomas.NONE)

            // restore subdiagonal value
            a10.Set(0, -1, v0)
        }
        // Compute householder to zero subdiagonal entries
        computeHouseholderVec(&a21, &tauq1)
        tauqv := tauq1.Get(0, 0)
		
        // set subdiagonal to unit
        v0 = a21.Get(0, 0)
        a21.Set(0, 0, 1.0)
        
        // y21 := tauq*A22*a21
        blasd.MVMultSym(&y21, &A22, &a21, tauqv, 0.0, gomas.LOWER)
        // w12 := A20.T*a21
        blasd.MVMult(&w12, &A20, &a21, 1.0, 0.0, gomas.TRANS)
        // y21 := y21 - Y20*(A20.T*a21)
        blasd.MVMult(&y21, &Y20, &w12, -tauqv, 1.0, gomas.NONE)
        // w12 := Y20.T*a21
        blasd.MVMult(&w12, &Y20, &a21, 1.0, 0.0, gomas.TRANS)
        // y21 := y21 - A20*(Y20.T*a21)
        blasd.MVMult(&y21, &A20, &w12, -tauqv, 1.0, gomas.NONE)
        
        // beta := tauq*a21.T*y21
        beta := tauqv*blasd.Dot(&a21, &y21)
        // y21  := y21 - 0.5*beta*a21
        blasd.Axpy(&y21, &a21, -0.5*beta)

        // ------------------------------------------------------
        k += 1
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x3to2x2(
            &YTL, nil,
            nil,  &YBR,   &Y00, &y11, &Y22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
    }
    // restore subdiagonal value
    A.Set(m(&ATL), n(&ATL)-1, v0)
}


func blkReduceTridiagLower(A, tauq, Y, W *cmat.FloatMatrix, lb int, conf *gomas.Config) {
    var ATL, ABR cmat.FloatMatrix
    var A00, A11, A21, A22 cmat.FloatMatrix
    var YT, YB, Y0, Y1, Y2 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var v0 float64
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition2x1(
        &YT,
        &YB,  Y, 0, util.PTOP)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PTOP)
	
    for m(&ABR)-lb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            nil,  &A11, nil,
            nil,  &A21, &A22,   A, lb, util.PBOTTOMRIGHT)
        util.Repartition2x1to3x1(&YT,
            &Y0,
            &Y1,
            &Y2,     Y, lb, util.PBOTTOM)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, lb, util.PBOTTOM)
        // ------------------------------------------------------
        unblkBuildTridiagLower(&ABR, &tauq1, &YB, W)

        // set subdiagonal entry to unit
        v0 = A21.Get(0, -1)
        A21.Set(0, -1, 1.0)

        // A22 := A22 - A21*Y2.T - Y2*A21.T
        blasd.Update2Sym(&A22, &A21, &Y2, -1.0, 1.0, gomas.LOWER, conf)

        // restore subdiagonal entry
        A21.Set(0, -1, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue3x1to2x1(
            &YT,
            &YB,   &Y0, &Y1,   Y, util.PBOTTOM)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PBOTTOM)
    }

    if m(&ABR) > 0 {
        unblkReduceTridiagLower(&ABR, &tqB, W)
    }
}


/*
 * Reduce upper triangular matrix to tridiagonal.
 *
 * Elementary reflectors Q = H(n-1)...H(2)H(1) are stored on upper
 * triangular part of A. Reflector H(n-1) saved at column A(n) and
 * scalar multiplier to tau[n-1]. If parameter `tail` is true then
 * this function is used to reduce tail part of partially reduced
 * matrix and tau-vector partitioning is starting from last position. 
 */
func unblkReduceTridiagUpper(A, tauq, W *cmat.FloatMatrix, tail bool) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a01, a11, A22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var y21 cmat.FloatMatrix
    var v0 float64
	
    toff := 1
    if tail {
        toff = 0
    }
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, toff, util.PBOTTOM)
	
    for n(&ATL) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01,  nil,
            nil,  &a11, nil,
            nil,  nil,  &A22,   A, 1, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PTOP)
        // set temp vectors for this round
        y21.SetBuf(n(&A00),  1, n(&A00),  W.Data())
        // ------------------------------------------------------

        // Compute householder to zero super-diagonal entries
        computeHouseholderRev(&a01, &tauq1)
        tauqv := tauq1.Get(0, 0)
		
        // set superdiagonal to unit
        v0 = a01.Get(-1, 0)
        a01.Set(-1, 0, 1.0)
        
        // y21 := A22*a12t
        blasd.MVMultSym(&y21, &A00, &a01, tauqv, 0.0, gomas.UPPER)
        // beta := tauq*a12t*y21
        beta := tauqv*blasd.Dot(&a01, &y21)
        // y21  := y21 - 0.5*beta*a125
        blasd.Axpy(&y21, &a01, -0.5*beta)
        // A22 := A22 - a12t*y21.T - y21*a12.T
        blasd.MVUpdate2Sym(&A00, &a01, &y21, -1.0, gomas.UPPER)

        // restore superdiagonal value
        a01.Set(-1, 0, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PTOP)
    }
}

/*
 * This is adaptation of TRIRED_LAZY_UNB algorithm from (1).
 */
func unblkBuildTridiagUpper(A, tauq, Y, W *cmat.FloatMatrix) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a01, A02, a11, a12, A22 cmat.FloatMatrix
    var YTL, YBR cmat.FloatMatrix
    var Y00, y01, Y02, y11, y12, Y22 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var w12 cmat.FloatMatrix
    var v0 float64
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x2(
        &YTL, nil,
        nil,  &YBR, Y, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 0, util.PBOTTOM)
	
    k := 0
    for k < n(Y) {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            nil,  &a11, &a12,
            nil,  nil,  &A22,   A, 1, util.PTOPLEFT)
        util.Repartition2x2to3x3(&YTL,
            &Y00, &y01, &Y02,
            nil,  &y11, &y12,
            nil,  nil,  &Y22,  Y, 1, util.PTOPLEFT)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, 1, util.PTOP)

        // set temp vectors for this round
        w12.SubMatrix(Y, -1, 0, 1, n(&Y02))
        // ------------------------------------------------------

        if n(&Y02) > 0 {
            aa := blasd.Dot(&a12, &y12)
            aa += blasd.Dot(&y12, &a12)
            a11.Set(0, 0, a11.Get(0, 0) - aa)
            
            // a01 := a01 - A02*y12
            blasd.MVMult(&a01, &A02, &y12, -1.0, 1.0, gomas.NONE)
            // a01 := a01 - Y02*a12
            blasd.MVMult(&a01, &Y02, &a12, -1.0, 1.0, gomas.NONE)

            // restore superdiagonal value
            a12.Set(0, 0, v0)
        }
        // Compute householder to zero subdiagonal entries
        computeHouseholderRev(&a01, &tauq1)
        tauqv := tauq1.Get(0, 0)
		
        // set sub&iagonal to unit
        v0 = a01.Get(-1, 0)
        a01.Set(-1, 0, 1.0)
        
        // y01 := tauq*A00*a01
        blasd.MVMultSym(&y01, &A00, &a01, tauqv, 0.0, gomas.UPPER)
        // w12 := A02.T*a01
        blasd.MVMult(&w12, &A02, &a01, 1.0, 0.0, gomas.TRANS)
        // y01 := y01 - Y02*(A02.T*a01)
        blasd.MVMult(&y01, &Y02, &w12, -tauqv, 1.0, gomas.NONE)
        // w12 := Y02.T*a01
        blasd.MVMult(&w12, &Y02, &a01, 1.0, 0.0, gomas.TRANS)
        // y01 := y01 - A02*(Y02.T*a01)
        blasd.MVMult(&y01, &A02, &w12, -tauqv, 1.0, gomas.NONE)
        
        // beta := tauq*a01.T*y01
        beta := tauqv*blasd.Dot(&a01, &y01)
        // y01  := y01 - 0.5*beta*a01
        blasd.Axpy(&y01, &a01, -0.5*beta)

        // ------------------------------------------------------
        k += 1
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        util.Continue3x3to2x2(
            &YTL, nil,
            nil,  &YBR,   &Y00, &y11, &Y22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PTOP)
    }
    // restore superdiagonal value
    A.Set(m(&ATL)-1, n(&ATL), v0)
}

func blkReduceTridiagUpper(A, tauq, Y, W *cmat.FloatMatrix, lb int, conf *gomas.Config) {
    var ATL, ABR cmat.FloatMatrix
    var A00, A01, A11, A22 cmat.FloatMatrix
    var YT, YB, Y0, Y1, Y2 cmat.FloatMatrix
    var tqT, tqB, tq0, tauq1, tq2 cmat.FloatMatrix
    var v0 float64
	
    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    util.Partition2x1(
        &YT,
        &YB,  Y, 0, util.PBOTTOM)
    util.Partition2x1(
        &tqT,
        &tqB,  tauq, 1, util.PBOTTOM)
	
    for m(&ATL)-lb > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            nil,  &A11, nil,
            nil,  nil,  &A22,   A, lb, util.PTOPLEFT)
        util.Repartition2x1to3x1(&YT,
            &Y0,
            &Y1,
            &Y2,     Y, lb, util.PTOP)
        util.Repartition2x1to3x1(&tqT,
            &tq0,
            &tauq1,
            &tq2,     tauq, lb, util.PTOP)
        // ------------------------------------------------------
        unblkBuildTridiagUpper(&ATL, &tauq1, &YT, W)
        
        // set subdiagonal entry to unit
        v0 = A01.Get(-1, 0)
        A01.Set(-1, 0, 1.0)

        // A22 := A22 - A01*Y0.T - Y0*A01.T
        blasd.Update2Sym(&A00, &A01, &Y0, -1.0, 1.0, gomas.UPPER, conf)

        // restore subdiagonal entry
        A01.Set(-1, 0, v0)
        // ------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &A11, &A22,   A, util.PTOPLEFT)
        util.Continue3x1to2x1(
            &YT,
            &YB,   &Y0, &Y1,   Y, util.PTOP)
        util.Continue3x1to2x1(
            &tqT,
            &tqB,   &tq0, &tauq1,   tauq, util.PTOP)
    }

    if m(&ATL) > 0 {
        unblkReduceTridiagUpper(&ATL, &tqT, W, true)
    }
}

/*
 * Reduce symmetric matrix to tridiagonal form by similiarity transformation A = Q*T*Q.T
 *
 * Arguments
 *  A      On entry, symmetric matrix with elemets stored in upper (lower) triangular
 *         part. On exit, diagonal and first super (sub) diagonals hold matrix T. The upper
 *         (lower) triangular part above (below) first super(sub)diagonal is used to store
 *         orthogonal matrix Q.
 *
 *  tau    Scalar coefficients of elementary reflectors.
 *
 *  W      Workspace
 *
 *  flags  LOWER or UPPER
 *
 *  confs  Optional blocking configuration
 *
 * If LOWER, then the matrix Q is represented as product of elementary reflectors
 *
 *   Q = H(1)H(2)...H(n-1).
 *
 * If UPPER, then the matrix Q is represented as product 
 * 
 *   Q = H(n-1)...H(2)H(1).
 *
 * Each H(k) has form I - tau*v*v.T.
 *
 * The contents of A on exit is as follow for N = 5.
 *
 *  LOWER                    UPPER
 *   ( d  .  .  .  . )         ( d  e  v1 v2 v3 )
 *   ( e  d  .  .  . )         ( .  d  e  v2 v3 )
 *   ( v1 e  d  .  . )         ( .  .  d  e  v3 )
 *   ( v1 v2 e  d  . )         ( .  .  .  d  e  )
 *   ( v1 v2 v3 e  d )         ( .  .  .  .  d  )
 */
func TRDReduce(A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var Y cmat.FloatMatrix

    ok := m(A) == n(A) && tau.Len() >= n(A)
    if ! ok {
        return gomas.NewError(gomas.ESIZE, "ReduceTridiag")
    }

    conf := gomas.CurrentConf(confs...)
    lb   := conf.LB
    wsmin := wsTridiag(A, 0)
    if W.Len() < wsmin {
        return gomas.NewError(gomas.EWORK, "ReduceTridiag", wsmin)
    }

    if flags & gomas.LOWER != 0 {
        if lb == 0 || n(A)-1 < lb {
            unblkReduceTridiagLower(A, tau, W)
        } else {
            Y.SetBuf(m(A), lb, m(A), W.Data())
            blkReduceTridiagLower(A, tau, &Y, W, lb, conf)
        }
    } else {
        if lb == 0 || n(A)-1 < lb {
            unblkReduceTridiagUpper(A, tau, W, false)
        } else {
            Y.SetBuf(m(A), lb, m(A), W.Data())
            blkReduceTridiagUpper(A, tau, &Y, W, lb, conf)
        }
    }
    return err
}

func wsTridiag(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 {
        return m(A)
    }
    return lb*m(A)
}

func TRDReduceWork(A *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsTridiag(A, conf.LB)
}

/*
 * 
 */
func TRDMult(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var Ch, Qh, tauh cmat.FloatMatrix
    
    if flags & gomas.LOWER != 0 {
        if flags & gomas.LEFT != 0 {
            Ch.SubMatrix(C, 1, 0, m(C)-1, n(C))
        } else {
            Ch.SubMatrix(C, 0, 1, m(C), n(C)-1)
        }
        Qh.SubMatrix(A, 1, 0, m(A)-1, m(A)-1)
        tauh.SubMatrix(tau, 0, 0, m(A)-1, 1)
        err = QRMult(&Ch, &Qh, &tauh, W, flags, confs...)
    } else {
        if flags & gomas.LEFT != 0 {
            Ch.SubMatrix(C, 0, 0, m(C)-1, n(C))
        } else {
            Ch.SubMatrix(C, 0, 0, m(C), n(C)-1)
        }
        Qh.SubMatrix(A, 0, 1, m(A)-1, m(A)-1)
        tauh.SubMatrix(tau, 0, 0, m(A)-1, 1)
        err = QLMult(&Ch, &Qh, &tauh, W, flags, confs...)
    }
    return err
}

func TRDMultWork(A *cmat.FloatMatrix, flags int, confs... *gomas.Config) int {
    if flags & gomas.UPPER != 0 { 
        return QLMultWork(A, flags, confs...)
    }
    return QRMultWork(A, flags, confs...)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:



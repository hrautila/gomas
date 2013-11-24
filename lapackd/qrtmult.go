
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
 * Blocked version for computing C = Q*C and C = Q.T*C with block reflector.
 *
 * Block reflector T is [T(0), T(1), ... T(k-1)], conf.LB*n(A) matrix where
 * where each T(n), expect T(k-1), is conf.LB*conf.LB. T(k-1) is IB*IB where
 * IB = imin(LB, K%LB)
 */
func blockedMultQTLeft(C, A, T, W *cmat.FloatMatrix, flags int, conf *gomas.Config) *gomas.Error {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A10, A11, A20, A21, A22 cmat.FloatMatrix
    var CT, CB, C0, C1, C2 cmat.FloatMatrix
    var TL, TR, T00, T01, T02  cmat.FloatMatrix
    var Wrk cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pCdir, pCstart, pTstart, pTdir util.Direction
    var bsz, mb, tb int

    lb := conf.LB
    if conf.LB == 0 {
        lb = m(T)
    }
    transpose := flags & gomas.TRANS != 0
    nb := lb
    //W0 := cmat.MakeMatrix(n(C), conf.LB, W.Data())

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from top-left to bottom-right to produce transposed sequence (Q.T*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pCstart = util.PTOP
        pCdir   = util.PBOTTOM
        pTstart = util.PLEFT
        pTdir   = util.PRIGHT
        mb      = 0
        tb      = nb
        Aref    = &ABR
    } else {
        // from bottom-right to top-left to produce normal sequence (Q*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pCstart = util.PBOTTOM
        pCdir   = util.PTOP
        pTstart = util.PRIGHT
        pTdir   = util.PLEFT
        mb      = m(A) - n(A)
        Aref    = &ATL
        // if N%LB != 0 then the last T is not of size LB and we need
        // adjust first repartitioning accordingly.
        tb      = n(A) % lb
    }

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, mb, 0, pAstart)
    util.Partition1x2(
        &TL, &TR,     T, 0, pTstart)
    util.Partition2x1(
        &CT,
        &CB,    C, mb, pCstart)

    nb = tb
    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pAdir)
        util.Repartition1x2to1x3(&TL,
            &T00, &T01, &T02,   T, nb, pTdir)

        bsz = n(&A11)        // must match A11 block size
        util.Repartition2x1to3x1(&CT,
            &C0,
            &C1,
            &C2,     C, bsz, pCdir)
        // --------------------------------------------------------
        tr, tc := T01.Size()
        // compute: Q.T*C == C - Y*(C.T*Y*T).T  transpose == true
        //          Q*C   == C - C*Y*T*Y.T      transpose == false
        Wrk.SubMatrix(W, 0, 0, n(&C1), bsz)
        if tr != tc {
            // this happens when n(A) not multiple of LB
            var Tmp cmat.FloatMatrix
            Tmp.SubMatrix(&T01, 0, 0, tc, tc)
            updateWithQTLeft(&C1, &C2, &A11, &A21, &Tmp, &Wrk, nb, transpose, conf)
        } else {
            updateWithQTLeft(&C1, &C2, &A11, &A21, &T01, &Wrk, nb, transpose, conf)
        }

        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &TL, &TR,     &T00, &T01,   T, pTdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,   &C0, &C1,   C, pCdir)

        nb = lb
    }
    return nil
}


/*
 * Blocked version for computing C = C*Q and C = C*Q.T with block reflector.
 *
 * Block reflector T is [T(0), T(1), ... T(k-1)], conf.LB*n(A) matrix where
 * where each T(n), expect T(k-1), is conf.LB*conf.LB. T(k-1) is IB*IB where
 * IB = imin(LB, K%LB)
 */
func blockedMultQTRight(C, A, T, W *cmat.FloatMatrix, flags int, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A10, A11, A20, A21, A22 cmat.FloatMatrix
    var CL, CR, C0, C1, C2 cmat.FloatMatrix
    //var TTL, TTR, TBL, TBR cmat.FloatMatrix
    //var T00, T01, T02, T11, T12, T22 cmat.FloatMatrix
    var TL, TR, T00, T01, T02  cmat.FloatMatrix
    var Wrk cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pCstart, pCdir, pTstart, pTdir util.Direction
    var bsz, cb, mb, tb int

    lb := conf.LB
    if conf.LB == 0 {
        lb = m(T)
    }
    transpose := flags & gomas.TRANS != 0
    nb := lb

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from bottom-right to top-left to produce transpose sequence (C*Q.T)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pCstart = util.PRIGHT
        pCdir   = util.PLEFT
        pTstart = util.PRIGHT
        pTdir   = util.PLEFT
        mb      = imax(0, m(A) - n(A))
        cb      = imax(0, n(C) - n(A))
        tb      = n(A) % lb
        Aref = &ATL
    } else {
        // from top-left to bottom-right to produce normal sequence (C*Q)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        pTstart = util.PLEFT
        pTdir   = util.PRIGHT
        mb = 0
        cb = 0
        tb = nb
        Aref = &ABR
    }

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, mb, 0, pAstart)
    util.Partition1x2(
        &TL, &TR,     T, 0, pTstart)
    util.Partition1x2(
        &CL, &CR,     C, cb, pCstart)

    nb = tb
    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pAdir)
        util.Repartition1x2to1x3(&TL,
            &T00, &T01, &T02,   T, nb, pTdir)

        bsz = n(&A11)
        util.Repartition1x2to1x3(&CL,
            &C0,  &C1,  &C2,    C, bsz, pCdir)
        // --------------------------------------------------------
        tr, tc := T01.Size()
        // compute: C*Q.T == C - C*Y*T.T*Y.T   transpose == true
        //          C*Q   == C - C*Y*T*Y.T     transpose == false
        Wrk.SubMatrix(W, 0, 0, m(&C1), bsz)
        if tr != tc {
            // this happens when n(A) not multiple of LB
            var Tmp cmat.FloatMatrix
            Tmp.SubMatrix(&T01, 0, 0, tc, tc)
            updateWithQTRight(&C1, &C2, &A11, &A21, &Tmp, &Wrk, nb, transpose, conf)
        } else {
            updateWithQTRight(&C1, &C2, &A11, &A21, &T01, &Wrk, nb, transpose, conf)
        }
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &TL, &TR,     &T00, &T01,   T, pTdir)
        util.Continue1x3to1x2(
            &CL,  &CR,    &C0, &C1,     C, pCdir)

        nb = lb
    }
}



/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors and block reflector T
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 * as returned by DecomposeQRT().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C. On exit C is overwritten by Q*C or Q.T*C.
 *
 *  A     QR factorization as returned by DecomposeQRT() where the lower trapezoidal
 *        part holds the elementary reflectors.
 *
 *  T     The block reflector computed from elementary reflectors as returned by
 *        DecomposeQRT() or computed from elementary reflectors and scalar coefficients
 *        by BuildT()
 *
 *  W     Workspace, size as returned by WorkspaceMultQT()
 *
 *  conf  Blocking configuration
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS, NOTRANS
 *       
 * Preconditions:
 *   a.   cols(A) == cols(T),
 *          columns A define number of elementary reflector, must match order of block reflector.
 *   b.   if conf.LB == 0, cols(T) == rows(T)
 *          unblocked invocation, block reflector T is upper triangular
 *   c.   if conf.LB != 0, rows(T) == conf.LB
 *          blocked invocation, T is sequence of triangular block reflectors of order LB
 *   d.   if LEFT, rows(C) >= cols(A) && cols(C) >= rows(A)
 *
 *   e.   if RIGHT, cols(C) >= cols(A) && rows(C) >= rows(A)
 *
 * Compatible with lapack.DGEMQRT
 */
func MultQT(C, A, T, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)

    wsz := WorksizeMultQT(C, T, flags, conf)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "MultQT", wsz)
    }
    ok := false
    switch flags & gomas.RIGHT {
    case gomas.RIGHT:
        ok = n(C) >= m(A)
    default:
        ok = m(C) >= n(A)
    }
    if ! ok {
        return gomas.NewError(gomas.ESIZE, "MultQT")
    }

    var Wrk cmat.FloatMatrix
    if flags & gomas.RIGHT != 0 {
        Wrk.SetBuf(m(C), conf.LB, m(C), W.Data())
        blockedMultQTRight(C, A, T, &Wrk, flags, conf)

    } else {
        Wrk.SetBuf(n(C), conf.LB, n(C), W.Data())
        blockedMultQTLeft(C, A, T, &Wrk, flags, conf)
    }
    return err
}

/*
 * Calculate workspace size needed to compute C*Q or Q*C with QR decomposition
 * computed with DecomposeQRT().
 */
func WorksizeMultQT(C, T *cmat.FloatMatrix, bits int, confs... *gomas.Config) (sz int) {
    conf := gomas.CurrentConf(confs...)

    switch bits & gomas.RIGHT {
    case gomas.RIGHT:
        sz = m(C)
    default:
        sz = n(C)
    }
    if conf.LB > 0 {
        // add space for intermediate reflector and account
        // for blocking factor
        sz = (sz + conf.LB)*conf.LB
    }
    return
}


/*
 * Solve a system of linear equations A*X = B with general M-by-N
 * matrix A using the QR factorization computed by DecomposeQRT().
 *
 * If flags&gomas.TRANS != 0:
 *   find the minimum norm solution of an overdetermined system A.T * X = B.
 *   i.e min ||X|| s.t A.T*X = B
 *
 * Otherwise:
 *   find the least squares solution of an overdetermined system, i.e.,
 *   solve the least squares problem: min || B - A*X ||.
 *
 * Arguments:
 *  B     On entry, the right hand side N-by-P matrix B. On exit, the solution matrix X.
 *
 *  A     The elements on and above the diagonal contain the min(M,N)-by-N upper
 *        trapezoidal matrix R. The elements below the diagonal with the matrix 'T', 
 *        represent the ortogonal matrix Q as product of elementary reflectors.
 *        Matrix A and T are as returned by DecomposeQRT()
 *
 *  T     The block reflector computed from elementary reflectors as returned by
 *        DecomposeQRT() or computed from elementary reflectors and scalar coefficients
 *        by BuildT()
 *
 *  W     Workspace, size as returned by WorkspaceMultQT()
 *
 *  flags Indicator flag
 *
 *  conf  Blocking configuration
 *
 * Compatible with lapack.GELS (the m >= n part)
 */
func SolveQRT(B, A, T, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var R, BT cmat.FloatMatrix
    conf := gomas.CurrentConf(confs...)

    if flags & gomas.TRANS != 0 {
        // Solve overdetermined system A.T*X = B

        // B' = R.-1*B
        R.SubMatrix(A, 0, 0, n(A), n(A))
        BT.SubMatrix(B, 0, 0, n(A), n(B))
        err = blasd.SolveTrm(&BT, &R, 1.0, gomas.LEFT|gomas.UPPER|gomas.TRANSA, conf)
        
        // Clear bottom part of B
        BT.SubMatrix(B, n(A), 0)
        BT.SetFrom(cmat.NewFloatConstSource(0.0))
        
        // X = Q*B'
        err = MultQT(B, A, T, W, gomas.LEFT, conf)
    } else {
        // solve least square problem min ||A*X - B||

        // B' = Q.T*B
        err = MultQT(B, A, T, W, gomas.LEFT|gomas.TRANS, conf)
        if err != nil {
            return err
        }

        // X = R.-1*B'
        R.SubMatrix(A, 0, 0, n(A), n(A))
        BT.SubMatrix(B, 0, 0, n(A), n(B))
        err = blasd.SolveTrm(&BT, &R, 1.0, gomas.LEFT|gomas.UPPER, conf)
    }
    return err
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

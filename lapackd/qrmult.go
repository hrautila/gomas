
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
 * Unblocked algorith for computing C = Q.T*C and C = Q*C.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th column
 * below diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces H(k)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(k) == Q and C = Q*C
 */
func unblockedMultQLeft(C, A, tau, w *cmat.FloatMatrix, flags int) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, A20, a21, A22 cmat.FloatMatrix
    var CT, CB, C0, c1t, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w1 cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart util.Direction
    var mb, tb, nb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from top-left to bottom-right to produce transposed sequence (Q.T*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        mb      = 0
        tb      = 0
        nb      = 0
        Aref    = &ABR
    } else {
        // from bottom-right to top-left to produce normal sequence (Q*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref    = &ATL
    }

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,  A, mb, nb, pAstart)
    util.Partition2x1(
        &CT,
        &CB,    C, mb, pStart)
    util.Partition2x1(
        &tT,
        &tB,    tau, tb, pStart)

    for  m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, pAdir)
        util.Repartition2x1to3x1(&CT,
            &C0,
            &c1t,
            &C2,     C,   1, pDir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, pDir)

        // --------------------------------------------------------

        w1.SubMatrix(w, 0, 0, c1t.Len(), 1)
        applyHouseholder2x1(&tau1, &a21, &c1t, &C2, &w1, gomas.LEFT)

        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, pAdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,    &C0, &c1t,    C, pDir)
        util.Continue3x1to2x1(
            &tT,
            &tB,    &t0, &tau1,   tau, pDir)
    }
}

/*
 * Unblocked algorith for computing C = C*Q.T and C = C*Q.
 *
 * Q = H(1)H(2)...H(k) where elementary reflectors H(i) are stored on i'th column
 * below diagonal in A.
 *
 *     Q.T = (H1(1)*H(2)*...*H(k)).T
 *         = H(k).T*...*H(2).T*H(1).T
 *         = H(k)...H(2)H(1)
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces C*H(1)H(2)...H(k) == C*Q.
 *
 * Progressing from bottom-right to top-left produces C*H(k)...H(2)H(1) == C*Q.T.
 */
func unblockedMultQRight(C, A, tau, w *cmat.FloatMatrix, flags int) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, A20, a21, A22 cmat.FloatMatrix
    var CL, CR, C0, c1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w1  cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart, pCstart, pCdir util.Direction
    var cb, mb, tb, nb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from bottom-right to top-left to produce transpose sequence (C*Q.T)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        pCstart = util.PRIGHT
        pCdir   = util.PLEFT
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        cb      = imax(0, n(C) - n(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref = &ATL
    } else {
        // from top-left to bottom-right to produce normal sequence (C*Q)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        mb = 0
        cb = 0
        tb = 0
        nb = 0
        Aref = &ABR
    }

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,  A, mb, nb, pAstart)
    util.Partition1x2(
        &CL, &CR,    C, cb, pCstart)
    util.Partition2x1(
        &tT,
        &tB,  tau, tb, pStart)

    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, pAdir)
        util.Repartition1x2to1x3(&CL,
            &C0, &c1, &C2,      C, 1, pCdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, pDir)

        // --------------------------------------------------------

        w1.SubMatrix(w, 0, 0, c1.Len(), 1)
        applyHouseholder2x1(&tau1, &a21, &c1, &C2, &w1, gomas.RIGHT)

        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &CL, &CR,     &C0, &c1,           C, pCdir)
        util.Continue3x1to2x1(
            &tT,
            &tB,          &t0, &tau1,         tau, pDir)
    }
}


/*
 * Blocked version for computing C = Q*C and C = Q.T*C from elementary reflectors
 * and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block reflector T.
 * Matrix C is updated by applying block reflector T using compact WY algorithm.
 */
func blockedMultQLeft(C, A, tau, W *cmat.FloatMatrix, flags, nb int, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR, AL cmat.FloatMatrix
    var A00, A10, A11, A20, A21, A22 cmat.FloatMatrix
    var CT, CB, C0, C1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix
    var Wrk, W0, Tw, Twork cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart util.Direction
    var bsz, mb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 || nb == n(A) {
        // from top-left to bottom-right to produce transposed sequence (Q.T*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        mb      = 0
        Aref    = &ABR
    } else {
        // from bottom-right to top-left to produce normal sequence (Q*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        mb      = imax(0, m(A) - n(A))
        Aref    = &ATL
    }

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, mb, 0, pAstart)
    util.Partition2x1(
        &CT,
        &CB,    C, mb, pStart)
    util.Partition2x1(
        &tT,
        &tB,    tau, 0, pStart)

    transpose := flags & gomas.TRANS != 0

    // intermediate reflector at start of workspace
    Twork.SetBuf(nb, nb, nb, W.Data())
    W0.SetBuf(n(C), nb, n(C), W.Data()[Twork.Len():])

    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pAdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, nb, pDir)
        bsz = n(&A11)
        util.Repartition2x1to3x1(&CT,
            &C0,
            &C1,
            &C2,     C, bsz, pDir)
        // --------------------------------------------------------
        // clear & build block reflector from current block
        util.Merge2x1(&AL, &A11, &A21)
        Tw.SubMatrix(&Twork, 0, 0, bsz, bsz)
        blasd.Scale(&Tw, 0.0)
        unblkQRBlockReflector(&Tw, &AL, &tau1)

        // compute: Q*T.C == C - Y*(C.T*Y*T).T  transpose == true
        //          Q*C   == C - C*Y*T*Y.T      transpose == false
        Wrk.SubMatrix(&W0, 0, 0, n(&C1), bsz)
        updateWithQTLeft(&C1, &C2, &A11, &A21, &Tw, &Wrk, transpose, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pAdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,     &C0, &C1,     C, pDir)
        util.Continue3x1to2x1(
            &tT,
            &tB,     &t0, &tau1,   tau, pDir)
    }

}

/*
 * Blocked version for computing C = C*Q and C = C*Q.T from elementary reflectors
 * and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block reflector T.
 * Matrix C is updated by applying block reflector T using compact WY algorithm.
 */
func blockedMultQRight(C, A, tau, W *cmat.FloatMatrix, flags, nb int, conf *gomas.Config) {
    var ATL, ATR, ABL, ABR, AL cmat.FloatMatrix
    var A00, A10, A11, A20, A21, A22 cmat.FloatMatrix
    var CL, CR, C0, C1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix
    var W0, Wrk, Tw, Twork cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart, pCstart, pCdir util.Direction
    var bsz, cb, mb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from bottom-right to top-left to produce transpose sequence (C*Q.T)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        pCstart = util.PRIGHT
        pCdir   = util.PLEFT
        mb      = imax(0, m(A) - n(A))
        cb      = n(C) - n(A)
        Aref = &ATL
    } else {
        // from top-left to bottom-right to produce normal sequence (C*Q)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        mb = 0
        cb = 0
        Aref = &ABR
    }

    // intermediate reflector at start of workspace
    Twork.SetBuf(nb, nb, nb, W.Data())
    W0.SetBuf(m(C), nb, m(C), W.Data()[Twork.Len():])

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, mb, 0, pAstart)
    util.Partition1x2(
        &CL, &CR,   C, cb, pCstart)
    util.Partition2x1(
        &tT,
        &tB,        tau, 0, pStart)

    transpose := flags & gomas.TRANS != 0

    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nb, pAdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, nb, pDir)

        bsz = n(&A11)        // C1 block size must match A11 
        util.Repartition1x2to1x3(&CL,
            &C0, &C1, &C2 ,     C, bsz, pCdir)
        // --------------------------------------------------------
        // clear & build block reflector from current block
        util.Merge2x1(&AL, &A11, &A21)
        Tw.SubMatrix(&Twork, 0, 0, bsz, bsz)
        blasd.Scale(&Tw, 0.0)
        unblkQRBlockReflector(&Tw, &AL, &tau1)

        // compute: C*Q.T == C - C*(Y*T*Y.T).T = C - C*Y*T.T*Y.T
        //          C*Q   == C - C*Y*T*Y.T
        Wrk.SubMatrix(&W0, 0, 0, m(&C1), bsz)
        updateWithQTRight(&C1, &C2, &A11, &A21, &Tw, &Wrk, transpose, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &CL,   &CR,   &C0, &C1,    C, pCdir)
        util.Continue3x1to2x1(
            &tT,
            &tB,   &t0, &tau1,   tau, pDir)
    }

}



/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(0) H(1) . . . H(K-1)
 *
 * as returned by QRFactor().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     QR factorization as returned by QRFactor() where the lower trapezoidal
 *        part holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors.
 *
 *  W     Workspace matrix,  required size is returned by WorksizeMultQ().
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block size. If it is zero
 *        unblocked invocation is assumed. Actual blocking size is adjusted
 *        to available workspace size and minimum of configured block size and
 *        block size implied by workspace is used.
 *
 * Compatible with lapack.DORMQR
 */
func QRMult(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)

    // n(A) is number of elementary reflectors defining the Q matrix
    ok := false
    wsizer := wsMultQLeft
    switch flags & gomas.RIGHT {
    case gomas.RIGHT:
        ok = n(C) == m(A)
        wsizer = wsMultQRight
    default:
        ok = m(C) == m(A)
    }
    if ! ok {
        return gomas.NewError(gomas.ESIZE, "QRMult")
    }

    // minimum workspace size
    wsz := wsizer(C, 0)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "QRMult", wsz)
    }

    // estimate blocking factor for current workspace
    lb := estimateLB(C, W.Len(), wsizer)
    lb = imin(lb, conf.LB)
    if lb == 0 {
        if flags & gomas.RIGHT != 0 {
            unblockedMultQRight(C, A, tau, W, flags)
        } else {
            unblockedMultQLeft(C, A, tau, W, flags)
        }
    } else {
        if flags & gomas.RIGHT != 0 {
            blockedMultQRight(C, A, tau, W, flags, lb, conf)
        } else {
            blockedMultQLeft(C, A, tau, W, flags, lb, conf)
        }
    }
    return err
}


/*
 * Solve a system of linear equations A*X = B with general M-by-N
 * matrix A using the QR factorization computed by QRFactor().
 *
 * If flags&TRANS != 0:
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
 *        trapezoidal matrix R. The elements below the diagonal with the vector 'tau', 
 *        represent the ortogonal matrix Q as product of elementary reflectors.
 *        Matrix A and T are as returned by DecomposeQR()
 *
 *  tau   The vector of N scalar coefficients that together with trilu(A) define
 *        the ortogonal matrix Q as Q = H(1)H(2)...H(N)
 *
 *  W     Workspace, size required returned QRSolveWork().
 *
 *  flags Indicator flags
 *
 *  conf  Optinal blocking configuration. If not given default will be used. Unblocked
 *        invocation is indicated with conf.LB == 0.
 *
 * Compatible with lapack.GELS (the m >= n part)
 */
func QRSolve(B, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var R, BT cmat.FloatMatrix

    conf := gomas.CurrentConf(confs...)

    msz := QRMultWork(B, gomas.LEFT, conf)
    if W.Len() < msz {
        return gomas.NewError(gomas.EWORK, "SolveQR", msz)
    }

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
        err = QRMult(B, A, tau, W, gomas.LEFT, conf)
    } else {
        // solve least square problem min ||A*X - B||

        // B' = Q.T*B
        err = QRMult(B, A, tau, W, gomas.LEFT|gomas.TRANS, conf)
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


/*
 * Calculate workspace size needed to compute C*Q or Q*C with QR decomposition
 * computed with DecomposeQR().
 */
func QRMultWork(C *cmat.FloatMatrix, bits int, confs... *gomas.Config) (sz int) {
    conf := gomas.CurrentConf(confs...)
    switch bits & gomas.RIGHT {
    case gomas.RIGHT:
        sz = wsMultQRight(C, conf.LB)
    default:
        sz = wsMultQLeft(C, conf.LB)
    }
    return 
}

func QRSolveWork(C *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsMultQLeft(C, conf.LB)
}

func wsMultQRight(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 || lb > n(A) {
        return m(A)
    }
    // need space for block reflector T and intermediate results
    return lb*(m(A)+lb)
}

func wsMultQLeft(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 || lb > n(A) {
        return n(A)
    }
    // need space for block reflector T and intermediate results
    return lb*(n(A)+lb)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

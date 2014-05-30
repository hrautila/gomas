
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/util"
    "github.com/hrautila/gomas/blasd"
    //"fmt"
)


/*
 * Compute C := Q*C or C := Q.T*C where Q is the M-by-N orthogonal matrix
 * defined K elementary reflectors. (K = A.cols)
 */
func unblkMultLeftQL(C, A, tau, w *cmat.FloatMatrix, flags int) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a01, a11, A22 cmat.FloatMatrix
    var CT, CB, C0, c1t, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w1 cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart util.Direction
    var mb, tb, nb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // A from bottom-right to top-left to produce transposed sequence (Q.T*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        mb      = 0
        tb      = 0
        nb      = 0
        Aref    = &ATL
    } else {
        // from top-left to bottom-right to produce normal sequence (Q*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref    = &ABR
    }

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  A, mb, nb, pAstart)
    util.Partition2x1(
        &CT,
        &CB,    C, mb, pStart)
    util.Partition2x1(
        &tT,
        &tB,    tau, tb, pStart)

    w1.SubMatrix(w, 0, 0, n(C), 1)

    for  n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, nil,
            nil,  &a11, nil,
            nil,  nil,  &A22,   A, 1, pAdir)
        util.Repartition2x1to3x1(&CT,
            &C0,
            &c1t,
            &C2,     C,   1, pDir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, pDir)
        // --------------------------------------------------------
        applyHouseholder2x1(&tau1, &a01, &c1t, &C0, &w1, gomas.LEFT)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, pAdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,    &C0, &c1t,    C, pDir)
        util.Continue3x1to2x1(
            &tT,
            &tB,    &t0, &tau1,   tau, pDir)
    }
}


func blkMultLeftQL(C, A, tau, W *cmat.FloatMatrix, flags, lb int, conf *gomas.Config) {
    var ATL, /*ATR, ABL,*/ ABR, AL cmat.FloatMatrix
    var A00, A01, A11, A22 cmat.FloatMatrix
    var CT, CB, C0, C1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2 cmat.FloatMatrix
    var T0, T, W0, Wrk cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart util.Direction
    var mb, tb, nb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // A from bottom-right to top-left to produce transposed sequence (Q.T*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        mb      = 0
        tb      = 0
        nb      = 0
        Aref    = &ATL
    } else {
        // from top-left to bottom-right to produce normal sequence (Q*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref    = &ABR
    }

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  A, mb, nb, pAstart)
    util.Partition2x1(
        &CT,
        &CB,    C, mb, pStart)
    util.Partition2x1(
        &tT,
        &tB,    tau, tb, pStart)

    transpose := flags & gomas.TRANS != 0
    // divide workspace for block reflector and temporart space
    T0.SetBuf(lb, lb, lb, W.Data())
    W0.SetBuf(n(C), lb, n(C), W.Data()[T0.Len():])

    for  n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            nil,  &A11, nil,
            nil,  nil,  &A22,   A, lb, pAdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, lb, pDir)
        bsz := n(&A11)
        util.Repartition2x1to3x1(&CT,
            &C0,
            &C1,
            &C2,     C,   bsz, pDir)
        // --------------------------------------------------------
        // build block reflector for current block
        util.Merge2x1(&AL, &A01, &A11)
        T.SubMatrix(&T0, 0, 0, bsz, bsz)
        blasd.Scale(&T, 0.0)
        unblkQLBlockReflector(&T, &AL, &tau1)

        // update with (I - Y*T*Y.T) or (I - Y*T*Y.T).T
        Wrk.SubMatrix(&W0, 0, 0, n(&C1), bsz)
        updateQLLeft(&C1, &C0, &A11, &A01, &T, &Wrk, transpose, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &A11, &A22,   A, pAdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,    &C0, &C1,    C, pDir)
        util.Continue3x1to2x1(
            &tT,
            &tB,    &t0, &tau1,   tau, pDir)
    }
}



/*
 * Compute C := C*Q or C := C*Q.T where Q is the M-by-N orthogonal matrix
 * defined K elementary reflectors. (K = A.cols)
 */
func unblkMultRightQL(C, A, tau, w *cmat.FloatMatrix, flags int) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a01, a11, A22 cmat.FloatMatrix
    var CL, CR, C0, c1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w1 cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart, pCdir, pCstart util.Direction
    var mb, tb, nb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from top-left to bottom-right to produce normal sequence (Q*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref    = &ABR
    } else {
        // A from bottom-right to top-left to produce transposed sequence (Q.T*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        pCstart = util.PRIGHT
        pCdir   = util.PLEFT
        mb      = 0
        tb      = 0
        nb      = 0
        Aref    = &ATL
    }

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  A, mb, nb, pAstart)
    util.Partition1x2(
        &CL, &CR,    C, mb, pCstart)
    util.Partition2x1(
        &tT,
        &tB,         tau, tb, pStart)

    w1.SubMatrix(w, 0, 0, m(C), 1)
    for  n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, nil,
            nil,  &a11, nil,
            nil,  nil,  &A22,   A, 1, pAdir)
        util.Repartition1x2to1x3(&CL,
            &C0, &c1, &C2,      C,   1, pCdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     tau, 1, pDir)
        // --------------------------------------------------------
        applyHouseholder2x1(&tau1, &a01, &c1, &C0, &w1, gomas.RIGHT)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   &A00, &a11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &CL,  &CR,    &C0, &c1,     C, pCdir)
        util.Continue3x1to2x1(
            &tT,
            &tB,          &t0, &tau1,   tau, pDir)
    }
}


func blkMultRightQL(C, A, tau, W *cmat.FloatMatrix, flags, lb int, conf *gomas.Config) {
    var ATL, ABR, AL cmat.FloatMatrix
    var A00, A01, A11, A22 cmat.FloatMatrix
    var CL, CR, C0, C1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2 cmat.FloatMatrix
    var T0, T, W0, Wrk cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart, pCdir, pCstart util.Direction
    var mb, tb, nb, cb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from top-left to bottom-right to produce transpose sequence (C*Q.T)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        cb      = imax(0, n(C) - n(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref    = &ABR
    } else {
        // A from bottom-right to top-left to produce normal sequence (C*Q)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        pCstart = util.PRIGHT
        pCdir   = util.PLEFT
        mb      = 0
        tb      = 0
        nb      = 0
        cb      = 0
        Aref    = &ATL
    }

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  /**/ A, mb, nb, pAstart)
    util.Partition1x2(
        &CL, &CR,    /**/ C, cb, pCstart)
    util.Partition2x1(
        &tT,
        &tB,         /**/ tau, tb, pStart)

    transpose := flags & gomas.TRANS != 0
    // divide workspace for block reflector and temporary work matrix
    T0.SetBuf(lb, lb, lb, W.Data())
    W0.SetBuf(m(C), lb, m(C), W.Data()[T0.Len():])
    
    for  n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, nil,
            nil,  &A11, nil,
            nil,  nil,  &A22,  /**/ A, lb, pAdir)
        bsz := n(&A11)
        util.Repartition1x2to1x3(&CL,
            &C0, &C1, &C2,     /**/ C,   bsz, pCdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     /**/ tau, bsz, pDir)
        // --------------------------------------------------------
        util.Merge2x1(&AL, &A01, &A11)
        T.SubMatrix(&T0, 0, 0, bsz, bsz)
        blasd.Scale(&T, 0.0)
        unblkQLBlockReflector(&T, &AL, &tau1)

        Wrk.SubMatrix(&W0, 0, 0, m(C), bsz)
        updateQLRight(&C1, &C0, &A11, &A01, &T, &Wrk, transpose, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   /**/ &A00, &A11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &CL,  &CR,    /**/ &C0, &C1,     C, pCdir)
        util.Continue3x1to2x1(
            &tT,
            &tB,          /**/ &t0, &tau1,   tau, pDir)
    }
}


/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of K first elementary reflectors.
 *
 *    Q = H(k) H(k-1) . . . H(1)
 *
 * as returned by DecomposeQL().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then N-by-M matrix
 *        On exit C is overwritten by Q*C or Q.T*C. If bit RIGHT is set then C is
 *        overwritten by C*Q or C*Q.T
 *
 *  A     QL factorization as returned by DecomposeQL() where the upper trapezoidal
 *        part holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors.
 *
 *  W     Workspace matrix, size as returned by WorksizeMultQL().
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block size. If it is zero
 *        unblocked invocation is assumed. Actual blocking size is adjusted
 *        to available workspace size and the smaller of configured block size and
 *        block size implied by workspace is used.
 *
 * Compatible with lapack.DORMQL
 */
func QLMult(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)

    // n(A) is number of elementary reflectors defining the Q matrix
    ok := false
    wsizer := wsMultLeftQL
    switch flags & gomas.RIGHT {
    case gomas.RIGHT:
        ok = n(C) == m(A)
        wsizer = wsMultRightQL
    default:
        ok = m(C) == m(A)
    }
    if ! ok {
        return gomas.NewError(gomas.ESIZE, "QLMult")
    }

    // minimum workspace size
    wsz := wsizer(C, 0)
    if W == nil || W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "QLMult", wsz)
    }

    // estimate blocking factor for current workspace
    lb := estimateLB(C, W.Len(), wsizer)
    lb = imin(lb, conf.LB)
    if lb == 0 {
        if flags & gomas.RIGHT != 0 {
            unblkMultRightQL(C, A, tau, W, flags)
        } else {
            unblkMultLeftQL(C, A, tau, W, flags)
        }
    } else {
        if flags & gomas.RIGHT != 0 {
            blkMultRightQL(C, A, tau, W, flags, lb, conf)
        } else {
            blkMultLeftQL(C, A, tau, W, flags, lb, conf)
        }
    }
    return err
}


func QLMultWork(C *cmat.FloatMatrix, bits int, confs... *gomas.Config) (sz int) {
    conf := gomas.CurrentConf(confs...)
    switch bits & gomas.RIGHT {
    case gomas.RIGHT:
        sz = wsMultRightQL(C, conf.LB)
    default:
        sz = wsMultLeftQL(C, conf.LB)
    }
    return 
}

func QLSolveWork(C *cmat.FloatMatrix, confs... *gomas.Config) int {
    conf := gomas.CurrentConf(confs...)
    return wsMultLeftQL(C, conf.LB)
}

func wsMultRightQL(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 {
        return m(A)
    }
    // need space for block reflector T and intermediate results
    return lb*(m(A)+lb)
}

func wsMultLeftQL(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 {
        return n(A)
    }
    // need space for block reflector T and intermediate results
    return lb*(n(A)+lb)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

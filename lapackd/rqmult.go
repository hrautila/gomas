
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
 * Q = H(0)H(1)...H(K-1) where elementary reflectors H(i) are stored on i'th row
 * right of  diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller row numbers
 * to larger, produces H(k)...H(2)H(1) == Q.T. and C = Q.T*C
 *
 * Progressing from bottom-right to top-left produces H(1)H(2)...H(k) == Q and C = Q*C
 */
func unblockedMultRQLeft(C, A, tau, w *cmat.FloatMatrix, flags int) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a10, a11, A22 cmat.FloatMatrix
    var CT, CB, C0, c1t, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2, w1 cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart util.Direction
    var mb, tb, nb, cb int

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from top-left to bottom-right to produce transpose sequence (Q.T*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        cb      = imax(0, m(C) - m(A))
        tb      = imax(0, tau.Len() - m(A))
        Aref    = &ABR
    } else {
        // from bottom-right to top-left to produce normal sequence (Q*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        mb      = 0
        tb      = 0
        nb      = 0
        cb      = 0
        Aref    = &ATL
    }

    util.Partition2x2(
        &ATL, nil,
        nil, &ABR,  /**/ A, mb, nb, pAstart)
    util.Partition2x1(
        &CT,
        &CB,    /**/ C, cb, pStart)
    util.Partition2x1(
        &tT,
        &tB,    /**/ tau, tb, pStart)

    w1.SubMatrix(w, 0, 0, n(C), 1)
    for  m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            nil,  nil,  &A22,   /**/ A, 1, pAdir)
        util.Repartition2x1to3x1(&CT,
            &C0,
            &c1t,
            &C2,     /**/ C,   1, pDir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     /**/ tau, 1, pDir)
        // --------------------------------------------------------
        applyHouseholder2x1(&tau1, &a10, &c1t, &C0, &w1, gomas.LEFT)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil, &ABR,   /**/ &A00, &a11, &A22,   A, pAdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,    /**/ &C0, &c1t,    C, pDir)
        util.Continue3x1to2x1(
            &tT,
            &tB,    /**/ &t0, &tau1,   tau, pDir)
    }
}

/*
 * Unblocked algorith for computing C = C*Q.T and C = C*Q.
 *
 * Q = H(0)H(1)...H(K-1) where elementary reflectors H(i) are stored on i'th row
 * right of diagonal in A.
 *
 * Progressing A from top-left to bottom-right i.e from smaller column numbers
 * to larger, produces C*H(0)H(1)...H(K-1) == C*Q.
 *
 * Progressing from bottom-right to top-left produces C*H(K-1)...H(1)H(0) == C*Q.T.
 */
func unblockedMultRQRight(C, A, tau, w *cmat.FloatMatrix, flags int) {
    var ATL, ABR cmat.FloatMatrix
    var A00, a10, a11, A22 cmat.FloatMatrix
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
        mb = 0
        nb = 0
        cb = 0
        tb = 0
        Aref = &ATL
    } else {
        // from top-left to bottom-right to produce normal sequence (C*Q)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        cb      = imax(0, n(C) - m(A))
        tb      = imax(0, tau.Len() - n(A))
        Aref = &ABR
    }

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  /**/ A, mb, nb, pAstart)
    util.Partition1x2(
        &CL, &CR,    /**/ C, cb, pCstart)
    util.Partition2x1(
        &tT,
        &tB,  /**/tau, tb, pStart)

    w1.SubMatrix(w, 0, 0, m(C), 1)

    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil, 
            &a10, &a11, nil,
            nil,  nil,  &A22,   /**/ A, 1, pAdir)
        util.Repartition1x2to1x3(&CL,
            &C0, &c1, &C2,      /**/ C, 1, pCdir)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     /**/ tau, 1, pDir)
        // --------------------------------------------------------
        applyHouseholder2x1(&tau1, &a10, &c1, &C0, &w1, gomas.RIGHT)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   /**/ &A00, &a11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &CL, &CR,     /**/ &C0, &c1,           C, pCdir)
        util.Continue3x1to2x1(
            &tT,
            &tB,          /**/ &t0, &tau1,         tau, pDir)
    }
}


/*
 * Blocked version for computing C = Q*C and C = Q.T*C from elementary
 * reflectors and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block
 * reflector T. Matrix C is updated by applying block reflector T using
 * compact WY algorithm.
 */
func blockedMultRQLeft(C, A, tau, W *cmat.FloatMatrix, flags, lb int, conf *gomas.Config) {
    var ATL, ABR, AL cmat.FloatMatrix
    var A00, A10, A11, A22 cmat.FloatMatrix
    var CT, CB, C0, C1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix
    var Wrk, W0, Tw, Twork cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart util.Direction
    var bsz, mb, nb, cb, tb int
    var transpose bool

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from top-left to bottom-right to produce transpose sequence (Q.T*C)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        cb      = imax(0, m(C) - m(A))
        tb      = imax(0, tau.Len() - m(A))
        Aref    = &ABR
        transpose = false
    } else {
        // from bottom-right to top-left to produce normal sequence (Q*C)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        mb      = 0
        nb      = 0
        cb      = 0
        tb      = 0
        Aref    = &ATL
        transpose = true
    }

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR,  /**/ A, mb, nb, pAstart)
    util.Partition2x1(
        &CT,
        &CB,    /**/ C, cb, pStart)
    util.Partition2x1(
        &tT,
        &tB,    /**/ tau, tb, pStart)

    // intermediate reflector at start of workspace
    Twork.SetBuf(lb, lb, lb, W.Data())
    W0.SetBuf(n(C), lb, n(C), W.Data()[Twork.Len():])

    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil, 
            &A10, &A11, nil,
            nil,  nil,  &A22,   /**/ A, lb, pAdir)
        bsz = m(&A11)
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     /**/ tau, bsz, pDir)
        util.Repartition2x1to3x1(&CT,
            &C0,
            &C1,
            &C2,     /**/ C, bsz, pDir)
        // --------------------------------------------------------
        // clear & build block reflector from current block
        util.Merge1x2(&AL, &A10, &A11)
        Tw.SubMatrix(&Twork, 0, 0, bsz, bsz)
        blasd.Scale(&Tw, 0.0)
        unblkBlockReflectorRQ(&Tw, &AL, &tau1)

        Wrk.SubMatrix(&W0, 0, 0, n(C), bsz)
        updateLeftRQ(&C1, &C0, &A11, &A10, &Tw, &Wrk, transpose, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   /**/ &A00, &A11, &A22,   A, pAdir)
        util.Continue3x1to2x1(
            &CT,
            &CB,     /**/ &C0, &C1,     C, pDir)
        util.Continue3x1to2x1(
            &tT,
            &tB,     /**/ &t0, &tau1,   tau, pDir)
    }

}

/*
 * Blocked version for computing C = C*Q and C = C*Q.T from elementary
 * reflectors and scalar coefficients.
 *
 * Elementary reflectors and scalar coefficients are used to build block
 * reflector T. Matrix C is updated by applying block reflector T using
 * compact WY algorithm.
 */
func blockedMultRQRight(C, A, tau, W *cmat.FloatMatrix, flags, lb int, conf *gomas.Config) {
    var ATL, ABR, AL cmat.FloatMatrix
    var A00, A10, A11, A22 cmat.FloatMatrix
    var CL, CR, C0, C1, C2 cmat.FloatMatrix
    var tT, tB cmat.FloatMatrix
    var t0, tau1, t2  cmat.FloatMatrix
    var W0, Wrk, Tw, Twork cmat.FloatMatrix

    var Aref *cmat.FloatMatrix
    var pAdir, pAstart, pDir, pStart, pCstart, pCdir util.Direction
    var bsz, cb, mb, nb, tb int
    var transpose bool

    // partitioning start and direction
    if flags & gomas.TRANS != 0 {
        // from bottom-right to top-left to produce transpose sequence (C*Q.T)
        pAstart = util.PBOTTOMRIGHT
        pAdir   = util.PTOPLEFT
        pStart  = util.PBOTTOM
        pDir    = util.PTOP
        pCstart = util.PRIGHT
        pCdir   = util.PLEFT
        mb = 0
        nb = 0
        cb = 0
        tb = 0
        Aref = &ATL
        transpose = false
    } else {
        // from top-left to bottom-right to produce normal sequence (C*Q)
        pAstart = util.PTOPLEFT
        pAdir   = util.PBOTTOMRIGHT
        pStart  = util.PTOP
        pDir    = util.PBOTTOM
        pCstart = util.PLEFT
        pCdir   = util.PRIGHT
        mb      = imax(0, m(A) - n(A))
        nb      = imax(0, n(A) - m(A))
        cb      = imax(0, n(C) - m(A))
        tb      = imax(0, tau.Len() - m(A))
        Aref = &ABR
        transpose = true
    }

    // intermediate reflector at start of workspace
    Twork.SetBuf(lb, lb, lb, W.Data())
    W0.SetBuf(m(C), lb, m(C), W.Data()[Twork.Len():])

    util.Partition2x2(
        &ATL, nil,
        nil,  &ABR, /**/ A, mb, nb, pAstart)
    util.Partition1x2(
        &CL, &CR,   /**/ C, cb, pCstart)
    util.Partition2x1(
        &tT,
        &tB,        /**/ tau, tb, pStart)

    for m(Aref) > 0 && n(Aref) > 0 {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            nil,  nil,  &A22,   /**/ A, lb, pAdir)
        bsz = m(&A11)        // C1 block size must match A11 
        util.Repartition2x1to3x1(&tT,
            &t0,
            &tau1,
            &t2,     /**/ tau, bsz, pDir)
        util.Repartition1x2to1x3(&CL,
            &C0, &C1, &C2 ,     /**/ C, bsz, pCdir)
        // --------------------------------------------------------
        // clear & build block reflector from current block
        util.Merge1x2(&AL, &A10, &A11)
        Tw.SubMatrix(&Twork, 0, 0, bsz, bsz)
        blasd.Scale(&Tw, 0.0)
        unblkBlockReflectorRQ(&Tw, &AL, &tau1)

        Wrk.SubMatrix(&W0, 0, 0, m(&C1), bsz)
        updateRightRQ(&C1, &C0, &A11, &A10, &Tw, &Wrk, transpose, conf)
        // --------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, nil,
            nil,  &ABR,   /**/ &A00, &A11, &A22,   A, pAdir)
        util.Continue1x3to1x2(
            &CL,   &CR,   /**/ &C0, &C1,    C, pCdir)
        util.Continue3x1to2x1(
            &tT,
            &tB,   /**/ &t0, &tau1,   tau, pDir)
    }

}



/*
 * Multiply and replace C with Q*C or Q.T*C where Q is a real orthogonal matrix
 * defined as the product of k elementary reflectors.
 *
 *    Q = H(0)H(1)...H(K-1)
 *
 * as returned by RQFactor().
 *
 * Arguments:
 *  C     On entry, the M-by-N matrix C or if flag bit RIGHT is set then
 *        N-by-M matrix.  On exit C is overwritten by Q*C or Q.T*C.
 *        If bit RIGHT is set then C is  overwritten by C*Q or C*Q.T
 *
 *  A     RQ factorization as returned by RQFactor() where the upper
 *        trapezoidal part holds the elementary reflectors.
 *
 *  tau   The scalar factors of the elementary reflectors.
 *
 *  W     Workspace matrix,  required size is returned by RQMultWork().
 *
 *  flags Indicators. Valid indicators LEFT, RIGHT, TRANS
 *       
 *  conf  Blocking configuration. Field LB defines block sized. If it is zero
 *        unblocked invocation is assumed.
 *
 * Compatible with lapack.DORMRQ
 *
 * Notes:
 *   m(A) is number of elementary reflectors
 *   n(A) is the order of the Q matrix
 *
 *   LEFT : m(C) >= n(Q) --> m(A) <= n(C) <= n(A)
 *   RIGHT: n(C) >= m(Q) --> m(A) <= m(C) <= n(A)
 */
func RQMult(C, A, tau, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var wsmin int
    var tauval float64
    var Qh, tauh cmat.FloatMatrix
    conf := gomas.CurrentConf(confs...)

    // default to multiply from left if side not defined
    if flags & (gomas.LEFT|gomas.RIGHT) == 0 {
        flags = flags | gomas.LEFT
    }
    // m(A) is number of elementary reflectors, Q is n(A)-by-n(A) matrix
    ok := false
    lb := 0
    hr, hc := m(A), n(A)
    switch flags & gomas.RIGHT {
    case gomas.RIGHT:
        ok = n(C) <= n(A) && m(A) <= n(C) 
        wsmin = wsMultRQRight(C, 0)
        hc = n(C)
        lb = estimateLB(C, W.Len(), wsMultRQRight)
    default:
        ok = m(C) <= n(A) && m(A) <= m(C)
        wsmin = wsMultRQLeft(C, 0)
        hc = m(C)
        lb = estimateLB(C, W.Len(), wsMultRQLeft)
    }
    if ! ok {
        return gomas.NewError(gomas.ESIZE, "MultRQ")
    }
    if W == nil || W.Len() < wsmin {
        return gomas.NewError(gomas.EWORK, "MultRQ", wsmin)
    }
    lb = imin(lb, conf.LB)
    Qh.SubMatrix(A, 0, 0, hr, hc)
    tauh.SubMatrix(tau, 0, 0, m(A), 1)
    if hc == hr {
        // m-by-m multiplication, H(K) is unit vector
        // set last tauval to zero, householder functions expect this
        tauval = tau.Get(hc-1, 0)
        tau.Set(hc-1, 0, 0.0)
    }
    if lb == 0 || m(A) <= lb {
        if flags & gomas.RIGHT != 0 {
            unblockedMultRQRight(C, &Qh, &tauh, W, flags)
        } else {
            unblockedMultRQLeft(C, &Qh, &tauh, W, flags)
        }
    } else {
        //lb = conf.LB
        if flags & gomas.RIGHT != 0 {
            blockedMultRQRight(C, &Qh, &tauh, W, flags, lb, conf)
        } else {
            blockedMultRQLeft(C, &Qh, &tauh, W, flags, lb, conf)
        }
    }
    if hc == hr {
        // restore tau value
        tau.Set(hc-1, 0, tauval)
    }
    return err
}



/*
 * Calculate workspace size needed to compute C*Q or Q*C with QR decomposition
 * computed with DecomposeQR().
 */
func RQMultWork(C *cmat.FloatMatrix, bits int, confs... *gomas.Config) (sz int) {
    conf := gomas.CurrentConf(confs...)
    switch bits & gomas.RIGHT {
    case gomas.RIGHT:
        sz = wsMultRQRight(C, conf.LB)
    default:
        sz = wsMultRQLeft(C, conf.LB)
    }
    return 
}

func wsMultRQRight(A *cmat.FloatMatrix, lb int) int {
    if lb == 0 || lb > n(A) {
        return m(A)
    }
    // need space for block reflector T and intermediate results
    return lb*(m(A)+lb)
}

func wsMultRQLeft(A *cmat.FloatMatrix, lb int) int {
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

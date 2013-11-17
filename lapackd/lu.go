
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


// unblocked LU decomposition w/o pivots, FLAME LU nopivots variant 5
func unblockedLUnoPiv(A *cmat.FloatMatrix, conf *gomas.Config) *gomas.Error {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 cmat.FloatMatrix
    var err *gomas.Error = nil

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PTOPLEFT)

    for m(&ATL) < m(A) {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)

        // a21 = a21/a11
        blasd.InvScale(&a21, a11.Get(0, 0))
        // A22 = A22 - a21*a12
        blasd.MVUpdate(&A22, &a21, &a12, -1.0, gomas.NONE)

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
    }
    return err
}


// blocked LU decomposition w/o pivots, FLAME LU nopivots variant 5
func blockedLUnoPiv(A *cmat.FloatMatrix, nb int, conf *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 cmat.FloatMatrix

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PTOPLEFT)


    for m(&ATL) < m(A) - nb {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22,   A, nb, util.PBOTTOMRIGHT)

        // A00 = LU(A00)
        unblockedLUnoPiv(&A11, conf)
        // A12 = trilu(A00)*A12.-1  (TRSM)
        blasd.SolveTrm(&A12, &A11, 1.0, gomas.LEFT|gomas.LOWER|gomas.UNIT)
        // A21 = A21.-1*triu(A00) (TRSM)
        blasd.SolveTrm(&A21, &A11, 1.0, gomas.RIGHT|gomas.UPPER)
        // A22 = A22 - A21*A12
        blasd.Mult(&A22, &A21, &A12, -1.0, 1.0, gomas.NONE)

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
    }
    // last block
    if m(&ATL) < m(A) {
        unblockedLUnoPiv(&ABR, conf)
    }
    return err
}


// unblocked LU decomposition with pivots: FLAME LU variant 3; Left-looking 
func unblockedLUpiv(A *cmat.FloatMatrix, p *Pivots, offset int, conf *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a10, a11, a12, A20, a21, A22 cmat.FloatMatrix
    var AL, AR, A0, a1, A2, aB1, AB0 cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots

    err = nil
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    util.Partition1x2(
        &AL, &AR, A, 0, util.PLEFT)
    partitionPivot2x1(
        &pT,
        &pB, *p, 0, util.PTOP)

    for m(&ATL) < m(A) && n(&ATL) < n(A) {
        util.Repartition2x2to3x3(&ATL, 
            &A00, &a01, &A02,
            &a10, &a11, &a12,
            &A20, &a21, &A22, /**/ A, 1, util.PBOTTOMRIGHT)
        util.Repartition1x2to1x3(&AL, 
            &A0, &a1, &A2,    /**/ A, 1, util.PRIGHT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,    /**/ *p, 1, util.PBOTTOM)

        // apply previously computed pivots on current column
        applyPivots(&a1, p0)

        // a01 = trilu(A00) \ a01 (TRSV)
        blasd.MVSolveTrm(&a01, &A00, 1.0, gomas.LOWER|gomas.UNIT)
        // a11 = a11 - a10 *a01 
        aval := a11.Get(0, 0) - blasd.Dot(&a10, &a01)
        a11.Set(0, 0, aval)
        // a21 = a21 -A20*a01
        blasd.MVMult(&a21, &A20, &a01, -1.0, 1.0, gomas.NONE)

        // pivot index on current column [a11, a21].T
        aB1.Column(&ABR, 0) 
        p1[0] = pivotIndex(&aB1)
        // pivots to current column
        applyPivots(&aB1, p1)
        
        // a21 = a21 / a11
        if aval == 0.0 {
            if err == nil {
                ij := m(&ATL) + p1[0] - 1
                err = gomas.NewError(gomas.ESINGULAR, "DecomposeLU", ij, ij)
            }
        } else {
            blasd.InvScale(&a21, a11.Get(0, 0))
        }

        // apply pivots to previous columns
        AB0.SubMatrix(&ABL, 0, 0)
        applyPivots(&AB0, p1)
        // scale last pivots to origin matrix row numbers
        p1[0] += m(&ATL)

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        util.Continue1x3to1x2(
            &AL, &AR,     &A0, &a1,   A, util.PRIGHT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    *p, util.PBOTTOM)
    }
    if n(&ATL) < n(A) {
        applyPivots(&ATR, *p)
        blasd.SolveTrm(&ATR, &ATL, 1.0, gomas.LEFT|gomas.UNIT|gomas.LOWER, conf)
    }
    return err
}

// blocked LU decomposition with pivots: FLAME LU variant 3; left-looking version
func blockedLUpiv(A *cmat.FloatMatrix, p *Pivots, nb int, conf *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 cmat.FloatMatrix
    var AL, AR, A0, A1, A2, AB1, AB0 cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,    A, 0, 0, util.PTOPLEFT)
    util.Partition1x2(
        &AL, &AR,      A, 0, util.PLEFT)
    partitionPivot2x1(
        &pT,
        &pB,     *p, 0, util.PTOP)

    for m(&ATL) < m(A) && n(&ATL) < n(A) {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22, /**/ A, nb, util.PBOTTOMRIGHT)
        util.Repartition1x2to1x3(&AL,
            &A0, &A1, &A2,    /**/ A, nb, util.PRIGHT)
        repartPivot2x1to3x1(&pT,
            &p0, &p1, &p2,   /**/ *p, nb, util.PBOTTOM)

        // apply previously computed pivots
        applyPivots(&A1, p0)

        // a01 = trilu(A00) \ a01 (TRSV)
        blasd.SolveTrm(&A01, &A00, 1.0, gomas.LOWER|gomas.UNIT)
        // A11 = A11 - A10*A01
        blasd.Mult(&A11, &A10, &A01, -1.0, 1.0, gomas.NONE)
        // A21 = A21 - A20*A01
        blasd.Mult(&A21, &A20, &A01, -1.0, 1.0, gomas.NONE)

        // LU_piv(AB1, p1)
        AB1.SubMatrix(&ABR, 0, 0, m(&ABR), n(&A11))
        unblockedLUpiv(&AB1, &p1, m(&ATL), conf)

        // apply pivots to previous columns
        AB0.SubMatrix(&ABL, 0, 0)
        applyPivots(&AB0, p1)
        // scale last pivots to origin matrix row numbers
        for k, _ := range p1 {
            p1[k] += m(&ATL)
        }

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR, /**/ &A00, &A11, &A22, A, util.PBOTTOMRIGHT)
        util.Continue1x3to1x2(
            &AL, &AR,   /**/ &A0, &A1, A, util.PRIGHT)
        contPivot3x1to2x1(
            &pT,
            &pB,        /**/ p0, p1, *p, util.PBOTTOM)
    }
    if n(&ATL) < n(A) {
        applyPivots(&ATR, *p)
        blasd.SolveTrm(&ATR, &ATL, 1.0, gomas.LEFT|gomas.UNIT|gomas.LOWER)
    }
    return err
}

/*
 * Compute an LU factorization of a general M-by-N matrix using
 * partial pivoting with row interchanges.
 *
 * Arguments:
 *   A      On entry, the M-by-N matrix to be factored. On exit the factors
 *          L and U from factorization A = P*L*U, the unit diagonal elements
 *          of L are not stored.
 *
 *   pivots On exit the pivot indices. 
 *
 *   nb     Blocking factor for blocked invocations. If bn == 0 or
 *          min(M,N) < nb unblocked algorithm is used.
 *
 * Returns:
 *  LU factorization and error indicator.
 *
 * Compatible with lapack.DGETRF
 */
func DecomposeLU(A *cmat.FloatMatrix, pivots Pivots, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    mlen := imin(m(A), n(A))
    if len(pivots) < mlen {
        return gomas.NewError(gomas.ESIZE_PIVOTS, "DecomposeLU")
    }
    // clear pivot array
    for k, _ := range pivots {
        pivots[k] = 0
    }
    if mlen <= conf.LB || conf.LB == 0 {
        err = unblockedLUpiv(A, &pivots, 0, conf)
    } else {
        err = blockedLUpiv(A, &pivots, conf.LB, conf)
    }
    return err
}

/*
 * Compute an LU factorization of a general M-by-N matrix without pivoting.
 *
 * Arguments:
 *   A   On entry, the M-by-N matrix to be factored. On exit the factors
 *       L and U from factorization A = P*L*U, the unit diagonal elements
 *       of L are not stored.
 *
 *   nb  Blocking factor for blocked invocations. If bn == 0 or
 *       min(M,N) < nb unblocked algorithm is used.
 *
 * Returns:
 *  LU factorization and error indicator.
 *
 * Compatible with lapack.DGETRF
 */
func DecomposeLUnoPiv(A *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    mlen := imin(m(A), n(A))
    if mlen <= conf.LB || conf.LB == 0 {
        err = unblockedLUnoPiv(A, conf)
    } else {
        err = blockedLUnoPiv(A, conf.LB, conf)
    }
    return err
}

/*
 * Solve a system of linear equations A*X = B or A.T*X = B with general N-by-N
 * matrix A using the LU factorization computed by DecomposeLU().
 *
 * Arguments:
 *  B      On entry, the right hand side matrix B. On exit, the solution matrix X.
 *
 *  A      The factor L and U from the factorization A = P*L*U as computed by
 *         DecomposeLU()
 *
 *  pivots The pivot indices from DecomposeLU().
 *
 *  flags  The indicator of the form of the system of equations.
 *         If flags&TRANSA then system is transposed. All other values
 *         indicate non transposed system.
 *
 * Compatible with lapack.DGETRS.
 */
func SolveLU(B, A *cmat.FloatMatrix, pivots Pivots, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    ar, ac := A.Size()
    br, _  := B.Size()
    if ar != ac {
        return gomas.NewError(gomas.ENOTSQUARE, "SolveLU")
    }
    if br != ac {
        return gomas.NewError(gomas.ESIZE, "SolveLU")
    }
    if pivots != nil {
        applyPivots(B, pivots)
    }
    if flags & gomas.TRANSA != 0 {
        // transposed X = A.-1*B == (L.T*U.T).-1*B == U.-T*(L.-T*B)
        blasd.SolveTrm(B, A, 1.0, gomas.LOWER|gomas.UNIT|gomas.TRANSA, conf)
        blasd.SolveTrm(B, A, 1.0, gomas.UPPER|gomas.TRANSA, conf)
    } else {
        // non-transposed X = A.-1*B == (L*U).-1*B == U.-1*(L.-1*B)
        blasd.SolveTrm(B, A, 1.0, gomas.LOWER|gomas.UNIT, conf)
        blasd.SolveTrm(B, A, 1.0, gomas.UPPER, conf)
    }
        
    return err
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

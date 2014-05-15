
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
    "math"
    //"fmt"
)

func unblockedLowerCHOL(A *cmat.FloatMatrix, flags int, nr int) (err *gomas.Error) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10, a11, A20, a21, A22 cmat.FloatMatrix

    err = nil
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PTOPLEFT)

    for m(&ATL) < m(A) {
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &a10, &a11, nil,
            &A20, &a21, &A22,   A, 1, util.PBOTTOMRIGHT)

        // a11 = sqrt(a11)
        aval := a11.Get(0, 0)
        if aval < 0.0 {
            if err == nil {
                err = gomas.NewError(gomas.ENEGATIVE, "DecomposeCHOL", m(&ATL)+nr)
            }
        } else {
            a11.Set(0, 0, math.Sqrt(aval))
        }

        // a21 = a21/a11
        blasd.InvScale(&a21, a11.Get(0, 0))
        // A22 = A22 - a21*a21' (SYR)
        blasd.MVUpdateSym(&A22, &a21, -1.0, gomas.LOWER)

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,  A, util.PBOTTOMRIGHT)

    }
    return
}

func unblockedUpperCHOL(A *cmat.FloatMatrix, flags int, nr int) (err *gomas.Error) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a11, a12, A22 cmat.FloatMatrix

    err = nil
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PTOPLEFT)

    for m(&ATL) < m(A) {
        util.Repartition2x2to3x3(&ATL,
            &A00, &a01, &A02,
            nil,  &a11, &a12,
            nil,  nil,  &A22,   A, 1, util.PBOTTOMRIGHT)

        aval := a11.Get(0, 0)
        if aval < 0.0 {
            if err == nil {
                err = gomas.NewError(gomas.ENEGATIVE, "DecomposeCHOL", nr+m(&ATL))
            }
        } else {
            a11.Set(0, 0, math.Sqrt(aval))
        }

        // a21 = a12/a11
        blasd.InvScale(&a12, a11.Get(0, 0))
        // A22 = A22 - a12'*a12 (SYR)
        blasd.MVUpdateSym(&A22, &a12, -1.0, gomas.UPPER)

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,  A, util.PBOTTOMRIGHT)

    }
    return
}

func blockedCHOL(A *cmat.FloatMatrix, flags int, conf *gomas.Config) *gomas.Error {
    var err, firstErr *gomas.Error
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A01, A02, A10, A11, A12, A20, A21, A22 cmat.FloatMatrix

    nb := conf.LB
    err = nil
    firstErr = nil
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PTOPLEFT)

    for m(A) - m(&ATL) > nb {
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            &A10, &A11, &A12,
            &A20, &A21, &A22,   A, nb, util.PBOTTOMRIGHT)


        if flags & gomas.LOWER != 0 {
            // A11 = chol(A11)
            err = unblockedLowerCHOL(&A11, flags, m(&ATL))
            // A21 = A21 * tril(A11).-1
            blasd.SolveTrm(&A21, &A11, 1.0, gomas.RIGHT|gomas.LOWER|gomas.TRANSA, conf)
            // A22 = A22 - A21*A21.T
            blasd.UpdateSym(&A22, &A21, -1.0, 1.0, gomas.LOWER, conf)
        } else {
            // A11 = chol(A11)
            err = unblockedUpperCHOL(&A11, flags, m(&ATL))
            // A12 = triu(A11).-1 * A12
            blasd.SolveTrm(&A12, &A11, 1.0, gomas.UPPER|gomas.TRANSA, conf)
            // A22 = A22 - A12.T*A12
            blasd.UpdateSym(&A22, &A12, -1.0, 1.0, gomas.UPPER|gomas.TRANSA, conf)
        }
        if err != nil && firstErr == nil {
            firstErr = err
        }

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,   A, util.PBOTTOMRIGHT)
    }

    if m(&ATL) < m(A) {
        // last block
        if flags & gomas.LOWER != 0 {
            unblockedLowerCHOL(&ABR, flags, 0)
        } else {
            unblockedUpperCHOL(&ABR, flags, 0)
        }
    }
    return firstErr
}

/*
 * Compute the Cholesky factorization of a symmetric positive definite
 * N-by-N matrix A.
 *
 * Arguments:
 *  A     On entry, the symmetric matrix A. If flags&UPPER the upper triangular part
 *        of A contains the upper triangular part of the matrix A, and strictly
 *        lower part A is not referenced. If flags&LOWER the lower triangular part
 *        of a contains the lower triangular part of the matrix A. Likewise, the
 *        strictly upper part of A is not referenced. On exit, factor U or L from the
 *        Cholesky factorization A = U.T*U or A = L*L.T
 *      
 *  flags The matrix structure indicator, UPPER for upper tridiagonal and LOWER for
 *        lower tridiagonal matrix.
 *
 *  confs Optional blocking configuration. If not provided default blocking configuration
 *        will be used.
 *
 * Compatible with lapack.DPOTRF
 */
func CHOLFactor(A *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    ar, ac := A.Size()
    if ac != ar {
        return gomas.NewError(gomas.ENOTSQUARE, "DecomposeCHOL")
    }
    if ac < conf.LB || conf.LB == 0 {
        if flags & gomas.UPPER != 0 {
            err = unblockedUpperCHOL(A, flags, 0)
        } else {
            err = unblockedLowerCHOL(A, flags, 0)
        }
    } else {
        err = blockedCHOL(A, flags, conf)
    }
    return err
}

/*
 * Solves a system system of linear equations A*X = B with symmetric positive
 * definite matrix A using the Cholesky factorization A = U.T*U or A = L*L.T
 * computed by DecomposeCHOL().
 *
 * Arguments:
 *  B   On entry, the right hand side matrix B. On exit, the solution
 *      matrix X.
 *
 *  A   The triangular factor U or L from Cholesky factorization as computed by
 *      DecomposeCHOL().
 *
 *  flags Indicator of which factor is stored in A. If flags&UPPER then upper
 *        triangle of A is stored. If flags&LOWER then lower triangle of A is
 *        stored.
 *
 * Compatible with lapack.DPOTRS.
 */
func CHOLSolve(B, A *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    // A*X = B; X = A.-1*B == (LU).-1*B == U.-1*L.-1*B == U.-1*(L.-1*B)
    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }
    ar, ac := A.Size()
    br, _  := B.Size()
    if ac != br || ar != ac {
        return gomas.NewError(gomas.ESIZE, "SolveCHOL")
    }
    if flags & gomas.UPPER != 0 {
        // X = (U.T*U).-1*B => U.-1*(U.-T*B)
        blasd.SolveTrm(B, A, 1.0, gomas.UPPER|gomas.TRANSA, conf)
        blasd.SolveTrm(B, A, 1.0, gomas.UPPER, conf)
    } else if flags & gomas.LOWER != 0 {
        // X = (L*L.T).-1*B = L.-T*(L.1*B)
        blasd.SolveTrm(B, A, 1.0, gomas.LOWER, conf)
        blasd.SolveTrm(B, A, 1.0, gomas.LOWER|gomas.TRANSA, conf)
    }
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

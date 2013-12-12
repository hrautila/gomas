
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
)

const bkALPHA = 0.6403882032022075   // (1.0 + sqrt(17.0))/8.0

/*
 * Compute LDL^T factorization of real symmetric matrix.
 *
 * Computes of a real symmetric matrix A using Bunch-Kauffman pivoting method.
 * The form of factorization is 
 *
 *    A = L*D*L.T  or A = U*D*U.T
 *
 * where L (or U) is product of permutation and unit lower (or upper) triangular matrix
 * and D is block diagonal symmetric matrix with 1x1 and 2x2 blocks.
 *
 * Arguments
 *  A     On entry, the N-by-N symmetric matrix A. If flags bit LOWER (or UPPER) is set then
 *        lower (or upper) triangular matrix and strictly upper (or lower) part is not
 *        accessed. On exit, the block diagonal matrix D and lower (or upper) triangular
 *        product matrix L (or U).
 *
 *  W     Workspace, size as returned by WorksizeBK().
 *
 *  ipiv  Pivot vector. On exit details of interchanges and the block structure of D. If
 *        ipiv[k] > 0 then D[k,k] is 1x1 and rows and columns k and ipiv[k]-1 were changed.
 *        If ipiv[k] == ipiv[k+1] < 0 then D[k,k] is 2x2. If A is lower then rows and
 *        columns k+1 and ipiv[k]-1  were changed. And if A is upper then rows and columns
 *        k and ipvk[k]-1 were changed.
 *
 *  flags Indicator bits, LOWER or UPPER.
 *
 *  confs Optional blocking configuration. If not provided then default blocking
 *        as returned by DefaultConf() is used. 
 *
 *  Unblocked algorithm is used if blocking configuration LB is zero or if N < LB.
 *
 *  Compatible with lapack.SYTRF.
 */
 func DecomposeBK(A, W *cmat.FloatMatrix, ipiv Pivots, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    
    for k, _ := range ipiv {
        ipiv[k] = 0
    }
    wsz := WorksizeBK(A, conf)
    if W.Len() < wsz {
        return gomas.NewError(gomas.EWORK, "DecomposeBK", wsz)
    }
    
    var Wrk cmat.FloatMatrix
    if n(A) < conf.LB || conf.LB == 0 {
        // make workspace rows(A)*2 matrix
        Wrk.SetBuf(m(A), 2, m(A), W.Data())
        if flags & gomas.LOWER != 0 {
            err, _ = unblkDecompBKLower(A, &Wrk, ipiv, conf)
        } else if flags & gomas.UPPER != 0 {
            err, _ = unblkDecompBKUpper(A, &Wrk, ipiv, conf)
        }
    } else {
        // make workspace rows(A)*(LB+1) matrix
        Wrk.SetBuf(m(A), conf.LB+1, m(A), W.Data())
        if flags & gomas.LOWER != 0 {
            err = blkDecompBKLower(A, &Wrk,  &ipiv, conf)
        } else if flags & gomas.UPPER != 0 {
            err = blkDecompBKUpper(A, &Wrk,  &ipiv, conf)
        }
    }        
    return err
}

/*
 * Solve A*X = B with symmetric real matrix A.
 *
 * Solves a system of linear equations A*X = B with a real symmetric matrix A using
 * the factorization A = U*D*U**T or A = L*D*L**T computed by DecomposeBK().
 *
 * Arguments
 *  B     On entry, right hand side matrix B. On exit, the solution matrix X.
 *
 *  A     Block diagonal matrix D and the multipliers used to compute factor U
 *        (or L) as returned by DecomposeBK().
 *
 *  ipiv  Block structure of matrix D and details of interchanges.
 *
 *  flags Indicator bits, LOWER or UPPER.
 *
 *  confs Optional blocking configuration.
 *
 * Currently only unblocked algorightm implemented. Compatible with lapack.SYTRS.
 */
 func SolveBK(B, A *cmat.FloatMatrix, ipiv Pivots, flags int, confs... *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    conf := gomas.CurrentConf(confs...)
    if n(A) != m(B) {
        return gomas.NewError(gomas.ESIZE, "SolveBK")
    }
    if flags & gomas.LOWER != 0 {
        // first part: Z = D.-1*(L.-1*B)
        err = unblkSolveBKLower(B, A, ipiv, 1, conf)
        // second part: X = L.-T*Z
        err = unblkSolveBKLower(B, A, ipiv, 2, conf)
    } else if flags & gomas.UPPER != 0 {
        // first part: Z = D.-1*(U.-1*B)
        err = unblkSolveBKUpper(B, A, ipiv, 1, conf)
        // second part: X = U.-T*Z
        err = unblkSolveBKUpper(B, A, ipiv, 2, conf)
    }
    return err
}



/*
 * Return worksize needed to compute Bunch-Kauffman LDL^T factorization.
 */
func WorksizeBK(A *cmat.FloatMatrix, conf *gomas.Config) int {
    if n(A) < conf.LB || conf.LB == 0 {
        return 2*m(A)
    }
    return m(A)*(conf.LB+1)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

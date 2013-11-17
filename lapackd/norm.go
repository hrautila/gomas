
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas/blasd"
)

type Norms int
const (
    NORM_ONE = Norms(1)
    NORM_TWO = Norms(2)
    NORM_INF = Norms(-1)
)

func mNorm1(A *cmat.FloatMatrix) float64 {
    var amax float64 = 0.0
    var col cmat.FloatMatrix
    _, acols := A.Size()
    for k := 0; k < acols; k++ {
        col.Column(A, k)
        cmax := blasd.ASum(&col)
        if cmax > amax {
            amax = cmax
        }
    }
    return amax
}

func mNormInf(A *cmat.FloatMatrix) float64 {
    var amax float64 = 0.0
    var row cmat.FloatMatrix
    arows, _ := A.Size()
    for k := 0; k < arows; k++ {
        row.Row(A, k)
        rmax := blasd.ASum(&row)
        if rmax > amax {
            amax = rmax
        }
    }
    return amax
}


/*
 * Compute matrix and vector norms.
 *
 * Arguments
 *  X    A real valued matrix or vector
 *
 *  norm Norm to compute
 *         NORM_ONE, NORM_TWO, NORM_INF
 *
 * Note: matrix NORM_TWO not yet implemented.
 */
func NormP(X *cmat.FloatMatrix, norm Norms) float64 {
    if X.IsVector() {
        switch norm {
        case NORM_ONE:
            return blasd.ASum(X)
        case NORM_TWO:
            return blasd.Nrm2(X)
        case NORM_INF:
            return blasd.Amax(X)
        }
        return 0.0
    }
    switch norm {
    case NORM_ONE:
        return mNorm1(X)
    case NORM_TWO:
        return 0.0
    case NORM_INF:
        return mNormInf(X)
    }
    return 0.0
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

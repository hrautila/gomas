
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas/blasd"
    "math"
)

func absSortVec(D *cmat.FloatMatrix, updown int) {
    var cval, tmpval float64

    j := 0
    for k := 0; k < D.Len(); k++ {
        cval = math.Abs(D.GetAt(k))
        for j = k; j > 0; j-- {
            tmpval = math.Abs(D.GetAt(j-1))
            if updown > 0 && tmpval >= cval {
                break
            }
            if updown < 0 && tmpval <= cval {
                break
            }
            D.SetAt(j, tmpval)
        }
        D.SetAt(j, cval)
    }
}


func sortVec(D *cmat.FloatMatrix, updown int) {
    var cval, tmpval float64

    j := 0
    for k := 0; k < D.Len(); k++ {
        cval = D.GetAt(k)
        for j = k; j > 0; j-- {
            tmpval = D.GetAt(j-1)
            if updown > 0 && tmpval >= cval {
                break
            }
            if updown < 0 && tmpval <= cval {
                break
            }
            D.SetAt(j, tmpval)
        }
        D.SetAt(j, cval)
    }
}

func vecMinMax(D *cmat.FloatMatrix, minmax int) int {
    var cval, tmpval float64
    cval = D.GetAt(0)
    ix := 0
    for k := 1; k < D.Len(); k++ {
        tmpval = D.GetAt(k)
        if minmax > 0 && tmpval > cval {
            cval = tmpval
            ix = k
        } else if minmax < 0 && tmpval < cval {
            cval = tmpval
            ix = k
        }
    }
    return ix
}

func absVecMinMax(D *cmat.FloatMatrix, minmax int) int {
    var cval, tmpval float64
    cval = math.Abs(D.GetAt(0))
    ix := 0
    for k := 1; k < D.Len(); k++ {
        tmpval = math.Abs(D.GetAt(k))
        if minmax > 0 && tmpval > cval {
            cval = tmpval
            ix = k
        } else if minmax < 0 && tmpval < cval {
            cval = tmpval
            ix = k
        }
    }
    return ix
}


func sortEigenVec(D, U, V, C *cmat.FloatMatrix, updown int) {
    var sD, m0, m1 cmat.FloatMatrix

    N := D.Len()
    for k := 0; k < N-1; k++ {
        sD.SubVector(D, k, N-k)
        pk := vecMinMax(&sD, -updown)
        if pk != 0 {
            t0 := D.GetAt(k)
            D.SetAt(k, D.GetAt(pk+k))
            D.SetAt(k+pk, t0)
            if U != nil {
                m0.Column(U, k)
                m1.Column(U, k+pk)
                blasd.Swap(&m1, &m0)
            }
            if V != nil {
                m0.Row(V, k)
                m1.Row(V, k+pk)
                blasd.Swap(&m1, &m0)
            }
            if C != nil {
                m0.Column(C, k)
                m1.Column(C, k+pk)
                blasd.Swap(&m1, &m0)
            }
        }
    }
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

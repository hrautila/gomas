
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import "github.com/hrautila/cmat"

func m(A *cmat.FloatMatrix) int {
    r, _ := A.Size()
    return r
}

func n(A *cmat.FloatMatrix) int {
    _, c := A.Size()
    return c
}

func imin(a, b int) int {
    if a < b {
        return a
    }
    return b
}

func imax(a, b int) int {
    if a > b {
        return a
    }
    return b
}

/*
 * Compute blocking factor that fits provided workspace.
 */
func estimateLB(A *cmat.FloatMatrix, wsz int, worksize func(*cmat.FloatMatrix, int)int) int {
    lb := 4
    wblk := worksize(A, 4)
    if wsz < wblk {
        // not enough for minimum blocking factor, fall to unblocked
        return 0
    }
    for wsz > wblk {
        lb += 2
        wblk = worksize(A, lb)
    }
    if wblk > wsz { lb -= 2 }
    return lb
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

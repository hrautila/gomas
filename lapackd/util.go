
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

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

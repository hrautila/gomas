
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    //"github.com/hrautila/cmat"
    "math"
)

// Compute Wilkinson shift from symmetric 2x2 trailing submatrix
// Stable formula from Hogben 2007, 42.3
func wilkinson(tn1, tnn1, tn float64) float64 {
    var d, tsq float64
    d = (tn1 - tn)/2.0
    tsq = math.Hypot(d, tnn1)
    u := tn - math.Copysign((tnn1/(math.Abs(d)+tsq))*tnn1, d)
    return u
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

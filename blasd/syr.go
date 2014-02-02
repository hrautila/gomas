
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package blasd

import "github.com/hrautila/cmat"
import "github.com/hrautila/gomas"


func MVUpdateSym(A, X *cmat.FloatMatrix, alpha float64, bits int, confs... *gomas.Config) *gomas.Error {
    ar, ac := A.Size()
    xr, xc := X.Size()

    if ar*ac == 0 {
        return nil
    }
    if xr != 1 && xc != 1 {
        return gomas.NewError(gomas.ENEED_VECTOR, "MVUpdateSym")
    }
    nx := X.Len()
    if  ac != nx || ar != ac {
        return gomas.NewError(gomas.ESIZE, "MVUpdateSym")
    }
    updtrmv(A, X, X, alpha, bits, nx, nx)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)

/*
 * Compute
 *   B = B*diag(D)      flags & RIGHT == true
 *   B = diag(D)*B      flags & LEFT  == true
 *
 * If flags is LEFT (RIGHT) then element-wise multiplies columns (rows) of B with vector D. 
 *
 * Arguments
 *   B     M-by-N matrix if flags&RIGHT == true or N-by-M matrix if flags&LEFT == true
 *
 *   D     N element column or row vector or N-by-N matrix
 *   
 *   flags Indicator bits, LEFT or RIGHT
 */
func MultDiag(B, D *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var c, d0 cmat.FloatMatrix
    var d *cmat.FloatMatrix

    conf := gomas.CurrentConf(confs...)
    d = D
    if ! D.IsVector() {
        d0.Diag(D)
        d = &d0
    }
    dn := d.Len()
    br, bc := B.Size()
    switch flags & (gomas.LEFT|gomas.RIGHT) {
    case gomas.LEFT:
        if br != dn {
            return gomas.NewError(gomas.ESIZE, "MultDiag")
        }
        // scale rows; for each column element-wise multiply with D-vector
        for k := 0; k < dn; k++ {
            c.Row(B, k)
            blasd.Scale(&c, d.GetAt(k), conf)
        }
    case gomas.RIGHT:
        if bc != dn {
            return gomas.NewError(gomas.ESIZE, "MultDiag")
        }
        // scale columns
        for k := 0; k < dn; k++ {
            c.Column(B, k)
            blasd.Scale(&c, d.GetAt(k), conf)
        }
    }
    return nil
}

/*
 * Compute
 *   B = B*diag(D).-1      flags & RIGHT == true
 *   B = diag(D).-1*B      flags & LEFT  == true
 *
 * If flags is LEFT (RIGHT) then element-wise divides columns (rows) of B with vector D. 
 *
 * Arguments:
 *   B     M-by-N matrix if flags&RIGHT == true or N-by-M matrix if flags&LEFT == true
 *
 *   D     N element column or row vector or N-by-N matrix
 *   
 *   flags Indicator bits, LEFT or RIGHT
 */
func SolveDiag(B, D *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var c, d0 cmat.FloatMatrix
    var d *cmat.FloatMatrix

    conf := gomas.CurrentConf(confs...)
    d = D
    if ! D.IsVector() {
        d0.Diag(D)
        d = &d0
    }
    dn := d0.Len()
    br, bc := B.Size()
    switch flags & (gomas.LEFT|gomas.RIGHT) {
    case gomas.LEFT:
        if br != dn {
            return gomas.NewError(gomas.ESIZE, "SolveDiag")
        }
        // scale rows; 
        for k := 0; k < dn; k++ {
            c.Row(B, k)
            blasd.InvScale(&c, d.GetAt(k), conf)
        }
    case gomas.RIGHT:
        if bc != dn {
            return gomas.NewError(gomas.ESIZE, "SolveDiag")
        }
        // scale columns
        for k := 0; k < dn; k++ {
            c.Column(B, k)
            blasd.InvScale(&c, d.GetAt(k), conf)
        }
    }
    return nil
}

/*
 * Generic rank update of diagonal matrix.
 *   diag(D) = diag(D) + alpha * x * y.T
 *
 * Arguments:
 *   D     N element column or row vector or N-by-N matrix
 *
 *   x, y  N element vectors
 *   
 *   alpha scalar 
 */
func MVUpdateDiag(D, x, y *cmat.FloatMatrix, alpha float64, confs... *gomas.Config) *gomas.Error {
    var d *cmat.FloatMatrix
    var d0 cmat.FloatMatrix

    if ! x.IsVector() || ! y.IsVector() {
        return gomas.NewError(gomas.ENEED_VECTOR, "MvUpdateDiag")
    }

    d = D
    if ! D.IsVector() {
        d0.Diag(D)
        d = &d0
    }
        
    for k := 0; k < d.Len(); k++ {
        val := d.GetAt(k)
        val += x.GetAt(k)*y.GetAt(k)*alpha
        d.SetAt(k, val)
    }
    return nil
}
// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

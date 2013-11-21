
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "math"
    //"fmt"
)

/* From LAPACK/dlapy2.f
 *
 * sqrtX2Y2() returns sqrt(x**2+y**2), taking care not to cause unnecessary
 * overflow.
 */
func sqrtX2Y2(x, y float64) float64 {
    xabs := math.Abs(x)
    yabs := math.Abs(y)
    w := xabs
    if yabs > w {
        w = yabs
    }
    z := xabs
    if yabs < z {
        z = yabs
    }
    if z == 0.0 {
        return w
    }
    return w * math.Sqrt(1.0 + (z/w)*(z/w))
}

/* From LAPACK/dlarfg.f
 *
 * DLARFG generates a real elementary reflector H of order n, such
 * that
 *
 *       H * ( alpha ) = ( beta ),   H**T * H = I.
 *           (   x   )   (   0  )
 *
 * where alpha and beta are scalars, and x is an (n-1)-element real
 * vector. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v**T ) ,
 *                     ( v )
 *
 * where tau is a real scalar and v is a real (n-1)-element
 * vector.
 *
 * If the elements of x are all zero, then tau = 0 and H is taken to be
 * the unit cmat.
 *
 * Otherwise  1 <= tau <= 2.
 */
func computeHouseholder(a11, x, tau *cmat.FloatMatrix, flags int) {
    
    // norm_x2 = ||x||_2
    norm_x2 := blasd.Nrm2(x)
    if norm_x2 == 0.0 {
        tau.Set(0, 0, 0.0)
        return
    }

    alpha := a11.Get(0, 0)
    sign := 1.0
    if math.Signbit(alpha) {
        sign = -1.0
    }
    // beta = -(alpha / |alpha|) * ||alpha x||
    //      = -sign(alpha) * sqrt(alpha**2, norm_x2**2)
    beta := -sign*sqrtX2Y2(alpha, norm_x2)

    // x = x /(a11 - beta)
    blasd.InvScale(x, alpha-beta)

    tau.Set(0, 0, (beta-alpha)/beta)
    a11.Set(0, 0, beta)
}

/* From LAPACK/dlarf.f
 *
 * Applies a real elementary reflector H to a real m by n matrix A,
 * from either the left or the right. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v.T )
 *                     ( v )
 *
 * where tau is a real scalar and v is a real vector.
 *
 * If tau = 0, then H is taken to be the unit cmat.
 *
 * A is /a1\   a1 := a1 - w1
 *      \A2/   A2 := A2 - v*w1
 *             w1 := tau*(a1 + A2.T*v) if side == LEFT
 *                := tau*(a1 + A2*v)   if side == RIGHT
 *
 * Allocates/frees intermediate work space matrix w1.
 */
func applyHouseholder(tau, v, a1, A2 *cmat.FloatMatrix, flags int) {

    tval := tau.Get(0, 0)
    if tval == 0.0 {
        return
    }
    w1 := cmat.NewCopy(a1)
    if flags & gomas.LEFT != 0 {
        // w1 = a1 + A2.T*v
        blasd.MVMult(w1, A2, v, 1.0, 1.0, gomas.TRANSA)
    } else {
        // w1 = a1 + A2*v
        blasd.MVMult(w1, A2, v, 1.0, 1.0, gomas.NONE)
    }

    // w1 = tau*w1
    blasd.Scale(w1, tval)

    // a1 = a1 - w1
    blasd.Axpy(a1, w1, -1.0)

    // A2 = A2 - v*w1
    blasd.MVUpdate(A2, v, w1, -1.0)
}

/* 
 * Applies a real elementary reflector H to a real m by n matrix A,
 * from either the left or the right. H is represented in the form
 *
 *       H = I - tau * ( 1 ) * ( 1 v.T )
 *                     ( v )
 *
 * where tau is a real scalar and v is a real vector.
 *
 * If tau = 0, then H is taken to be the unit cmat.
 *
 * A is /a1\   a1 := a1 - w1
 *      \A2/   A2 := A2 - v*w1
 *             w1 := tau*(a1 + A2.T*v) if side == LEFT
 *                := tau*(a1 + A2*v)   if side == RIGHT
 *
 * Intermediate work space w1 required as parameter, no allocation.
 */
func applyHouseholder2x1(tau, v, a1, A2, w1 *cmat.FloatMatrix, flags int) *gomas.Error {
    var err *gomas.Error = nil
    tval := tau.Get(0, 0)
    if tval == 0.0 {
        return err
    }

    // shape oblivious vector copy.
    blasd.Axpby(w1, a1, 1.0, 0.0)
    if flags & gomas.LEFT != 0 {
        // w1 = a1 + A2.T*v
        err = blasd.MVMult(w1, A2, v, 1.0, 1.0, gomas.TRANSA)
    } else {
        // w1 = a1 + A2*v
        err = blasd.MVMult(w1, A2, v, 1.0, 1.0, gomas.NONE)
    }
    // w1 = tau*w1
    blasd.Scale(w1, tval)

    // a1 = a1 - w1
    blasd.Axpy(a1, w1, -1.0)

    // A2 = A2 - v*w1
    if flags & gomas.LEFT != 0 {
        err = blasd.MVUpdate(A2, v, w1, -1.0)
    } else {
        err = blasd.MVUpdate(A2, w1, v, -1.0)
    }
    return err
}

func applyHouseholder1x1(tau, v, A2, w1 *cmat.FloatMatrix, flags int) {

    tval := tau.Get(0, 0)
    if tval == 0.0 {
        return
    }
    if flags & gomas.LEFT != 0 {
        // w1 = A2.T*v
        blasd.MVMult(w1, A2, v, 1.0, 0.0, gomas.TRANSA)
    } else {
        // w1 = A2*v
        blasd.MVMult(w1, A2, v, 1.0, 0.0, gomas.NONE)
    }

    // A2 = A2 - tau*v*w1
    blasd.MVUpdate(A2, v, w1, -tval)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


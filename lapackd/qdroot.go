
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import "math"

// Reference:
//  Kahan, On the cost of Floating-Point Computation Without Extra-Precise Arithmetic, 2004

const roundOffConst = float64(3)
const bigConst = float64((1 << 27) + 1)

// Break 53 sig. bit float64 to two 26 sig bit parts
func break2(x float64) (xh, xt float64) {
    var bigx, y float64
    bigx = x*bigConst
    y = x - bigx
    xh = y + bigx
    xt = x - xh
    return
}

// Compute dicriminant of quadratic equation extra-precisely if necessary
// to ensure accuracy to the lat sig bit or two.
func discriminant(a, b, c float64) float64 {
    var d, e, ah, at, bh, bt, ch, ct, p, q, dp, dq float64
    d = b*b - a*c
    e = b*b + a*c

    // good enough?
    if roundOffConst*math.Abs(d) > e {
        return d
    }
    p = b*b; q = a*c
    ah, at = break2(a)
    bh, bt = break2(b)
    ch, ct = break2(c)
    
    dp = ((bh*bh - p) + 2*bh*bt) + bt*bt
    dq = ((ah*ch - q) + (ah*ct + at*ch)) + at*ct
    d = (p - q) + (dp - dq)
    return d
}

// Compute roots of quadratic equation
func QuadraticRoots(a, b, c float64) (x1, x2 float64, realRoots bool) {
    var d, r, s, signb, zerob float64
    realRoots = true
    d = discriminant(a, b, c)
    if d < 0.0 {
        r = b/a
        s = math.Sqrt(math.Abs(d))/a
        x1 = -r/2.0
        x2 = s/2.0
        realRoots = false
        return
    }
    zerob = 0.0
    if b == 0.0 {
        zerob = 1.0
    }
    signb = math.Copysign(1.0, b)
    s = math.Sqrt(d) * (signb + zerob) + b
    x1 = s/a
    x2 = c/s
    return
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

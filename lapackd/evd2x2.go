
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/gomas"
    "math"
)

/*
 * Compute SVD of 2x2 bidiagonal matrix
 */
func bdSvd2x2(f, g, h float64) (smin, smax float64) {
    var C, fa, ga, ha, fhmax, fhmin, gs, d, t float64

    fa = math.Abs(f)
    ha = math.Abs(h)
    ga = math.Abs(g)

    if fa > ha {
        fhmax = fa
        fhmin = ha
    } else {
        fhmax = ha
        fhmin = fa
    }

    if fhmin == 0.0 {
        smin = 0.0
        if fhmax == 0.0 {
            smax = ga
        } else {
            if fhmax > ga {
                fhmin = ga
            } else {
                fhmin = fhmax
                fhmax = ga
            }
            smax = fhmax * math.Hypot(1.0, fhmin/fhmax)
        }
        return
    }

    t = 1.0 + fhmin/fhmax
    d = 2.0 - t
    if ga < fhmax {
        gs = ga/fhmax
        C = 0.5*(math.Sqrt(gs*gs + t*t) + math.Sqrt(gs*gs + d*d))
    } else {
        gs = fhmax/ga
        if gs == 0.0 {
            smin = (fhmin * fhmax) / ga
            smax = ga
            return
        }
        C = math.Sqrt(1.0 + (gs*t)*(gs*t)) + math.Sqrt(1.0 + (gs*d)*(gs*d))
        C /= 2.0*gs
    }
    smin = fhmin/C
    smax = fhmax*C
    return
}

func BDSvd2x2(f, g, h float64) (smin, smax float64) {
    return bdSvd2x2(f, g, h)
}

func bdSvd2x2Vec(f, g, h float64) (ssmin, ssmax, cosl, sinl, cosr, sinr float64) {
    var smin, smax, clt, crt, srt, slt, amax, amin, fhmax, fhmin float64
    var gt, ga, fa, ha, d, t, l, m, t2, m2, s, r, a, tsign float64
    swap := false
    gmax := false
    
    fa = math.Abs(f)
    ha = math.Abs(h)
    ga = math.Abs(g)

    if fa > ha {
        fhmax = f
        amax = fa
        fhmin = h
        amin = ha
    } else {
        fhmax = h
        amax = ha
        fhmin = f
        amin = fa
        swap = true
    }
    gt = g

    if ga == 0.0 {
        smin = amin
        smax = amax
        clt = 1.0
        crt = 1.0
        slt = 0.0
        srt = 0.0
        goto Signs
    }

    if ga > amax {
        gmax = true
        if amax/ga < gomas.Epsilon {
            smax = ga
            if amin > 1.0 {
                smin = amax / (ga/amin)
            } else {
                smin = (amax/ga)*amin
            }
            clt = 1.0
            slt = fhmin/gt
            crt = fhmax/gt
            srt = 1.0
            goto Signs
        }
    }
    
    // normal case
    d = amax - amin
    if d == amax {
        l = 1.0
    } else {
        l = d / amax
    }
    m = gt/fhmax
    t = 2.0 - l
    m2 = m*m
    t2 = t*t
    s = math.Sqrt(t2 + m2)
    if l == 0.0 {
        r = math.Abs(m)
    } else {
        r = math.Sqrt(l*l + m2)
    }
    a = 0.5*(s + r)
    smin = amin/a
    smax = amax*a
    
    // singular vectors
    if m2 == 0.0 {
        if l == 0.0 {
            t = math.Copysign(2.0, fhmax) * math.Copysign(1.0, gt)
        } else {
            t = gt / math.Copysign(d, fhmax) + m/t
        }
    } else {
        t = (m / (s+t) + m/(r+l)) * (1.0 + a)
    }
    l = math.Sqrt(t*t + 4)
    crt = 2.0/l
    srt = t/l
    clt = (crt + srt*m)/a
    slt = (fhmin/fhmax)*srt/a
    
Signs:
    if swap {
        cosr = slt
        sinr = clt
        cosl = srt
        sinl = crt
        tsign = math.Copysign(1.0, cosr)*math.Copysign(1.0, cosl)*math.Copysign(1.0,f)
    } else {
        cosr = crt
        sinr = srt
        cosl = clt
        sinl = slt
        tsign = math.Copysign(1.0, sinr)*math.Copysign(1.0, sinl)*math.Copysign(1.0,h)
    }
    if gmax {
        tsign = math.Copysign(1.0, sinr)*math.Copysign(1.0, cosl)*math.Copysign(1.0,g)
    }
    
    ssmax = math.Copysign(smax, tsign)
    ssmin = math.Copysign(smin, tsign*math.Copysign(1.0,f)*math.Copysign(1.0,h))
    return
}

func BDSvd2x2Vec(f, g, h float64) (ssmin, ssmax, cosl, sinl, cosr, sinr float64) {
    return bdSvd2x2Vec(f, g, h)
}

// Eigenvalues of 2x2 symmetrix matrix
func symEigen2x2(a, b, c float64) (z1, z2 float64) {
    var T, b2a, amca, Zt, acmax, acmin float64

    acmax = a; acmin = c
    if math.Abs(c) > math.Abs(a) {
        acmax = c; acmin = a
    }
    b2a = math.Abs(b + b)
    T = a + c
    amca = math.Abs(a - c)
    Zt = T + math.Copysign(math.Hypot(amca, b2a), T)
    z1 = 0.5*Zt
    z2 = 2.0*((acmax/Zt)*acmin - (b/Zt)*b)
    return
}

// Compute eigenvalues and eigenvector for symmetric 2x2 matrix
func symEigen2x2Vec(a, b, c float64) (z1, z2, cs, sn float64) {

    z1, z2 = symEigen2x2(a, b, c)

    ht := math.Hypot(1.0, (z1 - a)/b)
    cs = 1.0/ht
    sn = (z1 - a)/(b*ht)
    return
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

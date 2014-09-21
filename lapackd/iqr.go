
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import "github.com/hrautila/cmat"

// Functions here are copies from armas C package.

/*
 * Bidiagonal top to bottom implicit QR sweep
 */
func bdQRsweep(D, E, Cr, Sr, Cl, Sl *cmat.FloatMatrix, f0, g0 float64, saves bool) int {
    var d1, e1, d2, e2, f, g, cosr, sinr, cosl, sinl, r float64;
    N := D.Len()

    d1 = D.GetAtUnsafe(0)
    e1 = E.GetAtUnsafe(0)
    f = f0
    g = g0
    for k := 0; k < N-1; k++ {
        d2 = D.GetAtUnsafe(k+1)
        cosr, sinr, r = ComputeGivens(f, g)
        if k > 0 {
            E.SetAtUnsafe(k-1, r)
        }
        f, e1 = RotateGivens(d1, e1, cosr, sinr)
        g, d2 = RotateGivens(0.0, d2, cosr, sinr)
        cosl, sinl, r = ComputeGivens(f, g)
        d1 = r
        f, d2 = RotateGivens(e1, d2, cosl, sinl)
        if k < N-2 {
            e2 = E.GetAtUnsafe(k+1)
            g, e2 = RotateGivens(0.0, e2, cosl, sinl)
            E.SetAtUnsafe(k+1, e2)
        }
        D.SetAtUnsafe(k, d1)
        //D.SetAtUnsafe(k+1, d2)
        d1 = d2
        e1 = e2
        if saves {
            Cr.SetAtUnsafe(k, cosr)
            Sr.SetAtUnsafe(k, sinr)
            Cl.SetAtUnsafe(k, cosl)
            Sl.SetAtUnsafe(k, sinl)
        }
    }
    D.SetAtUnsafe(N-1, d1)
    E.SetAtUnsafe(N-2, f)
    return N-1
}

func BDQRsweep(D, E *cmat.FloatMatrix, f0, g0 float64) int {
    return bdQRsweep(D, E, nil, nil, nil, nil, f0, g0, false)
}

/*
 * Bidiagonal top to bottom implicit zero shift QR sweep
 */
func bdQRzero(D, E, Cr, Sr, Cl, Sl *cmat.FloatMatrix, saves bool) int {
    var d1, e1, d2, cosr, sinr, cosl, sinl, r float64;
    N := D.Len()

    d1 = D.GetAtUnsafe(0)
    cosr = 1.0
    cosl = 1.0
    for k := 0; k < N-1; k++ {
        e1 = E.GetAtUnsafe(k)
        d2 = D.GetAtUnsafe(k+1)
        cosr, sinr, r = ComputeGivens(d1*cosr, e1)
        if k > 0 {
            E.SetAtUnsafe(k-1, sinl*r)
        }
        cosl, sinl, r = ComputeGivens(cosl*r, sinr*d2)
        D.SetAtUnsafe(k, r)
        d1 = d2
        if saves {
            Cr.SetAtUnsafe(k, cosr)
            Sr.SetAtUnsafe(k, sinr)
            Cl.SetAtUnsafe(k, cosl)
            Sl.SetAtUnsafe(k, sinl)
        }
    }
    d2 = cosr*d2
    D.SetAtUnsafe(N-1, d2*cosl)
    E.SetAtUnsafe(N-2, d2*sinl)
    return N-1
}

func BDQRzero(D, E *cmat.FloatMatrix) int {
    return bdQRzero(D, E, nil, nil, nil, nil, false)
}

/*
 * Bidiagonal bottom to top implicit QL sweep
 */
func bdQLsweep(D, E, Cr, Sr, Cl, Sl *cmat.FloatMatrix, f0, g0 float64, saves bool) int {
    var d1, e1, d2, e2, f, g, cosr, sinr, cosl, sinl, r float64;
    N := D.Len()

    d1 = D.GetAtUnsafe(N-1)
    e1 = E.GetAtUnsafe(N-2)
    f = f0
    g = g0
    for k := N-1; k > 0; k-- {
        d2 = D.GetAtUnsafe(k-1)
        cosr, sinr, r = ComputeGivens(f, g)
        if k < N-1 {
            E.SetAt(k, r)
        }
        f, e1 = RotateGivens(d1, e1, cosr, sinr)
        g, d2 = RotateGivens(0.0, d2, cosr, sinr)
        cosl, sinl, r = ComputeGivens(f, g)
        d1 = r
        f, d2 = RotateGivens(e1, d2, cosl, sinl)
        if k > 1 {
            e2 = E.GetAtUnsafe(k-2)
            g, e2 = RotateGivens(0.0, e2, cosl, sinl)
            E.SetAtUnsafe(k-2, e2)
        }
        D.SetAtUnsafe(k, d1)
        //D.SetAtUnsafe(k-1, d2)
        d1 = d2
        e1 = e2
        if saves {
            Cr.SetAtUnsafe(k-1, cosr)
            Sr.SetAtUnsafe(k-1, -sinr)
            Cl.SetAtUnsafe(k-1, cosl)
            Sl.SetAtUnsafe(k-1, -sinl)
        }
    }
    D.SetAtUnsafe(0, d1)
    E.SetAt(0, f)
    return N-1
}

func BDQLsweep(D, E *cmat.FloatMatrix, f0, g0 float64) int {
    return bdQLsweep(D, E, nil, nil, nil, nil, f0, g0, false)
}

/*
 * Bidiagonal bottom to top implicit zero shift QL sweep
 */
func bdQLzero(D, E, Cr, Sr, Cl, Sl *cmat.FloatMatrix, saves bool) int {
    var d1, e1, d2, cosr, sinr, cosl, sinl, r float64;
    N := D.Len()

    d1 = D.GetAtUnsafe(N-1)
    cosr = 1.0
    cosl = 1.0
    for k := N-1; k > 0; k-- {
        e1 = E.GetAtUnsafe(k-1)
        d2 = D.GetAtUnsafe(k-1)
        cosr, sinr, r = ComputeGivens(d1*cosr, e1)
        if k < N-1 {
            E.SetAtUnsafe(k, sinl*r)
        }
        cosl, sinl, r = ComputeGivens(cosl*r, sinr*d2)
        D.SetAtUnsafe(k, r)
        d1 = d2
        if saves {
            Cr.SetAtUnsafe(k-1, cosr)
            Sr.SetAtUnsafe(k-1, -sinr)
            Cl.SetAtUnsafe(k-1, cosl)
            Sl.SetAtUnsafe(k-1, -sinl)
        }
    }
    d2 = cosr*D.GetAtUnsafe(0)
    D.SetAtUnsafe(0, d2*cosl)
    E.SetAtUnsafe(0, d2*sinl)
    return N-1
}

func BDQLzero(D, E *cmat.FloatMatrix) int {
    return bdQLzero(D, E, nil, nil, nil, nil, false)
}

/*
 * Tridiagonal top to bottom implicit QR sweep
 */
func trdQRsweep(D, E, Cr, Sr *cmat.FloatMatrix, f0, g0 float64, saves bool) int {
    var d0, e0, e1, d1, e0r, e0c, w0, f, g, cosr, sinr, r float64;
    N := D.Len()

    d0 = D.GetAt(0)
    e0 = E.GetAt(0)
    f = f0
    g = g0
    for k := 0; k < N-1; k++ {
        d1 = D.GetAt(k+1)
        cosr, sinr, r = ComputeGivens(f, g)
        if k > 0 {
            E.SetAt(k-1, r)
        }
        d0, e0c = RotateGivens(d0, e0, cosr, sinr)
        e0r, d1 = RotateGivens(e0, d1, cosr, sinr)
        d0, e0r = RotateGivens(d0, e0r, cosr, sinr)
        e0c, d1 = RotateGivens(e0c, d1, cosr, sinr)
        // here: e0c == e0r
        if k < N-2 {
            e1 = E.GetAt(k+1)
            w0, e1 = RotateGivens(0.0, e1, cosr, sinr)
        }
        D.SetAt(k, d0)
        d0 = d1
        e0 = e1
        f = e0r
        g = w0
        if saves {
            Cr.SetAt(k, cosr)
            Sr.SetAt(k, sinr)
        }
    }
    D.SetAt(N-1, d0)
    E.SetAt(N-2, e0r)
    return N-1
}


/*
 * Tridiagonal bottom to top implicit QL sweep
 */
func trdQLsweep(D, E, Cr, Sr *cmat.FloatMatrix, f0, g0 float64, saves bool) int {
    var d0, e0, e1, d1, e0r, e0c, w0, f, g, cosr, sinr, r float64;
    N := D.Len()

    d0 = D.GetAtUnsafe(N-1)
    e0 = E.GetAtUnsafe(N-2)
    f = f0
    g = g0
    for k := N-1; k > 0; k-- {
        d1 = D.GetAtUnsafe(k-1)
        cosr, sinr, r = ComputeGivens(f, g)
        if k < N-1 {
            E.SetAtUnsafe(k, r)
        }
        d0, e0c = RotateGivens(d0, e0, cosr, sinr)
        e0r, d1 = RotateGivens(e0, d1, cosr, sinr)
        d0, e0r = RotateGivens(d0, e0r, cosr, sinr)
        e0c, d1 = RotateGivens(e0c, d1, cosr, sinr)
        // here: e0c == e0r
        if k > 1 {
            e1 = E.GetAtUnsafe(k-2)
            w0, e1 = RotateGivens(0.0, e1, cosr, sinr)
        }
        D.SetAtUnsafe(k, d0)
        d0 = d1
        e0 = e1
        f = e0r
        g = w0
        if saves {
            Cr.SetAtUnsafe(k-1, cosr)
            Sr.SetAtUnsafe(k-1, -sinr)
        }
    }
    D.SetAtUnsafe(0, d0)
    E.SetAtUnsafe(0, e0c)
    return N-1
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "math"
    //"fmt"
)

/*
 * Demmel-Kahan SVD algorithm as described in
 * (1)  Demmel, Kahan 1990, Accurate Singular Values of Bidiagonal Matrices
 *      (Lapack Working Notes #3)
 * (2)  Hogben 2007, 45.2 algorithm 2a,2b
 */

// estimat singular values
func estimateSval(D, E *cmat.FloatMatrix) (smin, smax float64) {
    var ssmin, ssmax, mu, e0, e1, d1 float64
    
    N := D.Len()

    d1 = math.Abs(D.GetAt(0))
    e1 = math.Abs(E.GetAt(0))
    e0 = e1
    ssmax = d1
    if e1 > ssmax {
        ssmax = e1
    }
    mu = d1; ssmin = d1
    for k := 1; k < N; k++ {
        d1 = math.Abs(D.GetAt(k))
        if k < N-1 {
            e1 = math.Abs(E.GetAt(k))
        }
        if d1 > ssmax {
            ssmax = d1
        }
        if e1 > ssmax {
            ssmax = e1
        }
        if ssmin != 0.0 {
            mu = d1 * (mu / (mu + e0))
            if mu < ssmin {
                ssmin = mu
            }
        }
        e0 = e1
    }
    smax = ssmax
    smin = ssmin/math.Sqrt(float64(N))
    return 
}

func asrow(d, s *cmat.FloatMatrix) *cmat.FloatMatrix {
    d.SetBuf(1, s.Len(), 1, s.Data())
    return d
}

// Demmel-Kahan SVD algorithm as described in (1)
func bdSvdDK(D, E, U, V, CS *cmat.FloatMatrix, tol float64, conf *gomas.Config) int {

    var maxit, niter, k, nrot, ip, iq, ipold, iqold int
    var e0, e1, d0, d1, dp, f0, g0, ushift, slow, shigh, threshold, mu float64
    var sD, sE, Cr, Sr, Cl, Sl cmat.FloatMatrix

    N := D.Len()
    saves := false
    forwards := true
    zero := false
    abstol := conf.Flags & gomas.OPT_ABSTOL != 0
    maxit = 6*N*N

    if (U != nil || V != nil) {
        Cr.SubVector(CS, 0, N)
        Sr.SubVector(CS, N, N)
        Cl.SubVector(CS, 2*N, N)
        Sl.SubVector(CS, 3*N, N)
        saves = true
    }

    slow, shigh = estimateSval(D, E)
    if abstol {
        threshold = float64(maxit)*gomas.Safemin
        if tol*shigh < threshold {
            threshold = tol*shigh
        }
    } else {
        threshold = tol*slow
        if float64(maxit)*gomas.Safemin < threshold {
            threshold = float64(maxit)*gomas.Safemin
        }
    }

    ip = 0; iq = N; ipold = ip; iqold = iq
    work := true
    for niter = 0; work && maxit > niter && iq > 0; niter++ {
        d0 = math.Abs(D.GetAtUnsafe(iq-1))
        shigh = d0
        if abstol && d0 < threshold {
            D.SetAt(iq-1, 0.0)
            shigh = 0.0
        }
        slow = shigh

        for k = iq-1; k > 0; k-- {
            d0 = math.Abs(D.GetAtUnsafe(k-1))
            e0 = math.Abs(E.GetAtUnsafe(k-1))
            if e0 < threshold {
                E.SetAt(k-1, 0.0)
                if k == (iq - 1) {
                    //fmt.Printf("..deflate at bottom E[%] %e, [S.%d %e]\n", k-1, e0, k, D.GetAt(k))
                    iq = iq - 1
                    work = k > 0
                    goto Next
                }
                ip = k
                break
            }
            ip = k - 1
            slow = math.Min(slow, d0)
            shigh = math.Max(math.Max(shigh, d0), e0)
        }

        if iq <= ip {
            work = false
            continue
        }

        if (iq-ip) == 2 {
            // 2x2 block
            //fmt.Printf("..2x2 block at %d ... %d\n", ip, iq)
            d0 = D.GetAtUnsafe(ip)
            d1 = D.GetAtUnsafe(ip+1)
            e1 = E.GetAtUnsafe(ip)
            smin, smax, cosl, sinl, cosr, sinr := bdSvd2x2Vec(d0, e1, d1)
            D.SetAtUnsafe(ip, smax)
            D.SetAtUnsafe(ip+1, smin)
            E.SetAtUnsafe(ip, 0.0)
            //fmt.Printf("..smin=%e, smax=%e, d0=%e, e1=%e, d1=%e\n", smin, smax, d0, e1, d1)
            if U != nil {
                ApplyGivensRight(U, ip, ip+1, 0, m(U), cosl, sinl)
            }
            if V != nil {
                ApplyGivensLeft(V, ip, ip+1, 0, n(V), cosr, sinr)
            }
            iq -= 2
            goto Next
        }

        if niter == 0 || iq != iqold || ip != ipold {
            // disjoint block
            ipold = ip; iqold = iq
            d0 = math.Abs(D.GetAtUnsafe(ip))
            d1 = math.Abs(D.GetAtUnsafe(iq-1))
            forwards = d1 >= d0
            //fmt.Printf("..new disjoint block [%d,%d] forward=%v\n", ip, iq, forwards)
        }

        // convergence
        if forwards {
            e0 = math.Abs(E.GetAtUnsafe(iq-2))
            d0 = math.Abs(D.GetAtUnsafe(iq-1))
            if (abstol && e0 < threshold) || e0 < tol*d0 {
                E.SetAt(iq-2, 0.0)
                //fmt.Printf("..deflate (1a) E[%d] %e < %e [S.%d %e]\n", iq-2, e0, tol*d0, iq-1, d0)
                iq = iq - 1
                goto Next
            }
            if ! abstol {
                mu = math.Abs(D.GetAtUnsafe(ip))
                for k = ip; k < iq-1; k++ {
                    e0 = math.Abs(E.GetAtUnsafe(k))
                    if e0 <= tol*mu {
                        //fmt.Printf("..deflate (1a) E[%d] %e < %e\n", k, e0, tol*d0)
                        E.SetAt(k, 0.0)
                        goto Next
                    }
                    d0 = math.Abs(D.GetAtUnsafe(k+1))
                    mu = d0 * (mu / (mu + e0))
                }
            }
        } else {
            e0 = math.Abs(E.GetAtUnsafe(ip))
            d0 = math.Abs(D.GetAtUnsafe(ip))
            if (abstol && e0 < threshold) || e0 < tol*d0 {
                //fmt.Printf("..deflate (2a) E[%d] %e < %e [S.%d %e]\n", ip, e0, tol*d0, ip, d0)
                E.SetAt(ip, 0.0)
                ip = ip + 1
                goto Next
            }
            if ! abstol {
                mu = math.Abs(D.GetAtUnsafe(iq-1))
                for k = iq-1; k > ip; k-- {
                    e0 = math.Abs(E.GetAtUnsafe(k-1))
                    if e0 <= tol*mu {
                        //fmt.Printf("..deflate (2b) E[%d] %e < %e\n", k, e0, tol*mu)
                        E.SetAt(k-1, 0.0)
                        goto Next
                    }
                    d0 = math.Abs(D.GetAt(k-1))
                    mu = d0 * (mu / (mu + e0))
                }
            }
        }

        // compute shift
        if ! abstol && (float64(N)*tol*(slow/shigh)) <= math.Max(gomas.Epsilon, 0.01*tol) {
            zero = true
        } else {
            if forwards {
                d0 = math.Abs(D.GetAtUnsafe(ip))
                d1 = D.GetAtUnsafe(iq-2)
                e1 = E.GetAtUnsafe(iq-2)
                dp = D.GetAtUnsafe(iq-1)
            } else {
                d0 = math.Abs(D.GetAtUnsafe(iq-1))
                d1 = D.GetAtUnsafe(ip)
                e1 = E.GetAtUnsafe(ip)
                dp = D.GetAtUnsafe(ip+1)
            }
            ushift, _ = bdSvd2x2(d1, e1, dp)
            if d0 > 0.0 {
                zero = (ushift/d0)*(ushift/d0) <= gomas.Epsilon
            }
        }
        
        //fmt.Printf("..ushift = %e, d1=%e, e1=%e, dp=%e, zero=%v\n", ushift, d1, e1, dp, zero)
        sD.SubVector(D, ip, iq-ip)
        sE.SubVector(E, ip, iq-ip-1)

        // run the QR/QL sweep
        if forwards {
            if zero {
                //fmt.Printf("..zero QR sweep ...\n")
                nrot = bdQRzero(&sD, &sE, &Cr, &Sr, &Cl, &Sl, saves)
            } else {
                f0 = (math.Abs(d0) - ushift)*(math.Copysign(1.0,d0) + ushift/d0)
                g0 = E.GetAtUnsafe(ip)
                nrot = bdQRsweep(&sD, &sE, &Cr, &Sr, &Cl, &Sl, f0, g0, saves)
            }
            e0 = math.Abs(E.GetAtUnsafe(iq-2))
            if e0 <= threshold {
                //dp = D.GetAt(iq-1)
                //fmt.Printf("..converge (F) after qrsweep [%d], %e [S.%d %e]\n", iq-2, e0, iq-1, dp)
                E.SetAt(iq-2, 0.0)
            }
            if U != nil {
                UpdateGivens(U, ip, &Cl, &Sl, nrot, gomas.RIGHT)
            }
            if V != nil {
                UpdateGivens(V, ip, &Cr, &Sr, nrot, gomas.LEFT)
            }
        } else {
            // from bottom to top
            if zero {
                //fmt.Printf("..zero QL sweep ...\n")
                nrot = bdQLzero(&sD, &sE, &Cr, &Sr, &Cl, &Sl, saves)
            } else {
                f0 = (math.Abs(d0) - ushift)*(math.Copysign(1.0,d0) + ushift/d0)
                g0 = E.GetAtUnsafe(iq-2)
                //fmt.Printf("..ushift = %e, f0 = %e, g0 = %e\n", ushift, f0, g0);
                nrot = bdQLsweep(&sD, &sE, &Cr, &Sr, &Cl, &Sl, f0, g0, saves)
            }
            e0 = math.Abs(E.GetAtUnsafe(ip))
            if e0 <= threshold {
                //dp = D.GetAt(ip)
                //fmt.Printf("..converge (B) after qlsweep [%d] %e [S.%d %e]\n", ip, e0, ip, dp)
                E.SetAt(ip, 0.0)
            }
            if U != nil {
                UpdateGivens(U, ip, &Cr, &Sr, nrot, gomas.BACKWARD|gomas.RIGHT)
            }
            if V != nil {
                UpdateGivens(V, ip, &Cl, &Sl, nrot, gomas.BACKWARD|gomas.LEFT)
            }
        }
        // update singular vectors
    Next:
        // next round of iteration
        //var t cmat.FloatMatrix
        //fmt.Printf("D [%3d]: %v\n", niter, asrow(&t, &sD))
        //fmt.Printf("E [%3d]: %v\n", niter, asrow(&t, &sE))
        //blasd.Scale(CS, 0.0)
        ;
    }
    
    if niter < maxit {
        // converged
        for k = 0; k < N; k++ {
            d0 = D.GetAt(k)
            if d0 < 0.0 {
                D.SetAt(k, -d0)
                if V != nil {
                    sD.Row(V, k)
                    blasd.Scale(&sD, -1.0)
                }
            }
        }
        return 0
    }
    return -1;
}

// Rotate lower bidiagonal matrix to upper bidiagonal matrix
func bdMakeUpper(D, E, U, C, CS *cmat.FloatMatrix) {
    var Cl, Sl cmat.FloatMatrix
    var cosl, sinl, r, d0, e0, d1 float64

    saves := false
    N := D.Len()
    if U != nil || C != nil {
        Cl.SubVector(CS, 0, N)
        Sl.SubVector(CS, N, N)
        saves = true
    }
    d0 = D.GetAt(0)
    for k := 0; k < N-1; k++ {
        e0 = E.GetAt(k)
        d1 = D.GetAt(k+1)
        cosl, sinl, r = ComputeGivens(d0, e0)
        D.SetAt(k, r)
        E.SetAt(k, sinl*d1)
        d0 = cosl*d1
        //D.SetAt(k+1, d0)
        if saves {
            Cl.SetAt(k, cosl)
            Sl.SetAt(k, sinl)
        }
    }
    D.SetAt(N-1, d0)
    if U != nil {
        UpdateGivens(U, 0, &Cl, &Sl, N-1, gomas.RIGHT)
    }
    if C != nil {
        UpdateGivens(C, 0, &Cl, &Sl, N-1, gomas.LEFT)
    }
}

/*
 * \brief Compute SVD of bidiagonal matrix.
 *
 * Computes the singular values and, optionially, the left and/or right
 * singular vectors from the SVD of a N-by-N upper or lower bidiagonal
 * matrix. The SVD of B has the form
 *
 *    B = U*S*V.T
 *
 * where S is the diagonal matrix with singular values, U is an orthogonal
 * matrix of left singular vectors, and V.T is an orthogonal matrix of right
 * singular vectors. If singular vectors are requested they must be initialized
 * either to unit diagonal matrix or some other orthogonal matrices.
 *
 */
func BDSvd(D, E, U, V, W *cmat.FloatMatrix, flags int, confs... *gomas.Config) *gomas.Error {
    var CS cmat.FloatMatrix
    var uu, vv *cmat.FloatMatrix
    var tol float64 = 5.0

    conf := gomas.CurrentConf(confs...)
    N := D.Len()

    if !(D.IsVector() && E.IsVector()) {
        return gomas.NewError(gomas.ENEED_VECTOR, "BDSvd")
    }

    if flags & gomas.WANTU != 0 {
        if U == nil {
            return gomas.NewError(gomas.EVALUE, "BDSvd")
        }
        if n(U) != N {
            return gomas.NewError(gomas.ESIZE, "BDSvd")
        }
        uu = U
    }
    if flags & gomas.WANTV != 0 {
        if V == nil {
            return gomas.NewError(gomas.EVALUE, "BDSvd")
        }
        if m(V) != N {
            return gomas.NewError(gomas.ESIZE, "BDSvd")
        }
        if flags & gomas.WANTU != 0 && m(V) != n(U) {
            return gomas.NewError(gomas.ESIZE, "BDSvd")
        }
        vv = V
    }
    if (uu != nil || vv != nil) && W.Len() < 4*N {
        return gomas.NewError(gomas.EWORK, "BDSvd", 4*N)
    }
    if uu != nil || vv != nil {
        CS.SetBuf(4*N, 1, 4*N, W.Data())
    } else {
        CS.SetBuf(0, 0, 0, nil)
    }
    if flags & gomas.LOWER != 0 {
        // rotate to upper
        bdMakeUpper(D, E, uu, nil, &CS)
    }
    tol = 10.0*gomas.Epsilon
    if conf.TolMult != 0 {
        tol = float64(conf.TolMult)*gomas.Epsilon
    }
    err := bdSvdDK(D, E, uu, vv, &CS, tol, conf)
    if err != 0 {
        return gomas.NewError(gomas.ECONVERGE, "BDSvd")
    }
    sortEigenVec(D, uu, vv, nil, gomas.Descending)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

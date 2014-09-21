
// Copyright (c) Harri Rautila, 2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "math"
    //"fmt"
)

// Computes eigenvalues and eigenvectors of symmetric tridiagonal matrix
// using implicit QR iteration (Golub 8.3.3)
// (lapack.DSTEQR)
func trdEvdQR(D, E, V, CS *cmat.FloatMatrix, tol float64, conf *gomas.Config) int {

    var ip, iq, ipold, iqold, niter, maxiter, nrot int
    var e0, e1, d0, d1, f0, g0, ushift, cs, sn float64
    var Cr, Sr, sD, sE cmat.FloatMatrix
    
    N := D.Len()
    forwards := true
    stop := false
    saves := false

    if V != nil {
        Cr.SubVector(CS, 0, N)
        Sr.SubVector(CS, N, N)
        saves = true
    }

    maxiter = 6*N
    iq = N; ip = 0
    for niter = 0; !stop && niter < maxiter && iq > 0; niter++ {
        // deflate offdiagonals
        d0 = math.Abs(D.GetAt(iq-1))
        for k := iq-1; k > 0; k-- {
            e1 = math.Abs(E.GetAt(k-1))
            d1 = math.Abs(D.GetAt(k-1))
            if e1 < tol*(d0+d1) {
                E.SetAt(k-1, 0.0)
                if k == (iq-1) {
                    // convergence at bottom
                    iq = iq - 1
                    stop = k == 0
                    goto Next
                } else if (k-1 == ip) {
                    // convergence at top
                    ip = k
                    goto Next
                }
                ip = k
                break
            }
            ip = k - 1
            d0 = d1
        }

        if iq <= ip {
            stop = true
            continue
        }

        if (iq-ip) == 2 {
            // 2x2 block
            a := D.GetAt(ip)
            b := E.GetAt(ip)
            c := D.GetAt(ip+1)
            e0, e1, cs, sn = symEigen2x2Vec(a, b, c)
            D.SetAt(ip, e0)
            D.SetAt(ip+1, e1)
            E.SetAt(ip, 0.0)
            if V != nil {
                ApplyGivensRight(V, ip, ip+1, 0, m(V), cs, sn)
            }
            iq -= 2
            goto Next
        }

        // new disjoint block
        if niter == 0 || iq != iqold || ip != ipold {
            ipold = ip; iqold = iq
            d0 = math.Abs(D.GetAt(ip))
            d1 = math.Abs(D.GetAt(iq-1))
            forwards = d1 >= d0
        }
        
        sD.SubVector(D, ip, iq-ip)
        sE.SubVector(E, ip, iq-ip-1)

        if forwards {
            // implicit QR
            d0 = D.GetAt(iq-2)
            e0 = E.GetAt(iq-2)
            d1 = D.GetAt(iq-1)
            ushift = wilkinson(d0, e0, d1)
            f0 = sD.GetAt(0) - ushift
            g0 = sE.GetAt(0)
            nrot = trdQRsweep(&sD, &sE, &Cr, &Sr, f0, g0, saves)
            if V != nil {
                UpdateGivens(V, ip, &Cr, &Sr, nrot, gomas.RIGHT)
            }
        } else {
            // implicit QL
            d0 = D.GetAt(ip)
            e0 = E.GetAt(ip)
            d1 = D.GetAt(ip+1)
            ushift = wilkinson(d1, e0, d0)
            f0 = D.GetAt(iq-1) - ushift
            g0 = E.GetAt(iq-2)
            nrot = trdQLsweep(&sD, &sE, &Cr, &Sr, f0, g0, saves)
            if V != nil {
                UpdateGivens(V, ip, &Cr, &Sr, nrot, gomas.RIGHT|gomas.BACKWARD)
            }
        }
    Next:
        ;
    }
    if niter < maxiter {
        return 0
    }
    return -1
}

// Compute eigenvalues and eigenvectors of symmetric tridiagonal matrix.
func TRDEigen(D, E, V, W *cmat.FloatMatrix, bits int, confs... *gomas.Config) *gomas.Error {
    var CS cmat.FloatMatrix
    var vv *cmat.FloatMatrix

    conf := gomas.CurrentConf(confs...)
    tol := float64(10.0)
    N := D.Len()

    vv = nil
    if !(D.IsVector() && E.IsVector()) {
        return gomas.NewError(gomas.ENEED_VECTOR, "TRDEigen")
    }
    if bits & gomas.WANTV != 0 {
        if V == nil {
            return gomas.NewError(gomas.EVALUE, "TRDEigen")
        }
        if m(V) != N {
            return gomas.NewError(gomas.ESIZE, "TRDEigen")
        }
        vv = V
    }
    if E.Len() != N-1 {
        return gomas.NewError(gomas.ESIZE, "TRDEigen")
    }
    if vv != nil && W.Len() < 2*N {
        return gomas.NewError(gomas.EWORK, "TRDEigen")
    }
    if vv != nil {
        CS.SetBuf(2*N, 1, 2*N, W.Data())
    } else {
        CS.SetBuf(0, 0, 0, nil)
    }
    tol = tol*gomas.Epsilon
    if conf.TolMult > 0 {
        tol = float64(conf.TolMult)*gomas.Epsilon
    }
    err := trdEvdQR(D, E, vv, &CS, tol, conf)
    if err != 0 {
        return gomas.NewError(gomas.ECONVERGE, "TRDEigen")
    }
    sortEigenVec(D, vv, nil, nil, gomas.Ascending)
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

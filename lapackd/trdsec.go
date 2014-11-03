
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/util"
    "github.com/hrautila/gomas/blasd"
    "math"
)
// References
// (1) Ren-Chang Li
//         Solving Secular Equations Stably and Efficiently, 1993 (LAWN #93)
// (2) Demmel
//         Applied Numerical Linear Algebra, 1996, (section 5.3.3)
// (3) Gu, Eisenstat
//         A Stable and Efficient Algorithm for the Rank-one Modification of the
//         Symmetric Eigenproblem, 1992
// (4) trdsec.c in armas library

// Compute rational approximation forward
func rationalForward(z, delta *cmat.FloatMatrix, start, end int) (float64, float64) {
    var val, dval float64
    val, dval = 0.0, 0.0
    for i := start; i < end; i++ {
        dj := delta.GetAtUnsafe(i)
        zj := z.GetAtUnsafe(i)
        tval := zj/dj;
        val  = val + zj*tval
        dval = dval + tval*tval
    }
    return val, dval
}

// Compute rational approximation backward
func rationalBackward(z, delta *cmat.FloatMatrix, start, end int) (float64, float64) {
    var val, dval float64
    val, dval = 0.0, 0.0
    for i := end-1; i >= start; i-- {
        dj := delta.GetAtUnsafe(i)
        zj := z.GetAtUnsafe(i)
        tval := zj/dj;
        val  = val + zj*tval
        dval = dval + tval*tval
    }
    return val, dval
}

func computeDelta(delta, D *cmat.FloatMatrix, index int, tau float64) {
    d0 := D.GetAt(index)
    for i := 0; i < delta.Len(); i++ {
        delta.SetAt(i, D.GetAt(i) - d0 - tau)
    }
}

func updateDelta(delta *cmat.FloatMatrix, eta float64) {
    for i := 0; i < delta.Len(); i++ {
        delta.SetAt(i, delta.GetAt(i) - eta)
    }
}

// Initial guess for iteration
func trdsecInitialGuess(tau, taulow, tauhigh *float64,
    D, Z, delta *cmat.FloatMatrix, index int, rho float64) int {

    var d_k, d_k1, z_k, z_k1, diff, mpnt, A, B, C, F, G0, G1, Hx, dd float64
    var iN, iK, N int
    N = D.Len()
    last := false
    if index == N-1 {
        iN = N-2
        last = true
    } else {
        iN = index
    }

    d_k  = D.GetAt(iN)
    d_k1 = D.GetAt(iN+1)
    diff = d_k1 - d_k
    mpnt = diff/2.0
    if last {
        mpnt = rho/2.0
    }

    // compute delta = D[i] - D[index] - mpnt
    computeDelta(delta, D, index, mpnt)
    G0, _ = rationalForward(Z, delta, 0, iN+1)
    G1, _ = rationalBackward(Z, delta, iN+1, N)
    
    d_k  = delta.GetAt(iN)
    d_k1 = delta.GetAt(iN+1)
    z_k  = Z.GetAt(iN)
    z_k1 = Z.GetAt(iN+1)
    // F is f(x) at initial point, 1/rho + g(y) + h(y)
    F = 1.0/rho + G0 + G1;
    // Hx is h(y)
    Hx = z_k*(z_k/d_k) + z_k1*(z_k1/d_k1)
    // C is g(y) at initial point
    C = F - Hx
    if last {
        goto LastEntry
    }

    if (F > 0) {
        iK = index
        A  = z_k*z_k*diff
        B  = C*diff + z_k*z_k + z_k1*z_k1
        *taulow = 0.0
        *tauhigh = mpnt
    } else {
        iK = index + 1
        A  = -z_k1*z_k1*diff
        B  = -C*diff + z_k*z_k + z_k1*z_k1
        *taulow = -mpnt
        *tauhigh = 0.0
    }
    B = B / 2.0
    dd = discriminant(A, B, C)
    if (B > 0.0) {
        *tau = A/(B + math.Sqrt(dd))
    } else {
        *tau = (B - math.Sqrt(dd))/C
    }
    computeDelta(delta, D, iK, *tau)
    return iK

LastEntry:
    A = -z_k1*z_k1*diff
    B = (-C*diff + z_k*z_k + z_k1*z_k1)/2.0
    Hx = z_k*z_k/(diff+rho) + z_k1*z_k1/rho
    if F <= 0.0 && C <= Hx {
        *tau = rho
    } else {
        dd = discriminant(A, B, C)
        if B < 0.0 {
            *tau = A/(B - math.Sqrt(dd))
        } else {
            *tau = (B + math.Sqrt(dd))/C
        }
    }
    if F < 0.0 {
        *taulow = mpnt
        *tauhigh = rho
    } else {
        *taulow = 0.0
        *tauhigh = mpnt
    }
    computeDelta(delta, D, N-1, *tau)
    return index-1
}
    
// Compute i'th root of secular function by rational approximation.
func trdsecRoot(D, Z, delta *cmat.FloatMatrix, index int, rho float64) (float64, int) {
    var H, dH, G, dG, F, dF, Fa, A, B, C, tau, tau_low, tau_high, eta, eta0, dd, edif float64
    var delta_k, delta_k1, da_k, da_k1, lmbda float64
    var iK, iK1, niter, maxiter, N int

    N = D.Len()
    tau = 0.0

    iK = trdsecInitialGuess(&tau, &tau_low, &tau_high, D, Z, delta, index, rho)
    if iK == index {
        iK1 = index + 1
        delta_k = delta.GetAt(iK)
        if index < N-1 {
            delta_k1 = delta.GetAt(iK1)
        } else {
            delta_k1 = tau
        }
    } else {
        iK1 = index
        delta_k1 = delta.GetAt(iK1)
        if index < N-1 {
            delta_k = delta.GetAt(iK)
        } else {
            delta_k = tau
        }
    }

    eta = 0.0
    eta0 = 0.0
    maxiter = 30
    for niter = 0; niter < maxiter; niter++ {

        G, dG = rationalForward(Z, delta, 0, iK+1)
        H, dH = rationalBackward(Z, delta, iK+1, N)
        F = 1.0/rho + G + H
        dF = dG + dH
        Fa = 1/rho + math.Abs(G+H)

        // stopping criterion (1) eq50, (3) eq 3.4
        if math.Abs(F) < float64(N)*gomas.Epsilon*Fa {
            break
        }

        // stopping criterion (1) eq49
        da_k  = math.Abs(delta_k)
        da_k1 = math.Abs(delta_k1)
        if  da_k < da_k1 {
            edif  = da_k*(math.Abs(eta0) - math.Abs(eta))
        } else {
            edif  = da_k1*(math.Abs(eta0) - math.Abs(eta))
        }
        if eta*eta < gomas.Epsilon*edif {
            break
        }

        // update limits
        if F < 0.0 {
            tau_low = math.Max(tau_low, tau)
        } else {
            tau_high = math.Min(tau_high, tau)
        }
        
        A = F - delta_k*dG - delta_k1*dH
        B = ((delta_k + delta_k1)*F - delta_k*delta_k1*(dH + dG))/2.0
        C = delta_k*delta_k1*F
        dd = discriminant(A, B, C)
        eta0 = eta
        if B > 0.0 {
            eta = C/(B + math.Sqrt(dd))
        } else {
            eta = (B - math.Sqrt(dd))/A
        }

        // F and eta should be of differenct sign
        if F*eta > 0.0 {
            eta = -F/dF
        }
        // Adjust if overshooting
        if tau+eta > tau_high || tau+eta < tau_low {
            if F < 0.0 {
                eta = (tau_high - tau)/2.0
            } else {
                eta = (tau_low - tau)/2.0
            }
        }

        tau += eta
        delta_k -= eta
        delta_k1 -= eta
        updateDelta(delta, eta)
    }

    if index == N-1 {
        lmbda = D.GetAt(N-1) + tau
    } else {
        lmbda = D.GetAt(iK) + tau
    }
    if niter == maxiter {
        niter = -niter
    }
    // return new lambda and number of iterations 
    return lmbda, niter
}

// Compute i'th updated element of rank-1 update vector
func trdsecUpdateElemDelta(d, delta *cmat.FloatMatrix, index int, rho float64) float64 {
    var n0, n1, dn, dk, val, p0, p1 float64
    var k, N int

    N = d.Len()
    dk = d.GetAt(index)
    dn = delta.GetAt(N-1)

    // compute; prod j; (lambda_j - d_k)/(d_j - d_k), j = 0 .. index-1
    p0 = 1.0
    for k = 0; k < index; k++ {
        n0 = delta.GetAt(k)
        n1 = d.GetAt(k) - dk
        p0 = p0 * (n0/n1)
    }

    p1 = 1.0
    for k = index; k < N-1; k++ {
        n0 = delta.GetAt(k)
        n1 = d.GetAt(k+1) - dk
        p1 = p1 * (n0/n1)
    }
    val = p0*p1*(dn/rho)
    return math.Sqrt(math.Abs(val))
}

// Compute the updated rank-1 update vector with precomputed deltas
func trdsecUpdateVecDelta(z, Q, d *cmat.FloatMatrix, rho float64) {
    var delta cmat.FloatMatrix
    for i := 0; i < d.Len(); i++ {
        delta.Column(Q, i)
        zk := trdsecUpdateElemDelta(d, &delta, i, rho)
        z.SetAt(i, zk)
    }
}

// Compute eigenvector corresponding precomputed deltas
func trdsecEigenVecDelta(qi, delta, z *cmat.FloatMatrix) {
    var dk, zk float64

    for k := 0; k < delta.Len(); k++ {
        zk = z.GetAt(k)
        dk = delta.GetAt(k)
        qi.SetAt(k, zk/dk)
    }
    s := blasd.Nrm2(qi)
    blasd.InvScale(qi, s)
}


// Compute eigenmatrix Q for updated eigenvalues in 'dl'.
func trdsecEigenBuild(Q, z, Q2 *cmat.FloatMatrix) {
    var qi, delta cmat.FloatMatrix

    for k := 0; k < z.Len(); k++ {
        qi.Column(Q, k)
        delta.Row(Q2, k)
        trdsecEigenVecDelta(&qi, &delta, z)
    }
}

func trdsecEigenBuildInplace(Q, z *cmat.FloatMatrix) {
    var QTL, QBR, Q00, q11, q12, q21, Q22, qi cmat.FloatMatrix
    var zk0, zk1, dk0, dk1 float64

    util.Partition2x2(
        &QTL, nil,
        nil,  &QBR, /**/ Q, 0, 0, util.PTOPLEFT)
        
    for m(&QBR) > 0 {
        util.Repartition2x2to3x3(&QTL,
            &Q00, nil,  nil,
            nil,  &q11, &q12,
            nil,  &q21, &Q22, /**/ Q, 1, util.PBOTTOMRIGHT)
        //---------------------------------------------------------------
        k := m(&Q00)
        zk0 = z.GetAt(k)
        dk0 = q11.Get(0, 0)
        q11.Set(0, 0, zk0/dk0)
        
        for i := 0; i < q12.Len(); i++ {
            zk1 = z.GetAt(k+i+1)
            dk0 = q12.GetAt(i)
            dk1 = q21.GetAt(i)
            q12.SetAt(i, zk0/dk1)
            q21.SetAt(i, zk1/dk0)
        }
        //---------------------------------------------------------------
        util.Continue3x3to2x2(
            &QTL,  nil,
            nil,  &QBR,  /**/ &Q00, &q11, &Q22, /**/ Q, util.PBOTTOMRIGHT)
    }
    // scale column eigenvectors
    for k := 0; k < z.Len(); k++ {
        qi.Column(Q, k)
        t := blasd.Nrm2(&qi)
        blasd.InvScale(&qi, t)
    }
}

// Solve secular function arising in symmetric eigenproblems. On exit 'Y' contains new
// eigenvalues On entry 'D' holds original eigenvalues and 'Z' is the rank-one update vector.
// Parameter 'delta' is workspace needed for computation.
func TRDSecularSolve(Y, D, Z, delta *cmat.FloatMatrix, rho float64, confs... *gomas.Config) (err *gomas.Error) {
    var lmbda float64
    var e, ei int
    ei = 0
    err = nil
    if delta.Len() != D.Len() || Y.Len() != D.Len() || Z.Len() != D.Len() {
        err = gomas.NewError(gomas.ESIZE, "TRDSecularSolve")
        return
    }
    for i := 0; i < D.Len(); i++ {
        lmbda, e = trdsecRoot(D, Z, delta, i, rho)
        if e < 0 && ei == 0 {
            ei = -(i+1)
        }
        Y.SetAt(i, lmbda)
    }
    if ei != 0 {
        err = gomas.NewError(gomas.ECONVERGE, "TRDSecularSolve", ei)
    }
    return 
}

// Solve secular function arising in symmetric eigenproblems. On exit 'Y' contains new
// eigenvalues and 'V' the rank-one update vector corresponding new eigenvalues.
// The matrix Qd holds for each eigenvalue then computed deltas as row vectors. 
// On entry 'D' holds original eigenvalues and 'Z' is the rank-one update vector.
func TRDSecularSolveAll(y, v, Qd, d, z *cmat.FloatMatrix, rho float64, confs... *gomas.Config) (err *gomas.Error) {
    var delta cmat.FloatMatrix
    var lmbda float64
    var e, ei int

    ei = 0
    err = nil
    if y.Len() != d.Len() || z.Len() != d.Len() || m(Qd) != n(Qd) || m(Qd) != d.Len() {
        err = gomas.NewError(gomas.ESIZE, "TRDSecularSolveAll")
        return
    }
    for i := 0; i < d.Len(); i++ {
        delta.Row(Qd, i)
        lmbda, e = trdsecRoot(d, z, &delta, i, rho)
        if e < 0 && ei == 0 {
            ei = -(i+1)
        }
        y.SetAt(i, lmbda)
    }
    if ei == 0 {
        trdsecUpdateVecDelta(v, Qd, d, rho)
    } else {
        err = gomas.NewError(gomas.ECONVERGE, "TRDSecularSolveAll", ei)
    }
    return 
}

// Computes eigenvectors corresponding the updated eigenvalues and rank-one update vector.
// The matrix Qd holds precomputed deltas as returned by TRDSecularSolveAll(). If Qd is nil or
// Qd same as the matrix Q then computation is in-place and Q is assumed to hold precomputed
// deltas. On exit, Q holds the column eigenvectors.
func TRDSecularEigen(Q, v, Qd *cmat.FloatMatrix, confs... *gomas.Config) *gomas.Error {
    if m(Q) != n(Q) || (Qd != nil && (m(Qd) != n(Qd) || m(Qd) != m(Q))) {
        return gomas.NewError(gomas.ESIZE, "TRDSecularEigen")
    }
    if m(Q) != v.Len() {
        return gomas.NewError(gomas.ESIZE, "TRDSecularEigen")
    }
    if Qd == nil || Qd == Q {
        trdsecEigenBuildInplace(Q, v)
    } else {
        trdsecEigenBuild(Q, v, Qd)
    }
    return nil
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

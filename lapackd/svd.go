
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "math"
)

func svdCrossover(n1, n2 int) bool {
    return n1 > int(float64(n2)*1.6)
}

func svdSmall(S, U, V, A, W *cmat.FloatMatrix, bits int, conf *gomas.Config) (err *gomas.Error) {
    var r cmat.FloatMatrix
    var d0, d1, e0 float64

    err = nil
    K := m(A)
    if n(A) < K {
        K = n(A)
    }
    tau := cmat.NewMatrix(K, 1)
    if m(A) >= n(A) {
        if err = QRFactor(A, tau, W, conf); err != nil {
            return
        }
    } else {
        if err = LQFactor(A, tau, W, conf); err != nil {
            return
        }
    }
    if m(A) == 1 || n(A) == 1 {
        // either tall M-by-1 or wide 1-by-N
        S.SetAt(0, math.Abs(A.Get(0, 0)))

        if bits & gomas.WANTU != 0 {
            if n(A) == 1 {
                if n(U) == n(A) {
                    // U is M-by-1
                    U.Copy(A)
                    QRBuild(U, tau, W, n(A), conf)
                } else {
                    // U is M-by-M
                    eye := cmat.FloatDiagonalSource{1.0}
                    U.SetFrom(&eye, cmat.SYMM)
                    if err = QRMult(U, A, tau, W, gomas.RIGHT, conf); err != nil {
                        return
                    }
                }
            } else {
                U.Set(0, 0, -1.0)
            }
        }
        if bits & gomas.WANTV != 0 {
            if m(A) == 1 {
                if m(V) == m(A) {
                    // V is 1-by-N
                    V.Copy(A)
                    LQBuild(V, tau, W, m(A), conf)
                } else {
                    // V is N-by-N
                    eye := cmat.FloatDiagonalSource{1.0}
                    V.SetFrom(&eye, cmat.SYMM)
                    if err = LQMult(V, A, tau, W, gomas.RIGHT, conf); err != nil {
                        return
                    }
                }
            } else {
                // V is 1-by-1
                V.Set(0, 0, -1.0)
            }
        }
        return
    }
    
    // Use bdSvd2x2 functions
    d0 = A.Get(0, 0)
    d1 = A.Get(1, 1)
    if m(A) >= n(A) {
        e0 = A.Get(0, 1)
    } else {
        e0 = A.Get(1, 0)
    }

    if bits & (gomas.WANTU|gomas.WANTV) == 0 {
        // no vectors
        smin, smax := bdSvd2x2(d0, e0, d1)
        S.SetAt(0, math.Abs(smax))
        S.SetAt(1, math.Abs(smin))
        return
    }

    // at least left or right eigenvector wanted
    smin, smax, cosl, sinl, cosr, sinr := bdSvd2x2Vec(d0, e0, d1)
    if bits & gomas.WANTU != 0 {
        // left eigenvectors
        if m(A) >= n(A) {
            if n(U) == n(A) {
                U.Copy(A)
                if err = QRBuild(U, tau, W, n(A), conf); err != nil {
                    return
                }
            } else {
                // U is M-by-M
                eye := cmat.FloatDiagonalSource{1.0}
                U.SetFrom(&eye, cmat.SYMM)
                if err = QRMult(U, A, tau, W, gomas.RIGHT, conf); err != nil {
                    return
                }
            }
            ApplyGivensRight(U, 0, 1, 0, m(A), cosl, sinl)
        } else {
            // U is 2-by-2
            eye := cmat.FloatDiagonalSource{1.0}
            U.SetFrom(&eye, cmat.SYMM)
            ApplyGivensRight(U, 0, 1, 0, m(A), cosr, sinr)
        }
    }

    if bits & gomas.WANTV != 0 {
        if n(A) > m(A) {
            if m(V) == m(A) {
                V.Copy(A)
                if err = LQBuild(V, tau, W, m(A), conf); err != nil {
                    return
                }
            } else {
                eye := cmat.FloatDiagonalSource{1.0}
                V.SetFrom(&eye, cmat.SYMM)
                if err = LQMult(V, A, tau, W, gomas.RIGHT, conf); err != nil {
                    return
                }
            }
            ApplyGivensLeft(V, 0, 1, 0, n(A), cosl, sinl)
        } else {
            // V is 2-by-2
            eye := cmat.FloatDiagonalSource{1.0}
            V.SetFrom(&eye, cmat.SYMM)
            ApplyGivensLeft(V, 0, 1, 0, n(A), cosr, sinr)
        }
        if smax < 0.0 {
            r.Row(V, 0)
            blasd.Scale(&r, -1.0)
        }
        if smin < 0.0 {
            r.Row(V, 1)
            blasd.Scale(&r, -1.0)
        }
    }
    S.SetAt(0, math.Abs(smax))
    S.SetAt(1, math.Abs(smin))
    return
}

// Compute SVD when m(A) >= n(A)
func svdTall(S, U, V, A, W *cmat.FloatMatrix, bits int, conf *gomas.Config) (err *gomas.Error) {
    var uu, vv *cmat.FloatMatrix
    var tauq, taup, Wred, sD, sE, R, Un cmat.FloatMatrix

    if (bits & (gomas.WANTU|gomas.WANTV)) != 0 {
        if W.Len() < 4*n(A) {
            err = gomas.NewError(gomas.ESIZE, "SVD")
            return
        }
    }
    tauq.SetBuf(n(A), 1, n(A), W.Data())
    taup.SetBuf(n(A)-1, 1, n(A)-1, W.Data()[tauq.Len():])
    wrl := W.Len() - 2*n(A) - 1
    Wred.SetBuf(wrl, 1, wrl, W.Data()[2*n(A)-1:])

    if svdCrossover(m(A), n(A)) {
        goto do_m_much_bigger
    }

    // reduce to bidiagonal form
    if err = BDReduce(A, &tauq, &taup, &Wred, conf); err != nil {
        return
    }

    sD.Diag(A)
    sE.Diag(A, 1)
    blasd.Copy(S, &sD)

    // left vectors
    if bits & gomas.WANTU != 0 {
        if n(U) == n(A) {
            // U is M-by-N; copy and make lower triangular
            U.Copy(A)
            cmat.TriL(U, 0)
            if err = BDBuild(U, &tauq, &Wred, n(U), gomas.WANTQ, conf); err != nil {
                return
            }
        } else {
            // U is M-by-M
            eye := cmat.FloatDiagonalSource{1.0}
            U.SetFrom(&eye, cmat.SYMM)
            if err = BDMult(U, A, &tauq, &Wred, gomas.MULTQ|gomas.RIGHT, conf); err != nil {
                return
            }
        }
        uu = U
    }
    // right vectors
    if bits & gomas.WANTV != 0 {
        R.SubMatrix(A, 0, 0, n(A), n(A))
        V.Copy(&R)
        cmat.TriU(V, 0)
        if err = BDBuild(V, &taup, &Wred, m(V), gomas.WANTP, conf); err != nil {
            return
        }
        vv = V
    }
    err = BDSvd(S, &sE, uu, vv, W, bits|gomas.UPPER)
    return

do_m_much_bigger:
    // M >> N here; first use QR factorization
    if err = QRFactor(A, &tauq, &Wred, conf); err != nil {
        return
    }
    if bits & gomas.WANTU != 0 {
        if n(U) == n(A) {
            U.Copy(A)
            if err = QRBuild(U, &tauq, &Wred, n(U), conf); err != nil {
                return
            }
        } else {
            // U is M-by-M
            eye := cmat.FloatDiagonalSource{1.0}
            U.SetFrom(&eye, cmat.SYMM)
            if err = QRMult(U, A, &tauq, &Wred, gomas.LEFT, conf); err != nil {
                return
            }
        }
    }
    R.SubMatrix(A, 0, 0, n(A), n(A))
    cmat.TriU(&R, 0)

    // bidiagonal reduce
    if err = BDReduce(&R, &tauq, &taup, &Wred, conf); err != nil {
        return
    }

    if bits & gomas.WANTU != 0 {
        Un.SubMatrix(U, 0, 0, m(A), n(A))
        if err = BDMult(&Un, &R, &tauq, &Wred, gomas.MULTQ|gomas.RIGHT, conf); err != nil {
            return
        }
        uu = U
    }
    if bits & gomas.WANTV != 0 {
        V.Copy(&R)
        if err = BDBuild(V, &taup, &Wred, m(V), gomas.WANTP, conf); err != nil {
            return
        }
        vv = V
    }

    sD.Diag(A)
    sE.Diag(A, 1)
    blasd.Copy(S, &sD)

    err = BDSvd(S, &sE, uu, vv, W, bits|gomas.UPPER, conf)
    return
}

func svdWide(S, U, V, A, W *cmat.FloatMatrix, bits int, conf *gomas.Config) (err *gomas.Error) {
    var uu, vv *cmat.FloatMatrix
    var tauq, taup, Wred, sD, sE, L, Vm cmat.FloatMatrix

    if (bits & (gomas.WANTU|gomas.WANTV)) != 0 {
        if W.Len() < 4*n(A) {
            err = gomas.NewError(gomas.ESIZE, "SVD")
            return
        }
    }
    tauq.SetBuf(m(A)-1, 1, m(A)-1, W.Data())
    taup.SetBuf(m(A), 1, m(A), W.Data()[tauq.Len():])
    wrl := W.Len() - 2*m(A) - 1
    Wred.SetBuf(wrl, 1, wrl, W.Data()[2*m(A)-1:])

    if svdCrossover(n(A), m(A)) {
        goto do_n_much_bigger
    }

    // reduce to bidiagonal form
    if err = BDReduce(A, &tauq, &taup, &Wred, conf); err != nil {
        return
    }

    sD.Diag(A)
    sE.Diag(A, -1)
    blasd.Copy(S, &sD)

    // leftt vectors
    if bits & gomas.WANTU != 0 {
        L.SubMatrix(A, 0, 0, m(A), m(A))
        U.Copy(&L)
        cmat.TriL(U, 0)
        if err = BDBuild(U, &tauq, &Wred, m(U), gomas.WANTQ|gomas.LOWER, conf); err != nil {
            return
        }
        uu = U
    }
    // right vectors
    if bits & gomas.WANTV != 0 {
        if m(V) == m(A) {
            // V is M-by-N; copy and make upper triangular
            V.Copy(A)
            //cmat.TriU(V, 0)
            if err = BDBuild(V, &taup, &Wred, m(V), gomas.WANTP, conf); err != nil {
                return
            }
        } else {
            // V is N-by-N
            eye := cmat.FloatDiagonalSource{1.0}
            V.SetFrom(&eye, cmat.SYMM)
            err = BDMult(V, A, &taup, &Wred, gomas.MULTP|gomas.LEFT|gomas.TRANS, conf)
            if err != nil {
                return
            }
        }
        vv = V
    }
    err = BDSvd(S, &sE, uu, vv, W, bits|gomas.LOWER)
    return

do_n_much_bigger:
    // here N >> M, use LQ factor first
    if err = LQFactor(A, &taup, &Wred, conf); err != nil {
        return
    }
    if bits & gomas.WANTV != 0 {
        if m(V) == m(A) {
            V.Copy(A)
            if err = LQBuild(V, &taup, &Wred, m(A), conf); err != nil {
                return
            }
        } else {
            // V is N-by-N
            eye := cmat.FloatDiagonalSource{1.0}
            V.SetFrom(&eye, cmat.SYMM)
            if err = LQMult(V, A, &taup, &Wred, gomas.RIGHT, conf); err != nil {
                return
            }
        }
    }
    L.SubMatrix(A, 0, 0, m(A), m(A))
    cmat.TriL(&L, 0)

    // resize tauq/taup for UPPER bidiagonal reduction
    tauq.SetBuf(m(A), 1, m(A), W.Data())
    taup.SetBuf(m(A)-1, 1, m(A)-1, W.Data()[tauq.Len():])

    // bidiagonal reduce
    if err = BDReduce(&L, &tauq, &taup, &Wred, conf); err != nil {
        return
    }

    if bits & gomas.WANTV != 0 {
        Vm.SubMatrix(V, 0, 0, m(A), n(A))
        err = BDMult(&Vm, &L, &taup, &Wred, gomas.MULTP|gomas.LEFT|gomas.TRANS, conf)
        if err != nil {
            return
        }
        vv = V
    }
    if bits & gomas.WANTU != 0 {
        U.Copy(&L)
        if err = BDBuild(U, &tauq, &Wred, m(U), gomas.WANTQ, conf); err != nil {
            return
        }
        uu = U
    }

    sD.Diag(A)
    sE.Diag(A, 1)
    blasd.Copy(S, &sD)

    err = BDSvd(S, &sE, uu, vv, W, bits|gomas.UPPER, conf)
    return
}

/*
 * \brief Compute SVD of general M-by-N matrix.
 *
 * Computes the singular values and, optionally, the left and/or right
 * singular vectors from the SVD of a The SVD of A has the form
 *
 *    A = U*S*V.T
 *
 * where S is the diagonal matrix with singular values, U is an orthogonal
 * matrix of left singular vectors, and V.T is an orthogonal matrix of right
 * singular vectors.
 *
 * If left singular vectors are requested by setting bit gomas.WANTU and M >= N,  the matrix U is
 * either M-by-N or M-by-M. If M < N then U is M-by-M.
 *
 * If left singular vectors are requested by setting bit gomas.WANTV and M >= N,  the matrix V is
 * either N-by-N. If M < N then V is M-by-N or N-by-N.
 *
 */
func SVD(S, U, V, A, W *cmat.FloatMatrix, bits int, confs... *gomas.Config) (err *gomas.Error) {

    err = nil
    tall := m(A) >= n(A)
    conf := gomas.CurrentConf(confs...)
    
    if tall && S.Len() < n(A) {
        err = gomas.NewError(gomas.ESIZE, "SVD")
        return
    }
    if !tall && S.Len() < m(A) {
        err = gomas.NewError(gomas.ESIZE, "SVD")
        return
    }
    if bits & gomas.WANTU != 0 {
        if U == nil {
            err = gomas.NewError(gomas.EVALUE, "SVD")
            return
        }
        if tall {
            // if M >= N; U is either M-by-N or M-by-M
            if m(U) != m(A) || (n(U) != m(A) && n(U) != n(A)) {
                err = gomas.NewError(gomas.ESIZE, "SVD")
                return
            }
        } else {
            if m(U) != m(A) {
                err = gomas.NewError(gomas.ESIZE, "SVD")
                return
            }
            // U is square is M < N
            if m(U) != n(U) {
                err = gomas.NewError(gomas.ESIZE, "SVD")
                return
            }
        }
    }
    if bits & gomas.WANTV != 0 {
        if V == nil {
            err = gomas.NewError(gomas.EVALUE, "SVD")
            return
        }
        if tall {
            if n(V) != n(A) {
                err = gomas.NewError(gomas.ESIZE, "SVD")
                return
            }
            // V is square is M >= N
            if m(V) != n(V) {
                err = gomas.NewError(gomas.ESIZE, "SVD")
                return
            }
        } else {
            // if M < N; V is either M-by-N or N-by-N
            if n(V) != n(A) || (m(V) != m(A) && m(V) != n(A)) {
                err = gomas.NewError(gomas.ESIZE, "SVD")
                return
            }
        }
    }
    if tall {
        if n(A) <= 2 {
            err = svdSmall(S, U, V, A, W, bits, conf)
        } else {
            err = svdTall(S, U, V, A, W, bits, conf)
        }
    } else {
        if m(A) <= 2 {
            err = svdSmall(S, U, V, A, W, bits, conf)
        } else {
            err = svdWide(S, U, V, A, W, bits, conf)
        }
    }
    if err != nil {
        err.Update("SVD")
    }
    return
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

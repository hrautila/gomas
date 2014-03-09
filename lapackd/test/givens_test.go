
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package test

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/lapackd"
    "testing"
)

/*
 * Note:
 *
 * QR factorization with Givens rotations.
 *
 *  R = G(n)...G(2)G(1)*A  and  Q = G(1).TG(2).T...G(n).T -> Q.T = G(n)...G(2)G(1)
 *
 *  A = Q*R = (Q.T).T*R
 */

// Simple and slow QR decomposition with Givens rotations
func TestGivensQR(t *testing.T) {
    var d cmat.FloatMatrix
    M := 181
    N := 159
    A := cmat.NewMatrix(M, N)
    A1 := cmat.NewCopy(A)

    ones := cmat.NewFloatConstSource(1.0)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    A0 := cmat.NewCopy(A)
    
    Qt := cmat.NewMatrix(M, M)
    d.Diag(Qt)
    d.SetFrom(ones)

    // R = G(n)...G(2)G(1)*A; Q = G(1).T*G(2).T...G(n).T ;  Q.T = G(n)...G(2)G(1)

    // for all columns ...
    for j := 0; j < N; j++ {
        // ... zero elements below diagonal, starting from bottom
        for i := M-2; i >=j; i-- {
            c, s, r := lapackd.ComputeGivens(A.Get(i, j), A.Get(i+1, j))
            A.Set(i, j, r)
            A.Set(i+1, j, 0.0)
            // apply rotations on this row starting from column j, N-j column
            lapackd.ApplyGivensLeft(A, i, j+1, N-j-1, c, s)
            // update Qt = G(k)*Qt 
            lapackd.ApplyGivensLeft(Qt, i, 0, M, c, s)
        }
    }
    // check: A = Q*R
    blasd.Mult(A1, Qt, A, 1.0, 0.0, gomas.TRANSA)
    blasd.Plus(A0, A1, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||A - G(n)..G(1)*R||_1: %e\n", M, N, nrm)
}

// Simple and slow LQ decomposition with Givens rotations
func TestGivensLQ(t *testing.T) {
    var d cmat.FloatMatrix
    M := 149
    N := 167
    A := cmat.NewMatrix(M, N)
    A1 := cmat.NewCopy(A)

    ones := cmat.NewFloatConstSource(1.0)
    src := cmat.NewFloatNormSource()
    A.SetFrom(src)
    A0 := cmat.NewCopy(A)
    
    Qt := cmat.NewMatrix(N, N)
    d.Diag(Qt)
    d.SetFrom(ones)

    // R = G(n)...G(2)G(1)*A; Q = G(1).T*G(2).T...G(n).T ;  Q.T = G(n)...G(2)G(1)
    for i := 0; i < M; i++ {
        // zero elements right of diagonal
        for j := N-2; j >= i; j-- {
            c, s, r := lapackd.ComputeGivens(A.Get(i, j), A.Get(i, j+1))
            A.Set(i, j,   r)
            A.Set(i, j+1, 0.0)
            // apply rotation to this column starting from row i+1
            lapackd.ApplyGivensRight(A, i+1, j, M-i-1, c, s)
            // update Qt = G(k)*Qt
            lapackd.ApplyGivensRight(Qt, 0, j, N, c, s)
        }
    }
    // A = L*Q
    blasd.Mult(A1, A, Qt, 1.0, 0.0, gomas.TRANSB)
    blasd.Plus(A0, A1, 1.0, -1.0, gomas.NONE)
    nrm := lapackd.NormP(A0, lapackd.NORM_ONE)
    t.Logf("M=%d, N=%d ||A - L*G(1)..G(n)||_1: %e\n", M, N, nrm)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

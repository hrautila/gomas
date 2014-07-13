
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package gomas

const (
    LOWER  = 1 << iota  // lower triangular matrix
    UPPER               // upper triangular matrix
    SYMM                // symmetric matrix
    HERM                // hermitian matrix
    UNIT                // unit diagonal matrix
    LEFT                // multiply from left
    RIGHT               // multiply from right
    TRANSA              // A transposed
    TRANSB              // B transposed
    CONJA               // A is conjugate
    CONJB               // B is conjugate
    // bits above must be in sync with bits in C-header blasd/inc/interfaces.h
    // bits below are lapack spesific
    MULTQ               // multiply with orthogonal Q
    MULTP               // multiply with orthogonal P
    WANTQ               // generate orthogonal matrix Q
    WANTP               // generate orthogonal matrix P
    WANTU               // generate orthogonal left eigenvectors U
    WANTV               // generate orthogonal right eigenvectors V
    NULL = 0
    NONE = 0
    TRANS = TRANSA     // transposed matrix
    CONJ  = CONJA      // conjugate matrix
)


// calculated machine epsilon; smallest float64 number E for which 1.0 + E != 1.0
const Epsilon = 2.2204460492503131e-16        

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

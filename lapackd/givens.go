
// Copyright (c) Harri Rautila, 2013,2014

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "math"
)

/*
 * Givens rotation preserves vector length ie. || (a, b) ||_2 == || (r 0) ||_2 and therefore
 * r = sqrt(a*a + b*b).
 *
 * LEFT:  
 *  (c -s )( a ) = ( r )  == ( c*a - s*b ) = ( r )
 *  (s  c )( b )   ( 0 )     ( s*a + c*b )   ( 0 ) --> s = -(b/a)*c
 *
 *     c*a + (b/a)*b*c = r 
 *     c*(a^2 + b^2)/a = sqrt(a^2 + b^2)
 *     c = a/sqrt(a^2+b^2)
 *     c = a/r
 *     --> s = -b/r
 *
 * RIGHT:
 *  ( a  b )( c -s ) = (r  0) == (a*c+b*s  b*c-a*s) = ( r 0 ) --> s = (b/a)*c
 *         ( s  c )            
 *
 *      a*c + b*(b/a)*c = r
 *      c*(a^2 + b^2)/a = sqrt(a^2 + b^2)
 *      c = a/sqrt(a^2 + b^2)
 *      c = a/r
 *      --> s = b/r
 *
 * 
 *  r(R) ==  r(L) = sqrt(a^2 + b^2)
 *  c(R) ==  c(L) = a/r
 *  s(R) == -s(L) = -b/r, S(R) = b/r
 */

/*
 * Compute Givens rotation such that
 *
 *   G(s,c)*v = (r)   ==  ( c  -s ) ( a ) = ( r )
 *              (0)       ( s   c ) ( b )   ( 0 )
 *
 * or if bits RIGHT is set
 *
 *   v*G(s,c) = (r 0 ) == (a b ) ( c -s ) = ( r 0 )
 *                               ( s  c )
 *
 */
func ComputeGivens(a, b float64, bits int)  (c float64, s float64, r float64) {

    if b == 0.0 {
        if math.Signbit(a) {
            c = -a
        } else {
            c = a
        }
        s = 0.0
        r = math.Abs(a)
    } else if a == 0.0 {
        if math.Signbit(b) {
            s = -b
        } else  {
            s = b
        }
        c = 0.0
        r = math.Abs(b)
    } else if math.Abs(b) > math.Abs(a) {
        t := a/b
        u := math.Sqrt(1.0 + t*t)
        if math.Signbit(b) {
            u = -u
        }
        s = -1.0/u
        c = -s*t
        r = b*u
    } else {
        t := b/a
        u := math.Sqrt(1.0 + t*t)
        if math.Signbit(a) {
            u = -u
        }
        c = 1.0 / u
        s = -c*t
        r = a*u
    }
    if bits & gomas.RIGHT != 0 {
        s = -s
    }
    return 
}

/*
 * Compute A[i:i+1,j:j+nc] = G(c,s)*A[i:i+1,j:j+nc]
 *
 * Applies Givens rotation to nc columns on rows i:i+1 of A starting from column j.
 */
func ApplyGivensLeft(A *cmat.FloatMatrix, i, j, nc int, c, s float64) {
    if m(A)-i < 2 {
        // one row
        for k := j; k < j+nc; k++ {
            v0 := A.Get(i, k)
            A.Set(i, k, c*v0)
        }
        return
    }
    for k := j; k < j+nc; k++ {
        v0 := A.Get(i,   k)
        v1 := A.Get(i+1, k)
        y0 := c*v0 - s*v1
        y1 := s*v0 + c*v1
        A.Set(i,   k, y0)
        A.Set(i+1, k, y1)
    }
}

/*
 * Compute A[i:i+nr,j:j+1] = A[i:i+nr,j:j+1]*G(c,s)
 *
 * Applies Givens rotation to nr rows of columns j:j+1 of A starting at row i.
 */
func ApplyGivensRight(A *cmat.FloatMatrix, i, j, nr int, c, s float64) {
    if n(A)-j < 2 {
        // one column
        for k := i; k < i+nr; k++ {
            v0 := A.Get(k, j)
            A.Set(k, j, c*v0)
        }
        return
    }
    for k := i; k < i+nr; k++ {
        v0 := A.Get(k, j)
        v1 := A.Get(k, j+1)
        y0 := v0*c + v1*s
        y1 := v1*c - v0*s
        A.Set(k, j+0, y0)
        A.Set(k, j+1, y1)
    }
}

/*
 * If bits LEFT is set then compute
 *    ( y0 )  = G(c,s) * ( v0 )  
 *    ( y1 )             ( v1 )
 *
 * If bits RIGHT or ^LEFT
 *    ( y0 y1 ) = ( v0 v1 ) * G(c,s)
 */
func RotateGivens(v0, v1, c, s float64, bits int) (y0, y1 float64) {
    if bits & gomas.LEFT != 0 {
        y0 = c*v0 - s*v1
        y1 = s*v0 + c*v1
    } else {
        y0 = v0*c + v1*s
        y1 = v1*c - v0*s
    }
    return
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

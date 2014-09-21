
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
 *  ( a  b )( c  s ) = (r  0) == (a*c-b*s  a*s+b*c) = ( r 0 ) --> s = -(b/a)*c
 *          (-s  c )            
 *
 *      a*c + b*(b/a)*c = r
 *      c*(a^2 + b^2)/a = sqrt(a^2 + b^2)
 *      c = a/sqrt(a^2 + b^2)
 *      c = a/r
 *      --> s = -b/r
 *
 * 
 *  r(R) ==  r(L) = sqrt(a^2 + b^2)
 *  c(R) ==  c(L) = a/r
 *  s(R) ==  s(L) = b/r
 */

/*
 * Compute Givens rotation such that
 *
 *   G(s,c)*v = (r)   ==  (  c s ).T ( a ) = ( r )
 *              (0)       ( -s c )   ( b )   ( 0 )
 *
 * and 
 *
 *   v*G(s,c) = (r 0 ) == (a b ) (  c  s ) = ( r 0 )
 *                               ( -s  c )
 *
 */
func ComputeGivens(a, b float64)  (c float64, s float64, r float64) {

    if b == 0.0 {
        c = 1.0
        s = 0.0
        r = a
    } else if a == 0.0 {
        c = 0.0
        s = 1.0
        r = b
    } else if math.Abs(b) > math.Abs(a) {
        t := a/b
        u := math.Sqrt(1.0 + t*t)
        if math.Signbit(b) {
            u = -u
        }
        s = 1.0/u
        c = s*t
        r = b*u
    } else {
        t := b/a
        u := math.Sqrt(1.0 + t*t)
        r = a*u
        c = 1.0 / u
        s = c*t
    }
    return 
}

/*
 * Computes 
 *
 *    ( y0 )  = G(c,s) * ( v0 )  
 *    ( y1 )             ( v1 )
 *
 *    ( y0 y1 ) = ( v0 v1 ) * G(c,s)
 */
func RotateGivens(v0, v1, cos, sin float64) (y0, y1 float64) {
    y0 = cos*v0 + sin*v1
    y1 = cos*v1 - sin*v0
    return
}

/*
 * Compute A[i:i+1,j:j+nc] = G(c,s)*A[i:i+1,j:j+nc]
 *
 * Applies Givens rotation to nc columns on rows r1,r2 of A starting from col.
 */
func ApplyGivensLeft(A *cmat.FloatMatrix, r1, r2, col, ncol int, cos, sin float64) {
    if col >= n(A) {
        return
    }
    if r1 == r2 {
        // one row
        for k := col; k < col+ncol; k++ {
            v0 := A.Get(r1, k)
            A.Set(r1, k, cos*v0)
        }
        return
    }
    for k := col; k < col+ncol; k++ {
        v0 := A.Get(r1, k)
        v1 := A.Get(r2, k)
        y0 := cos*v0 + sin*v1
        y1 := cos*v1 - sin*v0
        A.Set(r1,  k, y0)
        A.Set(r2, k, y1)
    }
}

/*
 * Compute A[i:i+nr,c1;c2] = A[i:i+nr,c1;c2]*G(c,s)
 *
 * Applies Givens rotation to nr rows of columns c1, c2 of A starting at row.
 *
 */
func ApplyGivensRight(A *cmat.FloatMatrix, c1, c2, row, nrow int, cos, sin float64) {
    if row >= m(A) {
        return
    }
    if c1 == c2 {
        // one column
        for k := row; k < row+nrow; k++ {
            v0 := A.Get(k, c1)
            A.Set(k, c1, cos*v0)
        }
        return
    }
    for k := row; k < row+nrow; k++ {
        v0 := A.Get(k, c1)
        v1 := A.Get(k, c2)
        y0 := cos*v0 + sin*v1
        y1 := cos*v1 - sin*v0
        A.Set(k, c1, y0)
        A.Set(k, c2, y1)
    }
}

func UpdateGivens(A *cmat.FloatMatrix, start int, C, S *cmat.FloatMatrix, nrot, flags int) int {
    var k, l int 
    var cos, sin float64
    end := start + nrot 

    if flags & gomas.BACKWARD != 0 {
        if flags & gomas.LEFT != 0 {
            end = imin(m(A), end)
            k = end; l = nrot;
            for l > 0 && k > start {
                cos = C.GetAt(l-1)
                sin = S.GetAt(l-1)
                if  cos != 1.0 || sin != 0.0 {
                    ApplyGivensLeft(A, k-1, k, 0, n(A), cos, sin)
                }
                l--
                k--
            }
        } else {
            end = imin(n(A), end)
            k = end; l = nrot;
            for l > 0 && k > start {
                cos = C.GetAt(l-1)
                sin = S.GetAt(l-1)
                if  cos != 1.0 || sin != 0.0 {
                    ApplyGivensRight(A, k-1, k, 0, m(A), cos, sin)
                }
                l--
                k--
            }
        }
    } else {
        if flags & gomas.LEFT != 0 {
            end = imin(m(A), end)
            k = start; l = 0;
            for l < nrot && k < end {
                cos = C.GetAt(l)
                sin = S.GetAt(l)
                if  cos != 1.0 || sin != 0.0 {
                    ApplyGivensLeft(A, k, k+1, 0, n(A), cos, sin)
                }
                l++
                k++
            }
        } else {
            end = imin(n(A), end)
            k = start; l = 0;
            for l < nrot && k < end {
                cos = C.GetAt(l)
                sin = S.GetAt(l)
                if  cos != 1.0 || sin != 0.0 {
                    ApplyGivensRight(A, k, k+1, 0, m(A), cos, sin)
                }
                l++
                k++
            }
        }
    }
    return nrot
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

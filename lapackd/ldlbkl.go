
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
    "github.com/hrautila/cmat"
    "github.com/hrautila/gomas"
    "github.com/hrautila/gomas/blasd"
    "github.com/hrautila/gomas/util"
    "math"
    //"fmt"
)

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 *
 * LOWER triangular; moving from top-left to bottom-right
 *
 *    -----------------------
 *    | d 
 *    | x P1 x  x  x  P2     -- current row/col 'srcix'
 *    | x S2 d  x  x  x
 *    | x S2 x  d  x  x
 *    | x S2 x  x  d  x
 *    | x P2 D2 D2 D2 P3     -- swap with row/col 'dstix'
 *    | x S3 x  x  x  D3 d
 *    | x S3 x  x  x  D3 x d
 *         (AR)
 */
func applyBKPivotSymLower(AR *cmat.FloatMatrix, srcix, dstix int) {
    var s, d cmat.FloatMatrix
    // S2 -- D2
    s.SubMatrix(AR, srcix+1, srcix,   dstix-srcix-1, 1)
    d.SubMatrix(AR, dstix,   srcix+1, 1, dstix-srcix-1)
    blasd.Swap(&s, &d)
    // S3 -- D3
    s.SubMatrix(AR, dstix+1, srcix, m(AR)-dstix-1, 1)
    d.SubMatrix(AR, dstix+1, dstix, m(AR)-dstix-1, 1)
    blasd.Swap(&s, &d)
    // swap P1 and P3
    p1 := AR.Get(srcix, srcix)
    p3 := AR.Get(dstix, dstix)
    AR.Set(srcix, srcix, p3)
    AR.Set(dstix, dstix, p1)
    return
}

/*
 * α = (1 + sqrt(17))/8
 * λ = |a(r,1)| = max{|a(2,1)|, . . . , |a(m,1)|}
 * if λ > 0
 *     if |a(1,1)| ≥ αλ
 *         use a11 as 1-by-1 pivot
 *     else
 *         σ = |a(p,r)| = max{|a(1,r)|,..., |a(r−1,r)|, |a(r+1,r)|,..., |a(m,r)|}
 *         if |a(1,1) |σ ≥ αλ^2
 *             use a(1,1) as 1-by-1 pivot
 *         else if |a(r,r)| ≥ ασ
 *             use a(r,r) as 1-by-1 pivot
 *         else
 *                  a11 | ar1
 *             use  --------  as 2-by-2 pivot
 *                  ar1 | arr
 *         end
 *     end
 * end
 *
 *    -----------------------
 *    | d 
 *    | x P1 x  x  x  P2     -- current row/col 'srcix'
 *    | x S2 d  x  x  x
 *    | x S2 x  d  x  x
 *    | x S2 x  x  d  x
 *    | x P2 D2 D2 D2 P3     -- swap with row/col 'dstix'
 *    | x S3 x  x  x  D3 d
 *    | x S3 x  x  x  D3 x d
 *         (AR)
 */
func findBKPivotLower(A *cmat.FloatMatrix) (int, int) {
    var r, q int
    var rcol, qrow cmat.FloatMatrix
    if m(A) == 1 {
        return 0, 1
    }
    amax := math.Abs(A.Get(0, 0))
    // column below diagonal at [0, 0]
    rcol.SubMatrix(A, 1, 0, m(A)-1, 1)
    r = blasd.IAmax(&rcol) + 1
    // max off-diagonal on first column at index r
    rmax := math.Abs(A.Get(r, 0))
    if amax >= bkALPHA*rmax {
        // no pivoting, 1x1 diagonal
        return 0, 1
    }
    // max off-diagonal on r'th row at index q
    qrow.SubMatrix(A, r, 0, 1, r/*+1*/)
    q = blasd.IAmax(&qrow)
    qmax := math.Abs(A.Get(r, q/*+1*/))
    if r < m(A)-1 {
        // rest of the r'th row after diagonal
        qrow.SubMatrix(A, r+1, r, m(A)-r-1, 1)
        q = blasd.IAmax(&qrow)
        qmax2 := math.Abs(qrow.Get(q, 0))
        if qmax2 > qmax {
            qmax = qmax2
        }
    }
        
    if amax >= bkALPHA*rmax*(rmax/qmax) {
        // no pivoting, 1x1 diagonal
        return 0, 1
    }
    if math.Abs(A.Get(r,r)) >= bkALPHA*qmax {
        // 1x1 pivoting and interchange with k, r
        return r, 1 
    } else {
        // 2x2 pivoting and interchange with k+1, r
        return r, 2
    }
    return 0, 1
}


/*
 * Unblocked Bunch-Kauffman LDL factorization.
 *
 * Corresponds lapack.DSYTF2
 */
func unblkDecompBKLower(A, wrk *cmat.FloatMatrix, p Pivots, conf *gomas.Config) (*gomas.Error, int) {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10t, a11, A20, a21, A22, a11inv cmat.FloatMatrix
    var cwrk cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots

    err = nil
    nc := 0

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PTOPLEFT)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, util.PTOP)

    // permanent working space for symmetric inverse of a11
    a11inv.SubMatrix(wrk, 0, n(wrk)-2, 2, 2)
    a11inv.Set(1, 0, -1.0)
    a11inv.Set(0, 1, -1.0)

    for n(&ABR) > 0 {

        r, np := findBKPivotLower(&ABR)
        if r != 0 && r != np-1  { 
            // pivoting needed; do swaping here
            applyBKPivotSymLower(&ABR, np-1, r)
            if np == 2 {
                /*          [0,0] | [r,0]
                 * a11 ==   -------------  2-by-2 pivot, swapping [1,0] and [r,0]
                 *          [r,0] | [r,r]
                 */
                t := ABR.Get(1, 0)
                ABR.Set(1, 0, ABR.Get(r, 0))
                ABR.Set(r, 0, t)
            }
        }
        
        // repartition according the pivot size
        util.Repartition2x2to3x3(&ATL, 
            &A00,  nil,  nil,
            &a10t, &a11, nil,
            &A20,  &a21, &A22,   /**/ A, np, util.PBOTTOMRIGHT)
        repartPivot2x1to3x1(&pT, 
            &p0,
            &p1,
            &p2,   /**/ p, np, util.PBOTTOM)
        // ------------------------------------------------------------
        if np == 1 {
            // A22 = A22 - a21*a21.T/a11
            blasd.MVUpdateTrm(&A22, &a21, &a21, -1.0/a11.Get(0, 0), gomas.LOWER)
            // a21 = a21/a11
            blasd.InvScale(&a21, a11.Get(0, 0))
            // store pivot point relative to original matrix
            p1[0] = r + m(&ATL) + 1
        } else if np == 2 {
            /* from Bunch-Kaufmann 1977:
             *  (E2 C.T) = ( I2      0      )( E  0      )( I[n-2] E.-1*C.T )
             *  (C  B  )   ( C*E.-1  I[n-2] )( 0  A[n-2] )( 0      I2       )
             *
             *  A[n-2] = B - C*E.-1*C.T
             *
             *  E.-1 is inverse of a symmetric matrix, cannot use
             *  triangular solve. We calculate inverse of 2x2 matrix.
             *  Following is inspired by lapack.SYTF2
             *  
             *      a | b      1        d | -b         b         d/b | -1 
             *  inv ----- =  ------  * ------  =  ----------- * --------
             *      b | d    (ad-b^2)  -b |  a    (a*d - b^2)     -1 | a/b
             *
             */
            a := a11.Get(0, 0)
            b := a11.Get(1, 0)
            d := a11.Get(1, 1)
            a11inv.Set(0, 0,  d/b)
            a11inv.Set(1, 1,  a/b)
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale := 1.0 / ((a/b)*(d/b) - 1.0)
            scale /= b
            // cwrk = a21
            cwrk.SubMatrix(wrk, 2, 0, m(&a21), n(&a21))
            blasd.Copy(&cwrk, &a21)
            // a21 = a21*a11.-1
            blasd.Mult(&a21, &cwrk, &a11inv, scale, 0.0, gomas.NONE, conf)
            // A22 = A22 - a21*a11.-1*a21.T = A22 - a21*cwrk.T
            blasd.UpdateTrm(&A22, &a21, &cwrk, -1.0, 1.0, gomas.LOWER|gomas.TRANSB, conf)
            // store pivot point relative to original matrix
            p1[0] = -(r + m(&ATL) + 1)
            p1[1] = p1[0]
        }
        // ------------------------------------------------------------
        nc += np
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    p, util.PBOTTOM)
    }
    return err, nc
}


/*
 * Find diagonal pivot and build incrementaly updated block.
 *
 *  (AL)  (AR)                   (WL)  (WR)
 *  --------------------------   ----------    k'th row in W
 *  x x | c1                     w w | k kp1
 *  x x | c1 d                   w w | k kp1
 *  x x | c1 x  d                w w | k kp1
 *  x x | c1 x  x  d             w w | k kp1
 *  x x | c1 r2 r2 r2 r2         w w | k kp1
 *  x x | c1 x  x  x  r2 d       w w | k kp1
 *  x x | c1 x  x  x  r2 x d     w w | k kp1
 *         
 * Matrix AR contains the unfactored part of the matrix and AL the already
 * factored columns. Matrix WL is updated values of factored part ie.
 * w(i) = l(i)d(i). Matrix WR will have updated values for next column.
 * Column WR(k) contains updated AR(c1) and WR(kp1) possible pivot row AR(r2).
 */
func findAndBuildBKPivotLower(AL, AR, WL, WR *cmat.FloatMatrix, k int) (int, int) {
    var r, q int
    var rcol, qrow, src, wk, wkp1, wrow cmat.FloatMatrix

    // Copy AR column 0 to WR column 0 and update with WL[0:]
    src.SubMatrix(AR, 0, 0, m(AR), 1)
    wk.SubMatrix(WR, 0, 0, m(AR), 1)
    wk.Copy(&src)
    if k > 0 {
        wrow.SubMatrix(WL, 0, 0, 1, n(WL))
        blasd.MVMult(&wk, AL, &wrow, -1.0, 1.0, gomas.NONE)
    }
    if m(AR) == 1 {
        return 0, 1
    }
    amax := math.Abs(WR.Get(0, 0))

    // find max off-diagonal on first column. 
    rcol.SubMatrix(WR, 1, 0, m(AR)-1, 1)

    // r is row index and rmax is its absolute value
    r = blasd.IAmax(&rcol) + 1
    rmax := math.Abs(rcol.Get(r-1, 0))
    if amax >= bkALPHA*rmax {
        // no pivoting, 1x1 diagonal
        return 0, 1
    }
    // Now we need to copy row r to WR[:,1] and update it
    wkp1.SubMatrix(WR, 0, 1, m(AR), 1)
    qrow.SubMatrix(AR, r, 0, 1, r+1)
    blasd.Copy(&wkp1, &qrow)
    if  r < m(AR)-1 {
        var wkr cmat.FloatMatrix
        qrow.SubMatrix(AR, r, r, m(AR)-r, 1)
        wkr.SubMatrix(&wkp1, r, 0, m(&wkp1)-r, 1)
        blasd.Copy(&wkr, &qrow)
    }
    if k > 0 {
        // update wkp1
        wrow.SubMatrix(WL, r, 0, 1, n(WL))
        blasd.MVMult(&wkp1, AL, &wrow, -1.0, 1.0, gomas.NONE)
    }
    
    // set on-diagonal entry to zero to avoid finding it
    p1 := wkp1.Get(r, 0)
    wkp1.Set(r, 0, 0.0)  
    // max off-diagonal on r'th column/row at index q
    q = blasd.IAmax(&wkp1)
    qmax := math.Abs(wkp1.Get(q, 0))
    // restore on-diagonal entry
    wkp1.Set(r, 0, p1)  
        
    if amax >= bkALPHA*rmax*(rmax/qmax) {
        // no pivoting, 1x1 diagonal
        return 0, 1
    }
    // if q == r then qmax is not off-diagonal, qmax == WR[r,1] and
    // we get 1x1 pivot as following is always true
    if math.Abs(WR.Get(r, 1)) >= bkALPHA*qmax {
        // 1x1 pivoting and interchange with k, r
        // pivot row in column WR[:,1] to W[:,0]
        src.SubMatrix(WR,  0, 1, m(AR), 1)
        wkp1.SubMatrix(WR, 0, 0, m(AR), 1)
        blasd.Copy(&wkp1, &src)
        wkp1.Set(0, 0, src.Get(r, 0))
        wkp1.Set(r, 0, src.Get(0, 0))
        return r, 1 
    } else {
        // 2x2 pivoting and interchange with k+1, r
        return r, 2
    }
    return 0, 1
}


/*
 * Unblocked, bounded Bunch-Kauffman LDL factorization for at most ncol columns.
 * At most ncol columns are factorized and trailing matrix updates are restricted
 * to ncol columns. Also original columns are accumulated to working matrix, which
 * is used by calling blocked algorithm to update the trailing matrix with BLAS3
 * update.
 *
 * Corresponds lapack.DLASYF
 */
func unblkBoundedBKLower(A, wrk *cmat.FloatMatrix, p *Pivots, ncol int, conf *gomas.Config) (*gomas.Error, int) {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10t, a11, A20, a21, A22, a11inv cmat.FloatMatrix
    var w00, w10, w11 cmat.FloatMatrix
    var cwrk cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots

    err = nil
    nc := 0
    if ncol > n(A) {
        ncol = n(A)
    }

    // permanent working space for symmetric inverse of a11
    a11inv.SubMatrix(wrk, 0, n(wrk)-2, 2, 2)
    a11inv.Set(1, 0, -1.0)
    a11inv.Set(0, 1, -1.0)

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,    A, 0, 0, util.PTOPLEFT)
    partitionPivot2x1(
        &pT,
        &pB, *p, 0, util.PTOP)

    for n(&ABR) > 0 && nc < ncol {

        util.Partition2x2(
            &w00, nil,
            &w10, &w11, wrk, nc, nc, util.PTOPLEFT)

        r, np := findAndBuildBKPivotLower(&ABL, &ABR, &w10, &w11, nc)
        if np > ncol - nc {
            // next pivot does not fit into ncol columns, restore last column,
            // return with number of factorized columns
            return err, nc
        }
        if r != 0 && r != np-1  {
            // pivoting needed; do swaping here
            applyBKPivotSymLower(&ABR, np-1, r)
            // swap left hand rows to get correct updates
            swapRows(&ABL, np-1, r)
            swapRows(&w10, np-1, r)
            if np == 2 {
                /*
                 *          [0,0] | [r,0]
                 * a11 ==   -------------  2-by-2 pivot, swapping [1,0] and [r,0]
                 *          [r,0] | [r,r]
                 */
                t0 := w11.Get(1, 0)
                tr := w11.Get(r, 0)
                w11.Set(1, 0, tr) 
                w11.Set(r, 0, t0)
                // interchange diagonal entries on w11[:,1] 
                t0 = w11.Get(1, 1)
                tr = w11.Get(r, 1)
                w11.Set(1, 1, tr)
                w11.Set(r, 1, t0)
            }
        }
        // repartition according the pivot size
        util.Repartition2x2to3x3(&ATL, 
            &A00,  nil,  nil,
            &a10t, &a11, nil,
            &A20,  &a21, &A22,   /**/ A, np, util.PBOTTOMRIGHT)
        repartPivot2x1to3x1(&pT, 
            &p0,
            &p1,
            &p2,   /**/ *p, np, util.PBOTTOM)
        // ------------------------------------------------------------

        if np == 1 {
            // 
            cwrk.SubMatrix(&w11, np, 0, m(&a21), np)
            a11.Set(0, 0, w11.Get(0, 0))
            // a21 = a21/a11
            blasd.Copy(&a21, &cwrk)
            blasd.InvScale(&a21, a11.Get(0, 0))
            // store pivot point relative to original matrix
            p1[0] = r + m(&ATL) + 1
        } else if np == 2 {
            /*
             * See comments for this block in unblkDecompBKLower().
             */
            a := w11.Get(0, 0)
            b := w11.Get(1, 0)
            d := w11.Get(1, 1)
            a11inv.Set(0, 0,  d/b)
            a11inv.Set(1, 1,  a/b)
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale := 1.0 / ((a/b)*(d/b) - 1.0)
            scale /= b

            cwrk.SubMatrix(&w11, np, 0, m(&a21), np)
            // a21 = a21*a11.-1
            blasd.Mult(&a21, &cwrk, &a11inv, scale, 0.0, gomas.NONE, conf)
            a11.Set(0, 0, a)
            a11.Set(1, 0, b)
            a11.Set(1, 1, d)

            // store pivot point relative to original matrix
            p1[0] = -(r + m(&ATL) + 1)
            p1[1] = p1[0]
        }
        // ------------------------------------------------------------
        nc += np
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PBOTTOMRIGHT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    *p, util.PBOTTOM)

    }
    return err, nc
}



func blkDecompBKLower(A, W *cmat.FloatMatrix, p *Pivots, conf *gomas.Config) (err *gomas.Error) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A10, A11,  A20, A21, A22 cmat.FloatMatrix
    var wrk cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots
    var nblk int = 0

    err = nil
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PTOPLEFT)
    partitionPivot2x1(
        &pT,
        &pB, *p, 0, util.PTOP)

    nb := conf.LB
    for n(&ABR) >= nb {
        err, nblk = unblkBoundedBKLower(&ABR, W, &pB, nb, conf)
        // repartition nblk size
        util.Repartition2x2to3x3(&ATL,
            &A00, nil,  nil,
            &A10, &A11, nil,
            &A20, &A21, &A22,   A, nblk, util.PBOTTOMRIGHT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,   /**/ *p, nblk, util.PBOTTOM)

        // --------------------------------------------------------
        // here [A11;A21] has been decomposed by unblkBoundedBKLower()
        // Now we need update A22

        // wrk is original A21
        wrk.SubMatrix(W, nblk, 0, m(&A21), nblk)

        // A22 = A22 - L21*D1*L21.T = A22 - L21*W.T
        blasd.UpdateTrm(&A22, &A21, &wrk, -1.0, 1.0, gomas.LOWER|gomas.TRANSB)

        // partially undo row pivots left of diagonal
        for k := nblk; k > 0; k-- {
            var s, d cmat.FloatMatrix
            r := p1[k-1]
            rlen := k-1
            if r < 0 {
                r = -r
                rlen--
            }
            if r == k && p1[k-1] > 0 {
                // no pivot
                continue
            }
            s.SubMatrix(&ABR, k-1, 0, 1, rlen)
            d.SubMatrix(&ABR, r-1, 0, 1, rlen)
            blasd.Swap(&d, &s)

            if p1[k-1] < 0 {
                k-- // skip other entry in 2x2 pivots
            }
        }
        // shift pivot values
        for k, n := range p1 {
            if n > 0 {
                p1[k] += m(&ATL)
            } else {
                p1[k] -= m(&ATL)
            }
        }
        // zero work for debuging
        //blasd.Scale(W, 0.0)
        // ---------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,  A, util.PBOTTOMRIGHT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    *p, util.PBOTTOM)
    }

    // do the last part with unblocked code
    if n(&ABR) > 0 {
        unblkDecompBKLower(&ABR, W, pB, conf)
        // shift pivot values
        for k, n := range pB {
            if n > 0 {
                pB[k] += m(&ATL)
            } else {
                pB[k] -= m(&ATL)
            }
        }
    }
    return
}

func unblkSolveBKLower(B, A *cmat.FloatMatrix, p Pivots, phase int, conf *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a10t, a11, A20, a21, A22 /*, a11inv*/ cmat.FloatMatrix
    var Aref *cmat.FloatMatrix
    var BT, BB, B0, b1, B2, Bx cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots
    var aStart, aDir, bStart, bDir util.Direction
    var nc int

    np := 0

    if phase == 1 {
        aStart = util.PTOPLEFT
        aDir   = util.PBOTTOMRIGHT
        bStart = util.PTOP
        bDir   = util.PBOTTOM
        nc = 1
        Aref = &ABR
    } else {
        aStart = util.PBOTTOMRIGHT
        aDir   = util.PTOPLEFT
        bStart = util.PBOTTOM
        bDir   = util.PTOP
        nc = m(A)
        Aref = &ATL
    }
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, aStart)
    util.Partition2x1(
        &BT, 
        &BB,  B, 0, bStart)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, bStart)

    for n(Aref) > 0 {
        // see if next diagonal block is 1x1 or 2x2
        np = 1
        if p[nc-1] < 0 {
            np = 2
        }

        // repartition according the pivot size
        util.Repartition2x2to3x3(&ATL, 
            &A00,  nil,  nil,
            &a10t, &a11, nil,
            &A20,  &a21, &A22,   /**/ A, np, aDir)
        util.Repartition2x1to3x1(&BT, 
            &B0,
            &b1,
            &B2,   /**/ B, np, bDir)
        repartPivot2x1to3x1(&pT, 
            &p0,
            &p1,
            &p2,   /**/ p, np, bDir)
        // ------------------------------------------------------------

        switch phase {
        case 1:
            // computes D.-1*(L.-1*B)
            if np == 1 {
                if p1[0] != nc {
                    // swap rows on bottom part of B
                    swapRows(&BB, 0, p1[0]-m(&BT)-1)
                }
                // B2 = B2 - a21*b1
                blasd.MVUpdate(&B2, &a21, &b1, -1.0)
                // b1 = b1/d1 
                blasd.InvScale(&b1, a11.Get(0, 0))
                nc += 1
            } else if np == 2 {
                if p1[0] != -nc {
                    // swap rows on bottom part of B
                    swapRows(&BB, 1, -p1[0]-m(&BT)-1)
                }
                b := a11.Get(1, 0)
                apb := a11.Get(0, 0) / b
                dpb := a11.Get(1, 1) / b
                // (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
                scale := apb*dpb - 1.0
                scale *= b

                // B2 = B2 - a21*b1
                blasd.Mult(&B2, &a21, &b1, -1.0, 1.0, gomas.NONE, conf)
                // b1 = a11.-1*b1.T
                //(2x2 block, no subroutine for doing this in-place)
                for k := 0; k < n(&b1); k++ {
                    s0 := b1.Get(0, k) 
                    s1 := b1.Get(1, k) 
                    b1.Set(0, k, (dpb*s0-s1)/scale)
                    b1.Set(1, k, (apb*s1-s0)/scale)
                }
                nc += 2
            }
        case 2:
            if np == 1 {
                blasd.MVMult(&b1, &B2, &a21, -1.0, 1.0, gomas.TRANS)
                if p1[0] != nc {
                    // swap rows on bottom part of B
                    util.Merge2x1(&Bx, &b1, &B2)
                    swapRows(&Bx, 0, p1[0]-m(&BT))
                }
                nc -= 1
            } else if np == 2 {
                blasd.Mult(&b1, &a21, &B2, -1.0, 1.0, gomas.TRANSA, conf)
                if p1[0] != -nc {
                    // swap rows on bottom part of B
                    util.Merge2x1(&Bx, &b1, &B2)
                    swapRows(&Bx, 1, -p1[0]-m(&BT)+1)
                }
                nc -= 2
            }
        }

        // ------------------------------------------------------------

        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, aDir)
        util.Continue3x1to2x1(
            &BT,
            &BB,    &B0, &b1,    B, bDir)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    p, bDir)

    }
    return err
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

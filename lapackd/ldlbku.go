
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
 * UPPER triangular; moving from bottom-right to top-left
 *
 *    d x D3 x  x  x  S3 x |
 *      d D3 x  x  x  S3 x |
 *        P3 D2 D2 D2 P2 x |  -- dstinx
 *           d  x  x  S2 x |
 *              d  x  S2 x |
 *                 d  S2 x |
 *                    P1 x |  -- srcinx
 *                       d |
 *    ----------------------
 *               (ABR)
 */
func applyBKPivotSymUpper(AR *cmat.FloatMatrix, srcix, dstix int) {
    var s, d cmat.FloatMatrix
    // AL is ATL, AR is ATR; P1 is AL[srcix, srcix];
    // S2 -- D2
    s.SubMatrix(AR, dstix+1, srcix, srcix-dstix-1, 1)
    d.SubMatrix(AR, dstix,   dstix+1, 1, srcix-dstix-1)
    blasd.Swap(&s, &d)
    // S3 -- D3
    s.SubMatrix(AR, 0, srcix, dstix, 1)
    d.SubMatrix(AR, 0, dstix, dstix, 1)
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
 *    d x D3 x  x  x  S3 x |
 *      d D3 x  x  x  S3 x |
 *        P3 D2 D2 D2 P2 x |  -- dstinx
 *           d  x  x  S2 x |
 *              d  x  S2 x |
 *                 d  S2 x |
 *                    P1 x |  -- srcinx
 *                       d |
 *    ----------------------
 *               (ABR)
 */

func findBKPivotUpper(A *cmat.FloatMatrix) (int, int) {
    var r, q int
    var rcol, qrow cmat.FloatMatrix
    if m(A) == 1 {
        return 0, 1
    }
    lastcol := m(A) - 1
    amax := math.Abs(A.Get(lastcol, lastcol))
    // column above [A.Rows()-1, A.Rows()-1]
    rcol.SubMatrix(A, 0, lastcol, lastcol, 1)

    r = blasd.IAmax(&rcol)
    // max off-diagonal on first column at index r
    rmax := math.Abs(A.Get(r, lastcol))

    if amax >= bkALPHA*rmax {
        // no pivoting, 1x1 diagonal
        return -1, 1
    }
    // max off-diagonal on r'th row at index q
    //  a) rest of the r'th row above diagonal
    qmax := 0.0
    if r > 0 {
        qrow.SubMatrix(A, 0, r, r, 1)
        q = blasd.IAmax(&qrow)
        qmax = math.Abs(A.Get(q, r/*+1*/))
    }
    //  b) elements right of diagonal
    qrow.SubMatrix(A, r, r+1, 1, lastcol-r)
    q = blasd.IAmax(&qrow)
    qmax2 := math.Abs(qrow.Get(0, q))
    if qmax2 > qmax {
        qmax = qmax2
    }
        
    if amax >= bkALPHA*rmax*(rmax/qmax) {
        // no pivoting, 1x1 diagonal
        return -1, 1
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
func unblkDecompBKUpper(A, wrk *cmat.FloatMatrix, p Pivots, conf *gomas.Config) (*gomas.Error, int) {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a11, a12, A22, a11inv cmat.FloatMatrix
    var cwrk cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots

    err = nil
    nc := 0

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR, A, 0, 0, util.PBOTTOMRIGHT)
    partitionPivot2x1(
        &pT,
        &pB, p, 0, util.PBOTTOM)

    // permanent working space for symmetric inverse of a11
    a11inv.SubMatrix(wrk, 0, n(wrk)-2, 2, 2)
    a11inv.Set(1, 0, -1.0)
    a11inv.Set(0, 1, -1.0)

    for n(&ATL) > 0 {

        nr := m(&ATL) - 1
        r, np := findBKPivotUpper(&ATL)
        if r != -1  { 
            cwrk.SubMatrix(&ATL, 0, n(&ATL)-np, m(&ATL), np)
            // pivoting needed; do swaping here
            applyBKPivotSymUpper(&ATL, m(&ATL)-np, r)
            if np == 2 {
                /*          [r, r] | [r, nr]
                 * a11 ==   ----------------  2-by-2 pivot, swapping [nr-1,nr] and [r,nr]
                 *          [nr,r] | [nr,nr]  (nr is the current diagonal entry)
                 */
                t := ATL.Get(nr-1, nr)
                ATL.Set(nr-1, nr, ATL.Get(r, nr))
                ATL.Set(r, nr, t)
            }
        }
        
        // repartition according the pivot size
        util.Repartition2x2to3x3(&ATL, 
            &A00, &a01, &A02,
            nil,  &a11, &a12,
            nil,  nil,  &A22,   /**/ A, np, util.PTOPLEFT)
        repartPivot2x1to3x1(&pT, 
            &p0,
            &p1,
            &p2,   /**/ p, np, util.PTOP)
        // ------------------------------------------------------------

        if np == 1 {
            // A00 = A00 - a01*a01.T/a11
            blasd.MVUpdateTrm(&A00, &a01, &a01, -1.0/a11.Get(0, 0), gomas.UPPER)
            // a01 = a01/a11
            blasd.InvScale(&a01, a11.Get(0, 0))
            // store pivot point relative to original matrix
            if r == -1 {
                p1[0] = m(&ATL)
            } else {
                p1[0] = r + 1
            }
        } else if np == 2 {
            /* see comments on unblkDecompBKLower() */
            a := a11.Get(0, 0)
            b := a11.Get(0, 1)
            d := a11.Get(1, 1)
            a11inv.Set(0, 0,  d/b)
            a11inv.Set(1, 1,  a/b)
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale := 1.0 / ((a/b)*(d/b) - 1.0)
            scale /= b

            // cwrk = a21
            cwrk.SubMatrix(wrk, 2, 0, m(&a01), n(&a01))
            blasd.Copy(&cwrk, &a01)
            // a01 = a01*a11.-1
            blasd.Mult(&a01, &cwrk, &a11inv, scale, 0.0, gomas.NONE, conf)
            // A00 = A00 - a01*a11.-1*a01.T = A00 - a01*cwrk.T
            blasd.UpdateTrm(&A00, &a01, &cwrk, -1.0, 1.0, gomas.UPPER|gomas.TRANSB, conf)
            // store pivot point relative to original matrix
            p1[0] = -(r + 1)
            p1[1] = p1[0]
        }
        // ------------------------------------------------------------
        nc += np
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    p, util.PTOP)
    }
    return err, nc
}


/*
 * Find diagonal pivot and build incrementaly updated block.
 *
 *    d x r2 x  x  x  c1 | x x     kp1 k | w w 
 *      d r2 x  x  x  c1 | x x     kp1 k | w w 
 *        r2 r2 r2 r2 c1 | x x     kp1 k | w w 
 *           d  x  x  c1 | x x     kp1 k | w w 
 *              d  x  c1 | x x     kp1 k | w w 
 *                 d  c1 | x x     kp1 k | w w 
 *                    c1 | x x     kp1 k | w w 
 *   --------------------------   -------------
 *               (AL)     (AR)     (WL)   (WR) 
 *
 * Matrix AL contains the unfactored part of the matrix and AR the already
 * factored columns. Matrix WR is updated values of factored part ie.
 * w(i) = l(i)d(i). Matrix WL will have updated values for next column.
 * Column WL(k) contains updated AL(c1) and WL(kp1) possible pivot row AL(r2).
 *
 * On exit, for 1x1 diagonal the rightmost column of WL (k) holds the updated
 * value of AL(c1). If pivoting this required the WL(k) holds the actual pivoted
 * column/row.
 *
 * For 2x2 diagonal blocks WL(k) holds the updated AL(c1) and WL(kp1) holds
 * actual values of pivot column/row AL(r2), without the diagonal pivots.
 */
func findAndBuildBKPivotUpper(AL, AR, WL, WR *cmat.FloatMatrix, k int) (int, int) {
    var r, q int
    var rcol, qrow, src, wk, wkp1, wrow cmat.FloatMatrix

    lc := n(AL) - 1
    wc := n(WL) - 1
    lr := m(AL) - 1

    // Copy AL[:,lc] to WL[:,wc] and update with WR[0:]
    src.SubMatrix(AL, 0, lc, m(AL), 1)
    wk.SubMatrix(WL, 0, wc, m(AL), 1)
    blasd.Copy(&wk, &src)
    if k > 0 {
        wrow.SubMatrix(WR, lr, 0, 1, n(WR))
        blasd.MVMult(&wk, AR, &wrow, -1.0, 1.0, gomas.NONE)
    }
    if m(AL) == 1 {
        return -1, 1
    }
    // amax is on-diagonal element of current column 
    amax := math.Abs(WL.Get(lr, wc))

    // find max off-diagonal on first column. 
    rcol.SubMatrix(WL, 0, wc, lr, 1)
    // r is row index and rmax is its absolute value
    r = blasd.IAmax(&rcol) 
    rmax := math.Abs(rcol.Get(r, 0))
    if amax >= bkALPHA*rmax {
        // no pivoting, 1x1 diagonal
        return -1, 1
    }

    // Now we need to copy row r to WL[:,wc-1] and update it
    wkp1.SubMatrix(WL, 0, wc-1, m(AL), 1)
    if r > 0 {
        // above the diagonal part of AL
        qrow.SubMatrix(AL, 0, r, r, 1)
        blasd.Copy(&wkp1, &qrow)
    }
    var wkr cmat.FloatMatrix
    qrow.SubMatrix(AL, r, r, 1, m(AL)-r)
    wkr.SubMatrix(&wkp1, r, 0, m(AL)-r, 1)
    blasd.Copy(&wkr, &qrow)
    if k > 0 {
        // update wkp1
        wrow.SubMatrix(WR, r, 0, 1, n(WR))
        blasd.MVMult(&wkp1, AR, &wrow, -1.0, 1.0, gomas.NONE)
    }
    // set on-diagonal entry to zero to avoid hitting it.
    p1 := wkp1.Get(r, 0)
    wkp1.Set(r, 0, 0.0)

    // max off-diagonal on r'th column/row at index q
    q = blasd.IAmax(&wkp1)
    qmax := math.Abs(wkp1.Get(q, 0))
    wkp1.Set(r, 0, p1)
        
    if amax >= bkALPHA*rmax*(rmax/qmax) {
        // no pivoting, 1x1 diagonal
        return -1, 1
    }
    // if q == r then qmax is not off-diagonal, qmax == WR[r,1] and
    // we get 1x1 pivot as following is always true
    if math.Abs(WL.Get(r, wc-1)) >= bkALPHA*qmax {
        // 1x1 pivoting and interchange with k, r
        // pivot row in column WL[:,-2] to WL[:,-1]
        src.SubMatrix(WL,  0, wc-1, m(AL), 1)
        wkp1.SubMatrix(WL, 0, wc, m(AL), 1)
        blasd.Copy(&wkp1, &src)
        wkp1.Set(-1, 0, src.Get(r, 0))
        wkp1.Set( r, 0, src.Get(-1, 0))
        return r, 1 
    } else {
        // 2x2 pivoting and interchange with k+1, r
        return r, 2
    }
    return -1, 1
}


func unblkBoundedBKUpper(A, wrk *cmat.FloatMatrix, p *Pivots, ncol int, conf *gomas.Config) (*gomas.Error, int) {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a11, a12, A22, a11inv cmat.FloatMatrix
    var w00, w01, w11 cmat.FloatMatrix
    var cwrk cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots

    err = nil
    nc := 0
    if ncol > n(A) {
        ncol = n(A)
    }

    // permanent working space for symmetric inverse of a11
    a11inv.SubMatrix(wrk, m(wrk)-2, 0, 2, 2)
    a11inv.Set(0, 1, -1.0)
    a11inv.Set(1, 0, -1.0)

    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,    A, 0, 0, util.PBOTTOMRIGHT)
    partitionPivot2x1(
        &pT,
        &pB, *p, 0, util.PBOTTOM)

    for n(&ATL) > 0 && nc < ncol {

        util.Partition2x2(
            &w00, &w01,
            nil,  &w11, wrk, nc, nc, util.PBOTTOMRIGHT)

        r, np := findAndBuildBKPivotUpper(&ATL, &ATR, &w00, &w01, nc)
        if np > ncol - nc {
            // next pivot does not fit into ncol columns,
            // return with number of factorized columns
            return err, nc
        }
        cwrk.SubMatrix(&w00, 0, n(&w00)-np, m(&ATL), np)
        if r != -1  {
            // pivoting needed; do swaping here
            k := m(&ATL) - np
            applyBKPivotSymUpper(&ATL, k, r)
            // swap right hand rows to get correct updates
            swapRows(&ATR, k, r)
            swapRows(&w01, k, r)
            if np == 2 && r != k {
                /* for 2x2 blocks we need diagonal pivots.
                 *          [r, r] | [ r,-1]
                 * a11 ==   ----------------  2-by-2 pivot, swapping [1,0] and [r,0]
                 *          [-1,r] | [-1,-1]
                 */
                t0 := w00.Get(k, -1)
                tr := w00.Get(r, -1)
                w00.Set(k, -1, tr) 
                w00.Set(r, -1, t0)
                t0 = w00.Get(k, -2)
                tr = w00.Get(r, -2)
                w00.Set(k, -2, tr)
                w00.Set(r, -2, t0)
            }
        }
        // repartition according the pivot size
        util.Repartition2x2to3x3(&ATL, 
            &A00, &a01, &A02,
            nil,  &a11, &a12,
            nil,  nil,  &A22,   /**/ A, np, util.PTOPLEFT)
        repartPivot2x1to3x1(&pT, 
            &p0,
            &p1,
            &p2,   /**/ *p, np, util.PTOP)
        // ------------------------------------------------------------

        wlc := n(&w00) - np
        cwrk.SubMatrix(&w00, 0, wlc, m(&a01), n(&a01))
        if np == 1 {
            // 
            a11.Set(0, 0, w00.Get(m(&a01), wlc))
            // a21 = a21/a11
            blasd.Copy(&a01, &cwrk)
            blasd.InvScale(&a01, a11.Get(0, 0))
            // store pivot point relative to original matrix
            if r == -1 {
                p1[0] = m(&ATL)
            } else {
                p1[0] = r + 1
            }
        } else if np == 2 {
            /*          a | b                       d/b | -1
             *  w00 == ------  == a11 --> a11.-1 == -------- * scale
             *          . | d                        -1 | a/b
             */
            a := w00.Get(m(&ATL)-2, -2)
            b := w00.Get(m(&ATL)-2, -1)
            d := w00.Get(m(&ATL)-1, -1)
            a11inv.Set(0, 0,  d/b)
            a11inv.Set(1, 1,  a/b)
            // denominator: (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
            scale := 1.0 / ((a/b)*(d/b) - 1.0)
            scale /= b
            
            // a01 = a01*a11.-1
            blasd.Mult(&a01, &cwrk, &a11inv, scale, 0.0, gomas.NONE, conf)
            a11.Set(0, 0, a)
            a11.Set(0, 1, b)
            a11.Set(1, 1, d)

            // store pivot point relative to original matrix
            p1[0] = -(r + 1)
            p1[1] = p1[0]
        }
        // ------------------------------------------------------------
        nc += np
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &a11, &A22,   A, util.PTOPLEFT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    *p, util.PTOP)

    }
    return err, nc
}

/*
 */
func blkDecompBKUpper(A, W *cmat.FloatMatrix, p *Pivots, conf *gomas.Config) (err *gomas.Error) {
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, A01, A02, A11, A12, A22 cmat.FloatMatrix
    var wrk cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots
    var nblk int = 0

    err = nil
    util.Partition2x2(
        &ATL, &ATR,
        &ABL, &ABR,   A, 0, 0, util.PBOTTOMRIGHT)
    partitionPivot2x1(
        &pT,
        &pB, *p, 0, util.PBOTTOM)

    nb := conf.LB
    for n(&ATL) >= nb {
        err, nblk = unblkBoundedBKUpper(&ATL, W, &pT, nb, conf)
        // repartition nblk size
        util.Repartition2x2to3x3(&ATL,
            &A00, &A01, &A02,
            nil,  &A11, &A12,
            nil,  nil,  &A22,   A, nblk, util.PTOPLEFT)
        repartPivot2x1to3x1(&pT, 
            &p0, &p1, &p2,   /**/ *p, nblk, util.PTOP)

        // --------------------------------------------------------
        // here [A01;A11] has been decomposed by unblkBoundedBKUpper()
        // Now we need update A00

        // wrk is original A01
        wrk.SubMatrix(W, 0, n(W)-nblk, m(&A01), nblk)

        // A00 = A00 - L01*D1*L01.T = A22 - A01*W.T
        blasd.UpdateTrm(&A00, &A01, &wrk, -1.0, 1.0, gomas.UPPER|gomas.TRANSB)

        // partially undo row pivots right of diagonal
        for k := 0; k < nblk; k++ {
            var s, d cmat.FloatMatrix
            r := p1[k] 
            colno := n(&A00) + k
            np := 1
            if r < 0 {
                r = -r
                np = 2
            }
            rlen := n(&ATL) - colno - np
            if r == colno + 1 && p1[k] > 0 {
                // no pivot
                continue
            }
            s.SubMatrix(&ATL, colno, colno+np, 1, rlen)
            d.SubMatrix(&ATL, r-1,   colno+np, 1, rlen)
            blasd.Swap(&d, &s)

            if p1[k] < 0 {
                k++ // skip other entry in 2x2 pivots
            }
        }
        // ---------------------------------------------------------
        util.Continue3x3to2x2(
            &ATL, &ATR,
            &ABL, &ABR,   &A00, &A11, &A22,  A, util.PTOPLEFT)
        contPivot3x1to2x1(
            &pT,
            &pB,    p0, p1,    *p, util.PTOP)

    }

    // do the last part with unblocked code
    if n(&ATL) > 0 {
        unblkDecompBKUpper(&ATL, W, pT, conf)
    }
    return
}


/*
 * Unblocked solve A*X = B for Bunch-Kauffman decomposed symmetric real matrix.
 */
func unblkSolveBKUpper(B, A *cmat.FloatMatrix, p Pivots, phase int, conf *gomas.Config) *gomas.Error {
    var err *gomas.Error = nil
    var ATL, ATR, ABL, ABR cmat.FloatMatrix
    var A00, a01, A02, a11, a12t, A22 cmat.FloatMatrix
    var Aref *cmat.FloatMatrix
    var BT, BB, B0, b1, B2, Bx cmat.FloatMatrix
    var pT, pB, p0, p1, p2 Pivots
    var aStart, aDir, bStart, bDir util.Direction
    var nc int

    np := 0

    if phase == 2 {
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
            &A00, &a01, &A02,
            nil,  &a11, &a12t,
            nil,  nil,  &A22,   /**/ A, np, aDir)
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
                    swapRows(&BT, m(&BT)-1, p1[0]-1)
                }
                // B2 = B2 - a01*b1
                blasd.MVUpdate(&B2, &a01, &b1, -1.0)
                // b1 = b1/d1 
                blasd.InvScale(&b1, a11.Get(0, 0))
                nc -= 1
            } else if np == 2 {
                if p1[0] != -nc {
                    // swap rows on bottom part of B
                    swapRows(&BT, m(&BT)-2, -p1[0]-1)
                }
                b := a11.Get(0, 1)
                apb := a11.Get(0, 0) / b
                dpb := a11.Get(1, 1) / b
                // (a/b)*(d/b)-1.0 == (a*d - b^2)/b^2
                scale := apb*dpb - 1.0
                scale *= b

                // B2 = B2 - a01*b1
                blasd.Mult(&B2, &a01, &b1, -1.0, 1.0, gomas.NONE, conf)
                // b1 = a11.-1*b1.T
                //(2x2 block, no subroutine for doing this in-place)
                for k := 0; k < n(&b1); k++ {
                    s0 := b1.Get(0, k) 
                    s1 := b1.Get(1, k) 
                    b1.Set(0, k, (dpb*s0-s1)/scale)
                    b1.Set(1, k, (apb*s1-s0)/scale)
                }
                nc -= 2
            }
        case 2:
            if np == 1 {
                blasd.MVMult(&b1, &B2, &a01, -1.0, 1.0, gomas.TRANS)
                if p1[0] != nc {
                    // swap rows on bottom part of B
                    util.Merge2x1(&Bx, &B0, &b1)
                    swapRows(&Bx, m(&Bx)-1, p1[0]-1)
                }
                nc += 1
            } else if np == 2 {
                blasd.Mult(&b1, &a01, &B2, -1.0, 1.0, gomas.TRANSA, conf)
                if p1[0] != -nc {
                    // swap rows on bottom part of B
                    util.Merge2x1(&Bx, &B0, &b1)
                    swapRows(&Bx, m(&Bx)-2, -p1[0]-1)
                }
                nc += 2
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

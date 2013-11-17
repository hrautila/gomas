
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package lapackd

import (
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas/blasd"
	"github.com/hrautila/gomas/util"
	"github.com/hrautila/gomas"
    //"math"
)


type PPivots struct {
    Indexes []int
}

type Pivots []int

const (
    FORWARD  = 1
    BACKWARD = 2
)

func NewPPivot(N int) *PPivots {
    ipiv := make([]int, N, N)
    return &PPivots{ipiv}
}

func NewPivots(N int) Pivots {
    return Pivots(make([]int, N, N))
}

func (p Pivots) Shift(offset int) {
    for k, _ := range p {
        p[k] += offset
    }
}

/*
 * Partition p to 2 by 1 blocks.
 *
 *        pT
 *  p --> --
 *        pB
 *
 * Parameter nb is initial block size for pT (pTOP) or pB (pBOTTOM).  
 */
func partitionPivot2x1(pT, pB *Pivots, p Pivots, nb int, pdir util.Direction) {
    switch (pdir) {
    case util.PTOP:
        if nb == 0 {
            *pT = nil
        } else {
            *pT= p[:nb]
        }
        *pB = p[nb:]
    case util.PBOTTOM:
        if nb > 0 {
            *pT = p[:-nb]
            *pB = p[len(p)-nb:]
        } else {
            *pT = p
            *pB = nil
        }
    }
}

/*
 * Repartition 2 by 1 block to 3 by 1 block.
 * 
 *           pT      p0            pT       p0
 * pBOTTOM: --  --> --   ; pTOP:   --  -->  p1
 *           pB      p1            pB       --
 *                   p2                     p2
 *
 */
func repartPivot2x1to3x1(pT, p0, p1, p2 *Pivots, p Pivots, nb int, pdir util.Direction) {
    nT := len(*pT)
    switch (pdir) {
    case util.PBOTTOM:
        if nT + nb > len(p) {
            nb = len(p) - nT
        }
        *p0 = *pT
        *p1 = p[nT:nT+nb]
        *p2 = p[nT+nb:]
    case util.PTOP:
        if nb > nT {
            nb = nT
        }
        *p0 = p[:nT-nb]
        *p1 = p[nT-nb:nT]
        *p2 = p[nT:]
    }
}

/*
 * Continue with 2 by 1 block from 3 by 1 block.
 * 
 *           pT      p0            pT       p0
 * pBOTTOM: --  <--  p1   ; pTOP:   -- <--  --
 *           pB      --            pB       p1
 *                   p2                     p2
 *
 */
func contPivot3x1to2x1(pT, pB *Pivots, p0, p1, p Pivots, pdir util.Direction) {
    var n0, n1 int
    n0 = len(p0)
    n1 = len(p1)
    switch (pdir) {
    case util.PBOTTOM:
        *pT = p[:n0+n1]
        *pB = p[n0+n1:]
    case util.PTOP:
        *pT = p[:n0]
        *pB = p[n0:]
    }
}


func swapRows(A *cmat.FloatMatrix, src, dst int) {
    var r0, r1 cmat.FloatMatrix
    ar, ac := A.Size()
    if src == dst || ar == 0 {
        return
    }
    r0.SubMatrix(A, src, 0, 1, ac)
    r1.SubMatrix(A, dst, 0, 1, ac)
    blasd.Swap(&r0, &r1)
}

func swapCols(A *cmat.FloatMatrix, src, dst int) {
    var c0, c1 cmat.FloatMatrix
    ar, _ := A.Size()
    if src == dst || ar == 0 {
        return
    }
    c0.SubMatrix(A, 0, src, ar, 1)
    c1.SubMatrix(A, 0, dst, ar, 1)
    blasd.Swap(&c0, &c1)
}

func scalePivots(p Pivots, offset int) {
    for k, n := range p {
        if n > 0 {
            p[k] += offset
        }
    }
}

func applyPivots(A *cmat.FloatMatrix, p Pivots) {
    for k, n := range p {
        if n > 0 && n-1 != k {
            swapRows(A, n-1, k)
        }
    }
}

func applyRowPivots(A *cmat.FloatMatrix, p Pivots, offset, dir int) {
    if dir == FORWARD {
        for k, n := range p {
            if n-1 != k {
                swapRows(A, n-1-offset, k)
            }
        }
    } else if dir == BACKWARD {
        // 
        for k := len(p)-1; k >= 0; k-- {
            if p[k]-1 != k {
                swapRows(A, p[k]-1-offset, k)
            }
        }
    }
}

func applyColPivots(A *cmat.FloatMatrix, p Pivots, offset, dir int) {
    if dir == FORWARD {
        for k, n := range p {
            if n > 0 {
                swapCols(A, n-1-offset, k)
            }
        }
    } else if dir == BACKWARD {
        // 
        for k := len(p)-1; k >= 0; k-- {
            if p[k] > 0 {
                swapCols(A, p[k]-1-offset, k)
            }
        }
    }
}

// Find largest absolute value on column
func pivotIndex(A *cmat.FloatMatrix) int {
    return blasd.IAmax(A) + 1
}

/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 * AR[0,0] is on diagonal and AL is block to the left of diagonal and AR the
 * triangular diagonal block. Need to swap row and column.
 *
 * LOWER triangular; moving from top-left to bottom-right
 *
 *    d
 *    x  d
 *    x  x  d  |
 *    --------------------------
 *    S1 S1 S1 | P1 x  x  x  P2     -- current row 
 *    x  x  x  | S2 d  x  x  x
 *    x  x  x  | S2 x  d  x  x
 *    x  x  x  | S2 x  x  d  x
 *    D1 D1 D1 | P2 D2 D2 D2 P3     -- swap with row 'index'
 *    x  x  x  | S3 x  x  x  D3 d
 *    x  x  x  | S3 x  x  x  D3 x d
 *       (ABL)          (ABR)
 *
 * UPPER triangular; moving from bottom-right to top-left
 *
 *         (ATL)             (ATR)
 *    d  x  x  D3 x  x  x | S3 x  x
 *       d  x  D3 x  x  x | S3 x  x
 *          d  D3 x  x  x | S3 x  x 
 *             P3 D2 D2 D2| P2 D1 D1  
 *                d  x  x | S2 x  x
 *                   d  x | S2 x  x
 *                      d | S2 x  x
 *    -----------------------------
 *                        | P1 S1 S1
 *                        |    d  x
 *                        |       d
 *                           (ABR)
 */
func applyPivotSym(AL, AR *cmat.FloatMatrix, index int, flags int) {
    var s, d cmat.FloatMatrix
    lr, lc := AL.Size()
    rr, rc := AR.Size()

    if flags & gomas.LOWER != 0 {
        // AL is [ABL]; AR is [ABR]; P1 is AR[0,0], P2 is AR[index, 0]
        // S1 -- D1
        s.SubMatrix(AL, 0,     0, 1, lc)
        d.SubMatrix(AL, index, 0, 1, lc)
        blasd.Swap(&s, &d)
        // S2 -- D2
        s.SubMatrix(AR, 1,     0, index-1, 1)
        d.SubMatrix(AR, index, 1, 1, index-1)
        blasd.Swap(&s, &d)
        // S3 -- D3
        s.SubMatrix(AR, index+1, 0,     rr-index-1, 1)
        d.SubMatrix(AR, index+1, index, rr-index-1, 1)
        blasd.Swap(&s, &d)
        // swap P1 and P3
        p1 := AR.Get(0, 0)
        p3 := AR.Get(index, index)
        AR.Set(0, 0, p3)
        AR.Set(index, index, p1)
        return
    }
    if flags & gomas.UPPER != 0 {
        // AL is merged from [ATL, ATR], AR is [ABR]; P1 is AR[0, 0]; P2 is AL[index, -1]
        colno := lc - rc
        // S1 -- D1; S1 is on the first row of AR
        s.SubMatrix(AR, 0, 1, 1, rc-1)
        d.SubMatrix(AL, index, colno+1, 1, rc-1)
        blasd.Swap(&s, &d)
        // S2 -- D2
        s.SubMatrix(AL, index+1, colno, lr-index-2, 1)
        d.SubMatrix(AL, index,   index+1, 1, colno-index-1)
        blasd.Swap(&s, &d)
        // S3 -- D3
        s.SubMatrix(AL, 0, index, index, 1)
        d.SubMatrix(AL, 0, colno, index, 1)
        blasd.Swap(&s, &d)
        //fmt.Printf("3, AR=%v\n", AR)
        // swap P1 and P3
        p1 := AR.Get(0, 0)
        p3 := AL.Get(index, index)
        AR.Set(0, 0, p3)
        AL.Set(index, index, p1)
        return
    }
}


/*
 * Apply diagonal pivot (row and column swapped) to symmetric matrix blocks.
 * AR[0,0] is on diagonal and AL is block to the left of diagonal and AR the
 * triangular diagonal block. Need to swap row and column.
 *
 * LOWER triangular; moving from top-left to bottom-right
 *
 *    d
 *    x  d |
 *    --------------------------
 *    x  x | d 
 *    S1 S1| S1 P1 x  x  x  P2     -- current row/col 'srcix'
 *    x  x | x  S2 d  x  x  x
 *    x  x | x  S2 x  d  x  x
 *    x  x | x  S2 x  x  d  x
 *    D1 D1| D1 P2 D2 D2 D2 P3     -- swap with row/col 'dstix'
 *    x  x | x  S3 x  x  x  D3 d
 *    x  x | x  S3 x  x  x  D3 x d
 *    (ABL)          (ABR)
 *
 * UPPER triangular; moving from bottom-right to top-left
 *
 *         (ATL)                  (ATR)
 *    d  x  x  D3 x  x  x  S3 x | x
 *       d  x  D3 x  x  x  S3 x | x
 *          d  D3 x  x  x  S3 x | x 
 *             P3 D2 D2 D2 P2 D1| D1  -- dstinx
 *                d  x  x  S2 x | x
 *                   d  x  S2 x | x
 *                      d  S2 x | x
 *                         P1 S1| S1  -- srcinx
 *                            d | x
 *    -----------------------------
 *                              | d
 *                           (ABR)
 */
func applyPivotSym2(AL, AR *cmat.FloatMatrix, srcix, dstix int, flags int) {
    var s, d cmat.FloatMatrix
    _ , lc := AL.Size()
    rr, rc := AR.Size()
    if flags & gomas.LOWER != 0 {
        // AL is [ABL]; AR is [ABR]; P1 is AR[0,0], P2 is AR[index, 0]
        // S1 -- D1
        AL.SubMatrix(&s, srcix, 0, 1, lc)
        AL.SubMatrix(&d, dstix, 0, 1, lc)
        blasd.Swap(&s, &d)
        if srcix > 0 {
            AR.SubMatrix(&s, srcix, 0, 1, srcix)
            AR.SubMatrix(&d, dstix, 0, 1, srcix)
            blasd.Swap(&s, &d)
        }
        // S2 -- D2
        AR.SubMatrix(&s, srcix+1, srcix,   dstix-srcix-1, 1)
        AR.SubMatrix(&d, dstix,   srcix+1, 1, dstix-srcix-1)
        blasd.Swap(&s, &d)
        // S3 -- D3
        AR.SubMatrix(&s, dstix+1, srcix, rr-dstix-1, 1)
        AR.SubMatrix(&d, dstix+1, dstix, rr-dstix-1, 1)
        blasd.Swap(&s, &d)
        // swap P1 and P3
        p1 := AR.Get(srcix, srcix)
        p3 := AR.Get(dstix, dstix)
        AR.Set(srcix, srcix, p3)
        AR.Set(dstix, dstix, p1)
        return
    }
    if flags & gomas.UPPER != 0 {
        // AL is ATL, AR is ATR; P1 is AL[srcix, srcix];
        // S1 -- D1; 
        AR.SubMatrix(&s, srcix, 0, 1, rc)
        AR.SubMatrix(&d, dstix, 0, 1, rc)
        blasd.Swap(&s, &d)
        if srcix < lc-1 {
            // not the corner element
            AL.SubMatrix(&s, srcix, srcix+1, 1, srcix)
            AL.SubMatrix(&d, dstix, srcix+1, 1, srcix)
            blasd.Swap(&s, &d)
        }
        // S2 -- D2
        AL.SubMatrix(&s, dstix+1, srcix, srcix-dstix-1, 1)
        AL.SubMatrix(&d, dstix,   dstix+1, 1, srcix-dstix-1)
        blasd.Swap(&s, &d)
        // S3 -- D3
        AL.SubMatrix(&s, 0, srcix, dstix, 1)
        AL.SubMatrix(&d, 0, dstix, dstix, 1)
        blasd.Swap(&s, &d)
        //fmt.Printf("3, AR=%v\n", AR)
        // swap P1 and P3
        p1 := AR.Get(0, 0)
        p3 := AL.Get(dstix, dstix)
        AR.Set(srcix, srcix, p3)
        AL.Set(dstix, dstix, p1)
        return
    }
}


func ApplyRowPivots(A *cmat.FloatMatrix, ipiv []int, direction int) {
    applyRowPivots(A, ipiv, 0, direction)
}

func NumPivots(ipiv []int) int {
    count := 0
    for _, n := range ipiv {
        if n != 0 {
            count += 1
        }
    }
    return count
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

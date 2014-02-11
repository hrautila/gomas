
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package util

import (
    "github.com/hrautila/cmat"
)

// Functions here are support functions for libFLAME-like implementation
// of various linear algebra algorithms.

type Direction int
const (
    PLEFT = iota
    PRIGHT 
    PTOP
    PBOTTOM
    PTOPLEFT
    PBOTTOMRIGHT
)

/*
 * Partition p to 2 by 1 blocks.
 *
 *        AT
 *  A --> --
 *        AB
 *
 * Parameter nb is initial block size for AT (pTOP) or AB (pBOTTOM).  
 */
func Partition2x1(AT, AB, A *cmat.FloatMatrix, nb int, side Direction) {
    ar, ac := A.Size()
    if nb > ar {
        nb = ar
    }
    switch (side) {
    case PTOP:
        AT.SubMatrix(A, 0, 0, nb, ac)
        AB.SubMatrix(A, nb, 0, ar-nb, ac)
    case PBOTTOM:
        AT.SubMatrix(A, 0, 0, ar-nb, ac)
        AB.SubMatrix(A, ar-nb, 0, nb, ac)
    }
}

/*
 * Repartition 2 by 1 block to 3 by 1 block.
 *
 *           AT      A0            AT       A0
 * pBOTTOM: --  --> --   ; pTOP:   --  -->  A1
 *           AB      A1            AB       --
 *                   A2                     A2
 *
 */
func Repartition2x1to3x1(AT, A0, A1, A2, A *cmat.FloatMatrix, nb int, pdir Direction) {
    nT, _ := AT.Size()
    ar, ac := A.Size()
    switch (pdir) {
    case PBOTTOM:
        if nT + nb > ar {
            nb = ar - nT
        }
        A0.SubMatrix(A, 0,     0, nT, ac)
        A1.SubMatrix(A, nT,    0, nb, ac)
        A2.SubMatrix(A, nT+nb, 0, ar-nT-nb, ac)
    case PTOP:
        if nT < nb {
            nb = nT
        }
        A0.SubMatrix(A, 0,     0, nT-nb, ac)
        A1.SubMatrix(A, nT-nb, 0, nb,    ac)
        A2.SubMatrix(A, nT,    0, ar-nT, ac)
    }
}

/*
 * Continue with 2 by 1 block from 3 by 1 block.
 * 
 *           AT      A0            AT       A0
 * pBOTTOM: --  <--  A1   ; pTOP:   -- <--  --
 *           AB      --            AB       A1
 *                   A2                     A2
 */
func Continue3x1to2x1(AT, AB, A0, A1, A *cmat.FloatMatrix, pdir Direction) {
    ar, ac := A.Size()
    n0, _ := A0.Size()
    n1, _ := A1.Size()
    switch (pdir) {
    case PBOTTOM:
        AT.SubMatrix(A, 0,     0, n0+n1, ac)
        AB.SubMatrix(A, n0+n1, 0, ar-n0-n1, ac)
    case PTOP:
        AT.SubMatrix(A, 0,  0, n0, ac)
        AB.SubMatrix(A, n0, 0, ar-n0, ac)
    }
}

/*
 * Merge 1 by 1 block from 2 by 1 block.
 * 
 *          AT  
 * Abkl <-- --  
 *          AB  
 *
 */
func Merge2x1(ABLK, AT, AB *cmat.FloatMatrix) {
    tr, tc := AT.Size()
    br, _  := AB.Size()
    ABLK.SubMatrix(AT, 0, 0, tr+br, tc)
}

/*
 * Merge 1 by 1 block from 1 by 2 block. 
 * 
 * ABLK <--  AL | AR  
 *
 */
func Merge1x2(ABLK, AL, AR *cmat.FloatMatrix) {
    lr, lc := AL.Size()
    _ , rc := AR.Size()
    ABLK.SubMatrix(AL, 0, 0, lr, lc+rc)
}

/*
 * Partition A to 1 by 2 blocks.
 *
 *  A -->  AL | AR
 *
 * Parameter nb is initial block size for AL (pLEFT) or AR (pRIGHT).  
 */
func Partition1x2(AL, AR, A *cmat.FloatMatrix, nb int, side Direction) {
    ar, ac := A.Size()
    if nb > ac {
        nb = ac
    }
    switch (side) {
    case PLEFT:
        AL.SubMatrix(A, 0, 0,  ar, nb)
        AR.SubMatrix(A, 0, nb, ar, ac-nb)
    case PRIGHT:
        AL.SubMatrix(A, 0, 0,     ar, ac-nb)
        AR.SubMatrix(A, 0, ac-nb, ar, nb)
    }
}



/*
 * Repartition 1 by 2 blocks to 1 by 3 blocks.
 *
 * pRIGHT: AL | AR  -->  A0 | A1 A2 
 * pLEFT:  AL | AR  -->  A0 A1 | A2 
 *
 * Parameter As is left or right block of original 1x2 block.
 */
func Repartition1x2to1x3(AL, A0, A1, A2, A *cmat.FloatMatrix, nb int, pdir Direction) {
    ar, ac := A.Size()
    _, k := AL.Size()
    switch (pdir) {
    case PRIGHT:
        if k + nb > ac {
            nb = ac - k
        }
        // A0 is AL; [A1; A2] is AR
        A0.SubMatrix(A, 0, 0,    ar, k)
        A1.SubMatrix(A, 0, k,    ar, nb)
        A2.SubMatrix(A, 0, k+nb, ar, ac-nb-k)
    case PLEFT:
        if nb > k {
            nb = k
        }
        // A2 is AR; [A0; A1] is AL
        A0.SubMatrix(A, 0, 0,    ar, k-nb)
        A1.SubMatrix(A, 0, k-nb, ar, nb)
        A2.SubMatrix(A, 0, k,    ar, ac-k)
    }
}

/*
 * Repartition 1 by 2 blocks to 1 by 3 blocks.
 *
 * pRIGHT: AL | AR  --  A0 A1 | A2 
 * pLEFT:  AL | AR  <--  A0 | A1 A2 
 *
 */
func Continue1x3to1x2(AL, AR, A0, A1, A *cmat.FloatMatrix, pdir Direction) {
    ar, ac := A.Size()
    _ , k  := A0.Size()
    _ , nb := A1.Size()
    switch (pdir) {
    case PRIGHT:
        // AL is [A0; A1], AR is A2
        AL.SubMatrix(A, 0, 0,  ar, k+nb)
        _ , lc := AL.Size()
        AR.SubMatrix(A, 0, lc, ar, ac-lc)
    case PLEFT:
        // AL is A0; AR is [A1; A2]
        AL.SubMatrix(A, 0, 0, ar, k)
        AR.SubMatrix(A, 0, k, ar, ac-k)
    }
}

/*
 * Partition A to 2 by 2 blocks.
 *
 *           ATL | ATR
 *  A  -->   =========
 *           ABL | ABR
 *
 * Parameter nb is initial block size for ATL in column direction and mb in row direction.
 * ATR and ABL may be nil pointers.
 */
func Partition2x2(ATL, ATR, ABL, ABR, A *cmat.FloatMatrix, mb, nb int, side Direction) {
    ar, ac := A.Size()
    switch (side) {
    case PTOPLEFT:
        ATL.SubMatrix(A, 0, 0,  mb, nb)
        if ATR != nil {
            ATR.SubMatrix(A, 0, nb, mb, ac-nb)
        }
        if ABL != nil {
            ABL.SubMatrix(A, mb, 0, ar-mb, nb)
        }
        ABR.SubMatrix(A, mb, nb)
    case PBOTTOMRIGHT:
        ATL.SubMatrix(A, 0, 0,  ar-mb, ac-nb)
        if ATR != nil {
            ATR.SubMatrix(A, 0, ac-nb, ar-mb, nb)
        }
        if ABL != nil {
            ABL.SubMatrix(A, ar-mb, 0, mb, nb)
        }
        ABR.SubMatrix(A, ar-mb, ac-nb)
    }
}

/*
 * Repartition 2 by 2 blocks to 3 by 3 blocks.
 *
 *                      A00 | A01 : A02
 *   ATL | ATR   nb     ===============
 *   =========   -->    A10 | A11 : A12
 *   ABL | ABR          ---------------
 *                      A20 | A21 : A22
 *
 * ATR, ABL, ABR implicitely defined by ATL and A.
 * It is valid to have either the strictly upper or lower submatrices as nil values.
 * 
 */
func Repartition2x2to3x3(ATL, 
    A00, A01, A02, A10, A11, A12, A20, A21, A22, A *cmat.FloatMatrix, nb int, pdir Direction) {

    ar, ac := A.Size()
    kr, kc := ATL.Size()
    switch (pdir) {
    case PBOTTOMRIGHT:
        if kc + nb > ac {
            nb = ac - kc
        }
        if kr + nb > ar {
            nb = ar - kr
        }
        A00.SubMatrix(A, 0, 0,    kr, kc)
        if A01 != nil {
            A01.SubMatrix(A, 0, kc,    kr, nb)
        }
        if A02 != nil {
            A02.SubMatrix(A, 0, kc+nb, kr, ac-kc-nb)
        }

        if A10 != nil {
            A10.SubMatrix(A, kr, 0,    nb, kc)
        }
        A11.SubMatrix(A, kr, kc,    nb, nb)
        if A12 != nil {
            A12.SubMatrix(A, kr, kc+nb, nb, ac-kc-nb)
        }

        if A20 != nil {
            A20.SubMatrix(A, kr+nb, 0,    ar-kr-nb, kc)
        }
        if A21 != nil {
            A21.SubMatrix(A, kr+nb, kc,    ar-kr-nb, nb)
        }
        A22.SubMatrix(A, kr+nb, kc+nb)
    case PTOPLEFT:
        if nb > kc {
            nb = kc
        }
        if nb > kr {
            nb = kr
        }
        // move towards top left corner
        A00.SubMatrix(A, 0, 0,    kr-nb, kc-nb)
        if A01 != nil {
            A01.SubMatrix(A, 0, kc-nb, kr-nb, nb)
        }
        if A02 != nil {
            A02.SubMatrix(A, 0, kc, kr-nb, ac-kc)
        }

        if A10 != nil {
            A10.SubMatrix(A, kr-nb, 0, nb, kc-nb)
        }
        A11.SubMatrix(A, kr-nb, kc-nb,  nb, nb)
        if A12 != nil {
            A12.SubMatrix(A, kr-nb, kc, nb, ac-kc)
        }

        if A20 != nil {
            A20.SubMatrix(A, kr, 0,    ar-kr, kc-nb)
        }
        if A21 != nil {
            A21.SubMatrix(A, kr, kc-nb, ar-kr, nb)
        }
        A22.SubMatrix(A, kr, kc)
    }
}


/*
 * Redefine 2 by 2 blocks from 3 by 3 partition.
 *
 *                      A00 : A01 | A02
 *   ATL | ATR   nb     ---------------
 *   =========   <--    A10 : A11 | A12
 *   ABL | ABR          ===============
 *                      A20 : A21 | A22
 *
 * New division of ATL, ATR, ABL, ABR defined by diagonal entries A00, A11, A22
 */
func Continue3x3to2x2(
    ATL, ATR, ABL, ABR, 
    A00, A11, A22, A *cmat.FloatMatrix, pdir Direction) {

    ar, ac := A.Size()
    kr, kc := A00.Size()
    _, mb := A11.Size()
    switch (pdir) {
    case PBOTTOMRIGHT:
        ATL.SubMatrix(A, 0, 0,    kr+mb, kc+mb)
        if ATR != nil {
            ATR.SubMatrix(A, 0, kc+mb, kr+mb, ac-kc-mb)
        }
        if ABL != nil {
            ABL.SubMatrix(A, kr+mb, 0, ar-kr-mb, kc+mb)
        }
        ABR.SubMatrix(A, kr+mb, kc+mb)
    case PTOPLEFT:
        ATL.SubMatrix(A, 0, 0,  kr, kc)
        if ATR != nil {
            ATR.SubMatrix(A, 0, kc, kr, ac-kc)
        }
        if ABL != nil {
            ABL.SubMatrix(A, kr, 0, ar-kr, ac-kc)
        }
        ABR.SubMatrix(A, kr, kc)
    }
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

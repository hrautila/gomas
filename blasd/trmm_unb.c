
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b0
 *     0 |a11|a12  b1
 *     0 | 0 |a22  b2
 *
 *    b00 = a00*b0 + a01*b1 + a02*b2
 *    b10 =          a11*b1 + a12*b2
 *    b20 =                   a22*b2
 *
 *    --> work it forwards as b00, b01 not need for later elements; AXPY
 */
static void
__trmm_unb_upper(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                      int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;

  for (i = 0; i < nRE; i++) {
    // update all previous B-values with current A column and current B
    for (j = 0; j < nC; j++) {
      __vmult1axpy(&Bc[j*ldB], 1, &Ac[i*ldA], &Bc[i+j*ldB], 1, alpha, i);
      Bc[i+j*ldB] = unit ? Bc[i+j*ldB] : alpha*Bc[i+j*ldB]*Ac[i+i*ldA];
    }
  }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0  
 *  b1 =   0 |a11|a12  b'1  
 *  b2     0 | 0 |a22  b'2  
 *
 *  b0 = a00*b'0
 *  b1 = a01*b'0 + a11*b'1
 *  b2 = a02*b'0 + a12*b'1 + a22*b'2
 *
 *  --> work it backwards with DOT products 
 */
static void
__trmm_unb_u_trans(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                        int ldB, int ldA, int nRE, int nC)
{
  register int i, j;
  DTYPE xtmp;

  for (i = nRE; i > 0; i--) {
    for (j = 0; j < nC; j++) {
      xtmp = unit ? alpha*Bc[(i-1)+j*ldB] : 0.0;
      __vmult1dot(&xtmp, 1, &Ac[(i-1)*ldA], &Bc[j*ldB], 1, alpha, i-unit);
      Bc[(i-1)+j*ldB] = xtmp;
    }
  }
}

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = a00*b'0
 *  b1 = a10*b'0 + a11*b'1
 *  b2 = a20*b'0 + a21*b'1 + a22*b'2
 *
 *  --> work it backwards as b20 is not needed for b10, ...
 */
static void
__trmm_unb_lower(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                 int ldB, int ldA, int nRE, int nC)
{
  register int i, j;

  for (i = nRE; i > 0; i--) {
    for (j = 0; j < nC; j++) {
      // update all b-values below with the current A column and current B
      __vmult1axpy(&Bc[i+j*ldB], 1, &Ac[i+(i-1)*ldA], &Bc[(i-1)+j*ldB], 1, alpha, nRE-i);
      Bc[(i-1)+j*ldB] = alpha * (unit ? Bc[(i-1)+j*ldB] : Bc[(i-1)+j*ldB]*Ac[(i-1)+(i-1)*ldA]);
    }
  }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = a00*b'0 + a10*b'1 + a20*b'2
 *  b1 =           a11*b'1 + a21*b'2
 *  b2 =                     a22*b'2
 *
 *  --> work it forwards as b0 is not needed for b1, ...
 */
static void
__trmm_unb_l_trans(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                        int ldB, int ldA, int nRE, int nC)
{
  register int i, j;
  DTYPE xtmp;

  for (i = 0; i < nRE; i++) {
    for (j = 0; j < nC; j++) {
      xtmp = unit ? alpha*Bc[i+j*ldB] : 0.0;
      __vmult1dot(&xtmp, 1, &Ac[(i+unit)+i*ldA], &Bc[(i+unit)+j*ldB], 1, alpha, nRE-unit-i);
      Bc[i+j*ldB] = xtmp;
    }
  }
}


/*
 *  RIGHT-UPPER
 *  
 *                          a00|a01|a02
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12
 *                           0 | 0 |a22
 *
 *    b0 = b'0*a00
 *    b1 = b'0*a01 + a11*b'1
 *    b2 = b'0*a02 + a12*b'1 + a22*b'2
 *    
 *    --> work it backwards as b12 & b02 are not needed for b11, b01, ...
 */
static void
__trmm_unb_r_upper(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                        int ldB, int ldA, int nRE, int nC)
{
  // Y is 
  register int i, j;
  DTYPE btmp;

  for (j = nC; j > 0; j--) {
    for (i = 0; i < nRE; i++) {
      btmp = unit ? alpha*Bc[i+(j-1)*ldB] : 0.0;
      // calculate dot-product following Ar column and Br row
      __vmult1dot(&btmp, 1, &Ac[(j-1)*ldA], &Bc[i], ldB, alpha, j-unit);
      Bc[i+(j-1)*ldB] = btmp;
    }
  }
}

/*
 * LOWER, RIGHT,
 *  
 *                          a00| 0 | 0
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0
 *                          a20|a21|a22
 *
 *    b0 = b'0*a00 + b'1*a10 + b'2*a20
 *    b1 = b'1*a11 + b'2*a21
 *    b2 = b'2*a22
 *    
 *    --> work it forward as b00 are not needed for b01, b02, ... with DOT
 */
static void
__trmm_unb_r_lower(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                         int ldB, int ldA, int nRE, int nC)
{
  register int i, j;
  DTYPE btmp;

  for (j = 0; j < nC; j++) {
    for (i = 0; i < nRE; i++) {
      btmp = 0.0;
      // calculate dot-product following Ar column and Br row
      __vmult1dot(&btmp, 1, &Ac[(j+unit)+j*ldA], &Bc[i+(j+unit)*ldB], ldB, alpha, nC-j-unit);
      Bc[i+j*ldB] = unit ? btmp + alpha*Bc[i+j*ldB] : btmp;
    }
  }
}

/*
 *  RIGHT-UPPER-TRANS
 *  
 *                          a00|a01|a02
 *  b0|b1|b2 = b'0|b'1|b'2   0 |a11|a12
 *                           0 | 0 |a22
 *
 *    b0 = b'0*a00 + b'1*a01 + b'2*a02
 *    b1 =           b'1*a11 + b'2*a12
 *    b2 =                     b'2*a22
 */
static void
__trmm_unb_ru_trans(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                           int ldB, int ldA, int nRE, int nC)
{
  register int i, j;

  for (j = 0; j < nC; j++) {
    for (i = 0; i < nRE; i++) {
      // update preceeding elemnts
      __vmult1axpy(&Bc[i], ldB, &Ac[j*ldA], &Bc[i+j*ldB], ldB, alpha, j);
      // update current element on B rows
      Bc[i+j*ldB] *= unit ? alpha : alpha *Ac[j+j*ldA];
    }
  }
}

/* LOWER, RIGHT, TRANSA
 *  
 *                          a00| 0 | 0
 *  b0|b1|b2 = b'0|b'1|b'2  a10|a11| 0
 *                          a20|a21|a22
 *
 *    b0 = b'0*a00
 *    b1 = b'0*a10 + b'1*a11
 *    b2 = b'0*a20 + b'1*a21 + b'2*a22
 *    
 */
static void
__trmm_unb_rl_trans(DTYPE *Bc, const DTYPE *Ac, DTYPE alpha, int unit,
                         int ldB, int ldA, int nRE, int nC)
{
  register int i, j;

  for (j = nC; j > 0; j--) {
    for (i = 0; i < nRE; i++) {
      // update following elements on B row
      __vmult1axpy(&Bc[i+j*ldB], ldB, &Ac[j+(j-1)*ldA], &Bc[i+(j-1)*ldB], ldB, alpha, nC-j);
      // update current element on B rows
      Bc[i+(j-1)*ldB] *= unit ? alpha : alpha*Ac[(j-1)+(j-1)*ldA];
    }
  }
}

// X = A*X; unblocked version
void __trmm_unb(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int N, int S, int E)
{
  // indicates if diagonal entry is unit (=1.0) or non-unit.
  int unit = flags & GOMAS_UNIT ? 1 : 0;
  DTYPE *Bc; 
  
  if (flags & GOMAS_RIGHT) {
    // for X = X*op(A)
    Bc = &B->md[S];  // row of B
    if (flags & GOMAS_UPPER) {
      if (flags & GOMAS_TRANSA) {
        // axpy_left_fwd
        __trmm_unb_ru_trans(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      } else {
        // dot_bleft_backward
        __trmm_unb_r_upper(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      }
    } else {
      if (flags & GOMAS_TRANSA) {
        // axpy_bleft_backwd
        __trmm_unb_rl_trans(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      } else {
        // dot_bleft_fwd
        __trmm_unb_r_lower(Bc, A->md, alpha, unit, B->step, A->step, E-S, N);
      }
    }
  } else {
    // for X = op(A)*X
    Bc = &B->md[S*B->step]; // column of B
    if (flags & GOMAS_UPPER) {
      if (flags & GOMAS_TRANSA) {
        // dot_backward
        __trmm_unb_u_trans(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      } else {
        // axpy_forward
        __trmm_unb_upper(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      }
    } else {
      if (flags & GOMAS_TRANSA) {
        // dot_forward
        __trmm_unb_l_trans(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      } else {
        // axpy_backward
        __trmm_unb_lower(Bc, A->md, alpha, unit, B->step, A->step, N, E-S);
      }
    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:

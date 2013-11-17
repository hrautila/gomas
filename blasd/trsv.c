
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"

/*
 *  LEFT-UPPER
 *
 *    a00|a01|a02  b'0
 *     0 |a11|a12  b'1
 *     0 | 0 |a22  b'2
 *
 *    b0 = (b'0 - a01*b1 - a02*b2)/a00
 *    b1 =          (b'1 - a12*b2)/a11
 *    b2 =                     b'2/a22
 */
static inline
void __trsv_unb_lu(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int nRE)
{
  register int i;
   for (i = nRE; i > 0; i--) {
     X[(i-1)*incX] = unit ? X[(i-1)*incX] : X[(i-1)*incX]/Ac[(i-1)+(i-1)*ldA];
    // update all previous b-values with current A column and current B
    __vmult1axpy(&X[0], incX, &Ac[(i-1)*ldA], &X[(i-1)*incX], 1, -1.0, i-1);
  }
}

/*
 *  LEFT-UPPER-TRANS
 *
 *  b0    a00|a01|a02  b'0  
 *  b1 =   0 |a11|a12  b'1  
 *  b2     0 | 0 |a22  b'2  
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a01*b0)/a11
 *  b2 = (b'2 - a02*b0 - a12*b1)/a22
 */
static inline
void __trsv_unb_lut(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int nRE)
{
  register int i;
  DTYPE xtmp;

  for (i = 0; i < nRE; i++) {
    xtmp = 0.0;
    __vmult1dot(&xtmp, 1, &Ac[i*ldA], &X[0], incX, 1.0, i);
    xtmp = X[i*incX] - xtmp;
    X[i*incX] = unit ? xtmp : xtmp/Ac[i+i*ldA];
  }
}

/*
 *  LEFT-LOWER
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = b'0/a00
 *  b1 = (b'1 - a10*b0)/a11
 *  b2 = (b'2 - a20*b0 - a21*b1)/a22
 */
static inline
void __trsv_unb_ll(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int nRE)
{
  register int i;

  for (i = 0; i < nRE; i++) {
    X[i*incX] = unit ? X[i*incX] : X[i*incX]/Ac[i+i*ldA];
    // update all X-values below with the current A column and current X
    __vmult1axpy(&X[(i+1)*incX], incX, &Ac[(i+1)+i*ldA], &X[i*incX], incX, -1.0, nRE-i-1);
  }
}

/*
 *  LEFT-LOWER-TRANS
 *
 *  b0    a00| 0 | 0   b'0  
 *  b1 =  a10|a11| 0   b'1  
 *  b2    a20|a21|a22  b'2  
 *
 *  b0 = (b'0 - a10*b1 - a20*b2)/a00
 *  b1 =          (b'1 - a21*b2)/a11
 *  b2 =                     b'2/a22
 */
static inline
void __trsv_unb_llt(DTYPE *X, const DTYPE *Ac, int unit, int incX, int ldA, int N)
{
  register int i;
  DTYPE xtmp;

  for (i = N; i > 0; i--) {
    xtmp = 0.0;
    __vmult1dot(&xtmp, 1, &Ac[i+(i-1)*ldA], &X[i*incX], incX, 1.0, N-i);
    xtmp = X[(i-1)*incX] - xtmp;
    X[(i-1)*incX] = unit ? xtmp : xtmp/Ac[(i-1)+(i-1)*ldA];
  }
}



static
void __trsv_unb(mvec_t *X, const mdata_t *A, int flags, int N)
{
  int unit = flags & GOMAS_UNIT ? 1 : 0;
  switch (flags & (GOMAS_TRANS|GOMAS_UPPER|GOMAS_LOWER)){
  case GOMAS_UPPER|GOMAS_TRANS:
    __trsv_unb_lut(X->md, A->md, unit, X->inc, A->step, N);
    break;
  case GOMAS_UPPER:
    __trsv_unb_lu(X->md, A->md, unit, X->inc, A->step, N);
    break;
  case GOMAS_LOWER|GOMAS_TRANS:
    __trsv_unb_llt(X->md, A->md, unit, X->inc, A->step, N);
    break;
  case GOMAS_LOWER:
  default:
    __trsv_unb_ll(X->md, A->md, unit, X->inc, A->step, N);
    break;
  }
}

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *                                                     
 *    A00 | A01    x0         A00 |  0     x0
 *   ----------- * --        ----------- * --
 *     0  | A11    x1         A10 | A11    x1
 *
 *  upper:
 *    x'0 = A00*x0           --> x0 = trsv(x'0, A00)
 *    x'1 = A01*x0 + A11*x1  --> x1 = trsv(x'1 - A01*x0)
 *  lower:
 *    x'0 = A00*x0           --> x0 = trsv(x'0, A00)
 *    x'1 = A10*x0 + A11*x1  --> x1 = trsv(x'1 - A10*x0, A11)
 *
 *   Forward substitution.
 */
static
void __trsv_forward_recursive(mvec_t *X, const mdata_t *A, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  if (N < MIN_MVEC_SIZE) {
    __trsv_unb(X, A, flags, N);
    return;
  }

  // top part
  __subvector(&x0, X, 0);
  __subblock(&a0, A, 0, 0);
  __trsv_forward_recursive(&x0, &a0, flags, N/2);

  // update bottom with top
  __subvector(&x1, X, N/2);
  if (flags & GOMAS_UPPER) {
    __subblock(&a1, A, 0, N/2);
  } else {
    __subblock(&a1, A, N/2, 0);
  }
  __gemv_recursive(&x1, &a1, &x0, -1.0, 1.0, flags, 0, N/2, 0, N-N/2);


  // bottom part
  __subblock(&a1, A, N/2, N/2);
  __trsv_forward_recursive(&x1, &a1, flags, N-N/2);
}

/*
 *   LEFT-UPPER               LEFT-LOWER-TRANS
 *                                                     
 *    A00 | A01    x0         A00 |  0     x0
 *   ----------- * --        ----------- * --
 *     0  | A11    x1         A10 | A11    x1
 *
 *  upper:
 *    x'0 = A00*x0 + A01*x1  --> x0 = trsv(x'0 - A01*x1, A00)
 *    x'1 = A11*x1           --> x1 = trsv(x'1, A11)
 *  lower:
 *    x'0 = A00*x0 + A10*x1  --> x0 = trsv(x'0 - A10*x1, A00)
 *    x'1 = A11*x1           --> x1 = trsv(x'1, A11)
 *
 *   Backward substitution.
 */
static
void __trsv_backward_recursive(mvec_t *X, const mdata_t *A, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  if (N < MIN_MVEC_SIZE) {
    __trsv_unb(X, A, flags, N);
    return;
  }

  // bottom part
  __subvector(&x1, X, N/2);
  __subblock(&a1, A, N/2, N/2);
  __trsv_backward_recursive(&x1, &a1, flags, N-N/2);

  // update top with bottom
  __subvector(&x0, X, 0);
  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __gemv_recursive(&x0, &a0, &x1, -1.0, 1.0, flags, 0, N-N/2, 0, N/2);


  // top part
  __subblock(&a0, A, 0, 0);
  __trsv_backward_recursive(&x0, &a0, flags, N/2);
}

void __trsv_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  if (N < MIN_MVEC_SIZE) {
    __trsv_unb(X, A, flags, N);
    if (alpha != 1.0) {
      __vscale(X->md, X->inc, N, alpha);
    }
    return;
  }

  switch (flags & (GOMAS_UPPER|GOMAS_LOWER|GOMAS_TRANS)) {
  case GOMAS_LOWER|GOMAS_TRANS:
  case GOMAS_UPPER:
    __trsv_backward_recursive(X, A, flags, N);
    break;

  case GOMAS_UPPER|GOMAS_TRANS:
  case GOMAS_LOWER:
  default:
    __trsv_forward_recursive(X, A, flags, N);
    break;
  }
  if (alpha != 1.0) {
    __vscale(X->md, X->inc, N, alpha);
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:

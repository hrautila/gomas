
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
 *    a00|a01|a02  b0
 *     0 |a11|a12  b1
 *     0 | 0 |a22  b2
 *
 *    b00 = a00*b0 + a01*b1 + a02*b2
 *    b10 =          a11*b1 + a12*b2
 *    b20 =                   a22*b2
 */
static inline
void __trmv_unb_lu(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                 int incX, int ldA, int nRE)
{
  register int i;
  for (i = 0; i < nRE; i++) {
    // update all previous b-values with current A column and current B
    __vmult1axpy(&X[0], incX, &Ac[i*ldA], &X[i*incX], 1, alpha, i);
    X[i*incX] = unit ? X[i*incX] : alpha*X[i*incX]*Ac[i+i*ldA];
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
 */
static inline
void __trmv_unb_lut(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                    int incX, int ldA, int nRE)
{
  register int i;
  DTYPE xtmp;

  for (i = nRE; i > 0; i--) {
    xtmp = unit ? alpha*X[(i-1)*incX] : 0.0;
    __vmult1dot(&xtmp, 1, &Ac[(i-1)*ldA], &X[0], incX, alpha, i-unit);
    X[(i-1)*incX] = xtmp;
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
 */
static inline
void __trmv_unb_ll(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                   int incX, int ldA, int nRE)
{
  register int i;

  for (i = nRE; i > 0; i--) {
    // update all b-values below with the current A column and current B
    __vmult1axpy(&X[i*incX], 1, &Ac[i+(i-1)*ldA], &X[(i-1)*incX], 1, alpha, nRE-i);
    X[(i-1)*incX] = alpha * (unit ? X[(i-1)*incX] : X[(i-1)*incX]*Ac[(i-1)+(i-1)*ldA]);
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
 */
static inline
void __trmv_unb_llt(DTYPE *X, const DTYPE *Ac, const DTYPE alpha, int unit,
                    int incX, int ldA, int N)
{
  register int i;
  DTYPE xtmp;

  for (i = 0; i < N; i++) {
    xtmp = unit ? alpha*X[i*incX] : 0.0;
    __vmult1dot(&xtmp, 1, &Ac[(i+unit)+i*ldA], &X[(i+unit)*incX], incX, alpha, N-unit-i);
    X[i*incX] = xtmp;
  }
}



void __trmv_unb(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  int unit = flags & GOMAS_UNIT ? 1 : 0;
  switch (flags & (GOMAS_TRANS|GOMAS_UPPER|GOMAS_LOWER)){
  case GOMAS_UPPER|GOMAS_TRANS:
    __trmv_unb_lut(X->md, A->md, alpha, unit, X->inc, A->step, N);
    break;
  case GOMAS_UPPER:
    __trmv_unb_lu(X->md, A->md, alpha, unit, X->inc, A->step, N);
    break;
  case GOMAS_LOWER|GOMAS_TRANS:
    __trmv_unb_llt(X->md, A->md, alpha, unit, X->inc, A->step, N);
    break;
  case GOMAS_LOWER:
  default:
    __trmv_unb_ll(X->md, A->md, alpha, unit, X->inc, A->step, N);
    break;
  }
}

static
void __trmv_forward_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  if (N < MIN_MVEC_SIZE) {
    __trmv_unb(X, A, alpha, flags, N);
    return;
  }

  // top part
  __subvector(&x0, X, 0);
  __subblock(&a0, A, 0, 0);
  __trmv_forward_recursive(&x0, &a0, alpha, flags, N/2);

  // update top with bottom
  __subvector(&x1, X, N/2);
  if (flags & GOMAS_UPPER) {
    __subblock(&a1, A, 0, N/2);
  } else {
    __subblock(&a1, A, N/2, 0);
  }
  __gemv_recursive(&x0, &a1, &x1, alpha, 1.0, flags, 0, N-N/2, 0, N/2);


  // bottom part
  __subblock(&a1, A, N/2, N/2);
  __trmv_forward_recursive(&x1, &a1, alpha, flags, N-N/2);
}

static
void __trmv_backward_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  mvec_t x0, x1;
  mdata_t a0, a1;

  //printf("__trmv_bk_recursive: N=%d\n", N);
  if (N < MIN_MVEC_SIZE) {
    __trmv_unb(X, A, alpha, flags, N);
    return;
  }

  // bottom part
  __subvector(&x1, X, N/2);
  __subblock(&a1, A, N/2, N/2);
  __trmv_backward_recursive(&x1, &a1, alpha, flags, N-N/2);

  // update bottom with top
  __subvector(&x0, X, 0);
  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __gemv_recursive(&x1, &a0, &x0, alpha, 1.0, flags, 0, N/2, 0, N-N/2);


  // top part
  __subblock(&a0, A, 0, 0);
  __trmv_backward_recursive(&x0, &a0, alpha, flags, N/2);
}

void __trmv_recursive(mvec_t *X, const mdata_t *A, DTYPE alpha, int flags, int N)
{
  if (N < MIN_MVEC_SIZE) {
    __trmv_unb(X, A, alpha, flags, N);
    return;
  }

  switch (flags & (GOMAS_UPPER|GOMAS_LOWER|GOMAS_TRANS)) {
  case GOMAS_LOWER|GOMAS_TRANS:
  case GOMAS_UPPER:
    __trmv_forward_recursive(X, A, alpha, flags, N);
    break;

  case GOMAS_UPPER|GOMAS_TRANS:
  case GOMAS_LOWER:
  default:
    __trmv_backward_recursive(X, A, alpha, flags, N);
    break;
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:

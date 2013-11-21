
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"


static inline
void __update4axpy(mdata_t *A, const mvec_t *X, const mvec_t *Y, DTYPE alpha, int M)
{
  register int i;
  register DTYPE *a0, *a1, *a2, *a3;
  const DTYPE *x;
  register DTYPE y0, y1, y2, y3;

  y0 = alpha*Y->md[0];
  y1 = alpha*Y->md[Y->inc];
  y2 = alpha*Y->md[2*Y->inc];
  y3 = alpha*Y->md[3*Y->inc];
  a0 = &A->md[0];
  a1 = &A->md[A->step];
  a2 = &A->md[2*A->step];
  a3 = &A->md[3*A->step];
  for (i = 0; i < M-3; i += 4) {
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    a1[(i+0)] += X->md[(i+0)*X->inc]*y1;
    a2[(i+0)] += X->md[(i+0)*X->inc]*y2;
    a3[(i+0)] += X->md[(i+0)*X->inc]*y3;
    a0[(i+1)] += X->md[(i+1)*X->inc]*y0;
    a1[(i+1)] += X->md[(i+1)*X->inc]*y1;
    a2[(i+1)] += X->md[(i+1)*X->inc]*y2;
    a3[(i+1)] += X->md[(i+1)*X->inc]*y3;
    a0[(i+2)] += X->md[(i+2)*X->inc]*y0;
    a1[(i+2)] += X->md[(i+2)*X->inc]*y1;
    a2[(i+2)] += X->md[(i+2)*X->inc]*y2;
    a3[(i+2)] += X->md[(i+2)*X->inc]*y3;
    a0[(i+3)] += X->md[(i+3)*X->inc]*y0;
    a1[(i+3)] += X->md[(i+3)*X->inc]*y1;
    a2[(i+3)] += X->md[(i+3)*X->inc]*y2;
    a3[(i+3)] += X->md[(i+3)*X->inc]*y3;
  }
  if (i == M)
    return;
  switch (M-i) {
  case 3:
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    a1[(i+0)] += X->md[(i+0)*X->inc]*y1;
    a2[(i+0)] += X->md[(i+0)*X->inc]*y2;
    a3[(i+0)] += X->md[(i+0)*X->inc]*y3;
    i++;
  case 2:
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    a1[(i+0)] += X->md[(i+0)*X->inc]*y1;
    a2[(i+0)] += X->md[(i+0)*X->inc]*y2;
    a3[(i+0)] += X->md[(i+0)*X->inc]*y3;
    i++;
  case 1:
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    a1[(i+0)] += X->md[(i+0)*X->inc]*y1;
    a2[(i+0)] += X->md[(i+0)*X->inc]*y2;
    a3[(i+0)] += X->md[(i+0)*X->inc]*y3;
  }
}


static inline
void __update1axpy(mdata_t *A, const mvec_t *X, const mvec_t *Y, DTYPE alpha, int M)
{
  register int i;
  register DTYPE *a0;
  register DTYPE y0;

  y0 = alpha*Y->md[0];
  a0 = &A->md[0];
  for (i = 0; i < M-3; i += 4) {
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    a0[(i+1)] += X->md[(i+1)*X->inc]*y0;
    a0[(i+2)] += X->md[(i+2)*X->inc]*y0;
    a0[(i+3)] += X->md[(i+3)*X->inc]*y0;
  }
  if (i == M)
    return;
  switch (M-i) {
  case 3:
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    i++;
  case 2:
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
    i++;
  case 1:
    a0[(i+0)] += X->md[(i+0)*X->inc]*y0;
  }
}


/*
 * Unblocked update of triangular (M == N) and trapezoidial (M != N) matrix.
 */
void __update_ger_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                       DTYPE alpha, int N, int M)
{
  mvec_t y0;
  mdata_t A0;
  register int j;
  for (j = 0; j < N-3; j += 4) {
    __subblock(&A0, A, 0, j);
    __subvector(&y0, Y, j);
    __update4axpy(&A0, X, &y0, alpha, M);
  }
  if (j == N)
    return;

  for (; j < N; j++) {
    __subblock(&A0, A, 0, j);
    __subvector(&y0, Y, j);
    __update1axpy(&A0, X, &y0, alpha, M);
  }
}

void __update_ger_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                            DTYPE alpha, int N, int M)
{
  mvec_t x0, y0;
  mdata_t A0;
  int nd = min(M, N);

  if (M < MIN_MVEC_SIZE || N < MIN_MVEC_SIZE) {
    __update_ger_unb(A, X, Y, alpha, N, M);
    return;
  }

  //printf("update 1. block.. [0,0] - [%d,%d]\n", M/2, N/2);
  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  __update_ger_recursive(&A0, &x0, &y0, alpha, N/2, M/2);

  //printf("update 2. block... [%d,0] - [%d,%d]\n", M/2, M-M/2, N/2);
  __subvector(&x0, X, M/2);
  __subblock(&A0, A, M/2, 0);
  __update_ger_recursive(&A0, &x0, &y0, alpha, N/2, M-M/2);

  //printf("update 3. block... [0,%d] - [%d,%d]\n", N/2, M/2, N-N/2);
  __subvector(&x0, X, 0);
  __subvector(&y0, Y, N/2);
  __subblock(&A0, A, 0, N/2);
  __update_ger_recursive(&A0, &x0, &y0, alpha, N-N/2, M/2);

  //printf("update 4. block... [%d,%d] - [%d,%d]\n", M/2, N/2, M-M/2, N-N/2);
  __subvector(&x0, X, M/2);
  __subblock(&A0, A, M/2, N/2);
  __update_ger_recursive(&A0, &x0, &y0, alpha, N-N/2, M-M/2);
}



// Local Variables:
// indent-tabs-mode: nil
// End:

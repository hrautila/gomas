
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"

// update one column of A
static inline
void __mult_mv1axpy(DTYPE *Ac, DTYPE *Xc, int incX, DTYPE alpha, int N)
{
  register int i;
  for (i = 0; i < N-3; i += 4) {
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
    Ac[(i+1)] += Xc[(i+1)*incX]*alpha;
    Ac[(i+2)] += Xc[(i+2)*incX]*alpha;
    Ac[(i+3)] += Xc[(i+3)*incX]*alpha;
  }
  if (i == N)
    return;
  switch (N-i) {
  case 3:
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
    i++;
  case 2:
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
    i++;
  case 1:
    Ac[(i+0)] += Xc[(i+0)*incX]*alpha;
  }
}

/*
 * Unblocked update of triangular (M == N) and trapezoidial (M != N) matrix.
 * (M is rows, N is columns.)
 */
void __update_trmv_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                       DTYPE alpha, int flags, int N, int M)
{
  register int j;
  switch (flags & (GOMAS_UPPER|GOMAS_LOWER)) {
  case GOMAS_UPPER:
    for (j = 0; j < N; j++) {
      __mult_mv1axpy(&A->md[j*A->step], &X->md[0], X->inc, alpha*Y->md[j*Y->inc], min(M, j+1));
    }
    break;
  case GOMAS_LOWER:
  default:
    for (j = 0; j < N; j++) {
      __mult_mv1axpy(&A->md[j+j*A->step], &X->md[j*X->inc], X->inc, alpha*Y->md[j*Y->inc], M-j);
    }
  }
}

void __update_trmv_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N, int M)
{
  mvec_t x0, y0;
  mdata_t A0;
  int nd = min(M, N);

  if (M < MIN_MVEC_SIZE || N < MIN_MVEC_SIZE) {
    __update_trmv_unb(A, X, Y, alpha, flags, N, M);
    return;
  }

  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  if (nd/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, nd/2, nd/2);
  } else {
    __update_trmv_recursive(&A0, &x0, &y0, alpha, flags, nd/2, nd/2);
  }

  if (flags & GOMAS_UPPER) {
    __subvector(&x0, X, 0);
    __subvector(&y0, Y, nd/2);
    __subblock(&A0, A, 0, nd/2);
    __update_ger_recursive(&A0, &x0, &y0, alpha, N-nd/2, nd/2);
  } else {
    __subvector(&x0, X, nd/2);
    __subblock(&A0, A, nd/2, 0);
    __update_ger_recursive(&A0, &x0, &y0, alpha, nd/2, M-nd/2);
  }

  __subvector(&y0, Y, nd/2);
  __subvector(&x0, X, nd/2);
  __subblock(&A0, A, nd/2, nd/2);
  if (N-nd/2 < MIN_MVEC_SIZE || M-nd/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, N-nd/2, M-nd/2);
  } else {
  __update_trmv_recursive(&A0, &x0, &y0, alpha, flags, N-nd/2, M-nd/2);
  }
}



// Local Variables:
// indent-tabs-mode: nil
// End:

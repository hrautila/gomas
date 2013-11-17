
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"

// Y = alpha*A*X + beta*Y for rows R:E, A is M*N and 0 < R < E <= M, Update
// with S:L columns from A and correspoding elements from X.
// length of X. With matrix-vector operation will avoid copying data.
static
void __gemv_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                DTYPE alpha, /*DTYPE beta,*/ int flags,
                 int S, int L, int R, int E)
{
  int i, j;
  register DTYPE *y;
  register const DTYPE *x;
  register const DTYPE *a0, *a1, *a2, *a3;

  // L - S is columns in A, elements in X
  // E - R is rows in A, elements in Y 
  if (L - S <= 0 || E - R <= 0) {
    return;
  }

  if ((flags & GOMAS_TRANSA) || (flags & GOMAS_TRANS)) {

    x = &X->md[S*X->inc];
    for (i = R; i < E-3; i += 4) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[(i+0)*A->step];
      a1 = &A->md[(i+1)*A->step];
      a2 = &A->md[(i+2)*A->step];
      a3 = &A->md[(i+3)*A->step];
      __vmult4dot(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, L-S);
    }
    if (i == E)
      return;

    switch (E-i) {
    case 3:
    case 2:
      y = &Y->md[i*Y->inc];
      a0 = &A->md[(i+0)*A->step];
      a1 = &A->md[(i+1)*A->step];
      __vmult2dot(y, Y->inc, a0, a1, x, X->inc, alpha, L-S);
      i += 2;
    }
    if (i < E) {
      y = &Y->md[i*Y->inc];
      a0 = &A->md[(i+0)*A->step];
      __vmult1dot(y, Y->inc, a0, x, X->inc, alpha, L-S);
    }
    return;
  }

  // Non-Transposed A here

  y = &Y->md[R*Y->inc];
  for (j = S; j < L-3; j += 4) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    a2 = &A->md[R+(j+2)*A->step];
    a3 = &A->md[R+(j+3)*A->step];
    __vmult4axpy(y, Y->inc, a0, a1, a2, a3, x, X->inc, alpha, E-R);
  }

  if (j == L)
    return;

  switch (L-j) {
  case 3:
  case 2:
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    a1 = &A->md[R+(j+1)*A->step];
    __vmult2axpy(y, Y->inc, a0, a1, x, X->inc, alpha, E-R);
    j += 2;
  }
  if (j < L) {
    x = &X->md[j*X->inc];
    a0 = &A->md[R+(j+0)*A->step];
    __vmult1axpy(y, Y->inc, a0, x, X->inc, alpha, E-R);
  }
}


void __gemv_recursive(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                      DTYPE alpha, DTYPE beta, int flags,
                      int S, int L, int R, int E)
{
  mvec_t x0, y0;
  mdata_t A0;
  int ny = E - R;
  int nx = L - S;

  //printf("__gemv_recursive: S=%d, L=%d, R=%d, E=%d, nx=%d, ny=%d\n", S, L, R, E, nx, ny);

  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, S, L, R, E);
    return;
  }

  // 1st block
  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx/2, 0, ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx/2, 0, ny/2);
  }

  // 2nd block
  __subvector(&y0, Y, ny/2);
  if (flags & GOMAS_TRANS) {
    __subblock(&A0, A, 0, ny/2);
  } else {
    __subblock(&A0, A, ny/2, 0);
  }
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx/2, 0, ny-ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx/2, 0, ny-ny/2);
  }

  // 3rd block
  __subvector(&x0, X, nx/2);
  __subvector(&y0, Y, 0);
  if (flags & GOMAS_TRANS) {
    __subblock(&A0, A, nx/2, 0);
  } else {
    __subblock(&A0, A, 0, nx/2);
  }
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx-nx/2, 0, ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx-nx/2, 0, ny/2);
  }

  // 4th block
  __subvector(&y0, Y, ny/2);
  if (flags & GOMAS_TRANS) {
    __subblock(&A0, A, nx/2, ny/2);
  } else {
    __subblock(&A0, A, ny/2, nx/2);
  }
  if (ny < MIN_MVEC_SIZE || nx < MIN_MVEC_SIZE) {
    __gemv_unb(Y, A, X, alpha, flags, 0, nx-nx/2, 0, ny-ny/2);
  } else {
    __gemv_recursive(&y0, &A0, &x0, alpha, beta, flags, 0, nx-nx/2, 0, ny-ny/2);
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:


// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"

// return sum of absolute values
static inline
ABSTYPE __vec_asum(const mvec_t *X,  int N)
{
  register int i, k;
  register ABSTYPE c0, c1, c2, c3, a0, a1, a2, a3;
  register DTYPE z0, z1, z2, z3;

  c0 = c1 = c2 = c3 = 0.0;
  for (i = 0; i < N-3; i += 4) {
    z0 = X->md[(i+0)*X->inc];
    z1 = X->md[(i+1)*X->inc];
    z2 = X->md[(i+2)*X->inc];
    z3 = X->md[(i+3)*X->inc];
    a0 = __ABS(z0);
    a1 = __ABS(z1);
    a2 = __ABS(z2);
    a3 = __ABS(z3);
    c0 += a0;
    c1 += a1;
    c2 += a2;
    c3 += a3;
  }    
  if (i == N)
    goto update;
  
  k = i*X->inc;
  switch (N-i) {
  case 3:
    c0 += __ABS(X->md[k]);
    k += X->inc;
  case 2:
    c1 += __ABS(X->md[k]);
    k += X->inc;
  case 1:
    c2 += __ABS(X->md[k]);
  }
 update:
  return c0 + c1 + c2 + c3;
}

static
DTYPE __vec_asum_kahan(const mvec_t *X, int N)
{
  register int k;
  register DTYPE c0, s0, c1, s1;
  register DTYPE t0, y0, t1, y1;

  c0 = c1 = s0 = s1 = __ZERO;
  for (k = 0; k < N-1; k += 2) {
    y0 = __ABS(X->md[(k+0)*X->inc]) - c0;
    t0 = s0 + y0;
    c0 = (t0 - s0) - y0;
    s0 = t0;

    y1 = __ABS(X->md[(k+1)*X->inc]) - c1;
    t1 = s1 + y1;
    c1 = (t1 - s1) - y1;
    s1 = t1;
  }
  if (k == N)
    return s0 + s1;

  y0 = __ABS(X->md[(k+0)*X->inc]) - c0;
  t0 = s0 + y0;
  c0 = (t0 - s0) - y0;
  s0 = t0;
  return s0 + s1;
}

DTYPE __vec_asum_recursive(const mvec_t *X, int n)
{
  register DTYPE c0, c1, c2, c3;
  mvec_t x0;

  if (n < MIN_MVEC_SIZE)
    return __vec_asum(X, n);

  if (n/2 < MIN_MVEC_SIZE) {
    c0 = __vec_asum(__subvector(&x0, X, 0),   n/2);
    c1 = __vec_asum(__subvector(&x0, X, n/2), n-n/2);
    return c0+c1;
  }

  if (n/4 < MIN_MVEC_SIZE) {
    c0 = __vec_asum(__subvector(&x0, X, 0),       n/4);
    c1 = __vec_asum(__subvector(&x0, X, n/4),     n/2-n/4);
    c2 = __vec_asum(__subvector(&x0, X, n/2),     n/4);
    c3 = __vec_asum(__subvector(&x0, X, n/2+n/4), n-n/2-n/4);
    return c0 + c1 + c2 + c3;
  }

  c0 = __vec_asum_recursive(__subvector(&x0, X, 0),       n/4);
  c1 = __vec_asum_recursive(__subvector(&x0, X, n/4),     n/2-n/4);
  c2 = __vec_asum_recursive(__subvector(&x0, X, n/2),     n/4);
  c3 = __vec_asum_recursive(__subvector(&x0, X, n/2+n/4), n-n/2-n/4);
  return c0 + c1 + c2 + c3;
}


static inline
DTYPE __vec_sum(const mvec_t *X,  int N)
{
  register int i, k;
  register DTYPE c0, c1, c2, c3;
  register DTYPE z0, z1, z2, z3;

  c0 = X->md[0];
  c1 = X->md[X->inc];
  c2 = X->md[2*X->inc];
  c3 = X->md[3*X->inc];
  for (i = 4; i < N-3; i += 4) {
    z0 = X->md[(i+0)*X->inc];
    z1 = X->md[(i+1)*X->inc];
    z2 = X->md[(i+2)*X->inc];
    z3 = X->md[(i+3)*X->inc];
    c0 += z0;
    c1 += z1;
    c2 += z2;
    c3 += z3;
  }    
  if (i == N)
    goto update;
  
  k = i*X->inc;
  switch (N-i) {
  case 3:
    c0 += X->md[k];
    k += X->inc;
  case 2:
    c1 += X->md[k];
    k += X->inc;
  case 1:
    c2 += X->md[k];
  }
 update:
  return c0 + c1 + c2 + c3;
}

static
DTYPE __vec_sum_kahan(const mvec_t *X, int N)
{
  register int k;
  register DTYPE c0, s0, c1, s1;
  register DTYPE t0, y0, t1, y1;

  c0 = c1 = s0 = s1 = __ZERO;
  for (k = 0; k < N-1; k += 2) {
    y0 = X->md[(k+0)*X->inc] - c0;
    t0 = s0 + y0;
    c0 = (t0 - s0) - y0;
    s0 = t0;

    y1 = X->md[(k+1)*X->inc] - c1;
    t1 = s1 + y1;
    c1 = (t1 - s1) - y1;
    s1 = t1;
  }
  if (k == N)
    return s0 + s1;

  y0 = X->md[(k+0)*X->inc] - c0;
  t0 = s0 + y0;
  c0 = (t0 - s0) - y0;
  s0 = t0;
  return s0 + s1;
}


DTYPE __vec_sum_recursive(const mvec_t *X, int n)
{
  register DTYPE c0, c1, c2, c3;
  mvec_t x0;

  if (n < MIN_MVEC_SIZE)
    return __vec_sum(X, n);

  if (n/2 < MIN_MVEC_SIZE) {
    c0 = __vec_sum(__subvector(&x0, X, 0),   n/2);
    c1 = __vec_sum(__subvector(&x0, X, n/2), n-n/2);
    return c0 + c1;
  }

  if (n/4 < MIN_MVEC_SIZE) {
    c0 = __vec_sum(__subvector(&x0, X, 0),       n/4);
    c1 = __vec_sum(__subvector(&x0, X, n/4),     n/2-n/4);
    c2 = __vec_sum(__subvector(&x0, X, n/2),     n/4);
    c3 = __vec_sum(__subvector(&x0, X, n/2+n/4), n-n/2-n/4);
    return c0 + c1 + c2 + c3;
  }

  c0 = __vec_sum_recursive(__subvector(&x0, X, 0),       n/4);
  c1 = __vec_sum_recursive(__subvector(&x0, X, n/4),     n/2-n/4);
  c2 = __vec_sum_recursive(__subvector(&x0, X, n/2),     n/4);
  c3 = __vec_sum_recursive(__subvector(&x0, X, n/2+n/4), n-n/2-n/4);
  return c0 + c1 + c2 + c3;
}



// Local Variables:
// indent-tabs-mode: nil
// End:


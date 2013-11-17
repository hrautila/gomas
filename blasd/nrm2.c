
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"

// return vector norm; naive version
ABSTYPE __vec_nrm2_naive(const mvec_t *X,  int N)
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
    c0 += a0*a0;
    c1 += a1*a1;
    c2 += a2*a2;
    c3 += a3*a3;
  }    
  if (i == N)
    goto update;

  k = i*X->inc;
  switch (N-i) {
  case 3:
    a0 = __ABS(X->md[k]);
    c0 += a0*a0;
    k += X->inc;
  case 2:
    a1 = __ABS(X->md[k]);
    c1 += a1*a1;
    k += X->inc;
  case 1:
    a2 = __ABS(X->md[k]);
    c2 += a2*a2;
  }
 update:
  return __SQRT(c0 + c1 + c2 + c3);
}

/*
 * Nick Higham in Accurrancy and Precision:
 *   For about half of all machine numbers x, value of x^2 either
 *   underflows or overflows
 *
 * Overflow is avoided by summing squares of scaled numbers and
 * then multiplying then with the scaling factor. Following is
 * is by Hammarling and included in BLAS reference libary.
 */

ABSTYPE __vec_nrm2_scaled(const mvec_t *X,  int N)
{
  register int i, k;
  register ABSTYPE a0, sum, scale;

  sum = __ABSONE;
  scale = __ABSZERO;
  for (i = 0; i < N; i += 1) {
    if (X->md[(i+0)*X->inc] != __ZERO) {
      a0 = __ABS(X->md[(i+0)*X->inc]);
      if (a0 > scale) {
        sum = __ONE + sum * ((scale/a0)*(scale/a0));
        scale = a0;
      } else {
        sum = sum + (a0/scale)*(a0/scale);
      }
    }
  }    
  return scale*__SQRT(sum);
}

// Local Variables:
// indent-tabs-mode: nil
// End:


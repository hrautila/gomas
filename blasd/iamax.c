
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"

// return index of max absolute value
int __vec_iamax(const mvec_t *X,  int N)
{
  register int i, ix, n;
  register ABSTYPE max, c0, c1;

  if (N <= 1)
    return 0;

  max = 0.0;
  ix = 0;
  for (i = 0; i < N-1; i += 2) {
    c0 = __ABS(X->md[(i+0)*X->inc]);
    c1 = __ABS(X->md[(i+1)*X->inc]);
    if (c1 > c0) {
      n = 1;
      c0 = c1;
    }
    if (c0 > max) {
      ix = i+n;
      max = c0;
    }
    n = 0;
  }    
  if (i < N) {
    c0 = __ABS(X->md[i*X->inc]);
    ix = c0 > max ? N-1 : ix;
  }
  return ix;
}


ABSTYPE __vec_amax(const mvec_t *X,  int N)
{
  int ix = __vec_iamax(X, N);
  if (ix >= 0) {
    return X->md[ix*X->inc];
  }
  return __ABSZERO;
}


// Local Variables:
// indent-tabs-mode: nil
// End:



// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"


void __vec_swap(mvec_t *X,  mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double y0, y1, y2, y3, x0, x1, x2, x3;

  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    X->md[(i+0)*X->inc] = y0;
    X->md[(i+1)*X->inc] = y1;
    X->md[(i+2)*X->inc] = y2;
    X->md[(i+3)*X->inc] = y3;
    Y->md[(i+0)*Y->inc] = x0;
    Y->md[(i+1)*Y->inc] = x1;
    Y->md[(i+2)*Y->inc] = x2;
    Y->md[(i+3)*Y->inc] = x3;
  }    
  if (i == N)
    return;

  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky];
    Y->md[ky] = X->md[kx];
    X->md[kx] = y0;
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:


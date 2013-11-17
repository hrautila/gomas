
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"


void __vec_axpy(mvec_t *Y,  const mvec_t *X, DTYPE alpha, int N)
{
  register int i, kx, ky;
  register DTYPE y0, y1, y2, y3, x0, x1, x2, x3;

  // gcc uses different XMM target registers for yN, xN; 
  for (i = 0; i < N-3; i += 4) {
    y0 = Y->md[(i+0)*Y->inc];
    y1 = Y->md[(i+1)*Y->inc];
    y2 = Y->md[(i+2)*Y->inc];
    y3 = Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    y0 += alpha*x0;
    y1 += alpha*x1;
    y2 += alpha*x2;
    y3 += alpha*x3;
    Y->md[(i+0)*Y->inc] = y0;
    Y->md[(i+1)*Y->inc] = y1;
    Y->md[(i+2)*Y->inc] = y2;
    Y->md[(i+3)*Y->inc] = y3;
  }    
  if (i == N)
	return;

  kx = i*X->inc; ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
  }
}

void __vec_axpby(mvec_t *Y,  const mvec_t *X, DTYPE alpha, DTYPE beta, int N)
{
  register int i, kx, ky;
  register DTYPE y0, y1, y2, y3, x0, x1, x2, x3;

  // gcc uses different XMM target registers for yN, xN; 
  for (i = 0; i < N-3; i += 4) {
    y0 = beta*Y->md[(i+0)*Y->inc];
    y1 = beta*Y->md[(i+1)*Y->inc];
    y2 = beta*Y->md[(i+2)*Y->inc];
    y3 = beta*Y->md[(i+3)*Y->inc];
    x0 = X->md[(i+0)*X->inc];
    x1 = X->md[(i+1)*X->inc];
    x2 = X->md[(i+2)*X->inc];
    x3 = X->md[(i+3)*X->inc];
    y0 += alpha*x0;
    y1 += alpha*x1;
    y2 += alpha*x2;
    y3 += alpha*x3;
    Y->md[(i+0)*Y->inc] = y0;
    Y->md[(i+1)*Y->inc] = y1;
    Y->md[(i+2)*Y->inc] = y2;
    Y->md[(i+3)*Y->inc] = y3;
  }    
  if (i == N)
	return;

  kx = i*X->inc; ky = i*Y->inc;
  switch (N-i) {
  case 3:
    y0 = beta*Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 2:
    y0 = beta*Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
    kx += X->inc; ky += Y->inc;
  case 1:
    y0 = beta*Y->md[ky];
    Y->md[ky] = y0 + alpha*X->md[kx];
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:


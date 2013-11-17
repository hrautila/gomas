
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"

void __vec_scal(mvec_t *X,  DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] * alpha;
    f1 =  X->md[(i+1)*X->inc] * alpha;
    f2 =  X->md[(i+2)*X->inc] * alpha;
    f3 =  X->md[(i+3)*X->inc] * alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] *= alpha;
    k += X->inc;
  case 2:
    x0[k] *= alpha;
    k += X->inc;
  case 1:
    x0[k] *= alpha;
  }
}

void __vec_invscal(mvec_t *X,  DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] / alpha;
    f1 =  X->md[(i+1)*X->inc] / alpha;
    f2 =  X->md[(i+2)*X->inc] / alpha;
    f3 =  X->md[(i+3)*X->inc] / alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] /= alpha;
    k += X->inc;
  case 2:
    x0[k] /= alpha;
    k += X->inc;
  case 1:
    x0[k] /= alpha;
  }
}

void __vec_add(mvec_t *X,  DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X->md[(i+0)*X->inc] + alpha;
    f1 =  X->md[(i+1)*X->inc] + alpha;
    f2 =  X->md[(i+2)*X->inc] + alpha;
    f3 =  X->md[(i+3)*X->inc] + alpha;
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X->md[i*X->inc];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] += alpha;
    k += X->inc;
  case 2:
    x0[k] += alpha;
    k += X->inc;
  case 1:
    x0[k] += alpha;
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:


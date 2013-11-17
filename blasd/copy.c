
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"
#include "matcpy.h"

void __vec_copy(mvec_t *X,  const mvec_t *Y, int N)
{
  register int i, kx, ky;
  register double f0, f1, f2, f3;

  // gcc compiles loop body to use different target XMM registers
  for (i = 0; i < N-3; i += 4) {
    f0 = Y->md[(i+0)*Y->inc];
    f1 = Y->md[(i+1)*Y->inc];
    f2 = Y->md[(i+2)*Y->inc];
    f3 = Y->md[(i+3)*Y->inc];
    X->md[(i+0)*X->inc] = f0;
    X->md[(i+1)*X->inc] = f1;
    X->md[(i+2)*X->inc] = f2;
    X->md[(i+3)*X->inc] = f3;
  }    
  if (i == N)
    return;

  // calculate indexes only once
  kx = i*X->inc;
  ky = i*Y->inc;
  switch (N-i) {
  case 3:
    X->md[kx] = Y->md[ky];
    kx++; ky++;
  case 2:
    X->md[kx] = Y->md[ky];
    kx++; ky++;
  case 1:
    X->md[kx] = Y->md[ky];
  }
}

void __blk_copy(mdata_t *A,  const mdata_t *B, int M, int N)
{
  copy_plain_mcpy1(A->md, A->step, B->md, B->step, M, N);
}

// M is rows in source and cols in destination, N is cols in source
void __blk_transpose(mdata_t *A,  const mdata_t *B, int M, int N)
{
  copy_trans4x1(A->md, A->step, B->md, B->step, M, N);
}

// Local Variables:
// indent-tabs-mode: nil
// End:


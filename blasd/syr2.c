
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"


void __update_syr2_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N)
{
  mvec_t x0, y0;
  mdata_t A0;

  if (N < MIN_MVEC_SIZE) {
    __update_trmv_unb(A, X, Y, alpha, flags, N, N);
    __update_trmv_unb(A, Y, X, alpha, flags, N, N);
    return;
  }

  __subvector(&x0, X, 0);
  __subvector(&y0, Y, 0);
  __subblock(&A0, A, 0, 0);
  if (N/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, N/2, N/2);
    __update_trmv_unb(&A0, &y0, &x0, alpha, flags, N/2, N/2);
  } else {
    __update_syr2_recursive(&A0, &x0, &y0, alpha, flags, N/2);
  }

  if (flags & GOMAS_UPPER) {
    __subvector(&x0, X, 0);
    __subvector(&y0, Y, N/2);
    __subblock(&A0, A, 0, N/2);
    __update_ger_recursive(&A0, &x0, &y0, alpha, flags, N-N/2, N/2);
    __subvector(&x0, X, N/2);
    __subvector(&y0, Y, 0);
    __update_ger_recursive(&A0, &y0, &x0, alpha, flags, N-N/2, N/2);
  } else {
    __subvector(&y0, Y, 0);
    __subvector(&x0, X, N/2);
    __subblock(&A0, A, N/2, 0);
    __update_ger_recursive(&A0, &x0, &y0, alpha, flags, N/2, N-N/2);
    __subvector(&y0, Y, N/2);
    __subvector(&x0, X, 0);
    __update_ger_recursive(&A0, &y0, &x0, alpha, flags, N/2, N-N/2);
  }

  __subvector(&y0, Y, N/2);
  __subvector(&x0, X, N/2);
  __subblock(&A0, A, N/2, N/2);
  if (N-N/2 < MIN_MVEC_SIZE) {
    __update_trmv_unb(&A0, &x0, &y0, alpha, flags, N-N/2, N-N/2);
    __update_trmv_unb(&A0, &y0, &x0, alpha, flags, N-N/2, N-N/2);
  } else {
    __update_syr2_recursive(&A0, &x0, &y0, alpha, flags, N-N/2);
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:

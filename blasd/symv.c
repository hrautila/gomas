
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/armas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

/** @defgroup blas2 BLAS level 2 functions.
 *
 */
#include <stdio.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"


/*
 * Objective: read matrix A in memory order, along columns.
 *
 *  y0    a00 |  0   0     x0     y0 = a00*x0 + a10*x1 + a20*x2
 *  --    --------------   --
 *  y1    a10 | a11  0     x1     y1 = a10*x0 + a11*x1 + a21*x2
 *  y2    a20 | a21  a22   x2     y2 = a20*x0 + a21*x1 + a22*x2
 *
 *  y1 += (a11) * x1  
 *  y2    (a21)
 *
 *  y1 += a21.T*x2
 *
 * UPPER:
 *  y0    a00 | a01 a02   x0     y0 = a00*x0 + a01*x1 + a02*x2
 *  --    --------------   --
 *  y1     0  | a11 a12   x1     y1 = a01*x0 + a11*x1 + a12*x2
 *  y2     0  |  0  a22   x2     y2 = a02*x0 + a12*x1 + a22*x2
 *
 *  (y0) += (a01) * x1
 *  (y1)    (a11)
 */
void __symv_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                DTYPE alpha, int flags, int N)
{
  int i, j;
  mvec_t yy, xx, aa;
  
  if ( N <= 0 )
    return;

  if (flags & GOMAS_LOWER) {
    for (j = 0; j < N; j++) {
      __subvector(&yy, Y, j);
      __subvector(&xx, X, j);
      __colvec(&aa, A, j, j);
      __vmult1axpy(yy.md, yy.inc, aa.md, xx.md, xx.inc, alpha, N-j);
      __vmult1dot(yy.md, yy.inc, &aa.md[1], &xx.md[xx.inc], xx.inc, alpha, N-j-1);
    }
    return;
  }

  // Upper here;
  //  1. update elements 0:j with current column and x[j]
  //  2. update current element y[j] with product of a[0:j-1]*x[0:j-1]
  for (j = 0; j < N; j++) {
    __subvector(&xx, X, j);
    __colvec(&aa, A, 0, j);
    __vmult1axpy(Y->md, Y->inc, aa.md, xx.md, xx.inc, alpha, j+1);

    __subvector(&yy, Y, j);
    __vmult1dot(yy.md, yy.inc, aa.md, X->md, X->inc, alpha, j);
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:

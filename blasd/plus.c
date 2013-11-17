
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include "dtype.h"
#include "interfaces.h"

static inline
void __scale_plus_notrans(DTYPE * __restrict dst, int ldD,
                          const double * __restrict src, int ldS, int nR, int nC,
                          const double alpha, const DTYPE beta)
{
  register double *Dc, *Dr;
  register const double *Sc, *Sr;
  register int j, i;
  register int zero;

  zero = alpha == 0.0;

  Dc = dst; Sc = src;
  for (j = 0; j < nC; j++) {
    Dr = Dc;
    Sr = Sc;
    // incrementing Dr with ldD follows the dst row
    // and incrementing Sr with one follows the column
    for (i = 0; i < nR-1; i += 2) {
      Dr[0] = zero ? beta * Sr[0] : alpha*Dr[0] + beta * Sr[0];
      Dr[1] = zero ? beta * Sr[1] : alpha*Dr[1] + beta * Sr[1];
      Dr += 2;
      Sr += 2;
    }
    if (i < nR) {
      Dr[0] = zero ? beta * Sr[0] : alpha*Dr[0] + beta * Sr[0];
    }
    Dc += ldD;
    Sc += ldS;
  }
}


// dst = beta*dst + alpha*src.T
static inline
void __scale_plus_trans(DTYPE * __restrict dst, int ldD,
                        const double * __restrict src, int ldS, int nR, int nC,
                        const double alpha, const DTYPE beta)
{
  register double *Dc, *Dr;
  register const double *Sc, *Sr;
  register int j, i;
  register int zero;

  zero = alpha == 0.0;

  Dc = dst; Sc = src;
  for (j = 0; j < nC; j++) {
    Dr = Dc;
    Sr = Sc;
    // incrementing Dr with ldD follows the dst row
    // and incrementing Sr with one follows the column
    for (i = 0; i < nR-1; i += 2) {
      Dr[0]   = zero ? beta * Sr[0] : alpha*Dr[0]   + beta * Sr[0];
      Dr[ldD] = zero ? beta * Sr[1] : alpha*Dr[ldD] + beta * Sr[1];
      Dr += ldD << 1;
      Sr += 2;
    }
    if (i < nR) {
      Dr[0] = zero ? beta * Sr[0] : alpha*Dr[0] + beta * Sr[0];
    }
    // moves Dc pointer to next row on dst
    Dc++;
    // moves Sc pointer to next column on src
    Sc += ldS;
  }
}



/*
 * A = alpha*A + beta*B
 * A = alpha*A + beta*B.T
 */
void __d_scale_plus(mdata_t *A, const mdata_t *B,
                    double alpha, double beta, int flags,
                    int S, int L, int R, int E)
{
  double *Ac;
  const double *Bc;

  if (L-S <= 0 || E-R <= 0) {
    return;
  }

  Ac = &A->md[S*A->step + R];
  if (flags & (GOMAS_TRANSB|GOMAS_TRANS)) {
    Bc = &B->md[R*B->step + S];
    __scale_plus_trans(Ac, A->step, Bc, B->step, L-S, E-R, alpha, beta);
  } else {
    Bc = &B->md[S*B->step + R];
    __scale_plus_notrans(Ac, A->step, Bc, B->step, E-R, L-S, alpha, beta);
  }
}



// Local Variables:
// indent-tabs-mode: nil
// End:

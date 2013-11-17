
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"

// Recursive versions of TRMM

/*
 *   LEFT-UPPER              LEFT-LOWER-TRANS
 *                                                     
 *    A00 | A01   B0         A00 |  0     B0
 *   -----------  --        -----------   --
 *     0  | A11   B1         A10 | A11    B1
 *
 *
 *   B0 = A00*B0 + A01*B1    B0 = A00*B0 + A10*B1
 *   B1 = A11*B1             B1 = A11*B1
 */
static
void __mult_left_forward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                         int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0, a1;
  mdata_t *Acpy, *Bcpy;

  if (N < MIN_MBLOCK_SIZE) {
    __trmm_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b0, B, 0, S),
               __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S);
  } else {
    __mult_left_forward(__subblock(&b0, B, 0, S),
                        __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
  }

  // update B0 with A01*B1
  if (flags & GOMAS_UPPER) {
    __subblock(&a1, A, 0, N/2);
  } else {
    __subblock(&a1, A, N/2, 0);
  }
  __subblock(&b1, B, N/2, S);
  __gemm_colwise_inner_no_scale(&b0, &a1, &b1, alpha, flags,
                                  N-N/2, E-S, N/2, cache);
  
  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b0, B, N/2, S),
               __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S);
  } else {
    __mult_left_forward(__subblock(&b0, B, N/2, S),
                        __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
  }
}


/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *                                                     
 *    A00 | A01   B0         A00 |  0     B0
 *   -----------  --        -----------   --
 *     0  | A11   B1         A10 | A11    B1
 *
 *
 *   B0 = A00*B0            B0 = A00*B0
 *   B1 = A01*B0 + A11*B1   B1 = A10*B0 + A11*B1
 */
static
void __mult_left_backward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0, a1;

  if (N < MIN_MBLOCK_SIZE) {
    __trmm_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b1, B, N/2, S),
               __subblock(&a1, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S);
  } else {
    __mult_left_backward(__subblock(&b1, B, N/2, S),
                         __subblock(&a1, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
  }

  // update b1, with A10*B0/A01*b0
  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
    }
  __subblock(&b0, B, 0, S);
  __gemm_colwise_inner_no_scale(&b1, &a0, &b0, alpha, flags,
                                  N/2, E-S, N-N/2, cache);
    
  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b0, B, 0, S),
               __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S);
  } else {
    __mult_left_backward(__subblock(&b0, B, 0, S),
                         __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
  }
}

/*
 *   RIGHT-UPPER-TRANS         RIGHT-LOWER
 *
 *            A00 | A01                 A00 |  0 
 *   B0|B1 * -----------       B0|B1 * -----------
 *             0  | A11                 A10 | A11 
 *
 *   B0 = B0*A00 + B1*A01      B0 = B0*A00 + B1*A10
 *   B1 = B1*A11               B1 = B1*A11
 */
static
void __mult_right_forward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0, a1;
  int ar, ac, ops;

  if (N < MIN_MBLOCK_SIZE) {
    __trmm_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b0, B, S, 0),
               __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S);
  } else {
    __mult_right_forward(__subblock(&b0, B, S, 0),
                         __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
  }
  if (flags & GOMAS_UPPER) {
    __subblock(&a1, A, 0, N/2);
  } else {
    __subblock(&a1, A, N/2, 0);
  }
  __subblock(&b1, B, S, N/2);

  ops = flags & GOMAS_TRANSA ? GOMAS_TRANSB : GOMAS_NULL;
  __gemm_colwise_inner_no_scale(&b0, &b1, &a1, alpha, ops,
                                  N-N/2, N/2, E-S, cache);
  
  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b1, B, S,   N/2),
               __subblock(&a1, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S);
  } else {
    __mult_right_forward(__subblock(&b1, B, S,   N/2),
                         __subblock(&a1, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
  }
}


/*
 *   RIGHT-UPPER               RIGHT-LOWER-TRANSA
 *
 *            A00 | A01                 A00 |  0 
 *   B0|B1 * -----------       B0|B1 * -----------
 *             0  | A11                 A10 | A11 
 *
 *   B0 = B0*A00               B0 = B0*A00
 *   B1 = B0*A01 + B1*A11      B1 = B0*A10 + B1*A11
 */
static
void __mult_right_backward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0, a1;

  if (N < MIN_MBLOCK_SIZE) {
    __trmm_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b1, B, S,   N/2),
               __subblock(&a1, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S);
  } else {
    __mult_right_backward(__subblock(&b1, B, S,   N/2),
                          __subblock(&a1, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
  }

  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __subblock(&b0, B, S, 0);
  int flgs = flags & GOMAS_TRANSA ? GOMAS_TRANSB : 0;
  __gemm_colwise_inner_no_scale(&b1, &b0, &a0, alpha, flgs,
                                  N/2, N-N/2, E-S, cache);
    
  if (N/2 < MIN_MBLOCK_SIZE) {
    __trmm_unb(__subblock(&b0, B, S, 0),
               __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S);
  } else {
    __mult_right_backward(__subblock(&b0, B, S, 0),
                          __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
  }
}

void __trmm_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache)
{
  
  switch (flags&(GOMAS_UPPER|GOMAS_LOWER|GOMAS_RIGHT|GOMAS_TRANSA)) {
  case GOMAS_RIGHT|GOMAS_UPPER:
  case GOMAS_RIGHT|GOMAS_LOWER|GOMAS_TRANSA:
    __mult_right_backward(B, A, alpha, flags, N, S, E, cache);
    break;
    
  case GOMAS_RIGHT|GOMAS_LOWER:
  case GOMAS_RIGHT|GOMAS_UPPER|GOMAS_TRANSA:
    __mult_right_forward(B, A, alpha, flags, N, S, E, cache);
    break;

  case GOMAS_UPPER:
  case GOMAS_LOWER|GOMAS_TRANSA:
    __mult_left_forward(B, A, alpha, flags, N, S, E, cache);
    break;

  case GOMAS_LOWER:
  case GOMAS_UPPER|GOMAS_TRANSA:
  default:
    __mult_left_backward(B, A, alpha, flags, N, S, E, cache);
    break;
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:


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

// Recursive versions of TRSM

/*
 *   RIGHT-UPPER             RIGHT-LOWER
 *                                                     
 *            A00 | A01               A00 |  0  
 *   B0|B1 * -----------     B0|B1 * -----------
 *             0  | A11               A10 | A11 
 *
 */

/*
 *   LEFT-UPPER-TRANS        LEFT-LOWER
 *                                                     
 *    A00 | A01    B0         A00 |  0     B0
 *   ----------- * --        ----------- * --
 *     0  | A11    B1         A10 | A11    B1
 *
 *  upper:
 *    B'0 = A00*B0           --> B0 = trsm(B'0, A00)
 *    B'1 = A01*B0 + A11*B1  --> B1 = trsm(B'1 - A01*B0)
 *  lower:
 *    B'0 = A00*B0           --> B0 = trsm(B'0, A00)
 *    B'1 = A10*B0 + A11*B1  --> B1 = trsm(B'1 - A10*B0, A11)
 *
 *   Forward substitution.
 */
static
void __solve_left_forward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;
  mdata_t *Acpy, *Bcpy;

  //printf("__solve_left_forward: N=%d, S= %d, E=%d\n", N, S, E);
  if (N < MIN_MBLOCK_SIZE) {
    //printf("__solve_left_forward: direct ..\n");
    __solve_left_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  //printf("__solve_left_forward: sub-block [0:0, %d:%d]\n", 0, N/2);
  __solve_left_forward(__subblock(&b0, B, 0, S),
                       __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);

  // update B[0:N/2, S:E] with B[N/2:N, S:E]
  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __subblock(&b1, B, N/2, S);
  __gemm_colwise_inner_no_scale(&b1, &a0, &b0, -1.0, flags,
                                  N/2, E-S, N-N/2, cache);
  
  //printf("__solve_left_forward: sub-block [%d:%d, %d:%d]\n", N/2, N/2, N-N/2, N-N/2);
  __solve_left_forward(__subblock(&b0, B, N/2, S),
                       __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
}


/*
 *   LEFT-UPPER               LEFT-LOWER-TRANS
 *                                                     
 *    A00 | A01    B0         A00 |  0     B0
 *   ----------- * --        ----------- * --
 *     0  | A11    B1         A10 | A11    B1
 *
 *  upper:
 *    B'0 = A00*B0 + A01*B1  --> B0 = A00.-1*(B'0 - A01*B1)
 *    B'1 = A11*B1           --> B1 = A11.-1*B'1
 *  lower:
 *    B'0 = A00*B0 + A10*B1  --> B0 = trsm(B'0 - A10*B1, A00)
 *    B'1 = A11*B1           --> B1 = trsm(B'1, A11)
 *
 *   Backward substitution.
 */
static
void __solve_left_backward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;
  mdata_t *Acpy, *Bcpy;

  //printf("__solve_left_backward: N=%d, S= %d, E=%d\n", N, S, E);
  if (N < MIN_MBLOCK_SIZE) {
    //printf("__solve_left_backward: direct ..\n");
    __solve_left_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  // UPPER and LOWER TRANS are backward subsition, 
  //printf("__solve_left_backward: sub-block [%d:%d, %d:%d]\n", N/2, N/2, N, N);
  __solve_left_backward(__subblock(&b0, B, N/2, S),
                        __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);

  // update B[0:N/2, S:E] with B[N/2:N, S:E]
  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
  }
  __subblock(&b1, B, 0, S);
  __gemm_colwise_inner_no_scale(&b1, &a0, &b0, -1.0, flags,
                                  N-N/2, E-S, N/2, cache);
    
  //printf("__solve_left_backward: sub-block [0:0, %d:%d]\n", 0, N/2);
  __solve_left_backward(__subblock(&b0, B, 0, S),
                        __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
}


/*
 * Forward substitution for RIGHT-UPPER, RIGHT-LOWER-TRANSA
 */
static
void __solve_right_forward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;
  int ar, ac, ops;

  if (N < MIN_MBLOCK_SIZE) {
    __solve_right_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  //printf("__solve_right_forward: sub-block [0:0, %d:%d]\n", 0, N/2);
  __solve_right_forward(__subblock(&b0, B, S, 0),
                        __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);

  ar = flags & GOMAS_UPPER ? 0   : N/2;
  ac = flags & GOMAS_UPPER ? N/2 : 0;

  __subblock(&a0, A, ar, ac);
  __subblock(&b1, B, S, N/2);

  ops = flags & GOMAS_TRANSA ? GOMAS_TRANSB : GOMAS_NULL;
  __gemm_colwise_inner_no_scale(&b1, &b0, &a0, -1.0, ops,
                                  N/2, N-N/2, E-S, cache);
  
  //printf("__solve_right_forward: sub-block [%d:%d, %d:%d]\n", N/2, N/2, N, N);
  __solve_right_forward(__subblock(&b0, B, S,   N/2),
                        __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);
}


/*
 * Backward substitution for RIGHT-UPPER-TRANSA and RIGHT-LOWER
 */
static
void __solve_right_backward(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  mdata_t b0, b1, a0;
  mdata_t *Acpy, *Bcpy;
  int ops;

  //printf("__solve_right_backward: N=%d, S= %d, E=%d\n", N, S, E);
  if (N < MIN_MBLOCK_SIZE) {
    //printf("__solve_right_backward: direct ..\n");
    __solve_right_unb(B, A, alpha, flags, N, S, E);
    return;
  }

  //printf("__solve_right_backward: sub-block [%d:%d, %d:%d]\n", N/2, N/2, N, N);
  __solve_right_backward(__subblock(&b0, B, S,   N/2),
                         __subblock(&a0, A, N/2, N/2), alpha, flags, N-N/2, 0, E-S, cache);

  if (flags & GOMAS_UPPER) {
    __subblock(&a0, A, 0, N/2);
  } else {
    __subblock(&a0, A, N/2, 0);
    }
  __subblock(&b1, B, S, 0);
  ops = flags & GOMAS_TRANSA ? GOMAS_TRANSB : GOMAS_NULL;
  __gemm_colwise_inner_no_scale(&b1, &b0, &a0, -1.0, ops,
                                  N-N/2, N/2, E-S, cache);
    
  //printf("__solve_right_backward: sub-block [0:0, %d:%d]\n", 0, N/2);
  __solve_right_backward(__subblock(&b0, B, S, 0),
                         __subblock(&a0, A, 0, 0), alpha, flags, N/2, 0, E-S, cache);
}

void __solve_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache)
{
  
  switch (flags&(GOMAS_UPPER|GOMAS_LOWER|GOMAS_RIGHT|GOMAS_TRANSA)) {
  case GOMAS_RIGHT|GOMAS_UPPER:
  case GOMAS_RIGHT|GOMAS_LOWER|GOMAS_TRANSA:
    __solve_right_forward(B, A, alpha, flags, N, S, E, cache);
    break;
    
  case GOMAS_RIGHT|GOMAS_LOWER:
  case GOMAS_RIGHT|GOMAS_UPPER|GOMAS_TRANSA:
    __solve_right_backward(B, A, alpha, flags, N, S, E, cache);
    break;

  case GOMAS_UPPER:
  case GOMAS_LOWER|GOMAS_TRANSA:
    __solve_left_backward(B, A, alpha, flags, N, S, E, cache);
    break;

  case GOMAS_LOWER:
  case GOMAS_UPPER|GOMAS_TRANSA:
  default:
    __solve_left_forward(B, A, alpha, flags, N, S, E, cache);
    break;
  }
}

void __solve_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                       int flags, int N, int S, int E, int KB, int NB, int MB)
{
  mdata_t Acpy, Bcpy;
  cache_t cache;

  DTYPE Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-S <= 0 || N <= 0)
    return;

  // restrict block sizes as data is copied to aligned buffers of predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};
  
  switch (flags&(GOMAS_UPPER|GOMAS_LOWER|GOMAS_RIGHT|GOMAS_TRANSA)) {
  case GOMAS_RIGHT|GOMAS_UPPER:
  case GOMAS_RIGHT|GOMAS_LOWER|GOMAS_TRANSA:
    __solve_right_forward(B, A, alpha, flags, N, S, E, &cache);
    break;
    
  case GOMAS_RIGHT|GOMAS_LOWER:
  case GOMAS_RIGHT|GOMAS_UPPER|GOMAS_TRANSA:
    __solve_right_backward(B, A, alpha, flags, N, S, E, &cache);
    break;

  case GOMAS_UPPER:
  case GOMAS_LOWER|GOMAS_TRANSA:
    __solve_left_backward(B, A, alpha, flags, N, S, E, &cache);
    break;

  case GOMAS_LOWER:
  case GOMAS_UPPER|GOMAS_TRANSA:
    __solve_left_forward(B, A, alpha, flags, N, S, E, &cache);
    break;
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:

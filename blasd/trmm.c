
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"




/*
 * LEFT-UPPER
 *
 *   B0    A00 | A01 | A02   B0
 *   --   ----------------   --
 *   B1 =   0  | A11 | A12   B1
 *   --   ----------------   --
 *   B2     0  |  0  | A22   B2
 *
 *    B0 = A00*B0 + A01*B1 + A02*B2
 *    B1 = A11*B1 + A12*B2         
 *    B2 = A22*B2                  
 */
static
void __trmm_blk_upper(mdata_t *B, const mdata_t *A, const DTYPE alpha,
                      int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    // off diagonal part
    __subblock(&A0, A, i, i+nI);
    // diagonal block
    __subblock(&A1, A, i, i);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;
      __subblock(&B1, B, i, j);
      __subblock(&B0, B, i+nI, j);
      // update current part with diagonal
      __trmm_blk_recursive(&B1, &A1, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B1, &A0, &B0, alpha, 0, N-i-nI, nJ, nI, cache);
    }
  }
}
/*  LEFT-UPPER-TRANS
 *
 *  B0    A00 | A01 | A02   B0
 *  --   ----------------   --
 *  B1 =   0  | A11 | A12   B1
 *  --   ----------------   --
 *  B2     0  |  0  | A22   B2
 *
 *    B0 = A00*B0
 *    B1 = A01*B0 + A11*B1
 *    B2 = A02*B0 + A12*B1 + A22*B2                  
 */
static
void __trmm_blk_u_trans(mdata_t *B, const mdata_t *A, DTYPE alpha,
                        int flags, int N, int S, int L,  cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    __subblock(&A0, A, 0,    i-nI);
    __subblock(&A1, A, i-nI, i-nI);  // diagonal

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;
      __subblock(&B0, B, 0,    j);
      __subblock(&B1, B, i-nI, j);
      // update current part with diagonal
      __trmm_blk_recursive(&B1, &A1, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B1, &A0, &B0, alpha, GOMAS_TRANSA, i-nI, nJ, nI,
                                      cache); 
    }
  }
}


/*  LEFT-LOWER
 *
 *   B0     A00 |  0  |  0    B0
 *   --    ----------------   --
 *   B1 =   A10 | A11 |  0    B1
 *   --    ----------------   --
 *   B2     A20 | A21 | A22   B2
 *
 *    B0 = A00*B0                  
 *    B1 = A10*B0 + A11*B1         
 *    B2 = A20*B0 + A21*B1 + A22*B2
 */
static
void __trmm_blk_lower(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int L,  cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    __subblock(&A0, A, i-nI, 0);
    __subblock(&A1, A, i-nI, i-nI);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      __subblock(&B0, B, 0,    j);
      __subblock(&B1, B, i-nI, j);
      // update current part with diagonal
      __trmm_blk_recursive(&B1, &A1, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B1, &A0, &B0, alpha, 0, i-nI, nJ, nI, cache); 
    }
  }
}
/*
 *  LEFT-LOWER-TRANSA
 *
 *   B0     A00 |  0  |  0    B0  
 *   --    ----------------   --  
 *   B1  =  A10 | A11 |  0    B1  
 *   --    ----------------   --  
 *   B2     A20 | A21 | A22   B2  
 *
 *    B0 = A00*B0 + A10*B1 + A20*B2
 *    B1 = A11*B1 + A21*B2
 *    B2 = A22*B2
 */
static
void __trmm_blk_l_trans(mdata_t *B, const mdata_t *A,
                        DTYPE alpha, int flags, int N, int S, int L,  cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;

    __subblock(&A0, A, i,    i);
    __subblock(&A1, A, i+nI, i);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      __subblock(&B0, B, i,    j);
      __subblock(&B1, B, i+nI, j);
      
      // update current part with diagonal
      __trmm_blk_recursive(&B0, &A0, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B0, &A1, &B1, alpha, GOMAS_TRANSA,
                                      N-i-nI, nJ, nI, cache); 
    }
  }
}

/*
 *  RIGHT-UPPER
 *
 *                            A00 | A01 | A02 
 *                           ---------------- 
 *   B0|B1|B2  =  B0|B1|B2 *   0  | A11 | A12 
 *                           ---------------- 
 *                             0  |  0  | A22 
 *
 *    B0 = B0*A00                   = trmm_unb(B0, A00)
 *    B1 = B0*A01 + B1*A11          = trmm_unb(B1, A11) + B0*A01
 *    B2 = B0*A02 + B1*A12 + B2*A22 = trmm_unb(B2, A22) + [B0; B1]*[A02; A12].T
 */
static
void __trmm_blk_r_upper(mdata_t *B, const mdata_t *A, DTYPE alpha,
                        int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;
    __subblock(&A0, A, 0,    i-nI);
    __subblock(&A1, A, i-nI, i-nI);
    
    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      __subblock(&B0, B, j, 0);
      __subblock(&B1, B, j, i-nI);
      
      // update current part with diagonal
      __trmm_blk_recursive(&B1, &A1, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B1, &B0, &A0, alpha, 0, i-nI, nI, nJ, cache); 
    }
  }
}

/*
 *  RIGHT-UPPER-TRANS 
 *
 *                            A00 | A01 | A02 
 *                           ---------------- 
 *   B0|B1|B2  =  B0|B1|B2 *   0  | A11 | A12 
 *                           ---------------- 
 *                             0  |  0  | A22 
 *
 *  B0 = B0*A00 + B1*A01 + B2*A02  --> B0 = trmm(B0,A00) + [B1;B2]*[A01;A02].T
 *  B1 = B0*A11 + B2*A12           --> B1 = trmm(B1,A11) + B2*A12.T
 *  B2 = B2*A22                    --> B2 = trmm(B2,A22)
 */
static
void __trmm_blk_ru_trans(mdata_t *B, const mdata_t *A, DTYPE alpha,
                         int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    __subblock(&A0, A, i, i);
    __subblock(&A1, A, i, i+nI);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      __subblock(&B0, B, j, i);
      __subblock(&B1, B, j, i+nI);
      
      // update current part with diagonal; left with diagonal
      __trmm_blk_recursive(&B0, &A0, alpha, flags, nI, 0, nJ, cache);

      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B0, &B1, &A1, alpha, GOMAS_TRANSB,
                                      N-i-nI, nI, nJ, cache); 
    }
  }
}

/*
 * RIGHT-LOWER
 *
 *                             A00 |  0  |  0
 *                            ---------------- 
 *    B0|B1|B2  =  B0|B1|B2 *  A01 | A11 |  0
 *                            ---------------- 
 *                             A02 | A12 | A22 
 *
 *    B0 = B0*A00 + B1*A01 + B2*A02
 *    B1 = B1*A11 + B2*A12
 *    B2 = B2*A22
 */
static
void __trmm_blk_r_lower(mdata_t *B, const mdata_t *A, DTYPE alpha,
                        int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = 0; i < N; i += NB) {
    nI = N - i < NB ? N - i : NB;
    __subblock(&A0, A, i,    i);
    __subblock(&A1, A, i+nI, i);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      __subblock(&B0, B, j, i);
      __subblock(&B1, B, j, i+nI);
      
      // update current part with diagonal; left with diagonal
      __trmm_blk_recursive(&B0, &A0, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B0, &B1, &A1, alpha, 0,
                                      N-i-nI, nI, nJ, cache); 
    }
  }
}

/*
 *  RIGHT-LOWER-TRANSA
 *
 *                             A00 |  0  |  0
 *                            ---------------- 
 *    B0|B1|B2  =  B0|B1|B2 *  A01 | A11 |  0
 *                            ---------------- 
 *                             A02 | A12 | A22 
 *
 *    B0 = B0*A00
 *    B1 = B0*A01 + B1*A11
 *    B2 = B0*A02 + B1*A12 + B2*A22
 */
static
void __trmm_blk_rl_trans(mdata_t *B, const mdata_t *A, DTYPE alpha,
                         int flags, int N, int S, int L, cache_t *cache)
{
  register int i, j, nI, nJ;
  mdata_t A0, A1, B0, B1;
  int NB = cache->NB;

  for (i = N; i > 0; i -= NB) {
    nI = i < NB ? i : NB;

    __subblock(&A0, A, i-nI, 0);
    __subblock(&A1, A, i-nI, i-nI);

    for (j = S; j < L; j += NB) {
      nJ = L - j < NB ? L - j : NB;

      __subblock(&B0, B, j, 0);
      __subblock(&B1, B, j, i-nI);
      
      // update current part with diagonal
      __trmm_blk_recursive(&B1, &A1, alpha, flags, nI, 0, nJ, cache);
      // update current part with rest of the A, B panels
      __gemm_colwise_inner_no_scale(&B1, &B0, &A0, alpha, GOMAS_TRANSB,
                                      i-nI, nI, nJ, cache); 
    }
  }
}


void __trmm_blk(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags,
                int N, int S, int E, int KB, int NB, int MB)
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
  // set to zero in order avoid NaN values later
  memset(Abuf, 0, sizeof(Abuf));
  memset(Bbuf, 0, sizeof(Bbuf));

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

  if (flags & GOMAS_RIGHT) {
    // B = alpha*B*op(A)
    if (flags & GOMAS_UPPER) {
      if (flags & GOMAS_TRANSA) {
        __trmm_blk_ru_trans(B, A, alpha, flags, N, S, E, &cache);
      } else {
        __trmm_blk_r_upper(B, A, alpha, flags, N, S, E, &cache); 
      }
    } else {
      if (flags & GOMAS_TRANSA) {
        __trmm_blk_rl_trans(B, A, alpha, flags, N, S, E, &cache);
      } else {
        __trmm_blk_r_lower(B, A, alpha, flags, N, S, E, &cache); 
      }
    }

  } else {
    // B = alpha*op(A)*B
    if (flags & GOMAS_UPPER) {
      if (flags & GOMAS_TRANSA) {
        __trmm_blk_u_trans(B, A, alpha, flags, N, S, E, &cache); 
      } else {
        __trmm_blk_upper(B, A, alpha, flags, N, S, E, &cache); 
      }
    } else {
      if (flags & GOMAS_TRANSA) {
        __trmm_blk_l_trans(B, A, alpha, flags, N, S, E, &cache);
      } else {
        __trmm_blk_lower(B, A, alpha, flags, N, S, E, &cache); 
      }
    }
  }
}




// Local Variables:
// indent-tabs-mode: nil
// End:


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

/*
 * update diagonal block
 *
 *  l00           a00 a01   b00 b01 b02    u00 u01 u02
 *  l10 l11       a10 a11   b10 b11 b12        u11 u12
 *  l20 l21 l22   a20 a21                          u22
 *
 */
 static
 void __update_trm_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                        DTYPE alpha, DTYPE beta,
                        int flags,  int P, int nC, int nR, cache_t *cache)
{
  register int i, incA, incB, transA, transB;
  mdata_t A0, B0, C0;

  incA = flags & GOMAS_TRANSA ? A->step : 1;
  incB = flags & GOMAS_TRANSB ? 1 : B->step;

  __subblock(&A0, A, 0, 0);
  __subblock(&B0, B, 0, 0);

  if (flags & GOMAS_UPPER) {
    // index by row
    int M = min(nC, nR);
    for (i = 0; i < M; i++) {   
      // scale the target row with beta
      __subblock(&C0, C, i, i);
      __vscale(C0.md, C0.step, beta, nC-i);

      // update one row of C  (nC-i columns, 1 row)
      __gemm_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags,
                                      P, nC-i, 1, cache); 
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  } else {
    // index by column
    int N = min(nC, nR);
    for (i = 0; i < N; i++) {
      __subblock(&C0, C, i, i);
      // scale the target column with beta
      __vscale(C0.md, 1, beta, nR-i);
      // update one column of C  (1 column, nR-i rows)
      __gemm_colwise_inner_no_scale(&C0, &A0, &B0, alpha, flags,
                                      P, 1, nR-i, cache);
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incB; 
    }
  }
}

static
void __update_trm_naive(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        DTYPE alpha, DTYPE beta, int flags,
                        int P, int S, int L, int R, int E, int KB, int NB, int MB)
{
  mdata_t Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-R <= 0 || L-S <=0 || P <= 0) {
    return;
  }

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

  __update_trm_diag(C, A, B, alpha, beta, flags, P, L-S, E-R, &cache);
}

 static
 void __update_upper_recursive(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                               DTYPE alpha, DTYPE beta,
                               int flags,  int P, int N, int M, cache_t *cache)
{
  mdata_t c0, a0, b0;

  // can upper triangular (M == N) or upper trapezoidial (N > M)
  int nb = min(N, M);

  if (nb < MIN_MBLOCK_SIZE) {
    __update_trm_diag(C, A, B, alpha, beta, flags, P, N, M, cache);
    return;
  }

  // upper LEFT diagonal (square block)
  __subblock(&c0, C, 0, 0);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, 0, 0);
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  } else {
    __update_upper_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  }

  // upper RIGHT square
  __subblock(&c0, C, 0, nb/2);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, (flags & GOMAS_TRANSB ? nb/2 : 0), (flags & GOMAS_TRANSB ? 0 : nb/2));
  __gemm_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, nb-nb/2, 0, nb-nb/2, cache);

  // lower RIGHT diagonal
  __subblock(&c0, C, nb/2, nb/2);
  __subblock(&a0, A, (flags & GOMAS_TRANSA ? 0 : nb/2), (flags & GOMAS_TRANSA ? nb/2 : 0));
  __subblock(&b0, B, (flags & GOMAS_TRANSB ? nb/2 : 0), (flags & GOMAS_TRANSB ? 0 : nb/2));
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  } else {
    __update_upper_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  }
  if (M >= N)
    return;
  
  // right trapezoidal part 
  __subblock(&c0, C, 0, nb);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, (flags & GOMAS_TRANSB ? nb : 0), (flags & GOMAS_TRANSB ? 0 : nb));
  __gemm_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, N-nb, 0, nb, cache);
}

 static
 void __update_lower_recursive(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                               DTYPE alpha, DTYPE beta,
                               int flags,  int P, int N, int M, cache_t *cache)
{
  mdata_t c0, a0, b0;

  // can be lower triangular (M == N) or lower trapezoidial (M > N)
  int nb = min(M, N);

  //printf("__update_lower_rec: M=%d, N=%d, nb=%d\n", M, N, nb);
  if (nb < MIN_MBLOCK_SIZE) {
    __update_trm_diag(C, A, B, alpha, beta, flags, P, N, M, cache);
    return;
  }

  // upper LEFT diagonal
  __subblock(&c0, C, 0, 0);
  __subblock(&a0, A, 0, 0);
  __subblock(&b0, B, 0, 0);
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  } else {
    __update_lower_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb/2, nb/2, cache);
  }

  // lower LEFT square
  __subblock(&c0, C, nb/2, 0);
  __subblock(&a0, A, (flags & GOMAS_TRANSA ? 0 : nb/2), (flags & GOMAS_TRANSA ? nb/2 : 0));
  __subblock(&b0, B, 0,    0);
  __gemm_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, nb-nb/2, 0, nb-nb/2, cache);

  // lower RIGHT diagonal
  __subblock(&c0, C, nb/2, nb/2);
  __subblock(&a0, A, (flags & GOMAS_TRANSA ? 0 : nb/2), (flags & GOMAS_TRANSA ? nb/2 : 0));
  __subblock(&b0, B, (flags & GOMAS_TRANSB ? nb/2 : 0), (flags & GOMAS_TRANSB ? 0 : nb/2));
  //__subblock(&a0, A, nb/2, 0);
  //__subblock(&b0, B, 0,  nb/2);
  if (nb/2 < MIN_MBLOCK_SIZE) {
    __update_trm_diag(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  } else {
    __update_lower_recursive(&c0, &a0, &b0, alpha, beta, flags, P, nb-nb/2, nb-nb/2, cache);
  }
  if (M <= N)
    return;

  printf("__update_lower_rec: M > N: M-nb=%d rows \n", M-nb);
  // lower trapezoidial part
  __subblock(&c0, C, nb, 0);
  __subblock(&a0, A, (flags & GOMAS_TRANSA ? 0 : nb), (flags & GOMAS_TRANSA ? nb : 0));
  __subblock(&b0, B, 0,  0);
  __gemm_colwise_inner_scale_c(&c0, &a0, &b0, alpha, beta, flags,
                                 P, 0, nb, 0, M-nb, cache);
}

static
void __update_trm_recursive(mdata_t *C, const mdata_t *A, const mdata_t *B,
                            DTYPE alpha, DTYPE beta, int flags,
                            int P, int S, int L, int R, int E, int KB, int NB, int MB)
{
  mdata_t Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  //printf("__update_trm_rec: S=%d, L=%d, R=%d, E=%d\n", S, L, R, E);

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

  if (flags & GOMAS_UPPER) {
    __update_upper_recursive(C, A, B, alpha, beta, flags, P, L-S, E-R, &cache);
  } else {
    __update_lower_recursive(C, A, B, alpha, beta, flags, P, L-S, E-R, &cache);
  }
}

/*
 * Generic triangular matrix update:
 *      C = beta*op(C) + alpha*A*B
 *      C = beta*op(C) + alpha*A*B.T
 *      C = beta*op(C) + alpha*A.T*B
 *      C = beta*op(C) + alpha*A.T*B.T
 *
 * Some conditions on parameters that define the updated block:
 * 1. S == R && E == L 
 *    matrix is triangular square matrix
 * 2. S == R && L >  E
 *    matrix is trapezoidial with upper trapezoidial part right of triangular part
 * 3. S == R && L <  E
 *    matrix is trapezoidial with lower trapezoidial part below triangular part
 * 4. S != R && S >  E
 *    update is only to upper trapezoidial part right of triangular block
 * 5. S != R && R >  L
 *    update is only to lower trapezoidial part below triangular block
 * 6. S != R
 *    inconsistent update block spefication, will not do anything
 *            
 */
void __update_trm_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                      DTYPE alpha, DTYPE beta, int flags,
                      int P, int S, int L, int R, int E, int KB, int NB, int MB)
{
  register int i, j, nI, ar, ac, br, bc, N, M;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  cache_t cache;
  double Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-R <= 0 || L-S <= 0 || P <= 0) {
    return;
  }

  //printf("__update_trm_blk: S=%d, L=%d, R=%d, E=%d\n", S, L, R, E);

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

  if ( S != R && (S <= E || R <= L)) {
    // inconsistent update configuration
    return;
  }
  if (flags & GOMAS_UPPER) {
    // by rows; M is the last row; L-S is column count; implicitely S == R
    M = min(L, E);
    for (i = R; i < M; i += NB) {
      nI = M - i < NB ? M - i : NB;
    
      // 1. update block on diagonal (square block)
      br = flags & GOMAS_TRANSB ? i : 0;
      bc = flags & GOMAS_TRANSB ? 0 : i;
      ar = flags & GOMAS_TRANSA ? 0 : i;
      ac = flags & GOMAS_TRANSA ? i : 0;

      //printf("i=%dm nI=%d, L-i=%d, L-i-nI=%d\n", i, nI, L-i, L-i-nI);
      __subblock(&Cd, C, i,  i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __update_upper_recursive(&Cd, &Ad, &Bd, alpha, beta, flags, P,
                               nI, nI, &cache);

      // 2. update right of the diagonal block (rectangle, nI rows)
      br = flags & GOMAS_TRANSB ? i+nI : 0;
      bc = flags & GOMAS_TRANSB ? 0    : i+nI;
      ar = flags & GOMAS_TRANSA ? 0    : i;
      ac = flags & GOMAS_TRANSA ? i    : 0;

      __subblock(&Cd, C, i,  i+nI);
      __subblock(&Ad, A, ar, ac);
      __subblock(&Bd, B, br, bc);
      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, flags,
                                     P, 0, L-i-nI, 0, nI, &cache);

    }
   } else {
    // by columns; N is the last column, E-R is row count;
    N = min(L, E);
    for (i = S; i < N; i += NB) {
      nI = N - i < NB ? N - i : NB;
    
      // 1. update on diagonal (square block)
      br = flags & GOMAS_TRANSB ? i : 0;
      bc = flags & GOMAS_TRANSB ? 0 : i;
      ar = flags & GOMAS_TRANSA ? 0 : i;
      ac = flags & GOMAS_TRANSA ? i : 0;
      __subblock(&Cd, C, i, i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __update_lower_recursive(&Cd, &Ad, &Bd, alpha, beta, flags,
                               P, nI, nI, &cache);

      // 2. update block below the diagonal block (rectangle, nI columns)
      br = flags & GOMAS_TRANSB ? i    : 0;
      bc = flags & GOMAS_TRANSB ? 0    : i;
      ar = flags & GOMAS_TRANSA ? 0    : i+nI;
      ac = flags & GOMAS_TRANSA ? i+nI : 0;
      __subblock(&Cd, C, i+nI,  i);
      __subblock(&Bd, B, br, bc);
      __subblock(&Ad, A, ar, ac);
      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, flags,
                                     P, 0, nI, 0, E-i-nI, &cache);
    }
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:

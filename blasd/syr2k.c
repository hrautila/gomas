
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include "dtype.h"
#include "interfaces.h"


// SYR2K;
//   C = alpha*A*B.T + alpha*B*A.T + beta*C  
//   C = alpha*A.T*B + alpha*B.T*A + beta*C  if flags & GOMAS_TRANS
/*
 * Symmetric rank2 update
 *
 * upper
 *   C00 C01 C02    C00  C01 C02     A0               B0           
 *    0  C11 C12 =   0   C11 C12  +  A1 * B0 B1 B2 +  B1 * A0 A1 A2
 *    0   0  C22     0   0   C22     A2               B2           
 *
 * lower:
 *   C00  0   0    C00   0   0      A0               B0           
 *   C10 C11  0 =  C10  C11  0   +  A1 * B0 B1 B2 +  B1 * A0 A1 A2
 *   C20 C21 C22   C20  C21 C22     A2               B2           
 *
 */
void __rank2_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                 DTYPE alpha, DTYPE beta, int flags,
                 int P, int S, int E,  int KB, int NB, int MB)
{
  register int i, j, nI, nC;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-S <= 0 || P <= 0)
    return;

  if (NB > E-S) {
    NB = E-S;
  }

  Acpy = (mdata_t){Abuf, MAX_KB};
  Bcpy = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Acpy, &Bcpy, KB, NB, MB};

  if (flags & GOMAS_TRANSA || flags & GOMAS_TRANS) {
    //   C = alpha*A.T*B + alpha*B.T*A + beta*C 
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);
      __subblock(&Bd, B, 0, i);
      __rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, 0, i);
      if (flags & GOMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, B, 0, S);
        nC = E - i - nI;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Bd, B, 0, i+nI);
        nC = i;
      }
      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, GOMAS_TRANSA,
                                     P, 0, nC, 0, nI, &cache); 

      // 2nd part
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);
      __subblock(&Bd, B, 0, i);
      __rank_diag(&Cd, &Bd, &Ad, alpha, 1.0, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Bd, B, 0, i);
      if (flags & GOMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Ad, A, 0, S);
        nC = E - i - nI;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Ad, A, 0, i+nI);
        nC = i;
      }
      __gemm_colwise_inner_scale_c(&Cd, &Bd, &Ad, alpha, 1.0, GOMAS_TRANSA,
                                     P, 0, nC, 0, nI, &cache);

    }
  } else {
    //   C = alpha*A*B.T + alpha*B*A.T + beta*C  
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __subblock(&Bd, B, i, 0);
      __rank_diag(&Cd, &Ad, &Bd, alpha, beta, flags, P, nI, &cache); 

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, i, 0);
      if (flags & GOMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, B, S, 0);
        nC = i;
      } else {
        __subblock(&Cd, C, i,    i+nI);
        __subblock(&Bd, B, i+nI, 0);
        nC = E - i - nI;
      }

      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, GOMAS_TRANSB,
                                     P, 0, nC, 0, nI, &cache);

      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, B, i, 0);
      __subblock(&Bd, A, i, 0);
      __rank_diag(&Cd, &Ad, &Bd, alpha, 1.0, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, B, i, 0);
      if (flags & GOMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, A, S, 0);
        nC = i;
      } else {
        __subblock(&Cd, C, i,    i+nI);
        __subblock(&Bd, A, i+nI, 0);
        nC = E - i - nI;
      }

      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, 1.0, GOMAS_TRANSB,
                                   P, 0, nC, 0, nI, &cache);

    }
  }
}

// Local Variables:
// indent-tabs-mode: nil
// End:


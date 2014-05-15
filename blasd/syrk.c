
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#include "dtype.h"
#include "interfaces.h"
#include "mvec_nosimd.h"
#include "print.h"


void
__rank_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
              DTYPE alpha, DTYPE beta,  int flags,  int P, int nC, cache_t *cache)
{
  register int i, incA, incB, trans;
  //mdata_t C0 = {C->md, C->step};
  mdata_t A0 = {A->md, A->step};
  mdata_t B0 = {B->md, B->step};

  incA = flags & GOMAS_TRANSA ? A->step : 1;
  incB = flags & GOMAS_TRANSA ? 1 : B->step;
  trans = flags & GOMAS_TRANSA ? GOMAS_TRANSA : GOMAS_TRANSB;

  //printf("__rank_diag: P=%d, nC=%d, incA=%d, incB=%d\n", P, nC, incA, incB);
  if (flags & GOMAS_UPPER) {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      __vscale(C->md, C->step, beta, nC-i);
      // update one row of C  (nC-i columns, 1 row)
      __gemm_colwise_inner_no_scale(C, &A0, &B0, alpha, trans,
                                    P, nC-i, 1, cache); 
      // move along the diagonal to next row of C
      C->md += C->step + 1;
      // move A to next row
      A0.md += incA;
      // move B to next column
      B0.md += incA; 
    }
  } else {
    for (i = 0; i < nC; i++) {
      // scale the target row with beta
      __vscale(C->md, C->step, beta, i+1);
      // update one row of C  (nC-i columns, 1 row)
      __gemm_colwise_inner_no_scale(C, &A0, &B0, alpha, trans,
                                      P, i+1, 1, cache); 
      // move to next row of C
      C->md ++;
      // move A to next row
      A0.md += incA;
      //__blk_print(&C0, nC, nC, "C0", "%8.1e");
    }
  }
}

/*
 * Symmetric rank update
 *
 * upper
 *   C00 C01 C02    C00  C01 C02     A0  
 *    0  C11 C12 =   0   C11 C12  +  A1 * B0 B1 B2
 *    0   0  C22     0   0   C22     A2
 *
 * lower:
 *   C00  0   0    C00   0   0      A0  
 *   C10 C11  0 =  C10  C11  0   +  A1 * B0 B1 B2
 *   C20 C21 C22   C20  C21 C22     A2

 */
void __rank_blk(mdata_t *C, const mdata_t *A,
                DTYPE alpha, DTYPE beta,
                int flags,  int P, int S, int E, int KB, int NB, int MB)
{
  register int i, j, nI, nC;
  mdata_t Cd, Ad, Bd;
  mdata_t Acpy, Bcpy;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  if (E-S <= 0 || P <= 0)
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

  if ((flags & GOMAS_TRANSA) || (flags & GOMAS_TRANS)) {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, 0, i);

      // 1. update on diagonal
      __rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, 0, i);
      if (flags & GOMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, A, 0, S);
        nC = i;
      } else {
        __subblock(&Cd, C, i, i+nI);
        __subblock(&Bd, A, 0, i+nI);
        nC = E - i - nI;
      }

      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, GOMAS_TRANSA,
                                   P, 0, nC, 0, nI, &cache); 
    }
  } else {
    for (i = S; i < E; i += NB) {
      nI = E - i < NB ? E - i : NB;
    
      // 1. update on diagonal
      __subblock(&Cd, C, i, i);
      __subblock(&Ad, A, i, 0);
      __rank_diag(&Cd, &Ad, &Ad, alpha, beta, flags, P, nI, &cache);

      // 2. update block right of diagonal (UPPER) or left of diagonal (LOWER)
      __subblock(&Ad, A, i, 0);
      if (flags & GOMAS_LOWER) {
        __subblock(&Cd, C, i, 0);
        __subblock(&Bd, A, S, 0);
        nC = i;
      } else {
        __subblock(&Cd, C, i,    i+nI);
        __subblock(&Bd, A, i+nI, 0);
        nC = E - i - nI;
      }

      __gemm_colwise_inner_scale_c(&Cd, &Ad, &Bd, alpha, beta, GOMAS_TRANSB,
                                   P, 0, nC, 0, nI, &cache);
    }
  }
}



// Local Variables:
// indent-tabs-mode: nil
// End:


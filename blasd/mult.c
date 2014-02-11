
// Copyright (c) Harri Rautila, 2012,2013

// This file is part of github.com/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "interfaces.h"
#include "matcpy.h"

/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
#if defined(__AVX__) //&& defined(AVX_F64)
#include "mult_avx_f64.h"

#elif defined(__SSE__) //&& defined(SSE_F64)
#include "mult_sse_f64.h"

#else
#include "mult_nosimd.h"
#endif


/* ---------------------------------------------------------------------------------
 * 
 */

#include "kernel.h"


// update C block defined by nR rows, nJ columns, nP is A, B common dimension
// A, B data arranged for DOT operations, A matrix is the inner matrix block
// and is looped over nJ times
void __gemm_colblk_inner(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                         DTYPE alpha, int nJ, int nR, int nP)
{
  register int j;

  for (j = 0; j < nJ-3; j += 4) {
    __CMULT4(Cblk, Ablk, Bblk, alpha, j, nR, nP);
  }
  if (j == nJ)
    return;
  // the uneven column stripping part ....
  if (j < nJ-1) {
    __CMULT2(Cblk, Ablk, Bblk, alpha, j, nR, nP);
    j += 2;
  }
  if (j < nJ) {
    __CMULT1(Cblk, Ablk, Bblk, alpha, j, nR, nP);
    j++;
  }
}

// update block of C with A and B panels; A panel is nR*P, B panel is P*nSL
// C block is nR*nSL
void __gemm_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                   DTYPE alpha, int flags,
                                   int P, int nSL, int nRE,  cache_t *cache)
{
  int i, j, k, ip, jp, kp, nP, nI, nJ;
  mdata_t Ca;
  Ca.step = C->step;

  if (nRE <= 0 || nSL <= 0)
    return;

  for (jp = 0; jp < nSL; jp += cache->NB) {
    // in panels of N columns of C, B
    nJ = min(cache->NB, nSL-jp);

    for (kp = 0; kp < P; kp += cache->KB) {
      nP = min(cache->KB, P-kp);
    
      if (flags & GOMAS_TRANSB) {
        __CPTRANS(cache->Bcpy->md, cache->Bcpy->step, &B->md[jp+kp*B->step], B->step, nJ, nP);
      } else {
        __CP(cache->Bcpy->md, cache->Bcpy->step, &B->md[kp+jp*B->step], B->step, nP, nJ);
      }
	  
      for (ip = 0; ip < nRE; ip += cache->MB) {
        nI = min(cache->MB, nRE-ip);
        if (flags & GOMAS_TRANSA) {
          __CP(cache->Acpy->md, cache->Acpy->step, &A->md[kp+ip*A->step], A->step, nP, nI);
        } else {
          __CPTRANS(cache->Acpy->md, cache->Acpy->step, &A->md[ip+kp*A->step], A->step, nI, nP);
        }
        Ca.md = &C->md[ip+jp*C->step];
      
        for (j = 0; j < nJ-3; j += 4) {
          __CMULT4(&Ca, cache->Acpy, cache->Bcpy, alpha, j, nI, nP);
        }
        if (j == nJ)
          continue;
        // the uneven column stripping part ....
        if (j < nJ-1) {
          __CMULT2(&Ca, cache->Acpy, cache->Bcpy, alpha, j, nI, nP);
          j += 2;
        }
        if (j < nJ) {
          __CMULT1(&Ca, cache->Acpy, cache->Bcpy, alpha, j, nI, nP);
          j++;
        }
      }
    }
  }
}


void __gemm_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                  DTYPE alpha, DTYPE beta, int flags,
                                  int P, int S, int L, int R, int E, cache_t *cache)
{
  int i, j, k, ip, jp, kp, nP, nI, nJ;
  mdata_t Ca;
  int KB, NB, MB;
  mdata_t *Acpy, *Bcpy;

  if (L-S <= 0 || E-R <= 0 || P <= 0)
    return;

  Ca.step = C->step;
  KB = cache->KB; NB = cache->NB; MB = cache->MB;
  Acpy = cache->Acpy;
  Bcpy = cache->Bcpy;

  // loop over columns of C
  for (jp = S; jp < L; jp += NB) {
    // in panels of N columns of C, B
    nJ = min(NB, L-jp);
	
    // scale C block here.... jp, jp+nJ columns, E-R rows
    __subblock(&Ca, C, R, jp);
    __blk_scale(&Ca, beta, E-R, nJ);

    for (kp = 0; kp < P; kp += KB) {
      nP = min(KB, P-kp);

      if (flags & GOMAS_TRANSB) {
        __CPTRANS(Bcpy->md, Bcpy->step, &B->md[jp+kp*B->step], B->step, nJ, nP);
      } else {
        __CP(Bcpy->md, Bcpy->step, &B->md[kp+jp*B->step], B->step, nP, nJ);
      }
	  
      for (ip = R; ip < E; ip += MB) {
        nI = min(MB, E-ip);
        if (flags & GOMAS_TRANSA) {
          __CP(Acpy->md, Acpy->step, &A->md[kp+ip*A->step], A->step, nP, nI);
        } else {
          __CPTRANS(Acpy->md, Acpy->step, &A->md[ip+kp*A->step], A->step, nI, nP);
        }

        __subblock(&Ca, C, ip, jp);
        for (j = 0; j < nJ-3; j += 4) {
          __CMULT4(&Ca, Acpy, Bcpy, alpha, j, min(MB, E-ip), nP);
        }
        if (j == nJ)
          continue;
        // the uneven column stripping part ....
        if (j < nJ-1) {
          __CMULT2(&Ca, Acpy, Bcpy, alpha, j, min(MB, E-ip), nP);
          j += 2;
        }
        if (j < nJ) {
          __CMULT1(&Ca, Acpy, Bcpy, alpha, j, min(MB, E-ip), nP);
          j++;
        }
      }
    }
  }
}


void __gemm_inner(mdata_t *C, const mdata_t *A, const mdata_t *B,
                  DTYPE alpha, DTYPE beta, int flags,
                  int P, int S, int L, int R, int E, 
                  int KB, int NB, int MB)
{
  mdata_t Aa, Ba;
  cache_t cache;
  DTYPE Abuf[MAX_KB*MAX_MB], Bbuf[MAX_KB*MAX_NB] __attribute__((aligned(64)));

  Aa.md = Abuf;
  Aa.step = MAX_KB;
  Ba.md = Bbuf;
  Ba.step = MAX_KB;

  if (L-S <= 0 || E-R <= 0) {
    // nothing to do, zero columns or rows
    return;
  }

  // restrict block sizes as data is copied to aligned buffers of
  // predefined max sizes.
  if (NB > MAX_NB || NB <= 0) {
    NB = MAX_NB;
  }
  if (MB > MAX_MB || MB <= 0) {
    MB = MAX_MB;
  }
  if (KB  > MAX_KB || KB <= 0) {
    KB = MAX_KB;
  }

  if (alpha == 0.0) {
    __subblock(&Aa, C, R, S);
    __blk_scale(&Aa, beta, E-R, L-S);
    return;
  }

  Aa = (mdata_t){Abuf, MAX_KB};
  Ba = (mdata_t){Bbuf, MAX_KB};
  cache = (cache_t){&Aa, &Ba, KB, NB, MB};

  // update C using A as inner most matrix
  __gemm_colwise_inner_scale_c(C, A, B, alpha, beta, flags,
                               P, S, L, R, E, &cache); 
}

  
// Local Variables:
// indent-tabs-mode: nil
// End:

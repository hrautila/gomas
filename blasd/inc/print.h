
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef __PRINT_H
#define __PRINT_H 1

#include <stdio.h>

static inline
void __tile_print(const DTYPE *d, int ldD, int nR, int nC, const char *s, const char *efmt)
{
  int i, j;

  if (s)
    printf("%s\n", s);
  if (!efmt)
    efmt = __DATA_FORMAT;

  for (i = 0; i < nR; i++) {
    printf("[");
    for (j = 0; j < nC; j++) {
      if (j > 0)
        printf(", ");
      printf(efmt, __PRINTABLE(d[j*ldD+i]));
    }
    printf("]\n");
  }
  printf("\n");
}

static inline
void __blk_print(const mdata_t *A, int nR, int nC, const char *s, const char *efmt)
{
  __tile_print(A->md, A->step, nR, nC, s, efmt);
}
  
static inline
void __vec_print(const mvec_t *X, int N, const char *s, const char *efmt)
{
  register int i;
  if (s)
    printf("%s\n", s);
  if (!efmt)
    efmt = __DATA_FORMAT;
  for (i = 0; i < N; i++) {
    printf("[");
    printf(efmt, __PRINTABLE(X->md[i*X->inc]));
    printf("]\n");
  }
}

#endif

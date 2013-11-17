
// Copyright (c) Harri Rautila, 2012

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#include <stdio.h>

#include "dtype.h"
#include "interfaces.h"


// Scale a tile of M rows by N columns with leading index ldX.
void __blk_scale(mdata_t *A, DTYPE beta, int M, int N)
{
  register int i, j;
  DTYPE *X = A->md;
  int  ldX = A->step;
  if (beta == 1.0) {
    return;
  }

  // set to zero
  if (beta == 0.0) {
    for (j = 0; j < N-3; j += 4) {
      for (i = 0; i < M; i++) {
        X[i+(j+0)*ldX] = 0.0;
        X[i+(j+1)*ldX] = 0.0;
        X[i+(j+2)*ldX] = 0.0;
        X[i+(j+3)*ldX] = 0.0;
      }
    }
    if (j == N) 
      return;
    for (; j < N; j++) {
      for (i = 0; i < M; i++) {
        X[i+(j+0)*ldX] = 0.0;
      }
    }
    return;
  }
  // scale here
  for (j = 0; j < N-3; j += 4) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] *= beta;
      X[i+(j+1)*ldX] *= beta;
      X[i+(j+2)*ldX] *= beta;
      X[i+(j+3)*ldX] *= beta;
    }
  }
  if (j == N) 
    return;
  for (; j < N; j++) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] *= beta;
    }
  }
}

void __blk_invscale(mdata_t *A, DTYPE beta, int M, int N)
{
  register int i, j;
  DTYPE *X = A->md;
  int  ldX = A->step;
  if (beta == 1.0 || beta == 0.0) {
    return;
  }

  // scale here
  for (j = 0; j < N-3; j += 4) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] /= beta;
      X[i+(j+1)*ldX] /= beta;
      X[i+(j+2)*ldX] /= beta;
      X[i+(j+3)*ldX] /= beta;
    }
  }
  if (j == N) 
    return;
  for (; j < N; j++) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] /= beta;
    }
  }
}

void __blk_add(mdata_t *A, DTYPE beta, int M, int N)
{
  register int i, j;
  DTYPE *X = A->md;
  int ldX = A->step;

  if (beta == 0.0) {
    return;
  }

  for (j = 0; j < N-3; j += 4) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] += beta;
      X[i+(j+1)*ldX] += beta;
      X[i+(j+2)*ldX] += beta;
      X[i+(j+3)*ldX] += beta;
    }
  }
  if (j == N) 
    return;
  for (; j < N; j++) {
    for (i = 0; i < M; i++) {
      X[i+(j+0)*ldX] += beta;
    }
  }
}


// Local Variables:
// indent-tabs-mode: nil
// End:

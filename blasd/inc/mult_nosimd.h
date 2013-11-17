
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef _MULT_NOSIMD_H
#define _MULT_NOSIMD_H 1

// update single element of C; (mult1x1x4)
static inline
void __mult1c1(DTYPE *c, const DTYPE *a, const DTYPE *b, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3;
  y0 = 0.0;
  y1 = y0; y2 = y0; y3 = y0;
  for (k = 0; k < nR-3; k += 4) {
    y0 += a[k]*b[k];
    y1 += a[k+1]*b[k+1];
    y2 += a[k+2]*b[k+2];
    y3 += a[k+3]*b[k+3];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    y0 += a[k]*b[k];
    k++;
  case 2:
    y1 += a[k]*b[k];
    k++;
  case 1:
    y2 += a[k]*b[k];
    break;
  }
update:
  y0 += y1; y2 += y3; y0 += y2;
  c[0] += y0*alpha;
}


// update 1x2 block of C (mult2x1x4)
static inline
void __mult1c2(DTYPE *c0, DTYPE *c1,
               const DTYPE *a,
               const DTYPE *b0, const DTYPE *b1, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 =  y2 = y3 = y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR-3; k += 4) {
    y0 += a[k+0]*b0[k+0];
    y1 += a[k+0]*b1[k+0];
    y2 += a[k+1]*b0[k+1];
    y3 += a[k+1]*b1[k+1];
    y4 += a[k+2]*b0[k+2];
    y5 += a[k+2]*b1[k+2];
    y6 += a[k+3]*b0[k+3];
    y7 += a[k+3]*b1[k+3];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    y0 += a[k]*b0[k];
    y1 += a[k]*b1[k];
    k++;
  case 2:
    y2 += a[k]*b0[k];
    y3 += a[k]*b1[k];
    k++;
  case 1:
    y4 += a[k]*b0[k];
    y5 += a[k]*b1[k];
    k++;
  }
update:
  y0 += y2 + y4 + y6;
  y1 += y3 + y5 + y7;
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
}

// update 2x2 block of C (mult2x2x2)
static inline
void __mult2c2(DTYPE *c0, DTYPE *c1, 
               const DTYPE *a0, const DTYPE *a1,
               const DTYPE *b0, const DTYPE *b1,
               DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 = y2 = y3 = 0.0;
  y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR-1; k += 2) {
    y0 += a0[k+0]*b0[k+0];
    y1 += a0[k+0]*b1[k+0];
    y2 += a0[k+1]*b0[k+1];
    y3 += a0[k+1]*b1[k+1];
    y4 += a1[k+0]*b0[k+0];
    y5 += a1[k+0]*b1[k+0];
    y6 += a1[k+1]*b0[k+1];
    y7 += a1[k+1]*b1[k+1];
  }
  if (k == nR)
    goto update;

  y0 += a0[k+0]*b0[k+0];
  y1 += a0[k+0]*b1[k+0];
  y4 += a1[k+0]*b0[k+0];
  y5 += a1[k+0]*b1[k+0];
  k++;

update:
  c0[0] += (y0 + y2)*alpha;
  c1[0] += (y1 + y3)*alpha;
  c0[1] += (y4 + y6)*alpha;
  c1[1] += (y5 + y7)*alpha;
}



// update 1x4 block of C; (mult4x1x1)
static inline
void __mult1c4(DTYPE *c0, DTYPE *c1, DTYPE *c2, DTYPE *c3,
               const DTYPE *a,
               const DTYPE *b0, const DTYPE *b1,
               const DTYPE *b2, const DTYPE *b3, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += a[k]*b0[k];
    y1 += a[k]*b1[k];
    y2 += a[k]*b2[k];
    y3 += a[k]*b3[k];
  }
update:
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
  c2[0] += y2*alpha;
  c3[0] += y3*alpha;
}

// update 2x4 block of C; (mult4x2x1)
static inline
void __mult2c4(DTYPE *c0, DTYPE *c1, DTYPE *c2, DTYPE *c3,
               const DTYPE *a0, const DTYPE *a1,
               const DTYPE *b0, const DTYPE *b1,
               const DTYPE *b2, const DTYPE *b3, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 = y2 = y3 = 0.0;
  y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += a0[k]*b0[k];
    y1 += a0[k]*b1[k];
    y2 += a0[k]*b2[k];
    y3 += a0[k]*b3[k];
    y4 += a1[k]*b0[k];
    y5 += a1[k]*b1[k];
    y6 += a1[k]*b2[k];
    y7 += a1[k]*b3[k];
  }
update:
  c0[0] += y0*alpha;
  c1[0] += y1*alpha;
  c2[0] += y2*alpha;
  c3[0] += y3*alpha;
  c0[1] += y4*alpha;
  c1[1] += y5*alpha;
  c2[1] += y6*alpha;
  c3[1] += y7*alpha;
}



// update 4x1 block of C; (dmult4x1x1)
static inline
void __mult4c1(DTYPE *c0, 
               const DTYPE *a0, const DTYPE *a1, const DTYPE *a2, const DTYPE *a3,
               const DTYPE *b0, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += b0[k]*a0[k];
    y1 += b0[k]*a1[k];
    y2 += b0[k]*a2[k];
    y3 += b0[k]*a3[k];
  }
update:
  c0[0] += y0*alpha;
  c0[1] += y1*alpha;
  c0[2] += y2*alpha;
  c0[3] += y3*alpha;
}


// update 4x2 block of C; (mult4x2x1)
static inline
void __mult4c2(DTYPE *c0, DTYPE *c1,
               const DTYPE *a0, const DTYPE *a1, const DTYPE *a2, const DTYPE *a3,
               const DTYPE *b0, const DTYPE *b1, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = y1 = y2 = y3 = 0.0;
  y4 = y5 = y6 = y7 = 0.0;
  for (k = 0; k < nR; k += 1) {
    y0 += b0[k]*a0[k];
    y1 += b0[k]*a1[k];
    y2 += b0[k]*a2[k];
    y3 += b0[k]*a3[k];
    y4 += b1[k]*a0[k];
    y5 += b1[k]*a1[k];
    y6 += b1[k]*a2[k];
    y6 += b1[k]*a3[k];

  }
update:
  c0[0] += y0*alpha;
  c0[1] += y1*alpha;
  c0[2] += y2*alpha;
  c0[3] += y3*alpha;
  c1[0] += y4*alpha;
  c1[1] += y5*alpha;
  c1[2] += y6*alpha;
  c1[3] += y7*alpha;
}


// update 2x1 block of C; (dmult2x1x2)
static inline
void __mult2c1(DTYPE *c0, 
               const DTYPE *a0, const DTYPE *a1, 
               const DTYPE *b0, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE y0, y1, y2, y3;
  y0 = y1 = y2 = y3 = 0.0;
  for (k = 0; k < nR-1; k += 2) {
    y0 += b0[k+0]*a0[k+0];
    y1 += b0[k+0]*a1[k+0];
    y2 += b0[k+1]*a0[k+1];
    y3 += b0[k+1]*a0[k+1];
  }
  if (k == nR)
    goto update;

  y0 += b0[k]*a0[k];
  y1 += b0[k]*a1[k];

update:
  c0[0] += (y0 + y2)*alpha;
  c0[1] += (y1 + y3)*alpha;
}

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:


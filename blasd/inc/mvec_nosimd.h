
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef __ARMAS_MVEC_NOSIMD_H
#define __ARMAS_MVEC_NOSIMD_H 1

static inline
void __vscale(DTYPE *X,  int incx, DTYPE alpha, int N)
{
  register int i, k;
  register DTYPE f0, f1, f2, f3;
  register DTYPE *x0;

  // gcc compiles loop body to use 4 different XMM result registers
  // and results to 4 independent computations.
  for (i = 0; i < N-3; i += 4) {
    f0 =  X[(i+0)*incx] * alpha;
    f1 =  X[(i+1)*incx] * alpha;
    f2 =  X[(i+2)*incx] * alpha;
    f3 =  X[(i+3)*incx] * alpha;
    X[(i+0)*incx] = f0;
    X[(i+1)*incx] = f1;
    X[(i+2)*incx] = f2;
    X[(i+3)*incx] = f3;
  }    
  if (i == N)
    return;

  // do the index calculations only once.
  x0 = &X[i*incx];
  k = 0;
  switch(N-i) {
  case 3:
    x0[k] *= alpha;
    k += incx;
  case 2:
    x0[k] *= alpha;
    k += incx;
  case 1:
    x0[k] *= alpha;
  }
}


static inline
void __vmult1axpy(DTYPE *y, int incy,
                 const DTYPE *a0, const DTYPE *x, int incx,
                 DTYPE alpha, int nR)
{
  register int k;
  register DTYPE cf;

  cf = alpha*x[0];

  for (k = 0; k < nR-3; k += 4) {
    y[(k+0)*incy] += a0[k+0]*cf;
    y[(k+1)*incy] += a0[k+1]*cf;
    y[(k+2)*incy] += a0[k+2]*cf;
    y[(k+3)*incy] += a0[k+3]*cf;
  }
  if (k == nR)
    return;

  switch (nR-k) {
  case 3:
    y[(k+0)*incy] += a0[k+0]*cf;
    k++;
  case 2:
    y[(k+0)*incy] += a0[k+0]*cf;
    k++;
  case 1:
    y[(k+0)*incy] += a0[k+0]*cf;
  }
}

static inline
void __vmult1ysubax(DTYPE *y, int incy,
                    const DTYPE *a0, DTYPE alpha, int nR)
{
  register int k;
  register DTYPE cf;

  cf = alpha;

  for (k = 0; k < nR-3; k += 4) {
    y[(k+0)*incy] -= a0[k+0]*cf;
    y[(k+1)*incy] -= a0[k+1]*cf;
    y[(k+2)*incy] -= a0[k+2]*cf;
    y[(k+3)*incy] -= a0[k+3]*cf;
  }
  if (k == nR)
    return;

  switch (nR-k) {
  case 3:
    y[(k+0)*incy] -= a0[k+0]*cf;
    k++;
  case 2:
    y[(k+0)*incy] -= a0[k+0]*cf;
    k++;
  case 1:
    y[(k+0)*incy] -= a0[k+0]*cf;
  }
}



static inline
void __vmult2axpy(DTYPE *y, int incy,
                 const DTYPE *a0, const DTYPE *a1, const DTYPE *x, int incx,
                 DTYPE alpha, int nR)
{
  register int k;
  register DTYPE cf0, cf1, t0, t1, t2, t3, t4, t5, t6, t7;

  cf0 = alpha*x[0];
  cf1 = alpha*x[incx];

  for (k = 0; k < nR-3; k += 4) {
    t0 = a0[k+0]*cf0;
    t1 = a0[k+1]*cf0;
    t2 = a0[k+2]*cf0;
    t3 = a0[k+3]*cf0;
    t4 = a1[k+0]*cf1;
    t5 = a1[k+1]*cf1;
    t6 = a1[k+2]*cf1;
    t7 = a1[k+3]*cf1;
    y[(k+0)*incy] += t0 + t4;
    y[(k+1)*incy] += t1 + t5;
    y[(k+2)*incy] += t2 + t6;
    y[(k+3)*incy] += t3 + t7;
  }
  if (k == nR)
    return;

  switch (nR-k) {
  case 3:
    y[k*incy] += a0[k]*cf0 + a1[k]*cf1;
    k++;
  case 2:
    y[k*incy] += a0[k]*cf0 + a1[k]*cf1;
    k++;
  case 1:
    y[k*incy] += a0[k]*cf0 + a1[k]*cf1;
  }
}


static inline
void __vmult4axpy(DTYPE *y, int incy,
                 const DTYPE *a0, const DTYPE *a1, const DTYPE *a2, const DTYPE *a3,
                 const DTYPE *x, int incx,
                 DTYPE alpha, int nR)
{
  register int k;
  register DTYPE cf0, cf1, cf2, cf3, t0, t1, t2, t3, t4, t5, t6, t7;

  cf0 = alpha*x[0];
  cf1 = alpha*x[incx];
  cf2 = alpha*x[2*incx];
  cf3 = alpha*x[3*incx];

  for (k = 0; k < nR-1; k += 2) {
    t0 = a0[k+0]*cf0;
    t1 = a0[k+1]*cf0;

    t2 = a1[k+0]*cf1;
    t3 = a1[k+1]*cf1;

    t4 = a2[k+0]*cf2;
    t5 = a2[k+1]*cf2;

    t6 = a3[k+0]*cf3;
    t7 = a3[k+1]*cf3;

    y[(k+0)*incy] += t0 + t2 + t4 + t6;
    y[(k+1)*incy] += t1 + t3 + t5 + t7;
  }
  if (k == nR)
    return;

  t0 = a0[k+0]*cf0;
  t2 = a1[k+0]*cf1;
  t4 = a2[k+0]*cf2;
  t6 = a3[k+0]*cf3;
  y[k*incy] += t0 + t2 + t4 + t6;
}



static inline
void __vmult1dot(DTYPE *y, int incy,
                const DTYPE *a0, const DTYPE *x, int incx,
                DTYPE alpha, int nR)
{
  register int k;
  register DTYPE t0, t1, t2, t3;

  t0 = t1 = t2 = t3 = 0.0;
  for (k = 0; k < nR-3; k += 4) {
    t0 += a0[k+0]*x[(k+0)*incx];
    t1 += a0[k+1]*x[(k+1)*incx];
    t2 += a0[k+2]*x[(k+2)*incx];
    t3 += a0[k+3]*x[(k+3)*incx];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    t0 += a0[k]*x[k*incx];
    k++;
  case 2:
    t1 += a0[k]*x[k*incx];
    k++;
  case 1:
    t2 += a0[k]*x[k*incx];
  }
 update:
  t0 += t1; t2 += t3;
  y[0]    += (t0 + t2)*alpha;
}


static inline
void __vmult2dot(DTYPE *y, int incy,
                const DTYPE *a0, const DTYPE *a1, const DTYPE *x, int incx,
                DTYPE alpha, int nR)
{
  register int k;
  register DTYPE t0, t1, t2, t3, t4, t5, t6, t7;

  t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0.0;
  for (k = 0; k < nR-3; k += 4) {
    t0 += a0[k+0]*x[(k+0)*incx];
    t1 += a0[k+1]*x[(k+1)*incx];
    t2 += a0[k+2]*x[(k+2)*incx];
    t3 += a0[k+3]*x[(k+3)*incx];

    t4 += a1[k+0]*x[(k+0)*incx];
    t5 += a1[k+1]*x[(k+1)*incx];
    t6 += a1[k+2]*x[(k+2)*incx];
    t7 += a1[k+3]*x[(k+3)*incx];
  }
  if (k == nR)
    goto update;
  switch (nR-k) {
  case 3:
    t0 += a0[k]*x[k*incx];
    t4 += a1[k]*x[k*incx];
    k++;
  case 2:
    t1 += a0[k]*x[k*incx];
    t5 += a1[k]*x[k*incx];
    k++;
  case 1:
    t2 += a0[k]*x[k*incx];
    t6 += a1[k]*x[k*incx];
  }
 update:
  t0 += t1; t2 += t3;
  t4 += t5; t6 += t7;
  y[0]    += (t0 + t2)*alpha;
  y[incy] += (t4 + t6)*alpha;
}

static inline
void __vmult4dot(DTYPE *y, int incy,
                const DTYPE *a0, const DTYPE *a1, const DTYPE *a2, const DTYPE *a3,
                const DTYPE *x, int incx,
                DTYPE alpha, int nR)
{
  register int k;
  register DTYPE t0, t1, t2, t3, t4, t5, t6, t7;

  t0 = t1 = t2 = t3 = t4 = t5 = t6 = t7 = 0.0;
  for (k = 0; k < nR-1; k += 2) {
    t0 += a0[k+0]*x[(k+0)*incx];
    t1 += a0[k+1]*x[(k+1)*incx];

    t2 += a1[k+0]*x[(k+0)*incx];
    t3 += a1[k+1]*x[(k+1)*incx];

    t4 += a2[k+0]*x[(k+0)*incx];
    t5 += a2[k+1]*x[(k+1)*incx];

    t6 += a3[k+0]*x[(k+0)*incx];
    t7 += a3[k+1]*x[(k+1)*incx];
  }
  if (k == nR)
    goto update;

  t0 += a0[k]*x[k*incx];
  t2 += a1[k]*x[k*incx];
  t4 += a2[k]*x[k*incx];
  t6 += a3[k]*x[k*incx];

 update:
  t0 += t1; t2 += t3;
  t4 += t5; t6 += t7;
  y[0]      += t0*alpha;
  y[incy]   += t2*alpha;
  y[2*incy] += t4*alpha;
  y[3*incy] += t6*alpha;
}



#endif

// Local Variables:
// indent-tabs-mode: nil
// End:

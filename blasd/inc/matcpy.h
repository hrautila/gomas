
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef _GOMAS_MATCPY_H
#define _GOMAS_MATCPY_H 1


#include <string.h>

static inline
void copy_plain_mcpy1(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int j;
  for (j = 0; j < nC; j ++) {
    memcpy(&d[(j+0)*ldD], &s[(j+0)*ldS], nR*sizeof(DTYPE));
  }
}

static inline
void copy_trans1x4(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC; j ++) {
    for (i = 0; i < nR-3; i += 4) {
      d[(j+0)+(i+0)*ldD] = s[(i+0)+(j+0)*ldS];
      d[(j+0)+(i+1)*ldD] = s[(i+1)+(j+0)*ldS];
      d[(j+0)+(i+2)*ldD] = s[(i+2)+(j+0)*ldS];
      d[(j+0)+(i+3)*ldD] = s[(i+3)+(j+0)*ldS];
    }
    if (i == nR)
      continue;
    switch (nR-i) {
    case 3:
      d[j+i*ldD] = s[i+j*ldS];
      i++;
    case 2:
      d[j+i*ldD] = s[i+j*ldS];
      i++;
    case 1:
      d[j+i*ldD] = s[i+j*ldS];
    }
  }
}

static inline
void copy_trans4x1(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC-3; j += 4) {
    for (i = 0; i < nR; i ++) {
      d[(j+0)+(i+0)*ldD] = s[i+(j+0)*ldS];
      d[(j+1)+(i+0)*ldD] = s[i+(j+1)*ldS];
      d[(j+2)+(i+0)*ldD] = s[i+(j+2)*ldS];
      d[(j+3)+(i+0)*ldD] = s[i+(j+3)*ldS];
    }
  }
  if (j == nC)
    return;
  copy_trans1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

// Copy upper tridiagonal and fill lower part to form full symmetric matrix
// result is symmetric matrix A and A = A.T
static inline
void colcpy_fill_low(DTYPE *dst, int ldD, const DTYPE *src, int ldS, int nR, int nC, int unit)
{
  //assert(nR == nC);
  register int j, i;

  // fill dst row and column at the same time, following src columns
  for (j = 0; j < nC; j++) {
    for (i = 0; i < j; i++) {
      dst[i + j*ldD] = src[i + j*ldS];
      dst[j + i*ldD] = src[i + j*ldS];
    }
    // copy the diagonal entry
    dst[j + j*ldD] = unit ? 1.0 : src[j+j*ldS];
  }
}

// Copy lower tridiagonal and fill upper part to form full symmetric matrix;
// result is symmetric matrix A and A = A.T
static inline
void colcpy_fill_up(DTYPE *dst, int ldD, const DTYPE *src, int ldS, int nR, int nC, int unit)
{
  //assert(nR == nC);
  register int j, i;

  // fill dst row and column at the same time, following src columns
  for (j = 0; j < nC; j++) {
    dst[j + j*ldD] = unit ? 1.0 : src[j + j*ldS];

    // off diagonal entries
    for (i = j+1; i < nC; i++) {
      dst[i + j*ldD] = src[i + j*ldS];
      dst[j + i*ldD] = src[i + j*ldS];
    }
  }
}


static inline
void __CPTRANS(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  copy_trans4x1(d, ldD, s, ldS, nR, nC);
}

static inline
void __CP(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  copy_plain_mcpy1(d, ldD, s, ldS, nR, nC);
}

static inline
void __CPBLK_TRANS(mdata_t *d, const mdata_t *s, int nR, int nC) {
  copy_trans4x1(d->md, d->step, s->md, s->step, nR, nC);
}

static inline
void __CPBLK(mdata_t *d, const mdata_t *s, int nR, int nC) {
  copy_plain_mcpy1(d->md, d->step, s->md, s->step, nR, nC);
}

static inline
void __CPTRIL_UFILL(mdata_t *d, const mdata_t *s, int nR, int nC, int unit) {
  colcpy_fill_up(d->md, d->step, s->md, s->step, nR, nC, unit);
}

static inline
void __CPTRIU_LFILL(mdata_t *d, const mdata_t *s, int nR, int nC, int unit) {
  colcpy_fill_low(d->md, d->step, s->md, s->step, nR, nC, unit);
}


#if defined(__NEED_CONJUGATE)

#include <complex.h>
#define __CONJ(a) conj(a)

static inline
void copy_trans_conj1x4(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC; j ++) {
    for (i = 0; i < nR-3; i += 4) {
      d[(j+0)+(i+0)*ldD] = __CONJ(s[(i+0)+(j+0)*ldS]);
      d[(j+0)+(i+1)*ldD] = __CONJ(s[(i+1)+(j+0)*ldS]);
      d[(j+0)+(i+2)*ldD] = __CONJ(s[(i+2)+(j+0)*ldS]);
      d[(j+0)+(i+3)*ldD] = __CONJ(s[(i+3)+(j+0)*ldS]);
    }
    if (i == nR)
      continue;
    switch (nR-i) {
    case 3:
      d[j+i*ldD] = __CONJ(s[i+j*ldS]);
      i++;
    case 2:
      d[j+i*ldD] = __CONJ(s[i+j*ldS]);
      i++;
    case 1:
      d[j+i*ldD] = __CONJ(s[i+j*ldS]);
    }
  }
}

static inline
void __CPTRANS_CONJ(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC-3; j += 4) {
    for (i = 0; i < nR; i ++) {
      d[(j+0)+(i+0)*ldD] = __CONJ(s[i+(j+0)*ldS]);
      d[(j+1)+(i+0)*ldD] = __CONJ(s[i+(j+1)*ldS]);
      d[(j+2)+(i+0)*ldD] = __CONJ(s[i+(j+2)*ldS]);
      d[(j+3)+(i+0)*ldD] = __CONJ(s[i+(j+3)*ldS]);
    }
  }
  if (j == nC)
    return;
  copy_trans_conj1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

static inline
void copy_conj1x4(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC; j ++) {
    for (i = 0; i < nR-3; i += 4) {
      d[(i+0)+(j+0)*ldD] = __CONJ(s[(i+0)+(j+0)*ldS]);
      d[(i+1)+(j+0)*ldD] = __CONJ(s[(i+1)+(j+0)*ldS]);
      d[(i+2)+(j+0)*ldD] = __CONJ(s[(i+2)+(j+0)*ldS]);
      d[(i+3)+(j+0)*ldD] = __CONJ(s[(i+3)+(j+0)*ldS]);
    }
    if (i == nR)
      continue;
    switch (nR-i) {
    case 3:
      d[i+j*ldD] = __CONJ(s[i+j*ldS]);
      i++;
    case 2:
      d[i+j*ldD] = __CONJ(s[i+j*ldS]);
      i++;
    case 1:
      d[i+j*ldD] = __CONJ(s[i+j*ldS]);
    }
  }
}

static inline
void __CPCONJ(DTYPE *d, int ldD, const DTYPE *s, int ldS, int nR, int nC) {
  register int i, j;
  for (j = 0; j < nC-3; j += 4) {
    for (i = 0; i < nR; i ++) {
      d[i+(j+0)*ldD] = __CONJ(s[i+(j+0)*ldS]);
      d[i+(j+1)*ldD] = __CONJ(s[i+(j+1)*ldS]);
      d[i+(j+2)*ldD] = __CONJ(s[i+(j+2)*ldS]);
      d[i+(j+3)*ldD] = __CONJ(s[i+(j+3)*ldS]);
    }
  }
  if (j == nC)
    return;
  copy_conj1x4(&d[j], ldD, &s[j*ldS], ldS, nR, nC-j);
}

#else  /* __NEED_CONJUGATE */

#define __CPTRANS_CONJ __CPTRANS
#define __CPCONJ       __CP

#endif /* __NEED_CONJUGATE */

#endif

// Local Variables:
// indent-tabs-mode: nil
// End:

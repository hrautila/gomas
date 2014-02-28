

// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas library. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.


#ifndef __GOMAS_DBLE_INTERFACES_H
#define __GOMAS_DBLE_INTERFACES_H

#include "dtype.h"

enum armas_flags {
  GOMAS_NOTRANS = 0,
  GOMAS_NULL    = 0,
  // operand A is transposed
  GOMAS_TRANSA  = 0x1,          
  // operand B is transposed
  GOMAS_TRANSB  = 0x2,
  // matrix operand is transposed
  GOMAS_TRANS   = 0x4,
  // lower triangular matrix
  GOMAS_LOWER   = 0x8,
  // upper triangular matrix 
  GOMAS_UPPER   = 0x10,
  // multiplicaton from left
  GOMAS_LEFT    = 0x20,
  // multiplicaton from right
  GOMAS_RIGHT   = 0x40,
  // unit diagonal matrix
  GOMAS_UNIT    = 0x80,
  // operand A is conjugate transposed
  GOMAS_CONJA   = 0x100,
  // operand B is conjugate transposed
  GOMAS_CONJB   = 0x200,
  // matrix operand is conjugate transposed
  GOMAS_CONJ    = 0x400,
  // symmetric matrix
  GOMAS_SYMM    = 0x800,
  // hermitian matrix
  GOMAS_HERM    = 0x1000,
};



// maximum sizes

#ifndef MAX_KB
#define MAX_KB 192
#endif

#ifndef MAX_NB
#define MAX_NB 128
#endif

#ifndef MAX_MB
#define MAX_MB 128
#endif

// no recursion for vectors shorter than this
#ifndef MIN_MVEC_SIZE
#define MIN_MVEC_SIZE 48
#endif

#ifndef MIN_MBLOCK_SIZE
#define MIN_MBLOCK_SIZE 48
#endif


typedef struct mdata {
  double *md;
  int step;
} mdata_t;

typedef struct mvec {
  double *md;
  int inc;
} mvec_t;

typedef struct cache_buffer {
  mdata_t *Acpy;
  mdata_t *Bcpy;
  int KB;
  int NB;
  int MB;
} cache_t;


static inline
int min(int a, int b) {
  return a < b ? a : b;
}

// compute start of i'th block out of r blocks in sz elements
static inline
int __block_index4(int i, int n, int sz) {
    if (i == n) {
        return sz;
    }
    return i*sz/n - ((i*sz/n) & 0x3);
}

static inline
int __block_index2(int i, int n, int sz) {
    if (i == n) {
        return sz;
    }
    return i*sz/n - ((i*sz/n) & 0x1);
}

// make A subblock of B, starting from B[r,c] 
static inline
mdata_t *__subblock(mdata_t *A, const mdata_t *B, int r, int c)
{
  A->md = &B->md[r + c*B->step];
  A->step = B->step;
  return A;
}

// make X subvector of Y, starting at Y[n]
static inline
mvec_t *__subvector(mvec_t *X, const mvec_t *Y, int n)
{
  X->md = &Y->md[n*Y->inc];
  X->inc = Y->inc;
  return X;
}

static inline
mvec_t *__rowvec(mvec_t *X, const mdata_t *A, int r, int c)
{
  X->md = &A->md[r + c*A->step];
  X->inc = A->step;
  return X;
}

static inline
mvec_t *__colvec(mvec_t *X, const mdata_t *A, int r, int c)
{
  X->md = &A->md[r + c*A->step];
  X->inc = 1;
  return X;
}

#ifdef DEBUG
#define __DEBUG(a) do { a; } while (0)
#else
#define __DEBUG(a)
#endif


// ------------------------------------------------------------------------------------
// GO callable public functions.
extern
void __d_gemm_inner(mdata_t *C, const mdata_t *A, const mdata_t *B,
                    double alpha, double beta, int flags,
                    int P, int S, int L, int R, int E, int KB, int NB, int MB);

extern
void __d_symm_inner(mdata_t *C, const mdata_t *A, const mdata_t *B,
                    double alpha, double beta, int flags,
                    int P, int S, int L, int R, int E, int KB, int NB, int MB);

extern
void __d_rank_blk(mdata_t *C, const mdata_t *A, double alpha, double beta,
                  int flags,  int P, int S, int E, int KB, int NB, int MB);

extern
void __d_rank2_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                 double alpha, double beta, int flags,
                 int P, int S, int E,  int KB, int NB, int MB);

extern
void __d_scale_plus(mdata_t *A, const mdata_t *B, 
                    double alpha, double beta, int flags,
                    int S, int L, int R, int E);

extern 
void __d_trmm_blk(mdata_t *B, const mdata_t *A, double alpha, int flags,
                  int N, int S, int E, int KB, int NB, int MB);

extern
void __d_solve_blocked(mdata_t *B, const mdata_t *A, double alpha,
                       int flags, int N, int S, int E, int KB, int NB, int MB);

extern
void __d_update_trm_blk(mdata_t *C, const mdata_t *A, const mdata_t *B,
                        double alpha, double beta, int flags,
                        int P, int S, int L, int R, int E, int KB, int NB, int MB);

extern
void __d_gemv_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                  double alpha, int flags, int S, int L, int R, int E);

extern
void __d_symv_unb(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                  double alpha, int flags, int N);

extern
void __d_trmv_unb(mvec_t *X, const mdata_t *A, double alpha, int flags, int N);

extern
void __d_trsv_unb(mvec_t *X, const mdata_t *A,  int flags, int N);

extern
void __d_gemv_recursive(mvec_t *Y, const mdata_t *A, const mvec_t *X,
                        double alpha, double beta, int flags,
                        int S, int L, int R, int E);

extern
void __d_update_ger_unb(mdata_t *A, const mvec_t *Y, const mvec_t *X,
                        double alpha, int N, int M);

extern
void __d_update_trmv_unb(mdata_t *A, const mvec_t *Y, const mvec_t *X,
                         double alpha, int flags, int N, int M);

extern
void __d_update_syr2_unb(mdata_t *A, const mvec_t *Y, const mvec_t *X,
                         double alpha, int flags, int N);

extern
void __d_trmv_recursive(mvec_t *X, const mdata_t *A, 
                        double alpha, int flags, int N);

extern
void __d_trsv_recursive(mvec_t *X, const mdata_t *A, 
                        double alpha, int flags, int N);

extern
void __d_update_ger_recursive(mdata_t *A, const mvec_t *Y, const mvec_t *X,
                              double alpha, int N, int M);

extern
void __d_update_trmv_recursive(mdata_t *A, const mvec_t *Y, const mvec_t *X,
                               double alpha, int flags, int N, int M);

extern
void __d_update_syr2_recursive(mdata_t *A, const mvec_t *Y, const mvec_t *X,
                               double alpha, int flags, int N);

extern void __d_blk_scale(mdata_t *X, double beta, int M, int N);

extern void __d_blk_add(mdata_t *X, double beta, int M, int N);

extern void __d_vec_copy(mvec_t *X, const mvec_t *Y, int N);

extern void __d_vec_swap(mvec_t *X, mvec_t *Y, int N);

extern int __d_vec_iamax(const mvec_t *X, int N);

extern double __d_vec_amax(const mvec_t *X, int N);

extern double __d_vec_asum_recursive(const mvec_t *X, int N);

extern double __d_vec_sum_recursive(const mvec_t *X, int N);

extern double __d_vec_dot_recursive(const mvec_t *X, const mvec_t *Y, int N);

extern void __d_vec_axpy( mvec_t *Y, const mvec_t *X, double alpha, int N);

extern void __d_vec_axpby( mvec_t *Y, const mvec_t *X, double alpha, double beta, int N);

extern double __d_vec_nrm2_scaled(const mvec_t *X, int N);

extern void __d_vec_scal(mvec_t *X, double alpha, int N);

extern void __d_vec_invscal(mvec_t *X, double alpha, int N);

extern void __d_vec_add(mvec_t *X, double alpha, int N);

extern void __d_blk_scale(mdata_t *X, double beta, int M, int N);
extern void __d_blk_invscale(mdata_t *X, double beta, int M, int N);
extern void __d_blk_add(mdata_t *X, double beta, int M, int N);
extern void __d_blk_copy(mdata_t *A, const mdata_t *B, int M, int N);
extern void __d_blk_transpose(mdata_t *A, const mdata_t *B, int M, int N);


// -------------------------------------------------------------------------------------
// C callable public functions, type independent declarations

extern void __blk_scale(mdata_t *X, DTYPE beta, int M, int N);
extern void __blk_add(mdata_t *X, DTYPE beta, int M, int N);
//extern void __blk_print(const mdata_t *X, int M, int N, const char *s, const char *fmt);
//extern void __vec_print(const mvec_t *X, int N, const char *s, const char *fmt);

extern
void __gemm_colblk_inner(mdata_t *Cblk, const mdata_t *Ablk, const mdata_t *Bblk,
                         DTYPE alpha, int nJ, int nR, int nP);

extern
void __gemm_colwise_inner_no_scale(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                   DTYPE alpha, int flags,
                                   int P, int nSL, int nRE, cache_t *cache);

extern
void __gemm_colwise_inner_scale_c(mdata_t *C, const mdata_t *A, const mdata_t *B,
                                  DTYPE alpha, DTYPE beta, int flags,
                                  int P, int S, int L, int R, int E, cache_t *cache);

extern
void __rank_diag(mdata_t *C, const mdata_t *A, const mdata_t *B, 
                 DTYPE alpha, DTYPE beta, int flags,  int P, int nC, cache_t *cache);

extern
void __trmm_unb(mdata_t *B, const mdata_t *A, DTYPE alpha, int flags, int N, int S, int E);

extern
void __trmm_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E, int KB, int NB, int MB);

extern
void __trmm_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                          int flags, int N, int S, int E, cache_t *cache);


extern
void __solve_right_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                       int flags, int N, int S, int E);

extern
void __solve_left_unb(mdata_t *B, const mdata_t *A, DTYPE alpha,
                      int flags, int N, int S, int E);

extern
void __solve_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                       int flags, int N, int S, int E, int KB, int NB, int MB);

extern
void __solve_blk_recursive(mdata_t *B, const mdata_t *A, DTYPE alpha,
                           int flags, int N, int S, int E, cache_t *cache);



extern
void __update_ger_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                            DTYPE alpha, int N, int M);

extern
void __update_trmv_unb(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                       DTYPE alpha, int flags, int N, int M);

extern
void __update_trmv_recursive(mdata_t *A, const mvec_t *X, const mvec_t *Y,
                             DTYPE alpha, int flags, int N, int M);


#endif

// Local Variables:
// indent-tabs-mode: nil
// End:

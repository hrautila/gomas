
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

#ifndef __DBLE_DTYPE_H
#define __DBLE_DTYPE_H 1

#include <math.h>

/* ---------------------------------------------------------------------------
 * Definitions for double precision floating types.
 */
typedef double DTYPE;
typedef double ABSTYPE;

#define __ABSZERO (double)0.0
#define __ABSONE  (double)1.0
#define __ZERO    (double)0.0
#define __ONE     (double)1.0

#define __DATA_FORMAT "%8.1e"
#define __PRINTABLE(a) (a)

// functions from math.h
#define __ABS  fabs
#define __SQRT sqrt

// internally available functions
#define __blk_add        __d_blk_add
#define __blk_scale      __d_blk_scale
#define __blk_invscale   __d_blk_invscale
#define __blk_copy       __d_blk_copy
#define __blk_transpose  __d_blk_transpose

#define __blk_print    __d_blk_print
#define __vec_print    __d_vec_print

#define __gemm_colwise_inner_no_scale  __d_gemm_colwise_inner_no_scale
#define __gemm_colwise_inner_scale_c   __d_gemm_colwise_inner_scale_c
#define __gemm_colblk_inner      __d_gemm_colblk_inner
#define __gemm_inner             __d_gemm_inner
#define __gemv_recursive         __d_gemv_recursive
#define __symm_inner             __d_symm_inner
#define __rank_diag              __d_rank_diag
#define __rank2_blk              __d_rank2_blk
#define __rank_blk               __d_rank_blk
#define __trmm_unb               __d_trmm_unb
#define __trmm_blk               __d_trmm_blk
#define __trmm_recursive         __d_trmm_recursive
#define __trmm_blk_recursive     __d_trmm_blk_recursive
#define __trmv_recursive         __d_trmv_recursive
#define __trsv_recursive         __d_trsv_recursive
#define __solve_left_unb         __d_solve_left_unb
#define __solve_right_unb        __d_solve_right_unb
#define __solve_blocked          __d_solve_blocked
#define __solve_recursive        __d_solve_recursive
#define __solve_blk_recursive    __d_solve_blk_recursive
#define __update_trmv_unb        __d_update_trmv_unb
#define __update_trmv_recursive  __d_update_trmv_recursive
#define __update_trm_blk         __d_update_trm_blk
#define __update_ger_unb         __d_update_ger_unb
#define __update_ger_recursive   __d_update_ger_recursive
#define __update_syr2_recursive  __d_update_syr2_recursive

#define __vec_add            __d_vec_add
#define __vec_axpy           __d_vec_axpy
#define __vec_axpby          __d_vec_axpby
#define __vec_amax           __d_vec_amax
#define __vec_iamax          __d_vec_iamax
#define __vec_scal           __d_vec_scal
#define __vec_invscal        __d_vec_invscal
#define __vec_nrm2_scaled    __d_vec_nrm2_scaled
#define __vec_dot_recursive  __d_vec_dot_recursive
#define __vec_asum_recursive __d_vec_asum_recursive
#define __vec_sum_recursive  __d_vec_sum_recursive
#define __vec_swap           __d_vec_swap
#define __vec_copy           __d_vec_copy

#if 0
// public functions: blas level 3, matrix-matrix
#define __armas_mult        armas_d_mult
#define __armas_mult_sym    armas_d_mult_sym
#define __armas_mult_trm    armas_d_mult_trm
#define __armas_solve_trm   armas_d_solve_trm
#define __armas_update_sym  armas_d_update_sym
#define __armas_2update_sym armas_d_2update_sym
#define __armas_update_trm  armas_d_update_trm

// public functions: blas level 2, matrix-vector
#define __armas_mvmult        armas_d_mvmult
#define __armas_mvmult_sym    armas_d_mvmult_sym
#define __armas_mvmult_trm    armas_d_mvmult_trm
#define __armas_mvsolve_trm   armas_d_mvsolve_trm
#define __armas_mvupdate      armas_d_mvupdate
#define __armas_mvupdate_trm  armas_d_mvupdate_trm
#define __armas_mvupdate_sym  armas_d_mvupdate_sym
#define __armas_mv2update_sym armas_d_mv2update_sym

#define __armas_mvmult_diag   armas_d_mvmult_diag
#define __armas_mvsolve_diag  armas_d_mvsolve_diag

// public functions: blas level 1, vector-vector
#define __armas_nrm2    armas_d_nrm2
#define __armas_asum    armas_d_asum
#define __armas_scale   armas_d_scale
#define __armas_iamax   armas_d_iamax
#define __armas_dot     armas_d_dot
#define __armas_axpy    armas_d_axpy
#define __armas_swap    armas_d_swap
#define __armas_copy    armas_d_copy
// 
#define __armas_invscale armas_d_invscale
#define __armas_sum      armas_d_sum
#define __armas_amax     armas_d_amax
#define __armas_add      armas_d_add
#endif


#endif

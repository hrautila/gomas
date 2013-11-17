GO Matrix Algebra Subroutines
=============================

Almost complete implementation of BLAS level 1, 2 and 3 routines for double precision floating point.
All computation is in place. The implementation supports matrix views (submatrices of larger matrices)
and parallel execution of matrix operations in multiple threads. 

Low level matrix operations primitives use 256bit and 128bit vectorization instructions when possible.
At the moment there is no threading support for parallel execution of operations. 

Supported functionality is:

  Blas level 3

    Mult(C, A, B, alpha, beta, flags)           General matrix-matrix multiplication  (GEMM)
    MultSymm(C, A, B, alpha, beta, flags)       Symmetric matrix-matrix multipication (SYMM)
    MultTrm(B, A, alpha, flags)                 Triangular matrix-matrix multiplication (TRMM)  
    SolveTrm(B, A, alpha, flags)                Triangular solve with multiple RHS (TRSM)
    UpdateSym(C, A, alpha, beta,flags)          Symmetric matrix rank-k update (SYRK)
    Update2Sym(C, A, B, alpha, beta, flags)     Symmetric matrix rank-2k update (SYR2K)

  Blas level 2

    MVMult(X, A, Y, alpha, beta, flags)         General matrix-vector multiplication (GEMV)
    MVUpdate(A, X, Y, alpha, flags)             General matrix rank update (GER)
    MVUpdateSym(A, X, alpha, flags)             Symmetric matrix rank update (SYR)
    MVUpdate2Sym(A, X, Y, alpha, flags)         Symmetric matrix rank 2 update (SYR2)
    MVSolveTrm(X, A, alpha, flags)              Triangular solve (TRSV)
    MVMultTrm(X, A, flags)                      Triangular matrix-vector multiplication (TRMV)

  Blas level 1

    ASum(X)             Sum of absolute values sum(|x|) (ASUM)
    Axpy(Y, X, alpha)   Vector sum Y := alpha*X + Y (AXPY)
    IAMax(X)            Index of absolute maximum value (IAMAX)
    Norm2(X)            Vector norm sqrt(||x||^2) (NRM2)
    Dot(X, Y)           Inner product (DOT)
    Swap(X, Y)          Vector-vector swap (SWAP)
    Scale(X, alpha)     Scaling of X (SCAL)

  Additional

    Copy(A, B)                                  Copy B to A.
    InvScale(X, alpha)                          Inverse scaling of X 
    ScalePlus(A, B, alpha, beta, flags)         Calculate A = alpha*op(A) + beta*op(B)
    NormP(X, norm)                              Matrix or vector norm, _1, _2, _Inf
    UpdateTrm(C, A, B, alpha, beta, flags)      Triangular/trapezoidal matrix update
    MVUpdateTrm(C, X, Y, alpha, flags)          Triangular/trapezoidal matrix update with vectors.

  Lapack
  
    DecomposeCHOL(A, nb)                Cholesky factorization (DPOTRF)
    DecomposeLUnoPiv(A, nb)             LU factorization without pivoting
    DecomposeLU(A, pivots, nb)          LU factorization with pivoting (DGETRF)
    SolveCHOL(B, A, flags)              Solve Cholesky factorized linear system (DPOTRS)
    SolveLU(B, A, pivots, flags)        Solve LU factorized linear system (DGETRS)

  Other

    DefaultConf()             Get pointer to default blocking configuration 
    NewConf()                 Create a new blocking configuration
    NewError()                New error descriptor

This is still WORK IN PROGRESS. Consider this as beta level code, at best. 


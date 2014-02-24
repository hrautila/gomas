GO Matrix Algebra Subroutines
-----------------------------

Implementations of BLAS level 1, 2 and 3 routines and some LAPACK routines for double precision floating point.
This package is rewrite of my MATOPS package using different matrix implementation.

Some key ideas on implementation

- Good single threaded perfomance
- No internal memory allocation
- All operations in place. If workspace needed it must be provided explicitely.
- No assembler code. 
- Use SIMD instruction when available
- Higher level algorithms with libFLAME like implementation
- Unblocked and blocked versions
- Need to support parallelization
- More Go/C like call interface

At the moment there is no threading support for parallel execution of operations. 
This is WORK IN PROGRESS. Consider this as beta level code, at best. 

### Some numbers 

Performance of single threaded matrix-matrix multiplication (GEMM). Theoretical
maximum performance of Ivy Bridge CPU is 8 double precision floating point operations
per cpu cycle (4 addition, 4 multiplication). With peak performance 16.9034 Gflops
and 2.4 GHz clock rate we get 7.04 operations/cycle, ~ 88% of theoretical maximum.

    CPU: Intel(R) Core(TM) i7-3630QM CPU @ 2.40GHz

     200   12.5294   13.8528   14.4928 Gflops
     400   15.6345   16.2817   16.5653 Gflops
     600   16.4003   16.7994   16.9798 Gflops
     800   16.4187   16.6092   16.8357 Gflops
    1000   16.4117   16.5573   16.6710 Gflops
    1200   16.0345   16.5740   16.8028 Gflops
    1400   16.5978   16.7548   16.8224 Gflops
    1600   16.7352   16.8270   16.9034 Gflops
  

### Blas

  Level 3

    Mult(C, A, B, alpha, beta, flags)           General matrix-matrix multiplication  (GEMM)
    MultSymm(C, A, B, alpha, beta, flags)       Symmetric matrix-matrix multipication (SYMM)
    MultTrm(B, A, alpha, flags)                 Triangular matrix-matrix multiplication (TRMM)  
    SolveTrm(B, A, alpha, flags)                Triangular solve with multiple RHS (TRSM)
    UpdateSym(C, A, alpha, beta,flags)          Symmetric matrix rank-k update (SYRK)
    Update2Sym(C, A, B, alpha, beta, flags)     Symmetric matrix rank-2k update (SYR2K)

  Level 2

    MVMult(X, A, Y, alpha, beta, flags)         General matrix-vector multiplication (GEMV)
    MVUpdate(A, X, Y, alpha, flags)             General matrix rank update (GER)
    MVUpdateSym(A, X, alpha, flags)             Symmetric matrix rank update (SYR)
    MVUpdate2Sym(A, X, Y, alpha, flags)         Symmetric matrix rank 2 update (SYR2)
    MVSolveTrm(X, A, alpha, flags)              Triangular solve (TRSV)
    MVMultTrm(X, A, flags)                      Triangular matrix-vector multiplication (TRMV)

  Level 1

    ASum(X)                                     Sum of absolute values sum(|x|) (ASUM)
    Axpy(Y, X, alpha)                           Vector sum Y := alpha*X + Y (AXPY)
    IAmax(X)                                    Index of absolute maximum value (IAMAX)
    Nrm2(X)                                     Vector norm sqrt(||x||^2) (NRM2)
    Dot(X, Y)                                   Inner product (DOT)
    Swap(X, Y)                                  Vector-vector swap (SWAP)
    Scale(X, alpha)                             Scaling of X (SCAL)

  Additional

    Axpby(Y, X, alpha, beta)                    Vector sum Y := alpha*X + beta*Y 
    Amax(X)                                     Absolute maximum of X
    InvScale(X, alpha)                          Inverse scaling of X 
    Copy(A, B)                                  Copy B to A.
    Plus(A, B, alpha, beta, flags)              Calculate A = alpha*A + beta*op(B)
    NormP(X, norm)                              Matrix or vector norm, _1, _2, _Inf
    UpdateTrm(C, A, B, alpha, beta, flags)      Triangular/trapezoidal matrix update
    MVUpdateTrm(C, X, Y, alpha, flags)          Triangular/trapezoidal matrix update with vectors.

### Lapack
  
    DecomposeBK(A, W, flags, conf)              LDL.T factorization (DSYTRF)
    DecomposeCHOL(A, conf)                      Cholesky factorization (DPOTRF)
    DecomposeLUnoPiv(A, conf)                   LU factorization without pivoting
    DecomposeLU(A, pivots, conf)                LU factorization with pivoting (DGETRF)
    DecomposeLQ(A, tau, W, conf)                LQ factorization (DGELQF)
    DecomposeQR(A, tau, W, conf)                QR factorization (DGEQRF)
    DecomposeQRT(A, T, W, conf)                 QR factorization with compact WY transformation (DGEQRT)
    MultLQ(C, A, tau, W, flags, conf)           Multiply with Q or Q.T  (DORMLQ)
    MultQ(C, A, tau, W, flags, conf)            Multiply with Q or Q.T  (DORMQR)
    MultQT(C, A, T, W, flags, conf)             Multiply with Q or Q.T, compact WY transformation (DORGQR)
    MultQHess(C, A, tau, W, flags, conf)        Multiply with Hessengerg Q matrix (DORMHR)
    MultBidiag(C, A, tau, W, flags, conf)       Multiply with bidiagonal Q or P matrix (DORMBR)
    ReduceBidiag(A, tauq, taup, W, conf)        Bidiagonal reduction (DGEBRD)
    ReduceHess(A, tau, W, conf)                 Hessenberg reduction. (DGEHRD)
    SolveBK(B, A, W, flags, conf)               Solve LDL.T factorized linear system (DSYTRS)
    SolveCHOL(B, A, flags, conf)                Solve Cholesky factorized linear system (DPOTRS)
    SolveLU(B, A, pivots, flags, conf)          Solve LU factorized linear system (DGETRS)
    SolveLQ(B, A, W, flags, conf)               Solve LQ factorized linear system
    SolveQR(B, A, W, flags, conf)               Solve QR factorized linear system
    SolveQRT(B, A, W, flags, conf)              Solve QRWY factorized linear system

    WorksizeBK(A, conf)                         Compute worksize needed for LDL.T  factorization
    WorksizeQR(A, conf)                         Compute worksize needed for QR factorization
    WorksizeQRT(A, conf)                        Compute worksize needed for QRWY factorization
    WorksizeMultQ(A, conf)                      Compute worksize for MultQ
    WorksizeMultQT(A, conf)                     Compute worksize for MultQT
    Workspace(size)                             Create workspace

###  Other

    DefaultConf()                               Get default blocking configuration 
    NewConf()                                   Create a new blocking configuration
    NewError(err, name)                         New error descriptor



package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)

/*
 * test1: C0 = A*B; C1 = B.T*A.T; C0 == C1.T
 *    A[M,K], B[K,N], C[M,N] and M != N != K
 */
func TestDGemm1(t *testing.T) {
	M := 411
	N := 377
	K := 253

	A := cmat.NewMatrix(M, K)
	B := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(M, N)
	Ct := cmat.NewMatrix(N, M)
	T  := cmat.NewMatrix(M, N)

	zeromean :=  cmat.NewFloatNormSource()

	A.SetFrom(zeromean)
	B.SetFrom(zeromean)

	blasd.Mult(C, A, B, 1.0, 0.0, gomas.NONE)
	blasd.Mult(Ct, B, A, 1.0, 0.0, gomas.TRANSA|gomas.TRANSB)
	T.Transpose(Ct)
	ok := C.AllClose(T)
	t.Logf("gemm(A, B)   == transpose(gemm(B.T, A.T)): %v\n", ok)
}

/*
 * test2: C0 = A*B.T; C1 = B*A.T; C0 == C1.T
 *    A[M,K], B[K,N], C[M,N] and M != N == K
 */
func TestDGemm2(t *testing.T) {
	M := 411
	N := 377
	K := N
	A := cmat.NewMatrix(M, K)
	B := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(M, N)
	Ct := cmat.NewMatrix(N, M)
	T  := cmat.NewMatrix(M, N)

	zeromean :=  cmat.NewFloatNormSource()

	A.SetFrom(zeromean)
	B.SetFrom(zeromean)

	blasd.Mult(C,  A, B, 1.0, 0.0, gomas.TRANSB)
	blasd.Mult(Ct, B, A, 1.0, 0.0, gomas.TRANSB)
	T.Transpose(Ct)
	ok := C.AllClose(T)
	t.Logf("gemm(A, B.T) == transpose(gemm(B, A.T))  : %v\n", ok)
}

/*
 * test3: C0 = A.T*B; C1 = B.T*A; C0 == C1.T
 *    A[M,K], B[K,N], C[M,N] and M == K != N
 */
func TestDGemm3(t *testing.T) {
	M := 411
	N := 383
	K := M

	A := cmat.NewMatrix(M, K)
	B := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(M, N)
	Ct := cmat.NewMatrix(N, M)
	T  := cmat.NewMatrix(M, N)

	zeromean :=  cmat.NewFloatNormSource()

	A.SetFrom(zeromean)
	B.SetFrom(zeromean)

	blasd.Mult(C,  A, B, 1.0, 0.0, gomas.TRANSA)
	blasd.Mult(Ct, B, A, 1.0, 0.0, gomas.TRANSA)
	T.Transpose(Ct)
	ok := C.AllClose(T)
	t.Logf("gemm(A.T, B) == transpose(gemm(B.T, A))  : %v\n", ok)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

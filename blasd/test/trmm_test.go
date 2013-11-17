
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)

func TestDTrmm1(t *testing.T) {
	N := 563
	K := 171
	nofail := true

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(N, K)
	B0 := cmat.NewMatrix(N, K)
	C := cmat.NewMatrix(N, K)

	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.LOWER)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = A*B
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, L|L|N) == gemm(C, TriL(A), B)   : %v\n", ok)

	B.SetFrom(ones)
	// B = A.T*B
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT|gomas.TRANSA)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.TRANSA)
	ok = C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, L|L|T) == gemm(C, TriL(A).T, B) : %v\n", ok)
}


func TestDTrmm2(t *testing.T) {
	N := 563
	K := 171
	nofail := true

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(K, N)
	B0 := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(K, N)

	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.LOWER)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = B*A
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, R|L|N) == gemm(C, B, TriL(A))   : %v\n", ok)

	B.SetFrom(ones)
	// B = B*A.T
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT|gomas.TRANSA)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.TRANSB)
	ok = C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, R|L|T) == gemm(C, B, TriL(A).T) : %v\n", ok)
}


func TestDTrmm3(t *testing.T) {
	N := 563
	K := 171

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(N, K)
	B0 := cmat.NewMatrix(N, K)
	C := cmat.NewMatrix(N, K)

	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.UPPER)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = A*B
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	t.Logf("trmm(B, A, L|U|N) == gemm(C, TriU(A), B)   : %v\n", ok)

	B.SetFrom(ones)
	// B = A.T*B
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT|gomas.TRANSA)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.TRANSA)
	ok = C.AllClose(B)
	t.Logf("trmm(B, A, L|U|T) == gemm(C, TriU(A).T, B) : %v\n", ok)
}

func TestDTrmm4(t *testing.T) {
	N := 563
	K := 171

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(K, N)
	B0 := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(K, N)

	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.UPPER)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = B*A
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	t.Logf("trmm(B, A, R|U|N) == gemm(C, B, TriU(A))   : %v\n", ok)

	B.SetFrom(ones)
	// B = B*A.T
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT|gomas.TRANSA)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.TRANSB)
	ok = C.AllClose(B)
	t.Logf("trmm(B, A, R|U|T) == gemm(C, B, TriU(A).T) : %v\n", ok)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

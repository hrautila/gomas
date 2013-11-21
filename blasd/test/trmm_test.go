
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)

func TestDTrmmLower(t *testing.T) {
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

func TestDTrmmUnitLower(t *testing.T) {
    var d cmat.FloatMatrix
	N := 563
	K := 171
	nofail := true

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(N, K)
	B0 := cmat.NewMatrix(N, K)
	C := cmat.NewMatrix(N, K)

	zeros := cmat.NewFloatConstSource(0.0)
	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.LOWER|cmat.UNIT)
	B.SetFrom(ones)
	B0.SetFrom(ones)

	// B = A*B
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, L|L|N|U) == gemm(C, TriLU(A), B)   : %v\n", ok)

	B.SetFrom(ones)
	// B = A.T*B
    d.Diag(A).SetFrom(zeros)
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT|gomas.TRANSA|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.TRANSA)
	ok = C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, L|L|T|U) == gemm(C, TriLU(A).T, B) : %v\n", ok)
}



func TestDTrmmLowerRight(t *testing.T) {
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

func TestDTrmmUnitLowerRight(t *testing.T) {
    var d cmat.FloatMatrix
	N := 563
	K := 171
	nofail := true

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(K, N)
	B0 := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(K, N)

	zeros := cmat.NewFloatConstSource(0.0)
	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.LOWER|cmat.UNIT)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = B*A
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, R|L|N|U) == gemm(C, B, TriLU(A))   : %v\n", ok)

	B.SetFrom(ones)
	// B = B*A.T
    d.Diag(A).SetFrom(zeros)
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT|gomas.TRANSA|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.TRANSB)
	ok = C.AllClose(B)
	nofail = nofail && ok
	t.Logf("trmm(B, A, R|L|T|U) == gemm(C, B, TriLU(A).T) : %v\n", ok)
}


func TestDTrmmUpper(t *testing.T) {
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

func TestDTrmmUnitUpper(t *testing.T) {
    var d cmat.FloatMatrix
	N := 563
	K := 171

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(N, K)
	B0 := cmat.NewMatrix(N, K)
	C := cmat.NewMatrix(N, K)

	zeros := cmat.NewFloatConstSource(0.0)
	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.UPPER|cmat.UNIT)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = A*B
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	t.Logf("trmm(B, A, L|U|N|U) == gemm(C, TriUU(A), B)   : %v\n", ok)

	B.SetFrom(ones)
	// B = A.T*B
    d.Diag(A).SetFrom(zeros)
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT|gomas.TRANSA|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, A, B0, 1.0, 0.0, gomas.TRANSA)
	ok = C.AllClose(B)
	t.Logf("trmm(B, A, L|U|T|U) == gemm(C, TriUU(A).T, B) : %v\n", ok)
}

func TestDTrmmUpperRight(t *testing.T) {
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

func TestDTrmmUnitUpperRight(t *testing.T) {
    var d cmat.FloatMatrix
	N := 563
	K := 171

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(K, N)
	B0 := cmat.NewMatrix(K, N)
	C := cmat.NewMatrix(K, N)

	zeros := cmat.NewFloatConstSource(0.0)
	ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatNormSource()

	A.SetFrom(zeromean, cmat.UPPER|cmat.UNIT)
	B.SetFrom(ones)
	B0.SetFrom(ones)
	// B = B*A
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT|gomas.UNIT)
    d.Diag(A).SetFrom(ones)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.NONE)
	ok := C.AllClose(B)
	t.Logf("trmm(B, A, R|U|N|U) == gemm(C, B, TriUU(A))   : %v\n", ok)

	B.SetFrom(ones)
	// B = B*A.T
    d.SetFrom(zeros)
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT|gomas.TRANSA|gomas.UNIT)
    d.SetFrom(ones)
	blasd.Mult(C, B0, A, 1.0, 0.0, gomas.TRANSB)
	ok = C.AllClose(B)
	t.Logf("trmm(B, A, R|U|T|U) == gemm(C, B, TriUU(A).T) : %v\n", ok)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

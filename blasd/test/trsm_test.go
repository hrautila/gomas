
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)


func TestDTrsm1(t *testing.T) {
	nofail := true

    const N = 31
    const K = 4

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(N, K)
	B0 := cmat.NewMatrix(N, K)

    ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource(1.0, -0.5)

	A.SetFrom(zeromean, cmat.LOWER)
	B.SetFrom(ones)
	B0.Copy(B)
	// B = A*B
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT)
	blasd.SolveTrm(B, A, 1.0,gomas.LOWER|gomas.LEFT)
	ok := B0.AllClose(B)
	nofail = nofail && ok
	t.Logf("B == trsm(trmm(B, A, L|L|N), A, L|L|N) : %v\n", ok)
    if ! ok {
        t.Logf("B|B0:\n%v\n", cmat.NewJoin(cmat.AUGMENT, B, B0))
    }

	B.Copy(B0)
	// B = A.T*B
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT|gomas.TRANSA)
	blasd.SolveTrm(B, A, 1.0, gomas.LOWER|gomas.LEFT|gomas.TRANSA)
	ok = B0.AllClose(B)
	nofail = nofail && ok
	t.Logf("B == trsm(trmm(B, A, L|L|T), A, L|L|T) : %v\n", ok)
}


func TestDTrms2(t *testing.T) {
    const N = 31
    const K = 4

	nofail := true

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(K, N)
	B0 := cmat.NewMatrix(K, N)

    ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource(1.0, -0.5)

	A.SetFrom(zeromean, cmat.LOWER)
	B.SetFrom(ones)
	B0.Copy(B)

	// B = B*A
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT)
	blasd.SolveTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT)
	ok := B0.AllClose(B)
	nofail = nofail && ok
	t.Logf("B == trsm(trmm(B, A, R|L|N), A, R|L|N) : %v\n", ok)

	B.Copy(B0)
	// B = B*A.T
	blasd.MultTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT|gomas.TRANSA)
	blasd.SolveTrm(B, A, 1.0, gomas.LOWER|gomas.RIGHT|gomas.TRANSA)
	ok = B0.AllClose(B)
	nofail = nofail && ok
	t.Logf("B == trsm(trmm(B, A, R|L|T), A, R|L|T) : %v\n", ok)
}


func TestDTrsm3(t *testing.T) {
    const N = 31
    const K = 4

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(N, K)
	B0 := cmat.NewMatrix(N, K)

    ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource(1.0, -0.5)

	A.SetFrom(zeromean, cmat.UPPER)
	B.SetFrom(ones)
	B0.Copy(B)
	// B = A*B
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT)
	blasd.SolveTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT)
	ok := B0.AllClose(B)
	t.Logf("B == trsm(trmm(B, A, L|U|N), A, L|U|N) : %v\n", ok)

	B.Copy(B0)
	// B = A.T*B
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT|gomas.TRANSA)
	blasd.SolveTrm(B, A, 1.0, gomas.UPPER|gomas.LEFT|gomas.TRANSA)
	ok = B0.AllClose(B)
	t.Logf("B == trsm(trmm(B, A, L|U|T), A, L|U|T) : %v\n", ok)
}

func TestDTrsm4(t *testing.T) {
    const N = 31
    const K = 4

	A := cmat.NewMatrix(N, N)
	B := cmat.NewMatrix(K, N)
	B0 := cmat.NewMatrix(K, N)

    ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource(0.0, 1.0)

	A.SetFrom(zeromean, cmat.UPPER)
	B.SetFrom(ones)
	B0.Copy(B)
	// B = B*A
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT)
	blasd.SolveTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT)
	ok := B0.AllClose(B)
	t.Logf("B == trsm(trmm(B, A, R|U|N), A, R|U|N) : %v\n", ok)

	B.Copy(B0)
	// B = B*A.T
	blasd.MultTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT|gomas.TRANSA)
	blasd.SolveTrm(B, A, 1.0, gomas.UPPER|gomas.RIGHT|gomas.TRANSA)
	ok = B0.AllClose(B)
	t.Logf("B == trsm(trmm(B, A, R|U|T), A, R|U|T) : %v\n", ok)
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

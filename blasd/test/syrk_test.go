
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)

const N = 711
const K = 95

func TestDSyrkLower(t *testing.T) {
    var ok bool
    conf := gomas.NewConf()

	A  := cmat.NewMatrix(N, N)
    A0 := cmat.NewMatrix(N, N)
    B  := cmat.NewMatrix(N, K)
    Bt := cmat.NewMatrix(K, N)

    ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource()
    _, _ = ones, zeromean

	A.SetFrom(ones, cmat.LOWER)
    A0.Copy(A)
    B.SetFrom(ones)
    Bt.Transpose(B)

	// B = A*B
	blasd.UpdateSym(A, B, 1.0, 1.0, gomas.LOWER, conf)
    blasd.Mult(A0, B, B, 1.0, 1.0, gomas.TRANSB)
    cmat.TriL(A0, cmat.NONE)
    ok = A0.AllClose(A)
    t.Logf("UpdateSym(A, B, L|N) == TriL(Mult(A, B, B.T)) : %v\n", ok)
    if N < 10 {
        t.Logf("UpdateSym(A, B)\n%v\n", A)
        t.Logf("Mult(A, B.T, B)\n%v\n", A0)
    }
	A.SetFrom(ones, cmat.LOWER)
    A0.Copy(A)

	blasd.UpdateSym(A, Bt, 1.0, 1.0, gomas.LOWER|gomas.TRANSA, conf)
    blasd.Mult(A0, Bt, Bt, 1.0, 1.0, gomas.TRANSA)
    cmat.TriL(A0, cmat.NONE)
    ok = A0.AllClose(A)
    t.Logf("UpdateSym(A, B, L|T) == TriL(Mult(A, B.T, B)) : %v\n", ok)
    if N < 10 {
        t.Logf("UpdateSym(A, B)\n%v\n", A)
        t.Logf("Mult(A, B.T, B)\n%v\n", A0)
    }
}


func TestDSyrkUpper(t *testing.T) {
    var ok bool
    conf := gomas.NewConf()

	A  := cmat.NewMatrix(N, N)
    A0 := cmat.NewMatrix(N, N)
    B  := cmat.NewMatrix(N, K)
    Bt := cmat.NewMatrix(K, N)

    ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource()
    _, _ = ones, zeromean

	A.SetFrom(ones, cmat.UPPER)
    A0.Copy(A)
    B.SetFrom(ones)
    Bt.Transpose(B)

	// B = A*B
	blasd.UpdateSym(A, B, 1.0, 1.0, gomas.UPPER, conf)
    blasd.Mult(A0, B, B, 1.0, 1.0, gomas.TRANSB)
    cmat.TriU(A0, cmat.NONE)
    ok = A0.AllClose(A)
    t.Logf("UpdateSym(A, B, U|N) == TriU(Mult(A, B, B.T)) : %v\n", ok)
    if N < 10 {
        t.Logf("UpdateSym(A, B)\n%v\n", A)
        t.Logf("Mult(A, B.T, B)\n%v\n", A0)
    }
	A.SetFrom(ones, cmat.UPPER)
    A0.Copy(A)

	blasd.UpdateSym(A, Bt, 1.0, 1.0, gomas.UPPER|gomas.TRANSA, conf)
    blasd.Mult(A0, Bt, Bt, 1.0, 1.0, gomas.TRANSA)
    cmat.TriU(A0, cmat.NONE)
    ok = A0.AllClose(A)
    t.Logf("UpdateSym(A, B, U|T) == TriU(Mult(A, B.T, B)) : %v\n", ok)
    if N < 10 {
        t.Logf("UpdateSym(A, B)\n%v\n", A)
        t.Logf("Mult(A, B.T, B)\n%v\n", A0)
    }
}



// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

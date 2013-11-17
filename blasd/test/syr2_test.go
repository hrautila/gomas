
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)

const N = 911

func TestDSyr2(t *testing.T) {

	A := cmat.NewMatrix(N, N)
    X := cmat.NewMatrix(N, 1)
    Y := cmat.NewMatrix(N, 1)
    B := cmat.NewMatrix(N, N)

    ones := cmat.NewFloatConstSource(1.0)
    twos := cmat.NewFloatConstSource(2.0)
	zeromean := cmat.NewFloatUniformSource(0.5, 2.0)

	A.SetFrom(zeromean, cmat.LOWER)
	X.SetFrom(ones)
	Y.SetFrom(twos)
    B.Copy(A)

	// B = A*B
	blasd.MVUpdate(B, X, Y, 1.0, gomas.NONE)
	blasd.MVUpdate(B, Y, X, 1.0, gomas.NONE)
    cmat.TriL(B, cmat.NONE)
	blasd.MVUpdate2Sym(A, X, Y, 1.0, gomas.LOWER)
    ok := B.AllClose(A)
    if N < 10 {
        t.Logf("A:\n%v\n", A)
        t.Logf("B:\n%v\n", B)
    }
    t.Logf("MVUpdate2Sym(A, X, Y, L) == TriL(MVUpdate(A, X, Y);MVUpdate(A, Y, X)) : %v\n", ok)

	A.SetFrom(zeromean, cmat.UPPER)
    cmat.TriU(A, cmat.NONE)
    B.Copy(A)
	blasd.MVUpdate(B, X, Y, 1.0, gomas.NONE)
	blasd.MVUpdate(B, Y, X, 1.0, gomas.NONE)
    cmat.TriU(B, cmat.NONE)
	blasd.MVUpdate2Sym(A, X, Y, 1.0, gomas.UPPER)
    ok = B.AllClose(A)
    if N < 10 {
        t.Logf("A:\n%v\n", A)
        t.Logf("B:\n%v\n", B)
    }
    t.Logf("MVUpdate2Sym(A, X, Y, U) == TriU(MVUpdate(A, X, Y);MVUpdate(A, Y, X)) : %v\n", ok)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

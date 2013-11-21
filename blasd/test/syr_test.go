
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas"
	"github.com/hrautila/gomas/blasd"
)


func TestDSyr1(t *testing.T) {
    
    const N = 911

	A := cmat.NewMatrix(N, N)
    X := cmat.NewMatrix(N, 1)
    B := cmat.NewMatrix(N, N)

    //ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource(0.5, 2.0)

	A.SetFrom(zeromean, cmat.LOWER)
	X.SetFrom(zeromean)
    B.Copy(A)

	// B = A*B
	blasd.MVUpdate(B, X, X, 1.0)
    cmat.TriL(B, cmat.NONE)
	blasd.MVUpdateSym(A, X, 1.0, gomas.LOWER)
    ok := B.AllClose(A)
    t.Logf("MVUpdateSym(A, X, L) == TriL(MVUpdate(A, X, X)) : %v\n", ok)

	A.SetFrom(zeromean, cmat.UPPER)
    cmat.TriU(A, cmat.NONE)
    B.Copy(A)
	blasd.MVUpdate(B, X, X, 1.0)
    cmat.TriU(B, cmat.NONE)
	blasd.MVUpdateSym(A, X, 1.0, gomas.UPPER)
    ok = B.AllClose(A)
    t.Logf("MVUpdateSym(A, X, U) == TriU(MVUpdate(A, X, X)) : %v\n", ok)
}

func TestDSyrOther(t *testing.T) {

    const N = 911

    var vec, As, Bs cmat.FloatMatrix
    P := N/3
	A := cmat.NewMatrix(P, P)
    X := cmat.NewMatrix(P, 1)
    B := cmat.NewMatrix(P, P)

    //ones := cmat.NewFloatConstSource(1.0)
	zeromean := cmat.NewFloatUniformSource(0.5, 2.0)

	A.SetFrom(zeromean, cmat.UPPER)
	X.SetFrom(zeromean)
    B.Copy(A)

    // update submatrices
    for i := 1; i < P; i++ {
        vec.SubMatrix(A, i-1, i, 1, P-i)
        As.SubMatrix(A, i, i)
        Bs.SubMatrix(B, i, i)
        // update with normal and symmetric
	    blasd.MVUpdate(&Bs, &vec, &vec, 1.0)
	    blasd.MVUpdateSym(&As, &vec, 1.0, gomas.UPPER)
    }
    // make normal update triangular and compare
    cmat.TriU(B, cmat.NONE)
    ok := B.AllClose(A)
    t.Logf("submatrix updates on upper triangular : %v\n", ok)

	A.SetFrom(zeromean, cmat.LOWER)
    cmat.TriL(A, cmat.NONE)
    B.Copy(A)
    // update submatrices
    for i := 1; i < P; i++ {
        vec.SubMatrix(A, i-1, i, 1, P-i)
        As.SubMatrix(A, i, i)
        Bs.SubMatrix(B, i, i)
        // update with normal and symmetric
	    blasd.MVUpdate(&Bs, &vec, &vec, 1.0)
	    blasd.MVUpdateSym(&As, &vec, 1.0, gomas.LOWER)
    }
    // make normal update triangular and compare
    cmat.TriL(B, cmat.NONE)
    ok = B.AllClose(A)
    t.Logf("submatrix updates on lower triangular : %v\n", ok)
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

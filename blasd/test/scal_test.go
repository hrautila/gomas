
package test

import (
	"testing"
	"github.com/hrautila/cmat"
	"github.com/hrautila/gomas/blasd"
)


func TestDVecScal(t *testing.T) {

    const N = 911

    X := cmat.NewMatrix(N, 1)
    Y := cmat.NewMatrix(N, 1)
    
	zeromean := cmat.NewFloatUniformSource(2.0, 0.5)

	X.SetFrom(zeromean)
    Y.Copy(X)

	// B = A*B
	blasd.Scale(X, 2.0)
    blasd.InvScale(X, 2.0)
    ok := X.AllClose(Y)
    t.Logf("X = InvScale(Scale(X, 2.0), 2.0) : %v\n", ok)

}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


package blasd

// #cgo CFLAGS: -O3 -march=native -fomit-frame-pointer -ffast-math -Iinc -I.
// #cgo LDFLAGS: -lm
// #include "inc/interfaces.h"
import  "C"
import "unsafe"
import "github.com/hrautila/cmat"
import "github.com/hrautila/gomas"

func vadd(X *cmat.FloatMatrix, alpha float64, N int)  {
    var x C.mvec_t

    xr, _ := X.Size()
    x.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    x.inc = C.int(1)
    if xr == 1 {
        x.inc = C.int(X.Stride())
    }
    C.__d_vec_add(
        (*C.mvec_t)(unsafe.Pointer(&x)), C.double(alpha), C.int(N))
    return
}

func vscal(X *cmat.FloatMatrix, alpha float64, N int)  {
    var x C.mvec_t

    xr, _ := X.Size()
    x.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    x.inc = C.int(1)
    if xr == 1 {
        x.inc = C.int(X.Stride())
    }
    C.__d_vec_scal(
        (*C.mvec_t)(unsafe.Pointer(&x)), C.double(alpha), C.int(N))
    return
}

func vinvscal(X *cmat.FloatMatrix, alpha float64, N int)  {
    var x C.mvec_t

    xr, _ := X.Size()
    x.md = (*C.double)(unsafe.Pointer(&X.Data()[0]))
    x.inc = C.int(1)
    if xr == 1 {
        x.inc = C.int(X.Stride())
    }
    C.__d_vec_invscal(
        (*C.mvec_t)(unsafe.Pointer(&x)), C.double(alpha), C.int(N))
    return
}

func madd(A *cmat.FloatMatrix, alpha float64, M, N int)  {
    var a C.mdata_t

    a.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    a.step = C.int(A.Stride())
    C.__d_blk_add(
        (*C.mdata_t)(unsafe.Pointer(&a)), C.double(alpha), C.int(M), C.int(N))
    return
}

func mscale(A *cmat.FloatMatrix, alpha float64, M, N int)  {
    var a C.mdata_t

    a.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    a.step = C.int(A.Stride())
    C.__d_blk_scale(
        (*C.mdata_t)(unsafe.Pointer(&a)), C.double(alpha), C.int(M), C.int(N))
    return
}

func minvscale(A *cmat.FloatMatrix, alpha float64, M, N int)  {
    var a C.mdata_t

    a.md = (*C.double)(unsafe.Pointer(&A.Data()[0]))
    a.step = C.int(A.Stride())
    C.__d_blk_invscale(
        (*C.mdata_t)(unsafe.Pointer(&a)), C.double(alpha), C.int(M), C.int(N))
    return
}

func Add(A *cmat.FloatMatrix, alpha float64, confs ...*gomas.Config) *gomas.Error {
    if A.Len() == 0 {
        return nil
    }
    ar, ac := A.Size()
    if ar != 1 && ac != 1 {
        madd(A, alpha, ar, ac)
        return nil
    }
    vadd(A, alpha, A.Len())
    return nil
}

func Scale(A *cmat.FloatMatrix, alpha float64, confs ...*gomas.Config) *gomas.Error {
    if A.Len() == 0 {
        return nil
    }
    ar, ac := A.Size()
    if ar != 1 && ac != 1 {
        mscale(A, alpha, ar, ac)
        return  nil
    }
    vscal(A, alpha, A.Len())
    return nil
}

func InvScale(A *cmat.FloatMatrix, alpha float64, confs ...*gomas.Config) *gomas.Error {
    if A.Len() == 0 {
        return nil
    }
    ar, ac := A.Size()
    if ar != 1 && ac != 1 {
        minvscale(A, alpha, ar, ac)
        return nil
    }
    vinvscal(A, alpha, A.Len())
    return nil
}


// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

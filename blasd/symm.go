
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package blasd

// #cgo CFLAGS: -O3 -march=native -fomit-frame-pointer -ffast-math -Iinc -I.
// #cgo LDFLAGS: -lm
// #include "inc/interfaces.h"
import  "C"
import "unsafe"
import "github.com/hrautila/cmat"
import "github.com/hrautila/gomas"

func symm(Cc, Ac, Bc *cmat.FloatMatrix, alpha, beta float64, bits, P, S, L, R, E int, conf *gomas.Config) error {
    var Am, Cm, Bm C.mdata_t
    
    Am.md = (*C.double)(unsafe.Pointer(&Ac.Data()[0]))
    Am.step = C.int(Ac.Stride())
    Bm.md = (*C.double)(unsafe.Pointer(&Bc.Data()[0]))
    Bm.step = C.int(Bc.Stride())
    Cm.md = (*C.double)(unsafe.Pointer(&Cc.Data()[0]))
    Cm.step = C.int(Cc.Stride())

    C.__d_symm_inner(
        (*C.mdata_t)(unsafe.Pointer(&Cm)), 
        (*C.mdata_t)(unsafe.Pointer(&Am)), 
        (*C.mdata_t)(unsafe.Pointer(&Bm)),
        C.double(alpha), C.double(beta), C.int(bits),
        C.int(P), C.int(S), C.int(L), C.int(R), C.int(E),
        C.int(conf.KB), C.int(conf.NB), C.int(conf.MB))
    return nil
}

/*
 * Symmetric matrix-matrix multiplication.
 *
 * Computes C = beta*C + alpha*symm(A)*op(B) or C = beta*C + alpha*op(B)*symm(A) where
 * op is optional tranpose operation and symm defines symmetric matrix form, upper or
 * lower triangular.
 *
 * Matrix A is symmetric lower triangular if gomas.LOWER bit is set. If gomas.UPPER is
 * set then matrix A is upper triangular. First form, multiplication from left, is
 * indicated with bit gomas.LEFT. Second form, multiplication from right, is indicated
 * with bit gomas.RIGHT. Matrix B is transposed is bit gomas.TRANSB is set.
 */
func MultSym(Cc, A, B *cmat.FloatMatrix, alpha, beta float64, bits int, confs... *gomas.Config) *gomas.Error {

    conf := gomas.DefaultConf()
    if len(confs) > 0 {
        conf = confs[0]
    }

    // if A or B has zero columns or rows
    if A.Len() == 0 || B.Len() == 0 {
        return nil
    }
    ok := true
    cr, cc := Cc.Size()
    ar, ac := A.Size()
    br, bc := B.Size()
    P := ac
    L := cc
    E := cr
    switch bits & (gomas.LEFT|gomas.RIGHT) {
    case gomas.RIGHT:
        ok = cr == br && cc == ac && bc == ar && ar == ac
        P = ac
    case gomas.LEFT:
        fallthrough
    default:
        ok = cr == ar && cc == bc && ac == br && ar == ac
    }
    if !ok {
        return gomas.NewError(gomas.ESIZE, "MultSym")
    }
    // single threaded
    if conf.NProc == 1 || conf.WB <= 0 || Cc.Len() < conf.WB*conf.WB {
        symm(Cc, A, B, alpha, beta, bits, P, 0, L, 0, E, conf)
        return nil
    }

    // parallelized
    wait := make(chan int, 4)
    nM, nN := blocking(cr, cc, conf.WB)
    nT := int64(0)

    for j := 0; j < nN; j++ {
        jS := blockIndex(j, nN, conf.WB, cc)
        jL := blockIndex(j+1, nN, conf.WB, cc)
        for i := 0; i < nM; i++ {
            iR := blockIndex(i,   nM, conf.WB, cr)
            iE := blockIndex(i+1, nM, conf.WB, cr)
            task := func(q chan int) {
                symm(Cc, A, B, alpha, beta, bits, P, jS, jL, iR, iE, conf)
                q <- 1
            }
            nT += 1
            conf.Sched.Schedule(gomas.NewTask(task, wait))
        }
    }
    // wait for subtask to complete
    for nT > 0 {
        <- wait
        nT -= 1
    }
    return nil
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

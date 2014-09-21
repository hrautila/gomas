
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package gomas

import "fmt"

const (
    _ = iota
    ENEED_VECTOR
    ESIZE
    EVALUE
    ENEGATIVE
    ENOTSQUARE
    ESIZE_PIVOTS
    ESINGULAR
    EWORK
    ESMALL
    ECONVERGE
)


var errors = map[int]string {
    ENEED_VECTOR: "Vector argument needed",
    ESIZE: "Operand argument size mismatch",
    EVALUE: "Illegal value",
    ENEGATIVE: "Negative value",
    ENOTSQUARE: "Not a square matrix",
    ESIZE_PIVOTS: "Pivot array too small",
    ESINGULAR: "Zero on diagonal",
    EWORK: "Work space too small",
    ESMALL: "Argument matrix too small.", 
    ECONVERGE: "Did not converge.", 
}

type Error struct {
    // error code
    Err int
    // operator
    Op string
    // additional information
    Info int
}

func (e *Error) Error() string {
    var desc string
    var ok bool
    desc, ok = errors[e.Err]
    if !ok {
        desc = "Unknwon error code"
    }
    return fmt.Sprintf("%d [%s] %d: %s", e.Err, e.Op, e.Info, desc);
}

func NewError(err int, op string, infos... int) *Error {
    info := 0
    if len(infos) > 0 {
        info = infos[0]
    }
    return &Error{err, op, info}
}

func (e *Error) Set(err int, op string, info int) *Error {
    e.Err = err
    e.Op = op
    e.Info = info
    return e
}

// Prepend Op field with string s to form s.Op.
func (e *Error) Update(s string) *Error {
    e.Op = s + "." + e.Op
    return e
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


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
)

var errors = map[int]string {
    ENEED_VECTOR: "Vector argument needed",
    ESIZE: "Operand argument size mismatch",
    EVALUE: "Illegal value",
    ENEGATIVE: "Negative value",
    ENOTSQUARE: "Not a square matrix",
    ESIZE_PIVOTS: "Pivot array too small",
    ESINGULAR: "Zero on diagonal",
}

type Error struct {
    // error code
    Err int
    // operator
    Op string
    // possible invalid value location
    I, J int
}

func (e *Error) Error() string {
    var desc string
    var ok bool
    desc, ok = errors[e.Err]
    if !ok {
        desc = "Unknwon error code"
    }
    return fmt.Sprintf("%d [%s]: %s, (%d,%d)", e.Err, e.Op, desc, e.I, e.J);
}

func NewError(err int, op string, ijs... int) *Error {
    var i, j int
    switch len(ijs) {
    case 2:
        i = ijs[0]
        j = ijs[1]
    case 1:
        i = ijs[0]
        j = i
    }
    return &Error{err, op, i, j}
}

func (e *Error) Set(err int, op string, ijs... int) *Error {
    var i, j int
    switch (len(ijs)) {
    case 2:
        i = ijs[0]
        j = ijs[1]
    case 1:
        i = ijs[0]
        j = i
    }
    e.Err = err
    e.Op = op
    e.I = i
    e.J = j
    return e
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

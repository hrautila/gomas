
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package gomas

// these bits must be in sync with bits in C-header interfaces.h
const (
    TRANSA = 1 << iota
    TRANSB
    TRANS
    LOWER
    UPPER
    LEFT
    RIGHT
    UNIT
    CONJA
    CONJB
    CONJ
    SYMM
    NULL = 0
    NONE = 0
)
        

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:


// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package gomas


type Config struct {
    // row block size for blocked blasd.
    MB int
    // column block size for blocked blas
    NB int
    // row/column block size for blocked blas multiplication
    KB int
    // blocking factor for lapack routines
	LB int
    // last error
    LastErr Error
}

var __config Config = Config{64, 96, 160, 32, Error{0, "", 0, 0}}

func DefaultConf() *Config {
    return &__config
} 

func NewConf() *Config {
    return &Config{64, 96, 160, 32, Error{0, "", 0, 0}}
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

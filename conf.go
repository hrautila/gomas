
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING tile included in this archive.

package gomas

import (
    "os"
    "strings"
    "strconv"
)

// Blocking configuration for blas/lapack functions. Default configuration 
// may be changed by setting environment variable GOMASCONFIG. Its value should be
// comma separated list of fields. If field value is missing or is zero then compile
// time default is not changed. Fields are: MB,NB,KB,LB,NProc,WB
//
// Examples: GOMASCONFIG="80,96,160,44,2,400" or GOMASCONFIG=",,,,2"
//
type Config struct {
    // row block size for blocked blasd.
    MB int
    // column block size for blocked blas
    NB int
    // row/column block size for blocked blas multiplication
    KB int
    // blocking factor for lapack routines
	LB int
    // Number of workers used for parallel execution
    NProc int
    // block factor for parallel execution
    WB int
    // current scheduler
    Sched *Scheduler
}

var __config Config = Config{64, 96, 160, 32, 1, 480, nil}

func init() {
    var cval int64
    cstr := os.Getenv("GOMASCONFIG")
    if len(cstr) == 0 {
        return
    }
    elems := strings.Split(cstr, ",")

    // config.MB
    cval, _ = strconv.ParseInt(elems[0], 10, 64)
    if cval > 0 {
        __config.MB = int(cval)
    }
    if len(elems) == 1 { return }
    // config.NB
    cval, _ = strconv.ParseInt(elems[1], 10, 64)
    if cval > 0 {
        __config.NB = int(cval)
    }
    if len(elems) == 2 { return }
    // config.KB
    cval, _ = strconv.ParseInt(elems[2], 10, 64)
    if cval > 0 {
        __config.KB = int(cval)
    }
    if len(elems) == 3 { return }
    // config.LB
    cval, _ = strconv.ParseInt(elems[3], 10, 64)
    if cval > 0 {
        __config.LB = int(cval)
    }
    if len(elems) == 4 { return }
    // config.NProc
    cval, _ = strconv.ParseInt(elems[4], 10, 64)
    if cval > 0 {
        __config.NProc = int(cval)
    }
    if len(elems) == 5 { return }
    // config.WB
    cval, _ = strconv.ParseInt(elems[5], 10, 64)
    if cval > 0 {
        __config.WB = int(cval)
    }
    if __config.NProc > 1 {
        rrsched := false
        cstr = os.Getenv("GOMASSCHED")
        if len(cstr) >= 2 {
            rrsched = cstr[0] == 'R' && cstr[1] == 'R';
        }
        createDefaultScheduler(__config.NProc, rrsched)
        __config.Sched = CurrentScheduler()
    }
}

// Get default configuration
func DefaultConf() *Config {
    return &__config
} 

// Create new configuration block with initial values from default configuration.
func NewConf() *Config {
    return &Config{__config.MB, __config.NB, __config.KB, __config.LB, __config.NProc, __config.WB, __config.Sched}
}

// Get current configuration. If parameter list is not empty return first configuration
// in list. Otherwise returns default configuration.
func CurrentConf(confs... *Config) *Config {
    if len(confs) > 0 {
        return confs[0]
    }
    return &__config
}

        
// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

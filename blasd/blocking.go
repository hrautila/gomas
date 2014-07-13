
// Copyright (c) Harri Rautila, 2013

// This file is part of github.com/hrautila/gomas package. It is free software,
// distributed under the terms of GNU Lesser General Public License Version 3, or
// any later version. See the COPYING file included in this archive.

package blasd

import (
    //"github.com/hrautila/gomas"
)

// Calculate how many row/column blocks are needed with current blocking size.
func blocking(M, N, WB int) (int, int) {
    nM := M/WB
    nN := N/WB
    // if remaing block bigger the 10% of WB then will be separate block
    if M % WB > WB/10 {
        nM += 1
    }
    if N % WB > WB/10 {
        nN += 1
    }
    return nM, nN
}

// compute start of k'th out of nblk block when block size wb and total is K
// requires: K/wb == nblk or K/wb == nblk-1
func blockIndex(k, nblk, wb, K int) int {
    if k == nblk {
        return K
    }
    return k*wb
}

// Local Variables:
// tab-width: 4
// indent-tabs-mode: nil
// End:

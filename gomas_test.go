
package gomas

import "testing"

func TestCompile(t *testing.T) {
	t.Logf("OK\n")
}

func TestConfig(t *testing.T) {
	conf := DefaultConf()
	t.Logf("config.MB=%d\n", conf.MB)
	t.Logf("config.NB=%d\n", conf.NB)
	t.Logf("config.KB=%d\n", conf.KB)
	t.Logf("config.LB=%d\n", conf.LB)
	t.Logf("config.NProc=%d\n", conf.NProc)
	t.Logf("config.WB=%d\n", conf.WB)
}

func TestBits(t *testing.T) {
	t.Logf("LOWER  = 0x%05x\n", LOWER)
	t.Logf("UPPER  = 0x%05x\n", UPPER)
	t.Logf("SYMM   = 0x%05x\n", SYMM)
	t.Logf("HERM   = 0x%05x\n", HERM)
	t.Logf("UNIT   = 0x%05x\n", UNIT)
	t.Logf("LEFT   = 0x%05x\n", LEFT)
	t.Logf("RIGHT  = 0x%05x\n", RIGHT)
	t.Logf("TRANSA = 0x%05x\n", TRANSA)
	t.Logf("TRANS  = 0x%05x\n", TRANS)
	t.Logf("TRANSB = 0x%05x\n", TRANSB)
	t.Logf("CONJA  = 0x%05x\n", CONJA)
	t.Logf("CONJ   = 0x%05x\n", CONJ)
	t.Logf("CONJB  = 0x%05x\n", CONJB)
	t.Logf("MULTQ  = 0x%05x\n", MULTQ)
	t.Logf("MULTP  = 0x%05x\n", MULTP)
	t.Logf("WANTQ  = 0x%05x\n", WANTQ)
	t.Logf("WANTP  = 0x%05x\n", WANTP)
}

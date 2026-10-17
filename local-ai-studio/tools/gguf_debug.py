"""Debug: read KV pairs one by one with offset tracking to find misalignment."""
import struct, sys
from pathlib import Path

VT = {0:("B",),1:("b",),2:("H",),3:("h",),4:("I",),5:("i",),6:("f",),7:("?",),8:("s",),9:("a",),10:("Q",),11:("q",),12:("d",)}

def rs(fh):
    (n,) = struct.unpack("<Q", fh.read(8))
    if n > 1_000_000: raise ValueError(f"string len {n}")
    return fh.read(n).decode("utf-8", errors="replace")

def rv(fh, t, d=0):
    if t == 9:
        (at,) = struct.unpack("<I", fh.read(4))
        (cnt,) = struct.unpack("<Q", fh.read(8))
        if cnt > 200000: raise ValueError(f"array count {cnt}")
        if at == 8: return [rs(fh) for _ in range(cnt)]
        if at == 9: return [rv(fh, at, d+1) for _ in range(cnt)]
        f = VT.get(at, (None,))[0]
        if f is None: raise ValueError(f"bad array type {at}")
        sz = struct.calcsize("<"+f); data = fh.read(sz*cnt)
        return list(struct.unpack("<"+f*(len(data)//sz), data))
    if t == 8: return rs(fh)
    f = VT.get(t, (None,))[0]
    if f is None: raise ValueError(f"bad type {t}")
    (v,) = struct.unpack("<"+f, fh.read(struct.calcsize("<"+f)))
    return v

p = Path(sys.argv[1])
with p.open("rb") as fh:
    magic, ver = struct.unpack("<II", fh.read(8))
    (tc,) = struct.unpack("<Q", fh.read(8))
    (kc,) = struct.unpack("<Q", fh.read(8))
    print(f"magic=0x{magic:08x} ver={ver} tensors={tc} kv={kc}")
    ok = 0
    for i in range(kc):
        off = fh.tell()
        try:
            k = rs(fh)
            (t,) = struct.unpack("<I", fh.read(4))
            v = rv(fh, t)
            vs = str(v)
            print(f"[{i:3d}] off={off:>10} type={t:2d} {k} = {vs[:120]}")
            ok += 1
        except Exception as e:
            print(f"[{i:3d}] off={off:>10} FAILED: {e}")
            print("context bytes:", fh.seek(max(0, off-16)), fh.read(64))
            break
    print(f"parsed {ok}/{kc} KV pairs")

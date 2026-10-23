"""Compress raw entries of an existing SEC response cache in place (entries written before the client compressed them).
The client reads both forms, so this only frees disk. Run it while no pipeline stage is downloading.

    python tools/compact_sec_cache.py <HOT_ROOT>/sec_cache [--max-mb 400]   # --max-mb: delete entries larger than this
"""
from __future__ import annotations

import argparse
import os
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from secprice.sec.client import _ZMAGIC  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("--max-mb", type=float, default=0.0, help="delete (uncompressed) entries above this size; 0 = keep all")
    a = ap.parse_args()
    before = after = n = dropped = 0
    for p in Path(a.cache_dir).rglob("*.bin"):
        b = p.read_bytes()
        before += len(b)
        if b.startswith(_ZMAGIC) or b == b"__404__":
            if a.max_mb and b.startswith(_ZMAGIC) and len(zlib.decompress(b[len(_ZMAGIC):])) > a.max_mb * 1e6:
                p.unlink()
                dropped += 1
                continue
            after += len(b)
            continue
        if a.max_mb and len(b) > a.max_mb * 1e6:
            p.unlink()
            dropped += 1
            continue
        z = _ZMAGIC + zlib.compress(b, 6)
        tmp = p.with_suffix(".compact.tmp")
        tmp.write_bytes(z)
        os.replace(tmp, p)
        after += len(z)
        n += 1
    print(f"compressed {n:,} entries, dropped {dropped:,}; {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

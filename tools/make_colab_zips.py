"""Pack the local run folders into single zip files for Google Drive / Colab (one big file uploads far faster and more
reliably than thousands of small ones; the notebook unpacks them on first use).

    python tools/make_colab_zips.py <DATA_ROOT> <OUT_DIR> [profile]

Writes <OUT_DIR>/<profile>.zip (entries start with '<profile>/') and <OUT_DIR>/sec_inputs_for_colab.zip (entries relative to
inputs/). The S05 'exhibits' folder is left out: only S06 reads it, and S06 is already finished in the packed run.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def main() -> None:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    profile = sys.argv[3] if len(sys.argv) > 3 else "full"
    out.mkdir(parents=True, exist_ok=True)
    z = out / f"{profile}.zip"
    n = size = 0
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:   # parquet is already compressed
        for p in sorted((root / profile).rglob("*")):
            rel = p.relative_to(root).as_posix()
            if not p.is_file() or "/exhibits/" in rel or p.name.startswith(".tmp") or p.suffix == ".tmp":
                continue
            zf.write(p, rel)
            n += 1
            size += p.stat().st_size
    print(f"{z}: {n} files, {size / 1e9:.2f} GB")
    zi = out / "sec_inputs_for_colab.zip"
    with zipfile.ZipFile(zi, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zf:
        k = 0
        for p in sorted((root / "inputs").rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(root / "inputs").as_posix())
                k += 1
    print(f"{zi}: {k} files, {zi.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()

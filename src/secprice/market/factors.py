"""Kenneth French daily factors: FF5 (2x3) + momentum -> factors.parquet (decimal returns)."""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests


def _parse_french_zip(raw: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(raw))
    text = z.read(z.namelist()[0]).decode("latin1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",") and any(c.isalpha() for c in l))
    header = [h.strip() for h in lines[start].split(",")]
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def fetch_factors(ff5_url: str, mom_url: str, cache_dir=None) -> tuple[pd.DataFrame, dict]:
    import hashlib
    raws = {}
    for name, url in (("ff5", ff5_url), ("mom", mom_url)):
        cp = (cache_dir / f"{name}.zip") if cache_dir is not None else None
        if cp is not None and cp.exists():
            raws[name] = cp.read_bytes()
        else:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            raws[name] = r.content
            if cp is not None:
                cp.write_bytes(r.content)
    ff5 = _parse_french_zip(raws["ff5"]).rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RMW": "rmw",
                                                         "CMA": "cma", "RF": "rf"})
    mom = _parse_french_zip(raws["mom"])
    mom = mom.rename(columns={mom.columns[1]: "mom"})[["date", "mom"]]
    out = ff5.merge(mom, on="date", how="left")[["date", "mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]]
    prov = {k: hashlib.sha256(v).hexdigest() for k, v in raws.items()}
    prov.update({"last_date": str(out.date.max().date()), "first_date": str(out.date.min().date())})
    return out, prov

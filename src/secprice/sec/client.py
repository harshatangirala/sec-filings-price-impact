"""Thin SEC EDGAR client: declared User-Agent from env, global rate limit, cache, backoff, block detection
(SPEC.md hard rule 13, §13.5). No proxy rotation or evasion; a block pauses the run and is reported."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import zlib
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


_ZMAGIC = b"ZC1\x00"          # compressed cache entry marker (older raw entries are still read)


class SECBlockedError(RuntimeError):
    """Raised after repeated 403s: the run pauses and the owner is told (never evaded)."""


class UserAgentMissing(RuntimeError):
    pass


class RateLimiter:
    """Token bucket shared by all download threads."""

    def __init__(self, rps: float):
        self.interval = 1.0 / float(rps)
        self.lock = threading.Lock()
        self.next_t = time.monotonic()
        self.pause_until = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_t, self.pause_until)
            self.next_t = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def pause(self, seconds: float) -> None:
        with self.lock:
            self.pause_until = max(self.pause_until, time.monotonic() + seconds)


def resolve_user_agent(env_name: str = "SEC_USER_AGENT") -> str:
    ua = os.environ.get(env_name, "").strip()
    if not ua or "@" not in ua:
        raise UserAgentMissing(
            f"Set {env_name} to 'Your Name your.email@example.com' (Colab secret or environment variable). "
            "SEC fair access requires a declared contact; it is never hard-coded.")
    return ua


class SECClient:
    def __init__(self, user_agent: str, max_rps: float = 6, cache_dir: str | os.PathLike | None = None,
                 cache_days: float = 30, timeout_s: float = 60, max_retries: int = 6, backoff_base_s: float = 2.0,
                 blocked_403_limit: int = 20, log=None, cache_max_bytes: int = 0):
        self.ua = user_agent
        self.cache_max_bytes = int(cache_max_bytes or 0)   # 0 = cache every response; >0 = skip larger bodies (disk budget)
        self.limiter = RateLimiter(max_rps)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_s = float(cache_days) * 86400
        self.timeout = timeout_s
        self.max_retries = int(max_retries)
        self.backoff = float(backoff_base_s)
        self.blocked_limit = int(blocked_403_limit)
        self._consec_403 = 0
        self._lock = threading.Lock()
        self.log = log
        self.stats = {"requests": 0, "cache_hits": 0, "429": 0, "5xx": 0, "403": 0, "404": 0, "bytes": 0, "truncated": 0}
        self._local = threading.local()

    @classmethod
    def from_config(cls, cfg: dict, cache_dir, log=None) -> "SECClient":
        s = cfg["sec"]
        return cls(resolve_user_agent(s["user_agent_env"]), s["max_rps"], cache_dir, s["cache_days"], s["timeout_s"],
                   s["max_retries"], s["backoff_base_s"], s.get("blocked_403_limit", 20), log, s.get("cache_max_bytes", 0))

    def _session(self) -> requests.Session:
        sess = getattr(self._local, "sess", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"})
            self._local.sess = sess
        return sess

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        h = hashlib.sha1(url.encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.bin"

    def get(self, url: str, max_bytes: int | None = None, use_cache: bool = True) -> bytes | None:
        """Bytes of the URL, None on 404. Raises SECBlockedError on sustained 403."""
        key = url if not max_bytes or max_bytes >= 1_000_000 else f"{url}#prefix{max_bytes}"   # never mix a prefix with the full file
        cp = self._cache_path(key) if use_cache else None
        if cp and cp.exists() and (time.time() - cp.stat().st_mtime) < self.cache_s:
            with self._lock:
                self.stats["cache_hits"] += 1
            data = cp.read_bytes()
            if data.startswith(_ZMAGIC):
                data = zlib.decompress(data[len(_ZMAGIC):])
            return None if data == b"__404__" else data
        attempt = 0
        while True:
            attempt += 1
            self.limiter.wait()
            try:
                r = self._session().get(url, timeout=self.timeout, stream=True)
                status = r.status_code
                if status == 200:
                    buf = bytearray()
                    truncated = False
                    for chunk in r.iter_content(1 << 16):
                        buf.extend(chunk)
                        if max_bytes and len(buf) >= max_bytes:
                            truncated = True
                            break
                    r.close()
                    data = bytes(buf[:max_bytes] if max_bytes else buf)
                    with self._lock:
                        self.stats["requests"] += 1
                        self.stats["bytes"] += len(data)
                        self.stats["truncated"] += int(truncated)
                        self._consec_403 = 0
                    if cp and (not self.cache_max_bytes or len(data) <= self.cache_max_bytes):
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        tmp = cp.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                        tmp.write_bytes(_ZMAGIC + zlib.compress(data, 6))      # filings are text: ~5-10x smaller on disk
                        os.replace(tmp, cp)
                    return data
                r.close()
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                status = -1
                if attempt >= self.max_retries:
                    raise RuntimeError(f"network error after {attempt} attempts: {url}: {e}")
            with self._lock:
                self.stats["requests"] += 1
            if status == 404:
                with self._lock:
                    self.stats["404"] += 1
                if cp:
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    cp.write_bytes(b"__404__")
                return None
            if status == 403:
                with self._lock:
                    self.stats["403"] += 1
                    self._consec_403 += 1
                    blocked = self._consec_403 >= self.blocked_limit
                if blocked:
                    raise SECBlockedError(f"{self._consec_403} consecutive 403 responses from SEC (last: {url}). "
                                          "Pausing per fair-access policy; check the User-Agent and request rate.")
                self.limiter.pause(30 * attempt)
            elif status == 429 or status >= 500 or status == -1:
                with self._lock:
                    self.stats["429" if status == 429 else "5xx"] += 1
                self.limiter.pause(self.backoff ** min(attempt, 6))
            else:
                if attempt >= 2:
                    return None
            if attempt >= self.max_retries:
                if status == 403:
                    return None
                raise RuntimeError(f"HTTP {status} after {attempt} attempts: {url}")

    def get_json(self, url: str, use_cache: bool = True):
        b = self.get(url, use_cache=use_cache)
        return None if b is None else json.loads(b)


class FixtureSECClient:
    """Offline client: serves recorded/synthetic responses from tests/fixtures/sec/<host>/<path>."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.stats = {"requests": 0, "cache_hits": 0, "429": 0, "5xx": 0, "403": 0, "404": 0, "bytes": 0, "truncated": 0}
        self.ua = "fixture"

    def _path(self, url: str) -> Path:
        u = urlparse(url)
        return self.root / u.netloc / u.path.lstrip("/")

    def get(self, url: str, max_bytes: int | None = None, use_cache: bool = True) -> bytes | None:
        p = self._path(url)
        self.stats["requests"] += 1
        if not p.exists():
            self.stats["404"] += 1
            return None
        data = p.read_bytes()
        self.stats["bytes"] += len(data)
        return data[:max_bytes] if max_bytes else data

    def get_json(self, url: str, use_cache: bool = True):
        b = self.get(url)
        return None if b is None else json.loads(b)

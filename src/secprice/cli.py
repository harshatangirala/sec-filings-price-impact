"""secprice CLI (SPEC.md §17.2)."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    from . import pipeline as P
    ap = argparse.ArgumentParser(prog="secprice")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--profile", default=None)
    r.add_argument("--stages", nargs="+", required=True, help="S05, 'S00-S11' or a list")
    r.add_argument("--force", action="store_true")
    r.add_argument("--with-deps", action="store_true")
    r.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value (JSON values)")
    for name in ("status", "preflight", "prereg-freeze", "freeze", "report", "env"):
        p = sub.add_parser(name)
        p.add_argument("--profile", default=None)
        if name == "prereg-freeze":
            p.add_argument("--force", action="store_true")
    u = sub.add_parser("unlock-test")
    u.add_argument("--profile", default=None)
    u.add_argument("--acknowledge-reuse", action="store_true")
    g = sub.add_parser("gold")
    g.add_argument("action", choices=["export", "import", "agreement"])
    g.add_argument("--profile", default=None)
    s = sub.add_parser("sync")
    s.add_argument("direction", choices=["push", "pull"])
    s.add_argument("--profile", required=True)
    s.add_argument("--local", required=True)
    s.add_argument("--remote", required=True)
    a = ap.parse_args(argv)

    def ov(pairs):
        out = {}
        for kv in pairs:
            k, v = kv.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            cur = out
            parts = k.split(".")
            for p_ in parts[:-1]:
                cur = cur.setdefault(p_, {})
            cur[parts[-1]] = v
        return out

    if a.cmd == "run":
        print(P.run(a.stages, a.profile, ov(a.set), force=a.force, with_deps=a.with_deps))
    elif a.cmd == "status":
        print(P.status(a.profile).to_string(index=False))
    elif a.cmd == "preflight":
        print(P.run(["S13"], a.profile, force=True))
    elif a.cmd == "prereg-freeze":
        print(P.prereg_freeze(a.profile, force=a.force))
    elif a.cmd == "freeze":
        print(P.freeze(a.profile))
    elif a.cmd == "unlock-test":
        print(P.unlock_test(a.profile, acknowledge_reuse=a.acknowledge_reuse))
    elif a.cmd == "report":
        print(P.run(["S24"], a.profile, force=True))
    elif a.cmd == "env":
        print(P.run(["S00"], a.profile, force=True))
    elif a.cmd == "gold":
        print(P.run(["S12"], a.profile, force=True))
    elif a.cmd == "sync":
        print(P.sync(a.direction, a.local, a.remote, a.profile))
    return 0


if __name__ == "__main__":
    sys.exit(main())

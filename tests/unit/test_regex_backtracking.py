"""No catastrophic regex backtracking on adversarial filing lines (found in the full run: html.normalize spent hours on a
wide table row of empty cells). Every module-level compiled pattern in the parser modules, plus the normalisation path,
must finish fast on long pathological inputs."""
import re
import time

import pytest

from secprice.parse import eps_release, html, sections, taxonomy

ADVERSARIAL = [
    "a" + " | " * 4000 + "x",                                   # many empty cells, then text (trailing-pipe pattern fails)
    " | " * 4000,
    "Diluted" + " 1" * 4000 + " x",                             # long numeric row that fails the end anchor
    "Diluted $" + " $ 0" * 3000 + " end",
    "management " * 3000,
    "(" * 3000 + "0.12",
    " " * 20000 + "x",
    "1," * 5000 + "x",
    "per share " * 2000 + "$0.1",
]


def _patterns():
    for mod in (html, eps_release, sections, taxonomy):
        for name, v in vars(mod).items():
            if isinstance(v, re.Pattern):
                yield f"{mod.__name__}.{name}", v
            elif isinstance(v, (list, tuple)):
                for i, x in enumerate(v):
                    rx = x[0] if isinstance(x, tuple) and x and isinstance(x[0], re.Pattern) else x
                    if isinstance(rx, re.Pattern):
                        yield f"{mod.__name__}.{name}[{i}]", rx


@pytest.mark.parametrize("name,rx", list(_patterns()))
def test_pattern_is_linear_enough(name, rx):
    for s in ADVERSARIAL:
        t0 = time.perf_counter()
        rx.search(s)
        list(rx.finditer(s[:5000]))
        assert time.perf_counter() - t0 < 2.0, f"{name} is slow on {s[:40]!r}..."


def test_normalize_and_eps_parser_fast_on_wide_tables():
    row = "<tr>" + "<td></td>" * 3000 + "<td>Net income</td></tr>"
    doc = "<html><body><table>" + row * 3 + "</table></body></html>"
    t0 = time.perf_counter()
    txt, _, _ = html.to_text(doc)
    eps_release.parse_release_eps(txt + "\n" + "Diluted" + " | " * 3000 + "0.12")
    assert time.perf_counter() - t0 < 5.0

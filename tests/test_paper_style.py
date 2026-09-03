"""House style of the paper sources (user rules, 2026-09-03).

1. No double-dash expressions anywhere in the rendered text: neither the em dash (---) used as
   punctuation nor the en dash (--) used for ranges or compound names. Ranges are written with "to",
   compounds with a hyphen, table placeholders as n/a.
2. A link (\\url, \\href, a bare http URL) may appear only inside a footnote (\\footnote or
   \\tablefootnote) or in the bibliography; never inline in prose, captions or table cells.

The check strips LaTeX comments first, so a commented-out line does not count. It runs over every
source the paper inputs: main.tex, src/, tab/, figtex/.
"""
import os, re, glob
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "6a3af1ee773478ddf7390bb9")
FILES = [os.path.join(PAPER, "main.tex")] + sorted(glob.glob(os.path.join(PAPER, "src", "*.tex"))) + \
        sorted(glob.glob(os.path.join(PAPER, "tab", "*.tex"))) + sorted(glob.glob(os.path.join(PAPER, "figtex", "*.tex")))


def _stripped(path):
    out = []
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        # drop comments (a % not preceded by a backslash starts one)
        line = re.sub(r"(?<!\\)%.*$", "", line)
        out.append((i, line))
    return out


def _rel(p): return os.path.relpath(p, PAPER)


def test_paper_sources_exist():
    assert os.path.isfile(os.path.join(PAPER, "main.tex"))
    assert len(FILES) > 5


def test_no_double_dash_in_rendered_text():
    hits = [f"{_rel(p)}:{i}: {line.strip()[:90]}" for p in FILES for i, line in _stripped(p) if "--" in line]
    assert not hits, "double dashes in the paper sources:\n" + "\n".join(hits)


def test_links_only_in_footnotes():
    link = re.compile(r"\\url\{|\\href\{|https?://")
    inside = re.compile(r"\\(?:table)?footnote\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")   # footnote{...} with one nesting level
    hits = []
    for p in FILES:
        for i, line in _stripped(p):
            if not link.search(line):
                continue
            rest = inside.sub("", line)          # remove every footnote body, then look again
            if link.search(rest):
                hits.append(f"{_rel(p)}:{i}: {line.strip()[:90]}")
    assert not hits, "links outside footnotes:\n" + "\n".join(hits)

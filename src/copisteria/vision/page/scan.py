"""A scanned page.

A scanned PDF usually holds each page as one image (300 ppi 8-bit grey or 400 ppi 1-bit are typical).
``pdfimages`` pulls the scan out of the PDF byte for byte; re-rendering through a rasteriser would resample it, so
that is done only for a page the scan cannot be pulled from.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

RENDER_PPI = 300        # a page with no single embedded scan is rasterised at what the scanned pages carry


@dataclass
class Page:
    stem: str                               # file-safe name, e.g. "Dinah_p01"
    source: str                             # "Dinah.pdf#1"
    path: Path                              # the PNG on disk
    dpi: int
    size: tuple[int, int] = (0, 0)          # width, height in pixels

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def image(self) -> Image.Image:
        return Image.open(self.path).convert("L")


def from_pdf(pdf: str | Path, page_no: int, cache: str | Path) -> Page:
    """Page ``page_no`` (1-based) of ``pdf`` as a grey PNG in ``cache``, extracted once."""
    pdf, cache = Path(pdf), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{re.sub(r'[^A-Za-z0-9]+', '_', pdf.stem).strip('_')}_p{page_no:02d}"
    png = cache / f"{stem}.png"
    if not png.exists():
        _extract(pdf, page_no, png)
    with Image.open(png) as im:
        size = im.size
    return Page(stem=stem, source=f"{pdf.name}#{page_no}", path=png, dpi=_ppi(pdf, page_no), size=size)


def from_png(png: str | Path, dpi: int = 300) -> Page:
    png = Path(png)
    with Image.open(png) as im:
        size = im.size
    return Page(stem=png.stem, source=png.name, path=png, dpi=dpi, size=size)


def _extract(pdf: Path, page_no: int, png: Path) -> None:
    """Write the page's scan to ``png``: byte for byte when the page holds it as one image. A scan stored as
    several strips (one per staff or so, e.g. Abdallah.pdf) is rasterised instead, since only the rasteriser
    knows where each strip sits; so is a page with no image at all (one re-encoded as drawing commands)."""
    prefix = png.with_name(f".{png.stem}")

    def written(cmd: list[str]) -> list[Path]:
        subprocess.run([*cmd, *_one_page(pdf, page_no), str(prefix)], check=True)
        return sorted(png.parent.glob(f"{prefix.name}-*.png"))

    parts = written(["pdfimages", "-png"])
    if len(parts) != 1:
        _unlink(parts)
        parts = written(["pdftoppm", "-r", str(RENDER_PPI), "-gray", "-png"])
        if not parts:
            raise FileNotFoundError(f"{pdf} page {page_no}: no embedded image and nothing rendered")
    with Image.open(parts[0]) as im:
        im.convert("L").save(png)
    _unlink(parts)


def _ppi(pdf: Path, page_no: int) -> int:
    """The resolution of the page's first embedded image (``pdfimages -list``: column 13 is x-ppi)."""
    out = subprocess.run(["pdfimages", "-list", *_one_page(pdf, page_no)],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        cols = line.split()
        if len(cols) > 13 and cols[0] == str(page_no) and cols[2] == "image":
            return int(cols[12])
    return RENDER_PPI


def _one_page(pdf: Path, page_no: int) -> list[str]:
    return ["-f", str(page_no), "-l", str(page_no), str(pdf)]


def _unlink(paths: list[Path]) -> None:
    for p in paths:
        p.unlink()

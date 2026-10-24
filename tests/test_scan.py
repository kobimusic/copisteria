"""A page pulled out of a PDF: a scan stored turned is read as the page's /Rotate shows it."""
import zlib
from pathlib import Path

from PIL import Image

from copisteria.vision.page import scan

ROTATED = Path(__file__).parent / "data" / "rotated.pdf"      # regenerate with: python tests/test_scan.py
TURNS = (0, 90, 180, 270)                                     # page i + 1 carries /Rotate TURNS[i]


def _upright():
    """The scan as it should read: portrait, dark above, light below."""
    im = Image.new("L", (40, 60), 255)
    im.paste(0, (0, 0, 40, 30))
    return im


def _stream(head, data):
    return b"<< %s /Length %d >>\nstream\n" % (head, len(data)) + data + b"\nendstream"


def _pdf(path):
    """A page per turn, its scan stored turned back against /Rotate so that a viewer shows every page upright."""
    kids = b" ".join(b"%d 0 R" % (3 + 3 * i) for i in range(len(TURNS)))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(TURNS))]
    for turn in TURNS:
        im = _upright().rotate(turn, expand=True)             # /Rotate turns clockwise, PIL anticlockwise
        w, h, n = *im.size, len(objs) + 1
        objs += [b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Rotate %d /Resources << /XObject << /Im %d 0 R"
                 b" >> >> /Contents %d 0 R >>" % (w, h, turn, n + 1, n + 2),
                 _stream(b"/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceGray "
                         b"/BitsPerComponent 8 /Filter /FlateDecode" % (w, h), zlib.compress(im.tobytes())),
                 _stream(b"", b"q %d 0 0 %d 0 0 cm /Im Do Q" % (w, h))]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (n, body)
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(out)


def test_a_scan_stored_turned_is_read_as_the_page_shows_it(tmp_path):
    def upright(page_no):
        im = scan.from_pdf(ROTATED, page_no, tmp_path).image()
        return im.height > im.width and im.crop((0, 0, im.width, im.height // 2)).getextrema()[1] < 128
    assert [turn for page_no, turn in enumerate(TURNS, 1) if not upright(page_no)] == []


if __name__ == "__main__":
    _pdf(ROTATED)

"""A missing file the user has to fetch says what to fetch, not a torch or subprocess traceback."""
import pytest

from copisteria import paths
from copisteria.vision.page import scan


def test_missing_weights_say_where_they_come_from(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"hf download kobimusic/copista-28m"):
        paths.require(tmp_path / "evidence-2m.pt")


def test_present_weights_pass_through(tmp_path):
    f = tmp_path / "w.pt"
    f.write_bytes(b"")
    assert paths.require(f) == f


def test_a_pdf_without_poppler_says_to_install_it(tmp_path, monkeypatch):
    monkeypatch.setattr(scan.shutil, "which", lambda tool: None)
    with pytest.raises(FileNotFoundError, match="poppler-utils"):
        scan.from_pdf(tmp_path / "score.pdf", 1, tmp_path)

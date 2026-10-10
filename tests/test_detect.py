import json
from types import SimpleNamespace

from PIL import Image

from copisteria import detect, pipeline
from copisteria.detect import imgsz_for, measure_space, staff_space


def measures(sp, n=8, conf=0.9):
    return [{"cls": "measure", "conf": conf, "xyxy": [0, 100 * k, 200, 100 * k + 4 * sp]} for k in range(n)]


def test_measure_boxes_give_a_quarter_of_their_median_height():
    assert measure_space(measures(9.0) + measures(30.0, n=3)) == 9.0
    assert measure_space(measures(9.0, conf=0.3)) == 0.0                 # unsure boxes say nothing
    assert measure_space([{"cls": "note_quarter", "conf": 0.9, "xyxy": [0, 0, 10, 10]}]) == 0.0


def test_the_inks_staff_space_stands_while_the_measure_boxes_agree():
    assert staff_space(8.6, 8.7) == 8.6
    assert staff_space(8.6, 0.0) == 8.6                                  # no measure boxes: nothing to check it by


def test_a_staff_space_the_line_finder_reads_at_half_or_twice_is_the_measure_boxes():
    assert staff_space(4.87, 8.90) == 8.90                               # half the spacing: read far too large
    assert staff_space(42.9, 9.07) == 9.07                               # a multiple: read far too small
    assert staff_space(0.0, 8.4) == 8.4                                  # no staves in the ink


def test_a_page_whose_ink_staff_space_is_half_is_read_at_the_measure_boxes_size(tmp_path, monkeypatch):
    monkeypatch.delenv("COPISTERIA_DETECTOR", raising=False)
    png = tmp_path / "page.png"
    Image.new("L", (1024, 1407), 255).save(png)
    sizes = []

    class Fake:
        prepare = staticmethod(lambda im, imgsz: (imgsz, 1.0))

        def run_array(self, imgsz, s):
            sizes.append(imgsz)
            return measures(8.9) if imgsz == 2048 else measures(8.9)[:2]

    monkeypatch.setattr(detect, "Detector", Fake)
    monkeypatch.setattr("copisteria.vision.page.scan.from_png", lambda p: SimpleNamespace(image=lambda: None))
    monkeypatch.setattr("copisteria.vision.page.staves.find_staves", lambda im: SimpleNamespace(spacing=4.87))
    dets, imgsz = pipeline.detections(png, want_size=True)
    assert imgsz == imgsz_for(8.9, 1407) == 1664 and sizes == [2048, 1664]
    assert json.loads((tmp_path / "page.dets_v7_1664.json").read_text()) == dets


def test_a_page_read_at_the_first_reads_size_is_read_once(tmp_path, monkeypatch):
    monkeypatch.delenv("COPISTERIA_DETECTOR", raising=False)
    png = tmp_path / "page.png"
    Image.new("L", (1024, 1400), 255).save(png)
    sizes = []

    class Fake:
        prepare = staticmethod(lambda im, imgsz: (imgsz, 1.0))

        def run_array(self, imgsz, s):
            sizes.append(imgsz)
            return measures(7.25)

    monkeypatch.setattr(detect, "Detector", Fake)
    monkeypatch.setattr("copisteria.vision.page.scan.from_png", lambda p: SimpleNamespace(image=lambda: None))
    monkeypatch.setattr("copisteria.vision.page.staves.find_staves", lambda im: SimpleNamespace(spacing=7.25))
    _, imgsz = pipeline.detections(png, want_size=True)
    assert imgsz == 2048 and sizes == [2048]

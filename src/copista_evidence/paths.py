"""Where the reader finds its files: the repository's models/ folder (run from the repository's root, installed
editable)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
EVIDENCE = MODELS / "evidence-2m.pt"                          # the evidence model (kobimusic/copista-evidence)
# the v7 symbol detectors (kobimusic/copista-evidence): size -> (weights, tag of the page caches they write)
DETECTORS = {
    "28m": (MODELS / "small" / "v7_obj-recall-30m_best.pt", "v7"),         # copista-28m, the default
    "2m": (MODELS / "small" / "v7_obj-recall_best.pt", "v7small"),         # copista-2m
}
DETECTOR = DETECTORS["28m"][0]


def detector() -> tuple[Path, str]:
    """(weights, cache tag) of the detector in use: COPISTA_DETECTOR = 28m (default), 2m, or a weights file of the same
    taxonomy (cached under its file name)."""
    name = os.environ.get("COPISTA_DETECTOR", "28m")
    if name in DETECTORS:
        return DETECTORS[name]
    p = Path(name)
    return p, p.stem

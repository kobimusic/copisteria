"""Where the reader finds its files: the repository's models/ folder (run from the repository's root, installed
editable)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
EVIDENCE = MODELS / "evidence-2m.pt"                          # the evidence model (kobimusic/copista-evidence)
# the v7 symbol detectors (kobimusic/copista-evidence): name -> (weights, tag of the page caches they write)
DETECTORS = {
    "v7": (MODELS / "small" / "v7_obj-recall-30m_best.pt", "v7"),          # 28.7M parameters, the default
    "small": (MODELS / "small" / "v7_obj-recall_best.pt", "v7small"),      # 1.94M parameters
}
DETECTOR = DETECTORS["v7"][0]


def detector() -> tuple[Path, str]:
    """(weights, cache tag) of the detector in use: COPISTA_DETECTOR = v7 (default), small, or a weights file of the
    same taxonomy (cached under its file name)."""
    name = os.environ.get("COPISTA_DETECTOR", "v7")
    if name in DETECTORS:
        return DETECTORS[name]
    p = Path(name)
    return p, p.stem

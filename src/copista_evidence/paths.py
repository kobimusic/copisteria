"""Where the reader finds its files: the repository's models/ folder (run from the repository's root, installed
editable)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
DETECTOR = MODELS / "small" / "v7_obj-recall-30m_best.pt"     # the v7 symbol detector (kobimusic/copista-evidence)
EVIDENCE = MODELS / "evidence-2m.pt"                          # the evidence model (kobimusic/copista-evidence)

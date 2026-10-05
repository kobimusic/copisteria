"""LMX-style target serialization for the image->music model.

Turns a positional-label JSON (from labels.extract_labels) into a flat token sequence
the decoder predicts — in reading order, with **staff_position as a first-class token**
(per the architecture spec: position is a discrete symbolic attribute, pitch is derivable
from position + clef + key at de-linearization time).

Token scheme (single stream):
  <bos>  clef_<shape><line>[_<dis>]  key_<sig>  time_<count>_<unit>
  then per event in order:
    note  -> pos_<staff_position>  dur_<type>  [dot_<n>]
    rest  -> rest_<type>  [dot_<n>]
    barLine -> bar
  <eos>

Unpitched notes with no staff_position (rare 1-line percussion) emit pos_na.
De-linearization back to MusicXML (pos+clef+key -> pitch) is a later step; this module is
the encode side + vocab, which is what training needs first.
"""
from __future__ import annotations

import json
from pathlib import Path

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def remap_path(p: str, data_root="."):
    """Map a repo-relative index path (``<tree>/<id>/<file>``) onto ``data_root`` so the
    same index.jsonl works wherever the dataset tree lives (repo root, or the VM)."""
    parts = Path(p).parts
    root = Path(data_root)
    return root / parts[-3] / parts[-2] / parts[-1] if len(parts) >= 3 else root / Path(p).name


def tokens_from_labels(d: dict) -> list[str]:
    """Serialize one labels dict into a list of target tokens (no padding)."""
    out = ["<bos>"]
    for e in d.get("elements", []):
        t = e.get("type")
        a = e.get("attrs", {})
        if t == "clef":
            shape, line = a.get("shape", "?"), a.get("line", "?")
            dis = f"_{a['dis']}{a.get('dis.place', '')[:1]}" if a.get("dis") else ""
            out.append(f"clef_{shape}{line}{dis}")
        elif t == "keySig":
            out.append(f"key_{a.get('sig', '0')}")
        elif t == "meterSig":
            if "count" in a and "unit" in a:
                out.append(f"time_{a['count']}_{a['unit']}")
            else:
                out.append("time_other")
        elif t == "barLine":
            out.append("bar")
        elif t == "note":
            sp = e.get("staff_position")
            out.append(f"pos_{sp}" if sp is not None else "pos_na")
            out.append(f"dur_{a.get('dur', '?')}")
            if a.get("dots"):
                out.append(f"dot_{a['dots']}")
        elif t == "rest":
            out.append(f"rest_{a.get('dur', '?')}")
            if a.get("dots"):
                out.append(f"dot_{a['dots']}")
    out.append("<eos>")
    return out


class Tokenizer:
    """Maps target tokens <-> ids. Build the vocab once from the dataset, then save/load."""

    def __init__(self, vocab: list[str]):
        self.itos = list(vocab)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            t = self.itos[i] if 0 <= i < len(self.itos) else "<unk>"
            if t == "<eos>":
                break
            if t not in ("<bos>", "<pad>"):
                out.append(t)
        return out

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def build(cls, index_jsonl: str | Path, min_count: int = 1, data_root="."):
        """Build a tokenizer by scanning every label file in an index.jsonl."""
        from collections import Counter
        counts: Counter = Counter()
        lengths = []
        for line in Path(index_jsonl).read_text().splitlines():
            row = json.loads(line)
            labels = json.loads(remap_path(row["labels"], data_root).read_text())
            toks = tokens_from_labels(labels)
            lengths.append(len(toks))
            counts.update(toks)
        vocab = list(SPECIAL) + sorted(t for t, c in counts.items()
                                       if c >= min_count and t not in SPECIAL)
        tok = cls(vocab)
        tok.lengths = lengths  # type: ignore[attr-defined]
        return tok


if __name__ == "__main__":
    import sys
    idx = sys.argv[1] if len(sys.argv) > 1 else "out/dataset2000/index.jsonl"
    root = sys.argv[2] if len(sys.argv) > 2 else "."
    tk = Tokenizer.build(idx, data_root=root)
    L = sorted(tk.lengths)
    print(f"vocab size: {len(tk)}")
    print(f"sequences: {len(L)}  len min/median/p95/max: "
          f"{L[0]}/{L[len(L)//2]}/{L[int(len(L)*0.95)]}/{L[-1]}")
    print("sample vocab:", tk.itos[4:34])

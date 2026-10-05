"""Detection taxonomy: the class list (type+subtype "titles") + per-event attribute value
vocabularies, all built from the dataset labels (analogous to lmx.Tokenizer for detection).

- A **class** is ``f"{type}_{subtype}"`` (e.g. note_eighth, clef_G2, accid_sharp,
  barLine_double, beam, slur) — auto-derived from whatever appears in the data.
- **Attributes** are a fixed broad schema the model also predicts per detection; each is a
  categorical value or "<na>" (not applicable to that class). Value vocabularies are built
  from the data so e.g. staff_position only carries the integer positions actually seen.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..lmx import remap_path

# attributes predicted per detection (besides class + bbox). Each maps an element to a
# categorical value or None (= not applicable -> "<na>").
ATTRS = ("staff_position", "stem_dir", "dots", "place", "grace", "voice_slot")


# Typeset text the detector does NOT predict: title/composer/publisher blocks, part labels, tempo and expression
# words, chord symbols, lyrics, rehearsal marks, fingering. A conventional OCR/text-detection model does this far
# better than a music-symbol detector can, and trying to do it here cost capacity and produced text-vs-glyph
# confusion (handwritten chord symbols detected as articulations). ``drop_text`` removes them from the taxonomy
# and skips their boxes at export; ``merge_text`` (the earlier experiment) instead folds them into one ``text``
# class. Musical numerals stay in every mode -- volta/ending numbers and tuplet numbers are notation, not prose,
# and so are dynamics, which Verovio renders as SMuFL glyphs rather than text.
TEXT_TYPES = ("pgHead", "pgFoot", "fig", "dir", "tempo", "harm", "label", "labelAbbr", "syl", "reh", "fing")


def class_name(e: dict, merge_text: bool = False) -> str:
    if merge_text and e["type"] in TEXT_TYPES:
        return "text"
    return f"{e['type']}_{e['subtype']}" if e.get("subtype") else e["type"]


def is_text(e: dict) -> bool:
    """True for typeset-text elements that ``drop_text`` excludes."""
    return e.get("type") in TEXT_TYPES


def attr_values(e: dict) -> dict:
    a = e.get("attrs", {})
    return {
        "staff_position": e.get("staff_position"),   # int (notes)
        "stem_dir": a.get("stem.dir"),               # up / down
        "dots": a.get("dots"),                       # "1" / "2"
        "place": a.get("place"),                     # above / below (artic/dynam)
        "grace": a.get("grace"),                     # acc / unacc (grace notes) -> None for normal notes
        # which voice on THIS staff (1 = first, 2 = second, ...). 34.7% of pages carry more than one
        # voice, and stem direction does not separate them (voice 1 is 42% down / 34% up), so without
        # this the consumer has to guess which stream a note belongs to.
        "voice_slot": e.get("voice_slot"),
    }


class Taxonomy:
    def __init__(self, classes, attr_vocab):
        self.classes = list(classes)
        self.merge_text = "text" in self.classes
        # a taxonomy carrying no text role at all was built with drop_text
        self.drop_text = not any(c.partition("_")[0] in TEXT_TYPES or c == "text" for c in self.classes)
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        self.attr_vocab = {a: list(v) for a, v in attr_vocab.items()}
        self.attr_id = {a: {v: i for i, v in enumerate(vs)} for a, vs in self.attr_vocab.items()}
        # rare class -> its family bucket (``meterSig_31_8`` -> ``meterSig_x``), see ``bucket_rare``
        self.bucket: dict[str, str] = {}

    @property
    def num_classes(self):
        return len(self.classes)

    def attr_size(self, attr):
        return len(self.attr_vocab[attr])

    def encode_class(self, e):
        name = class_name(e, self.merge_text)
        return self.cls2id.get(self.bucket.get(name, name))

    def bucket_rare(self, counts, min_count=20):
        """Route every class with fewer than ``min_count`` boxes to its family's ``_x`` bucket when that
        bucket is already a class (``meterSig_x``, ``tuplet_x``, ``ending_x``, ...). The class LIST is
        untouched -- a fixed taxonomy keeps its ids and its trained class head; the rare class merely
        receives no boxes. Families without a bucket are left alone and returned so the caller can
        report them. 62 of 213 music classes had under 20 instances in the training corpus, 82 of the
        classes are time signatures, and a head trained on a dozen examples of ``meterSig_31_8``
        spends capacity on noise. Returns (bucketed {cls: bucket}, unbucketable [cls])."""
        self.bucket = {}
        unbucketable = []
        for c, n in counts.items():
            if n >= min_count or c not in self.cls2id:
                continue
            b = f"{c.partition('_')[0]}_x"
            if b != c and b in self.cls2id:
                self.bucket[c] = b
            else:
                unbucketable.append(c)
        return dict(self.bucket), sorted(unbucketable)

    def encode_attrs(self, e):
        out = {}
        for a, v in attr_values(e).items():
            key = str(v) if v is not None else "<na>"
            out[a] = self.attr_id[a].get(key, 0)  # 0 == "<na>"
        return out

    def decode_attr(self, attr, idx):
        return self.attr_vocab[attr][idx]

    def save(self, path):
        Path(path).write_text(json.dumps({"classes": self.classes, "attr_vocab": self.attr_vocab}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["classes"], d["attr_vocab"])

    @staticmethod
    def count_classes(index_jsonl, data_root=".", merge_text=False, drop_text=False):
        """One pass over the index: (class -> box count, attr -> set of values seen)."""
        ccount = Counter()
        avals = {a: set() for a in ATTRS}
        for line in Path(index_jsonl).read_text().splitlines():
            row = json.loads(line)
            d = json.loads(remap_path(row["labels"], data_root).read_text())
            for e in d.get("elements", []):
                b = e.get("bbox")
                if not b or b[2] < 1 or b[3] < 1:   # skip degenerate/zero-area boxes
                    continue
                if drop_text and is_text(e):
                    continue
                ccount[class_name(e, merge_text)] += 1
                for a, v in attr_values(e).items():
                    avals[a].add(str(v) if v is not None else "<na>")
        return ccount, avals

    @classmethod
    def build(cls, index_jsonl, data_root=".", min_count=1, merge_text=False, drop_text=False):
        ccount, avals = cls.count_classes(index_jsonl, data_root, merge_text, drop_text)
        classes = sorted(c for c, n in ccount.items() if n >= min_count)
        # "<na>" first (id 0) in every attribute vocab
        attr_vocab = {a: ["<na>"] + sorted(v for v in vs if v != "<na>") for a, vs in avals.items()}
        t = cls(classes, attr_vocab)
        t.counts = ccount  # type: ignore[attr-defined]
        t.merge_text = merge_text
        t.drop_text = drop_text
        return t


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a not in ("--merge-text", "--drop-text")]
    idx = args[0]
    root = args[1] if len(args) > 1 else "."
    tx = Taxonomy.build(idx, data_root=root, merge_text="--merge-text" in sys.argv,
                        drop_text="--drop-text" in sys.argv)
    print(f"classes: {tx.num_classes}")
    for a in ATTRS:
        print(f"  attr {a}: {tx.attr_size(a)} values -> {tx.attr_vocab[a][:12]}"
              f"{'...' if tx.attr_size(a) > 12 else ''}")
    print("top classes:", [f"{c}:{n}" for c, n in tx.counts.most_common(15)])

"""The writer's alteration decode and chord handling; the optional refinement pass."""
import xml.etree.ElementTree as ET

import torch

from copista_evidence import front, read, write
from copista_evidence.model import EvidenceNet
from copista_evidence.read import Reading

from .test_front_write import det, staff_row


def _bar(L):
    return {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "measure"}, "key": {"v": 0}, "time": {"v": "4/4"},
            "clef": {"v": "G2"}}


def _note(pos, alter="key", p=0.6, chord=0.0, typ="note_quarter"):
    return {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": typ}, "dots": {"v": 0}, "pos": {"v": pos},
            "voice": {"v": 1}, "tup": {"v": "none"}, "alter": {"v": alter, "p": p}, "grace": {"v": "none"},
            "chord": chord, "tie": 0.0}


def _notes(xml):
    root = ET.fromstring(xml.split("\n", 2)[2])
    return root, root.findall(".//note")


def test_accidental_carries_along_its_line_unless_the_model_is_sure():
    # F5 (top line, y 100) three times; a sharp before the first; the model reads the later two as the key's
    dets = staff_row(100, [0, 300]) + [det("note_quarter", (x, 95, x + 10, 105), staff_position=4) for x in (50, 120, 190)]
    dets.append(det("accid_sharp", (35, 85, 45, 115)))
    L = front.build(dets, 400, 300)
    tok = {}
    for s in L.syms:
        if s.fam == "measure":
            tok[s.i] = _bar(L)
        elif s.fam == "accid":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "accid_sharp"}}
        else:
            tok[s.i] = _note(4, p=0.99 if s.cx > 150 else 0.6)     # the last one: the model is sure it is natural
    _, notes = _notes(write.write(Reading(layout=L, tok=tok)))
    assert [n.findtext("pitch/step") + str(n.findtext("pitch/octave")) for n in notes] == ["F5"] * 3
    assert [n.findtext("pitch/alter") for n in notes] == ["1", "1", None]
    assert notes[0].findtext("accidental") == "sharp" and notes[1].find("accidental") is None


def test_a_flat_pairs_with_the_head_at_its_bowl():
    # a flat's box centre sits about half a staff space above its head: the head (B4, y 120) gets it, not the C5
    # above it a little further right
    dets = staff_row(100, [0, 300]) + [det("note_quarter", (50, 115, 60, 125), staff_position=0),
                                      det("note_quarter", (62, 110, 72, 120), staff_position=1),
                                      det("accid_flat", (40, 100, 47, 126))]
    L = front.build(dets, 400, 300)
    tok = {}
    for s in L.syms:
        if s.fam == "measure":
            tok[s.i] = _bar(L)
        elif s.fam == "accid":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "accid_flat"}}
        else:
            tok[s.i] = _note(0 if s.cx < 61 else 1)
    _, notes = _notes(write.write(Reading(layout=L, tok=tok)))
    by_step = {n.findtext("pitch/step"): n.findtext("pitch/alter") for n in notes}
    assert by_step == {"B": "-1", "C": None}


def test_words_on_a_chord_member_do_not_break_the_chord():
    dets = staff_row(100, [0, 300]) + [det("note_quarter", (50, 115, 60, 125), staff_position=0),
                                      det("note_quarter", (52, 105, 62, 115), staff_position=2),
                                      det("note_quarter", (150, 115, 160, 125), staff_position=0)]
    L = front.build(dets, 400, 300)
    tok = {}
    for s in L.syms:
        if s.fam == "measure":
            tok[s.i] = _bar(L)
        else:
            tok[s.i] = _note(0 if s.cy > 116 else 2, chord=1.0 if s.cx == 57 else 0.0)
    texts = [{"text": "dolce", "role": "dir", "xyxy": [57, 60, 80, 70]}]  # nearest the member in x
    root, notes = _notes(write.write(Reading(layout=L, tok=tok), texts=texts))
    meas = root.find(".//measure")
    kids = [c.tag for c in meas if c.tag in ("note", "direction")]
    assert kids[:3] == ["direction", "note", "note"]                     # the words before the chord's root
    assert notes[1].find("chord") is not None and notes[2].find("chord") is None


def test_a_refinement_pass_fresh_from_its_base_reads_as_the_base():
    # the optional refinement pass (zero-initialised second pass) changes nothing until it is trained
    torch.manual_seed(0)
    base = EvidenceNet().eval()
    dets = staff_row(100, [0, 150, 300]) + [det("note_quarter", (x, 115, x + 10, 125), staff_position=0)
                                           for x in (30, 80, 180, 230)]
    fresh = EvidenceNet(refine=2, fb=39).eval()
    fresh.load_state_dict(base.state_dict(), strict=False)
    fresh.init_refine_from_base()
    fresh.onset_trained = base.onset_trained = True
    a = read.read(dets, 400, 300, base, attention=False)
    b = read.read(dets, 400, 300, fresh, attention=False)
    notes = [i for i in a.tok if a.layout.syms[i].fam == "note"]
    assert len(notes) == 4
    assert all(abs(a.tok[i]["real"]["p"] - b.tok[i]["real"]["p"]) < 1e-5 for i in a.tok)
    assert all(a.tok[i]["onset"] == b.tok[i]["onset"] for i in notes)

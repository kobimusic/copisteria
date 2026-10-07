from fractions import Fraction

import xml.etree.ElementTree as ET

from copisteria import front, write
from copisteria.read import Reading
from copisteria.vocab import CLS_ID


def det(cls, box, conf=0.9, **attrs):
    d = {"cls": cls, "conf": conf, "xyxy": list(box)}
    if attrs:
        d["attrs"] = {k: str(v) for k, v in attrs.items()}
        d["attr_p"] = {k: 0.9 for k in attrs}
    return d


def staff_row(y0, xs, sp=10.0):
    return [det("measure", (a, y0, b, y0 + 4 * sp)) for a, b in zip(xs, xs[1:])]


def test_merge_same_family_only():
    syms = front.merge([det("note_quarter", (0, 0, 10, 10), 0.8), det("note_half", (1, 0, 11, 10), 0.4),
                        det("dots_1", (0, 0, 10, 10), 0.5)])
    assert len(syms) == 2
    note = next(s for s in syms if s.fam == "note")
    assert note.cls == "note_quarter" and note.n == 2
    assert abs(note.post[CLS_ID["note_quarter"]] - 0.8 / 1.2) < 1e-6
    assert abs(note.p - (1 - 0.2 * 0.6)) < 1e-6


def test_staff_chains_across_a_missed_bar_and_fills_it():
    dets = staff_row(100, [0, 100, 200]) + staff_row(100, [300, 400])        # bar 200-300 missed
    dets.append(det("note_quarter", (240, 115, 250, 125), staff_position=0))
    L = front.build(dets, 500, 300)
    assert len(L.staves) == 1
    st = L.staves[0]
    assert len(st.bars) == 4
    gap = L.syms[st.bars[2]]
    assert gap.synthetic and gap.box[0] == 200 and gap.box[2] == 300
    note = next(s for s in L.syms if s.fam == "note")
    assert note.bar == 2


def test_aligned_staves_form_a_system_unaligned_do_not(monkeypatch):
    from copisteria import links
    monkeypatch.setattr(links, "load", lambda: None)              # the hand rule, not the fitted model
    dets = staff_row(100, [0, 100, 210, 300]) + staff_row(180, [0, 101, 209, 300]) + \
        staff_row(400, [0, 140, 230, 300])
    L = front.build(dets, 400, 600)
    assert [sy.staves for sy in L.systems] == [[0, 1], [2]]


def test_a_scans_frame_is_not_ink():
    import numpy as np
    page = np.full((400, 300), 230, np.uint8)
    page[100:104, 50:250] = 40                                    # a staff line
    page[60:300, 50:52] = 40                                      # a system's opening line
    framed = page.copy()
    framed[:, :12] = 25                                           # the scanner bed around the sheet
    framed[:, -9:] = 25
    framed[:10, :] = 25
    assert front.unframe(page) is page                            # no frame: the page as it is
    assert (front.ink_mask(framed) == front.ink_mask(page)).all()
    gap = front.ink_mask(framed)[150:250]                         # the gap below the staff, left of the line
    assert not gap[:, :45].any() and gap[:, 50:52].all()


def test_pitch_under_clefs():
    assert write._pitch("G2", 0) == ("B", 4)
    assert write._pitch("G2", -6) == ("C", 4)
    assert write._pitch("F4", 0) == ("D", 3)
    assert write._pitch("C3", 0) == ("C", 4)
    assert write._pitch("G2_8b", 0) == ("B", 3)


def test_durations():
    assert write._duration("quarter", 1, "none") == Fraction(3, 2)
    assert write._duration("eighth", 0, "3/2") == Fraction(1, 3)
    assert write._duration("half", 2, "none") == Fraction(7, 2)


def _reading(L, tok):
    return Reading(layout=L, tok=tok)


def test_write_one_bar():
    dets = staff_row(100, [0, 200]) + [det("note_quarter", (50, 115, 60, 125), staff_position=0),
                                      det("note_quarter", (100, 115, 110, 125), staff_position=2),
                                      det("meterSig_2_4", (20, 100, 30, 140))]
    L = front.build(dets, 300, 300)
    tok = {}
    for s in L.syms:
        if s.fam == "measure":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "measure"},
                        "key": {"v": -1}, "time": {"v": "2/4"}, "clef": {"v": "G2"}}
        elif s.fam == "meterSig":
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "meterSig_2_4"}}
        else:
            pos = 0 if s.cx < 80 else 2
            tok[s.i] = {"real": {"p": 1.0, "det": 1.0, "ev": 0}, "cls": {"v": "note_quarter"}, "dots": {"v": 0},
                        "pos": {"v": pos}, "voice": {"v": 1}, "tup": {"v": "none"}, "alter": {"v": "key" if pos == 0 else 0},
                        "grace": {"v": "none"}, "chord": 0.0, "tie": 0.0}
    xml = write.write(_reading(L, tok))
    root = ET.fromstring(xml.split("\n", 2)[2])
    notes = root.findall(".//note")
    assert [n.findtext("pitch/step") for n in notes] == ["B", "D"]
    assert notes[0].findtext("pitch/alter") == "-1"
    assert root.findtext(".//key/fifths") == "-1" and root.findtext(".//time/beats") == "2"


def test_chord_member_joins_nearest_root():
    E = write.Event
    root, member, other = E(x=10, kind="note", typ="quarter"), E(x=11, kind="note", typ="eighth", chord=True), \
        E(x=40, kind="note", typ="quarter")
    vs = write._voices([member, other, root], sp=10)
    evs = vs[1]
    assert [e is m for e, m in zip(evs, (root, member, other))] == [True, True, True]
    assert member.chord and member.dur == root.dur == 1


def test_lone_member_stands_alone_and_tuplets_close():
    E = write.Event
    a = E(x=10, kind="note", typ="eighth", tup="3/2")
    b = E(x=20, kind="note", typ="eighth", tup="3/2", chord=True)      # no root within 1.5 sp: its own onset
    c = E(x=30, kind="note", typ="eighth", tup="3/2")
    write._voices([a, b, c], sp=5)
    assert not b.chord
    assert (a.tuplet_mark, b.tuplet_mark, c.tuplet_mark) == ("start", "", "stop")
    assert a.dur == Fraction(1, 3)


def _staff_bar(*voices, meter="3/4"):
    return {"voices": dict(enumerate(voices, 1)), "marks": {"meter": False, "right": None},
            "state": {"time": meter}}


def test_an_overfull_voice_is_cut_at_the_length_most_staves_give_the_bar():
    E = write.Event
    q = lambda x, **kw: E(x=x, kind="note", typ="quarter", dur=Fraction(1), **kw)         # noqa: E731
    full = [q(10), q(20), q(30)]
    over = [q(10, beams=[(1, "begin")]), q(20, beams=[(1, "end")]), E(x=30, kind="note", typ="half", dur=Fraction(2)),
            E(x=40, kind="mrest", tie_stop=True)]
    over[2].tie = True
    over[3].step, over[3].octave = over[2].step, over[2].octave
    built = [[{"staves": [_staff_bar(full)]}], [{"staves": [_staff_bar(over)]}], [{"staves": [_staff_bar(list(full))]}]]
    assert write._column_lengths(built) == [3]                       # 3, 5 + a bar, 3: the bar is 3 beats
    out = write._chop(over, Fraction(3))
    assert [(e.typ, e.dur) for e in out] == [("quarter", 1), ("quarter", 1), ("quarter", 1)]    # the half ends at 3
    assert not out[2].tie                                            # its partner was cut away
    assert write._chop(full, Fraction(3)) is full                    # a voice in time is left as it is


def test_a_cut_closes_the_beams_and_tuplets_it_breaks():
    E = write.Event
    t = lambda x, mark="": E(x=x, kind="note", typ="eighth", tup="3/2", dur=Fraction(1, 3), tuplet_mark=mark)  # noqa
    evs = [E(x=0, kind="note", typ="half", dur=Fraction(2))] + [t(10, "start"), t(20), t(30, "stop")]
    for e, b in zip(evs[1:], ("begin", "continue", "end")):
        e.beams = [(1, b)]
    out = write._chop(evs, Fraction(8, 3))                           # the last triplet eighth falls past the cut
    assert len(out) == 3
    assert (out[1].tuplet_mark, out[2].tuplet_mark) == ("start", "stop")
    assert (out[1].beams, out[2].beams) == ([(1, "begin")], [(1, "end")])


def _dist(onset, tup_none=0.6):
    beat = [0.0] * 16; frac = [0.0] * 48
    b = int(onset); f = round((onset - b) * 48)
    beat[b] = 0.9; frac[f] = 0.9
    for k in range(16):
        beat[k] += 0.1 / 16
    for k in range(48):
        frac[k] += 0.1 / 48
    tup = [tup_none, 1 - tup_none] + [0.0] * 6
    return {"beat": beat, "frac": frac, "tup": tup}


def test_rhythm_decoding_reads_unmarked_triplets_from_onsets():
    E = write.Event
    evs = [E(x=10 * k, kind="note", typ="eighth", dist=_dist(k / 3)) for k in range(6)]
    write._decode_rhythm(evs)
    assert [e.tup for e in evs[:-1]] == ["3/2"] * 5
    assert [round(e.onset, 3) for e in evs] == [round(k / 3, 3) for k in range(6)]


def test_rhythm_decoding_keeps_plain_eighths():
    E = write.Event
    evs = [E(x=10 * k, kind="note", typ="eighth", dist=_dist(k / 2, tup_none=0.4)) for k in range(4)]
    write._decode_rhythm(evs)
    assert [e.tup for e in evs[:-1]] == ["none"] * 3

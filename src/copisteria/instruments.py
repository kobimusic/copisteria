"""Instrument label -> instrument facts (GM program, written->sounding transposition, clef).

Data: ``instruments.json`` -- facts extracted from MuseScore's instruments.xml. On top of it a hand-curated ALIASES table for the shorthand real parts and
section labels actually use ("1st Alto", "Trbs", "Bass", "Drums", "Repiano", "Eb Bass"), which
neither that table nor music21 resolve (music21 maps "1st Alto" / "Bass" to the VOCAL alto/bass and
has no entry at all for Flugelhorn / Trbs / Drums -- tested 2026-09-01). Pipeline per label:

    normalize -> strip ordinals ("1st", "II", OCR'd "Ist"/"lst") + role words ("Solo")
              -> whole-string alias/name -> key-in-name hint -> bare voice word by page context
              -> exact (name / short / id) -> token subset -> fuzzy -> None
    then: keyed variant ("Clarinet in A") -> ensemble override (brass band) -> clef variant

The CLEF rule matters more than it looks: MuseScore keeps two entries for brass that read either
clef -- ``trombone`` (bass clef, concert) vs ``trombone-treble`` (treble clef, sounds a major 9th
below written), same for euphonium / baritone horn / E♭ and B♭ tuba. The label can't tell them
apart; the clef the detector found on that staff can. A British brass band score is the extreme
case: 16 of 17 staves are treble-clef transposing parts.

``match()`` never guesses silently: an unmatched label returns None and the caller reports it.
Transposition is kept as MuseScore stores it (total diatonic/chromatic) and converted to the
MusicXML ``<transpose>`` form (diatonic / chromatic / octave-change) on demand.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_JSON = Path(__file__).with_name("instruments.json")

_ACCIDENTALS = {"♭": "b", "♯": "#", "♮": ""}
# "1st" / "2nd" / "12" / roman "II" -- and tesseract's usual damage to a "1": "Ist", "lst", "|st"
_ORDINAL_TOKEN = re.compile(r"^(\d+|[il|]{1,3})(st|nd|rd|th)?$|^(iv|v|vi{1,3})$")
_ROLE_TOKENS = {"solo", "principal", "asst", "assistant", "lead"}
_KEY_IN = re.compile(r"\bin\s+([a-g])\s*(b|#|flat|sharp)?\b")
_KEY_PREFIX = re.compile(r"^([a-g])(b|#)\s+")

# Bare voice-type words: (saxophone, choir voice, default). "Alto" / "Tenor" on a chart are
# saxes unless the page says choir; "Baritone" alone is the baritone HORN (band and brass band
# alike -- a sax part is labelled "Bari"/"Baritone Sax"), while "Bari" alone is only ever the sax.
VOICE_WORDS = {"soprano": ("soprano-saxophone", "soprano", "soprano-saxophone"),
               "alto": ("alto-saxophone", "alto", "alto-saxophone"),
               "tenor": ("tenor-saxophone", "tenor", "tenor-saxophone"),
               "baritone": ("baritone-saxophone", "baritone", "baritone-horn"),
               "bari": ("baritone-saxophone", "baritone", "baritone-saxophone")}
_SAX_CTX = ("sax", "saxes", "saxophone", "saxophones", "reeds")
_CHOIR_CTX = ("choir", "chorus", "voice", "voices", "vocal", "vocals", "satb")

# Ensemble detection from the OTHER labels on the page. Cornets / Repiano / E♭-B♭ Bass only
# occur on brass-band scores; any woodwind label rules it out (a concert band has cornets too).
_BRASS_BAND_SIGNS = ("cornet", "repiano", "eb bass", "bb bass")
_WOODWIND_SIGNS = ("flute", "clarinet", "sax", "oboe", "bassoon", "piccolo")
# Resolved id -> brass-band reading. "Horn" in a brass band is the E♭ tenor horn, not the F horn.
_BRASS_BAND_OVERRIDES = {"horn": "eb-alto-horn", "acoustic-bass": "bb-tuba"}

# normalized label -> instruments.json id. Abbreviations the table's shortName misses, section
# shorthand, and the default readings for ambiguous rhythm-section names.
ALIASES = {
    # classical scores' names (Italian, German, French) the table does not carry
    "pianoforte": "piano", "klavier": "piano", "clavier": "piano", "singstimme": "voice", "gesang": "voice",
    "stimme": "voice", "canto": "voice", "chant": "voice", "geige": "violin", "viole": "viola",
    "bratsche": "viola", "kontrabass": "contrabass", "flauto": "flute", "flauti": "flute", "flöte": "flute",
    "flote": "flute", "flûte": "flute", "oboi": "oboe", "hoboe": "oboe", "hautbois": "oboe",
    "fagotto": "bassoon", "fagotti": "bassoon", "fagott": "bassoon", "corno": "horn", "corni": "horn",
    "tromba": "trumpet", "trompete": "trumpet", "trompette": "trumpet", "posaune": "trombone", "pauken": "timpani",
    "arpa": "harp", "harfe": "harp",
    # saxophones
    "sax": "alto-saxophone", "saxes": "alto-saxophone",
    "alto sax": "alto-saxophone", "a sax": "alto-saxophone", "asax": "alto-saxophone",
    "as": "alto-saxophone", "alto saxophone": "alto-saxophone",
    "tenor sax": "tenor-saxophone", "t sax": "tenor-saxophone", "tsax": "tenor-saxophone",
    "ts": "tenor-saxophone",
    "bari sax": "baritone-saxophone", "baritone sax": "baritone-saxophone",
    "bar sax": "baritone-saxophone", "bs": "baritone-saxophone",
    "sop sax": "soprano-saxophone", "soprano sax": "soprano-saxophone", "s sax": "soprano-saxophone",
    # brass -- orchestra / big band
    "trumpets": "trumpet", "tpt": "trumpet", "tpts": "trumpet", "trpt": "trumpet",
    "trpts": "trumpet", "trp": "trumpet", "tr": "trumpet",
    "flugel": "flugelhorn", "flug": "flugelhorn", "flgh": "flugelhorn", "flghn": "flugelhorn",
    "fluegelhorn": "flugelhorn", "flugel horn": "flugelhorn",
    "trombones": "trombone", "tbn": "trombone", "tbns": "trombone", "trb": "trombone",
    "trbs": "trombone", "tb": "trombone", "bone": "trombone", "bones": "trombone",
    "b tbn": "bass-trombone", "btb": "bass-trombone", "bass tbn": "bass-trombone",
    "horns": "horn", "hn": "horn", "hns": "horn", "f horn": "horn", "french horn": "horn",
    "tba": "tuba", "euph": "euphonium", "bari horn": "baritone-horn",
    # brass -- brass band
    "cornet": "bb-cornet", "cornets": "bb-cornet", "cor": "bb-cornet", "cnt": "bb-cornet",
    "repiano": "bb-cornet", "repiano cornet": "bb-cornet", "rep cornet": "bb-cornet",
    "rep": "bb-cornet", "ripieno": "bb-cornet",
    "soprano cornet": "eb-cornet", "sop cornet": "eb-cornet", "eb cornet": "eb-cornet",
    "tenor horn": "eb-alto-horn", "eb tenor horn": "eb-alto-horn", "eb horn": "eb-alto-horn",
    "eb bass": "eb-tuba", "bb bass": "bb-tuba", "basses": "bb-tuba",
    # woodwinds
    "fl": "flute", "picc": "piccolo", "cl": "clarinet", "clar": "clarinet", "clt": "clarinet",
    "b cl": "bass-clarinet", "bcl": "bass-clarinet", "ob": "oboe", "bsn": "bassoon", "fg": "bassoon",
    # rhythm section (jazz / band defaults for the bare words)
    "bass": "acoustic-bass", "string bass": "acoustic-bass", "upright bass": "acoustic-bass",
    "e bass": "electric-bass", "el bass": "electric-bass", "cb": "contrabass", "db": "double-bass",
    "guitar": "guitar-steel", "gtr": "guitar-steel", "gt": "guitar-steel",
    "el guitar": "electric-guitar", "e guitar": "electric-guitar",
    "pno": "piano", "keys": "piano", "keyboard": "piano", "pf": "piano",
    "drums": "drumset", "drum set": "drumset", "drum kit": "drumset", "kit": "drumset",
    "dr": "drumset", "d set": "drumset", "percussion": "drumset", "perc": "drumset",
    "vibes": "vibraphone", "vib": "vibraphone",
    # strings
    "vln": "violin", "vn": "violin", "vla": "viola", "va": "viola", "vc": "violoncello",
    "cello": "violoncello", "vcl": "violoncello",
    # voice
    "vocal": "voice", "vox": "voice", "vocals": "voice",
}


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    program: int          # 0-based General MIDI program (MusicXML <midi-program> is this + 1)
    chromatic: int        # written -> sounding, total semitones (MuseScore convention)
    diatonic: int
    clef: str
    perc: bool

    @property
    def transposing(self) -> bool:
        return self.chromatic != 0

    def musicxml_transpose(self) -> dict:
        return to_musicxml_transpose(self.diatonic, self.chromatic)


@dataclass(frozen=True)
class Match:
    instrument: Instrument
    label: str
    key: str | None       # key-in-name hint that was parsed out ("a" for "Clarinet in A"), if any
    how: str              # exact | alias | context | subset | fuzzy
    ensemble: str | None  # page-level reading that influenced the pick ("brassband"), if any


_TABLE: dict[str, Instrument] = {}
_INDEX: dict[str, str] = {}      # normalized name / short / track / id -> id
_KEYS: list[str] = []


def normalize(s: str) -> str:
    s = (s or "").lower()
    for k, v in _ACCIDENTALS.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\w#\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_key(norm: str) -> tuple[str, str | None]:
    """'clarinet in a' -> ('clarinet', 'a'); 'bb clarinet' -> ('clarinet', 'bb'); else unchanged."""
    m = _KEY_IN.search(norm)
    if m:
        acc = {"b": "b", "flat": "b", "#": "#", "sharp": "#"}.get(m.group(2) or "", "")
        return (norm[:m.start()] + norm[m.end():]).strip(), m.group(1) + acc
    m = _KEY_PREFIX.match(norm)
    if m:
        return norm[m.end():].strip(), m.group(1) + m.group(2)
    return norm, None


def strip_ordinals(norm: str) -> str:
    """Drop part numbers ("1st", "II", OCR'd "Ist") and role words ("Solo") -- neither changes
    which instrument it is."""
    return " ".join(t for t in norm.split()
                    if not _ORDINAL_TOKEN.match(t) and t not in _ROLE_TOKENS)


_OCR_ONE = re.compile(r"(?<![\w])[Ili|](?=(st|ST)\b)")


def clean_label(text: str) -> str:
    """Undo tesseract's one systematic damage to part labels: a "1" read as I / l / | in an
    ordinal ("Ist Horn", "lst Baritone" -> "1st ..."). Display text only; matching normalizes
    these itself."""
    return _OCR_ONE.sub("1", text or "")


def ensemble_of(context) -> str | None:
    """What kind of score the page is, from all its labels. Only 'brassband' is recognised so far:
    it flips the reading of Horn / Baritone / Bass and is the one ensemble where nearly every
    staff is a treble-clef transposing part."""
    ctx = [normalize(c) for c in context if c]
    if any(w in c for w in _WOODWIND_SIGNS for c in ctx):
        return None
    signs = {s for s in _BRASS_BAND_SIGNS if any(s in c for c in ctx)}
    return "brassband" if len(signs) >= 2 else None


def _load():
    if _TABLE:
        return
    rows = json.loads(_JSON.read_text())
    # index order = preference order on collisions: everyday instruments first, then shorter ids
    def pref(r):
        g = set(r.get("genres", ()))
        return ("common" not in g, "jazz" not in g, "concertband" not in g, "orchestra" not in g,
                len(r["id"]))
    for r in sorted(rows, key=pref):
        inst = Instrument(r["id"], r["name"], int(r["program"]), int(r["chromatic"]),
                          int(r["diatonic"]), r.get("clef", "G"), bool(r.get("perc")))
        _TABLE[inst.id] = inst
        for key in (r["name"], r.get("short"), r.get("track"), r["id"].replace("-", " ")):
            k = normalize(key or "")
            if k:
                _INDEX.setdefault(k, inst.id)
    for k, iid in ALIASES.items():
        if iid in _TABLE:
            _INDEX[k] = iid
    _KEYS[:] = sorted(_INDEX)


def table() -> dict[str, Instrument]:
    _load()
    return _TABLE


def _with_key(iid: str, key: str | None) -> str:
    """Prefer the keyed variant when the label named one: clarinet + 'a' -> a-clarinet."""
    if key:
        base = re.sub(r"^(bb|eb|ab|db|gb|f#|c#|[a-g])-", "", iid)
        for cand in (f"{key}-{base}", f"{key}-{iid}"):
            if cand in _TABLE:
                return cand
    return iid


def _by_clef(iid: str, clef) -> str:
    """Pick the treble- or bass-clef entry of a brass instrument that reads either, from the clef
    the detector found on that staff. The two entries differ in TRANSPOSITION, not just clef."""
    sign = str(clef or "")[:1].upper()
    if sign == "G" and f"{iid}-treble" in _TABLE:
        return f"{iid}-treble"
    if sign == "F" and iid.endswith("-treble") and iid[:-7] in _TABLE:
        return iid[:-7]
    return iid


def match(label: str, context=(), clef=None) -> Match | None:
    """Resolve one part label. ``context`` = the other label texts on the page (section headers
    like "Saxes" disambiguate bare voice words; cornets + Repiano mark a brass band). ``clef`` =
    the clef sign detected on this part's staff ("G" / "F"), which selects the treble-vs-bass
    entry for brass that read either."""
    _load()
    base0 = strip_ordinals(normalize(label))
    if not base0:
        return None
    ctx = " ".join(normalize(c) for c in context if c).split()
    ensemble = ensemble_of(context)

    def finish(iid, key, how):
        iid = _with_key(iid, key)
        if ensemble == "brassband":
            iid = _BRASS_BAND_OVERRIDES.get(iid, iid)
        return Match(_TABLE[_by_clef(iid, clef)], label, key, how, ensemble)

    def voice_word(word, key):
        # BEFORE any table lookup: MuseScore has vocal entries literally named "Alto" /
        # "Baritone", which would otherwise win the exact match over the page context.
        sax_id, voice_id, default_id = VOICE_WORDS[word]
        if any(w in _CHOIR_CTX for w in ctx) and not any(w in _SAX_CTX for w in ctx):
            iid = voice_id if voice_id in _TABLE else sax_id
        elif any(w in _SAX_CTX for w in ctx):
            iid = sax_id
        else:
            iid = default_id
        return finish(iid, key, "context")

    if base0 in VOICE_WORDS:
        return voice_word(base0, None)
    # whole string next: "eb bass" / "bb clarinet" are names, not a key hint + a bare word
    iid = _INDEX.get(base0)
    if iid:
        return finish(iid, None, "alias" if base0 in ALIASES else "exact")
    base, key = split_key(base0)
    if not base:
        return None
    if base in VOICE_WORDS:
        return voice_word(base, key)
    iid = _INDEX.get(base)
    if iid:
        return finish(iid, key, "alias" if base in ALIASES else "exact")
    toks = set(base.split())
    subset = [k for k in _KEYS if len(k) > 2 and set(k.split()) <= toks]
    if subset:
        return finish(_INDEX[max(subset, key=len)], key, "subset")
    # fuzzy: OCR damage only ("trumpe1"), never a different word ("repiano" is not "piano")
    close = [k for k in difflib.get_close_matches(base, _KEYS + list(VOICE_WORDS), n=3, cutoff=0.85)
             if k[:1] == base[:1]]
    if close:
        # an OCR-damaged voice word ("Alito") must take the same context route as the clean one,
        # not land on the table's vocal "Alto" entry
        if close[0] in VOICE_WORDS:
            return voice_word(close[0], key)
        return finish(_INDEX[close[0]], key, "fuzzy")
    return None


# ---- transposition arithmetic -----------------------------------------------------------------
def to_musicxml_transpose(diatonic: int, chromatic: int) -> dict:
    """MuseScore total interval -> MusicXML <transpose> parts. Bari sax -12/-21 becomes
    diatonic -5, chromatic -9, octave-change -1 (the form engravers and players expect)."""
    octave = int(chromatic / 12)                     # truncate toward zero
    return {"diatonic": diatonic - 7 * octave, "chromatic": chromatic - 12 * octave,
            "octave_change": octave}


def concert_fifths(written_fifths: int, chromatic: int) -> int:
    """Concert-pitch key signature implied by a written key on an instrument that sounds
    ``chromatic`` semitones from written (alto sax -9: written G/1# -> concert Bb/-2)."""
    shift = (7 * chromatic) % 12
    if shift > 6:
        shift -= 12
    return written_fifths + shift


def concert_key_conflicts(parts) -> list[dict]:
    """``parts`` = iterable of (part_id, written_fifths, chromatic). Every part on a page shares
    one concert key, so a part whose implied concert key disagrees with the majority has either a
    misread label (wrong transposition) or a misread key signature."""
    implied = [(pid, concert_fifths(wf, ch)) for pid, wf, ch in parts]
    if len(implied) < 2:
        return []
    majority = Counter(f for _, f in implied).most_common(1)[0][0]
    return [{"part": pid, "implied_concert_fifths": f, "majority_concert_fifths": majority}
            for pid, f in implied if f != majority]

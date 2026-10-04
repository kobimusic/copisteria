# copista-evidence: architecture

The idea: do almost nothing by hand. A front end decides the page's geometry (which symbols there are, on which
staff and in which bar) and a small model decides everything musical, from evidence it learned on its own. Nowhere
in the code is there a rule like "a natural against the key signature means the key is wrong" or "a bar that adds
up to three beats under 4/4 has a misread duration"; the model sees the whole system at once and has learned which
contexts make which readings likely, and how far to trust the detector against them.

```
page image ──► detector (v7) ──► front end ──► tokens ──► evidence model ──► readings ──► writer ──► MusicXML
                 270 classes      geometry       per        2.03M params      per symbol    rhythm and
                 + attributes     only           symbol     transformer       + evidence    alteration
                                                                                            decoding
```

## 1. Detector

`detect.py` runs the v7 symbol detector (`detector/small`, architecture `obj-recall-30m`) over the page at the size
that puts the staff space at 10.6 px (the staff space comes from the staff lines in the ink; when the line finder
finds none, from a first pass's measure boxes, which are four staff spaces tall). Every detection has a class (270,
`models/yolo_v7_taxonomy.json`), a confidence and a box; notes and rests also carry attribute distributions: staff
position, stem direction, dots, grace, voice slot. Detections at confidence 0.05 and up are kept and cached beside
the page.

The two sizes are named after their detector: copista-28m reads with that one (28.7M parameters), copista-2m
(`--detector 2m`) with the small v7 detector (architecture `obj-recall`, 1.94M parameters, the same classes and
attributes), so the whole reader is about 4M parameters. The evidence model is the same in both; it was trained on
copista-28m's detector, not the small one.

## 2. Front end (`front.py`, `links.py`)

Only geometry, and only what the writer cannot do without:

* **Merge.** Boxes of one family (note, rest, accid, barLine, ...) that overlap at IoU 0.5 or more are one symbol.
  Its class posterior is the boxes' confidence mass per class; its probability of being real is their noisy-or.
  Two families never merge: two readings of one glyph stay two symbols and the model decides which one is real.
* **Staves.** From the staff lines in the page's ink (`vision/page/staves.py`: five long thin runs at one spacing),
  each checked against the staff space of the detector's measure boxes; a row of confident measure boxes no ink
  staff explains is a staff too, and a "staff" with no measure box and fewer than two notes (a ruled line, a
  border) is none. The line finder runs on the page as it is and on a cleaned copy (the paper's shading divided
  out: a tinted or shaded scan reads as ink to it), and the set of staves the detector's confident symbols confirm
  better is kept. Where the line finder fails (too few staves found), staves are chains of side-by-side, vertically
  overlapping measure boxes, and a gap between two of them that holds notes or rests is a bar of its own.
* **Systems.** Two consecutive staves are one system when a vertical ink run crosses most of the gap between them
  at the system's left (the opening line, brace or bracket) or their bar lines' ink runs on from one into the other;
  a white gap separates them whatever the boxes say. In between, a brace or bracket the detector saw over both, or
  the `links` model: a logistic model over the evidence a pair gives (a group symbol, how their bar lines line up,
  the gap), fitted on rendered pages and decoded with a preference for one staves-per-system count on a page.
* **Bars.** Decided once per system for all its staves. A candidate x (a measure box's edge or a detected bar line,
  on any staff, or a column of ink through every staff) is a bar line where most staves show one: a detection and
  a vertical stroke over the staff's full height, or the stroke through all of them. On a single staff, a detection
  and the stroke, or a firm detection alone; the stroke is looked for across the candidates' span (a double bar
  line's centre is the gap between its strokes). Bars narrower than two staff spaces fold into their neighbour, an
  empty first or last stretch (staff lines running into the margin, a cautionary key after the last bar line) is no
  bar, and inside a system an empty stretch under four staff spaces is a bar-line group (a repeat sign, a thick
  double bar), its two cuts one.
* **Revision** (`revise.py`). Once the page is read, a voice whose durations add up to 1.75 bars of its meter or
  more has lost a bar line: where its running sum reaches a whole number of bars between two notes and a stroke
  over the staff's full height stands clear of every note head (and on at least half of the system's other
  staves), the bar line is added and the page is read again.
* **Assignment.** Every other symbol goes to the staff it is nearest to (a note to the staff where its own staff
  position reading fits) and to the bar its centre falls in.

## 3. Tokens (`features.py`)

Every assigned symbol is a token: its top three classes and their probabilities, P(real), the attribute
distributions, and its geometry in staff spaces: x within its bar, height relative to the bar's middle line, its
size, its bar's width, its position in the system, whether it is in the first or last bar, whether its bar was
filled in by the front end. Plus the indices the model's attention uses: system, staff within the system (and the
system's staff count), bar column. Tokens are raw readings; the model builds embeddings and priors from them, so
training can corrupt the raw readings and everything downstream stays consistent.

## 4. The evidence model (`model.py`)

`EvidenceNet`: d = 160, 6 layers, 8 heads, feed-forward 640, 2,026,222 parameters.

**Residual evidence.** Every head that has a detector reading behind it is a residual on that reading's
log-probability:

```
final logits = a_head · log P_detector(value) + evidence(value | context)
```

so the evidence is what the context adds to the detector's local reading, in nats, and with no evidence the
detector stands. These heads are: real (is the symbol there at all), class, dots, staff position, voice, grace.

**Free heads**, read from context alone: a note's sounding alteration (relative to the key, so the head learns
"as the key gives it" and "an accidental's"), chord membership, tie, tuplet, and its onset in the bar (whole
quarters, 16 classes, and 48ths of a quarter, 48 classes); for each bar, the clef, key and meter in effect and
whether the bar sets them.

**Attention.** The bias between every pair of tokens is a learned weight per head on 13 fixed relation maps: same
staff row or not, by bar distance (0, 1, 2, 3, 4-8), farther than 8 bars, the other token in a later bar, same
system.
Locality is a prior the model can override, not a wall.

**Refinement pass** (optional; the released model has none). `EvidenceNet(refine=2)` reads the window a second
time with the first reading's conclusions fed back in, per token: its onset and how sure it was, the running sum of
the durations read before it in its voice and bar and how far that sum is from its onset, collisions with the other
notes of its voice, the voice's total in the bar, the nearest note of another voice and how its onset lines up, the
nearest accidental glyph at the note's height (a flat at its bowl), the alteration read for the last note on the
same line, and the bar's length under the meter in view. Two more transformer layers turn that into the final
reading. The feedback layer and the two layers' outputs start at zero, so a refinement model starts out reading
exactly as the model it is fine-tuned from. Fine-tuned from the released model it read the string quartets better
and the Lieder scans worse, and no better overall: it is not released.

**Reading** (`read.py`). Each system is read in a window with its neighbours (the previous and the next system, up
to 1024 tokens), and only its own tokens are taken from that window. A symbol's reading keeps, per head, the
detector's choice, the final choice, its probability, and the evidence for the final choice over the detector's.

**Optional inputs.** The model and the training loop also take the detector's image features at every token and
its sub-threshold candidates (detections at confidence 0.01-0.05 as extra tokens, flagged, for the real head to
accept or reject); the tokenizer writes both. The released model does not use them.

## 5. Writer (`write.py`)

Transcription only: every decision was the model's. A symbol is written when its P(real) is 0.5 or more.

* **Pitch**: staff position under the clef in effect (a clef inside a bar takes over from its x on), plus the
  sounding alteration, decoded per staff and bar as notation defines it: each accidental glyph goes to one head
  (the nearest at its anchor's height to its right, one-to-one; a flat's anchor is its bowl) and sets that head's
  alteration, which then holds for the later heads on the same step and octave to the bar's end; a head with
  neither takes the model's reading, and a model at least 97 % sure of another alteration keeps its own. A printed
  accidental is written where its glyph agrees with the alteration written.
* **Rhythm**: notes go to voices by the model's voice reading; chord members attach to their nearest root. Each
  note is placed at the onset the model reads for it when it is confident enough (`COPISTA_ONSET_P`), and each
  voice's durations and tuplets are decoded jointly by a Viterbi search over 48ths of a quarter: the model's
  duration, onset and tuplet distributions, with penalties for gaps, overlaps, two notes at one onset and running
  past the bar (strict for single-staff parts). One misread duration does not shift the rest of the bar.
* **Bars**: keys, meters and clefs as the bar heads read them, printed where they change; a meter only where the
  page prints one. A bar with nothing in it gets a `<forward>`, never an invented rest.
* **Marks**, attached by position: beams (the notes whose stems a beam box reaches: a head under the beam has its
  stem at its right, a head over it at its left; a beam between the two staves of a keyboard part joins notes of
  both; levels from the note types), slurs (from their corners, across systems), ties,
  articulations, ornaments, fermatas, dynamics, arpeggios, octave lines (the pitches shifted), pedal, segno and coda,
  bar line styles and repeats, part groups from brackets and braces.
* **Text**, from `<page>.texts.json` when present: OCR text boxes with a role. `pgHead_title`, `pgHead_composer`
  give the title and composer; `label`, `labelAbbr` the part names; `dir`, `tempo`, `expression`, `words` become
  directions before the nearest note (only words that are mostly letters), and a word followed by a run of dashes
  in the ink ("cresc. - - -") starts dashes that stop at the last note they reach; digits over a multi-measure rest
  give its count. A direction that falls on a chord member is written with the chord's root: between a root and
  its `<chord/>` notes it would break the chord.

## 6. Training

### Data

Rendered pages from PDMX scores, each with a labels JSON whose every element carries `src`, what it means in the
score:

```
{"image": {"width": W, "height": H}, "staff_space": 10.2,
 "elements": [{"id": ..., "type": "note", "subtype": "quarter", "bbox": [x, y, w, h],
               "src": {"part": ..., "staff": ..., "system": ..., "measure": ..., "mi": ..., "voice": 1,
                       "pitch": "F4", "alter": 1, "onset": "3/2", "dots": 1, "tie_start": true,
                       "tm": "3/2", "grace": false}}, ...],
 "staves": [{"part": ..., "staff": ..., "system": ..., "x0": ..., "x1": ..., "lines": [y, ...],
             "clef": ..., "key": ..., "time": ...}, ...]}
```

Every `measure` element is one staff's bar (`src`: part, staff, system, index, number); clefs, key and meter
signatures carry what they set. An element's `type` and `subtype` name the detector class it is drawn as. `training/tokenize.py` runs v7 over every page and builds its tokens; `match.py`
gives every symbol its targets. A symbol is real when a label of its family overlaps it (greedy one-to-one by IoU,
0.3 or more), and a real symbol's targets are that label's meaning.

The released model saw 31,800 pages rendered from 28,000 PDMX scores, in random engraving styles with scan effects
(ink bleed, warps, rotation, noise, show-through), measure repeats and multi-measure rests (which also turn
ensemble scores into single parts). 2 % of the scores (by a hash of the source) are held out for validation.

### Corruption

A training sample is a window of 1-4 consecutive systems of one page. The detector's readings are corrupted on the
fly at rates drawn per window, so the model meets everything from a clean read to a bad one: classes masked or
swapped for a sibling, attributes masked or nudged, staff positions misread confidently, bars turned into
filled-in bars, symbols dropped, ghost symbols (moved, re-classed, not real), a confidence calibration shift, and
whole families the page does not print (tuplet marks left out after the first). The targets never change.

Tuplet emphasis (`COPISTA_TUP_UNMARK`, `COPISTA_TUP_WEIGHT`): in a share of the windows each staff row keeps only
its first tuplet mark (engravers print a tuplet's number once), and pages with tuplets are drawn more often.
`COPISTA_TUP_DROPALL` leaves every tuplet mark out of a share of the windows (a run marked pages earlier shows none
on this page), `COPISTA_MULTIVOICE_WEIGHT` draws pages with two voices more often.

### Recipe

`train.py`'s docstring has the released model's two stages: 70,000 steps from scratch (batch 24, learning rate
1e-3) and a 25,000-step fine-tune with the tuplet emphasis (learning rate 3e-4). With a refinement pass
(`--refine`) the first pass is trained too, at half weight. Validation reports each head's accuracy against the
detector's own argmax, with and without the corruption.

## 7. Known limits

* Structure is the front end's hand-made rules. They were tuned on real scans; structure models learned from
  renders did not transfer to other engravings.
* Triplets a page does not mark (marked once, often pages earlier) are still often read as plain notes: with every
  tuplet mark removed from rendered pages, 65 % of tuplet notes are written as tuplets (96 % with the marks). The
  writer's rhythm decoding cannot recover the rest: it is bounded by what the model believes.
* Lyrics are not written.

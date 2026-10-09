# StudySnap

### A source-cited study companion that runs entirely on the device

Students record every lecture. Almost none of them can do anything useful with the
recordings and the obvious fix, uploading the audio to a cloud service, fails for reasons
that are economic and infrastructural rather than technical.

StudySnap turns a lecture recording **and** a photograph of the whiteboard into revision
material without a single byte leaving the device.

---

## The problem

| Barrier | Reality for a typical student |
|---|---|
| Cost | Cloud ASR is priced in dollars; a student budgets in hundreds of rupees |
| Data | 1 hour of lecture audio is ~115 MB uploaded before a single word comes back |
| Connectivity | Campus Wi-Fi drops in lecture halls. So does 4G in older buildings |
| Privacy | Lecture audio contains other students' voices and internal course material |
| Battery | Streaming audio to a server drains a laptop through a 3-hour lab session |

So the recording sits in storage and gets re-watched at 2× speed the night before the exam,
which is the least effective possible way to revise. Then there is the second problem:
**the board**. The equations, the definitions, the diagrams — none of that is in the audio
at all.

**The students who most need a study tool are the ones who can least afford the
infrastructure the cloud version requires.** That is the constraint every design decision
below follows from.

---

## What it does

Four runnable modules in one codebase. The core study workflow runs on-device; the optional Qualcomm AI Hub profiling path requires a network connection:

```
   lecture audio ──────────┐
                           ├──►  source-cited knowledge base  ──►  citations
   board / textbook photo ─┘              │
                                          ▼
                              adaptive assessment engine
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                  questions at the                 mastery tracking +
                  right difficulty                  spaced repetition
```

1. **Audio** (`app.py`) — transcript → summary → flashcards
2. **Vision** (`whiteboard.py`) — photograph a board or page → structured notes
3. **Knowledge base** (`knowledge_base.py`) — every fact carries its source, precisely
4. **Adaptive assessment** (`adaptive.py`) — question selection and scheduling per student

### The citations are the hard part

Citing a source is easy. Citing it *precisely* is not, and a citation that lands in the
wrong place is worse than no citation — the student clicks through and finds nothing.

The first implementation recorded only where each *chunk* began. A question about learning
rates was answered correctly but cited `lecture.mp4 @ 00:00`, when the content is at
`02:10`. Chunks are ~90 words and deliberately overlap, so a chunk's start point can be
minutes away from the answer inside it.

Provenance is therefore tracked per **segment**: every chunk knows the timestamp or page
number of each line inside it, and the answer's own sentence is located before citing. The
same answer is now cited as `lecture.mp4 @ 02:10`.

### The system says "I don't know"

`KnowledgeBase.ask()` returns `grounded=False` when nothing clears the relevance threshold,
rather than returning the closest thing it can find:

```
Q: what is the capital of France?
   confidence 0.0  grounded False
   I could not find this in the provided material.
```

For a tool aimed at students, a confident wrong answer is the worst possible failure mode.
An extractive system can guarantee this property; a generative one cannot.

### The questions adapt to the student

Question difficulty and student ability are measured on the same Elo scale. Selection
targets the *desirable difficulty* band — roughly 70–85% expected success, where retrieval
practice produces the most durable learning. Questions the student always gets right teach
nothing; questions they always fail teach nothing either.

```
SIMULATED SESSION — one student, 30 questions
 #  concept                diff P(correct)  result  ability      Δ
 1  overfitting model      1052       0.70       ✓   1209.6   +9.6
 2  gradient descent       1043       0.71       ✗   1177.2  -22.8
 3  momentum               1043       0.71       ✓   1209.2   +9.2

realised success rate: 80%   (target band 70-85%)
ability moved 1200 -> 1225
```

A correct answer to a hard question moves ability a lot; a correct answer to an easy one
barely moves it. A wrong answer costs more than a right answer gains, which is what makes
the estimate conservative and honest.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  on-device                                                       │
│                                                                  │
│   audio ──► Whisper ──► transcript ──┐                           │
│                                      │                           │
│   photo ──► preprocess ──► OCR ──► notes ──┐                     │
│             (deskew, Otsu,                 │                     │
│              illumination)                 ▼                     │
│                              ┌─────────────────────────┐         │
│                              │  source-cited KB (BM25) │         │
│                              │  timestamp / page       │         │
│                              └───────────┬─────────────┘         │
│                                          │                       │
│                          ┌───────────────┴──────────────┐        │
│                          ▼                              ▼        │
│                   cited answers                  item generation │
│                  (refuses to guess)                      │       │
│                                            ┌─────────────┴────┐  │
│                                            ▼                  ▼  │
│                                    Elo ability      spaced       │
│                                    estimation       repetition   │
│                                                                  │
│   ── core study workflow runs locally; no network calls ──       │
└──────────────────────────────────────────────────────────────────┘
```

---

## How to run it

Every module has a demo that runs with **no model downloads and no network**.

```bash
python3 app.py --demo            # audio pipeline: transcript → summary → flashcards
python3 whiteboard.py --demo     # vision pipeline: board text → structured notes
python3 knowledge_base.py --demo # cited question answering + refusal behaviour
python3 adaptive.py --demo       # adaptive assessment + mastery report
```

### Real inputs

```bash
pip install faster-whisper pypdf pytesseract   # + the tesseract binary

python3 app.py --audio lecture.m4a

python3 whiteboard.py --image board.jpg --transcript lecture.txt --debug-dir debug/

python3 knowledge_base.py \
    --transcript lecture.txt \
    --pdf textbook.pdf \
    --slides slides.md \
    --ask "why does a large learning rate cause problems?"

python3 adaptive.py --kb knowledge_base.json
```

---

## Optional: hardware acceleration

The default path is CPU. `qualcomm_npu.py` adds an optional compile-and-profile path via
**Qualcomm AI Hub** for Snapdragon hardware. If you run it, it can collect on-device
latency, peak memory and the per-layer NPU/CPU execution split. This profiling path
requires a network connection; it is separate from the offline study workflow.

```bash
pip install qai-hub qai-hub-models
export QAI_HUB_API_TOKEN="<token from app.aihub.qualcomm.com>"
python3 qualcomm_npu.py --device "Snapdragon X Elite CRD"
```

> **Not measured yet.** The compile path is implemented; no profiling job has been run,
> so no numbers are claimed here.

| Metric | Value |
|---|---|
| Target device | *pending* |
| Model | Whisper-Small |
| Compile job ID | *pending* |
| Inference latency | *pending* |
| Peak memory | *pending* |
| Compute unit | *pending* |

This is an optimisation, not a dependency. Everything above runs without it.

---

## Two bugs worth reading about

Both were silent failures — the dangerous kind. They are documented because a system that
claims to have no bugs is a system nobody has tested.

### 1. The deskew that did nothing

`whiteboard.py` corrects camera tilt by rotating the image over a range of angles and
picking the one that maximises the variance of the horizontal projection profile — text
lines become sharp peaks when level.

The first version profiled the whole frame. **Rotating an image fills the corners with a
constant colour, and those blank wedges dominate the variance**, so the score increased
monotonically with the angle and the search always returned its boundary value. It looked
like it was working and was a no-op. Profiling only the central region, excluding the fill
border, fixed it: the estimator now recovers 4.2° on an image rotated by 4.5°.

### 2. The concepts that never accumulated

`adaptive.py` tracks mastery per concept. The first version labelled each *sentence* with
its longest word, which turned a 31-question assessment into 26 distinct "concepts".

The effect was not obvious in the output — the session ran, questions were asked, answers
were recorded — but **every question arrived on a fresh concept with a fresh novice
rating**, so ability never accumulated and mastery stayed pinned near zero no matter how
well the student performed. The trace showed ability repeatedly resetting to 1200.

Labelling per *chunk* instead of per sentence gives one concept per coherent passage. The
session now tracks 8 concepts and ability moves 1200 → 1225 as it should.

---

## Honest limitations

- **Lecture video ingestion is not implemented.** Audio works; extracting slide frames from
  a video is designed but not built. This is the largest remaining gap.
- **Retrieval is BM25, not embeddings.** No model weights, instant on CPU, fully
  inspectable — but it matches on terms, not meaning. "How do I stop a model memorising the
  training data?" will not find a passage about overfitting unless the words overlap. A
  hybrid with embeddings is the roadmapped upgrade; `BM25Index.search()` is the single
  method that changes.
- **Answers are extractive.** Nothing is generated, so nothing is hallucinated — but the
  output can be a listing rather than an explanation. The two-sentence assembly in `ask()`
  mitigates this; a generative layer with citation grounding would do better.
- **Whisper-Small is English-first.** Indian-English accents, code-switching and technical
  jargon will degrade accuracy.
- **OCR needs the tesseract binary**, which is a system dependency, not a pip one.
- **Adaptive difficulty seeds are heuristic**, derived from sentence properties (definition
  vs prose, clause structure). They self-calibrate as items are answered, but the first pass
  for a new item pool is approximate.

---

## Repository layout

```
├── app.py             audio pipeline: transcribe → summarise → flashcards → report
├── whiteboard.py      vision pipeline: photo → preprocess → OCR → structured notes
├── knowledge_base.py  cited knowledge base + BM25 retrieval + grounded QA
├── adaptive.py        Elo ability estimation + adaptive item selection + spacing
├── qualcomm_npu.py    optional AI Hub compile + profile path
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Roadmap

1. **Lecture video ingestion** — extract audio *and* detect slide-change frames, so one
   video yields both the transcript and the slides.
2. **Hybrid retrieval** — BM25 plus embeddings, fused by reciprocal rank.
3. **Fine-tune Whisper** on Indian-English lecture audio
4. **Kannada and Hindi** lecture support via multilingual Whisper
5. **Anki / Quizlet export** for the generated decks

---

## Background

The constraint driving every decision: **build for the student who currently has the least
access.** That student is on a campus with unreliable Wi-Fi, paying for mobile data by the
gigabyte, and cannot upload their lectures to a remote server. On-device
inference is what makes the tool reach them.

StudySnap has been developed and iterated across several student AI competitions during
2026, and extended between them.

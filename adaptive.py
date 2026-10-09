"""
adaptive.py — adaptive assessment engine for StudySnap.

Track D asks for "adaptive assessments and personalized tutoring". This module
is the adaptive half: it decides *what to ask next* and *when to ask it again*,
per student, per concept.

Three mechanisms, each doing a specific job:

1. **Ability estimation (Elo).** Each concept carries a student ability rating
   and each question carries a difficulty rating on the same scale. After every
   answer the ability moves toward the truth by an amount proportional to the
   surprise — a correct answer to a hard question moves it a lot, a correct
   answer to an easy question barely moves it. This is the same mechanism used
   to rate chess players, applied to concept mastery.

2. **Item selection at the desired-difficulty point.** Questions are not chosen
   at random and not chosen to be easy. Retrieval-practice research puts the
   most durable learning when the learner succeeds roughly 70-85% of the time.
   On the Elo scale that is a question ~130 points above current ability, so
   that is what we target. Asking questions the student always gets right
   teaches nothing; asking ones they always get wrong teaches nothing either.

3. **Spaced repetition (SM-2 style).** A concept answered correctly is
   rescheduled further into the future; one answered incorrectly resets to a
   short interval. The interval multiplier is scaled by how far above the
   student's ability the item was, so hard wins earn longer gaps.

Every item carries the citation of the passage it was generated from, so a
wrong answer can be traced straight back to the lecture timestamp or page.

Run:  python3 adaptive.py --demo
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta


# ============================================================================
# Data model
# ============================================================================

@dataclass
class Item:
    """One question, with the passage it came from."""
    item_id: str
    question: str
    answer: str
    concept: str
    difficulty: float = 1200.0      # Elo scale
    source: str = ""
    locator: str = ""
    seen: int = 0
    correct: int = 0

    def citation(self) -> str:
        return f"{self.source} @ {self.locator}" if self.source else ""


@dataclass
class ConceptState:
    """Per-concept mastery for one student."""
    concept: str
    ability: float = 1200.0          # Elo scale, same units as Item.difficulty
    attempts: int = 0
    correct: int = 0
    interval_days: float = 0.0
    ease: float = 2.5                # SM-2 ease factor
    due: str = ""                    # ISO date

    def mastery(self) -> float:
        """
        Map ability onto a 0-1 mastery score for display.

        1200 is the starting point (novice, mastery 0); 1400 is treated as
        solid command (mastery 1).

        The width of this window is a presentation choice, not a measurement.
        Elo ability moves slowly and deliberately — a 30-question session might
        move it 25 points — so anchoring mastery at 1650 rendered real progress
        as 5% and told the student nothing. A 200-point window keeps the number
        responsive while staying monotonic in ability. Clamped so a long
        correct streak cannot report 140%.
        """
        return max(0.0, min(1.0, (self.ability - 1200.0) / 200.0))


# ============================================================================
# The engine
# ============================================================================

class AdaptiveEngine:
    K = 32.0                 # Elo update rate
    # Offset from current ability at which to aim the next question.
    #
    # Sign check, because this was initially wrong: with Elo expected score
    # p = 1 / (1 + 10^((difficulty - ability)/400)), a POSITIVE offset makes the
    # item harder than the student and p < 0.5. A +130 offset yields p = 0.32 —
    # questions the student mostly fails, which teaches little. To land in the
    # desirable-difficulty band around p = 0.70 we need the item somewhat
    # EASIER than current ability: solving gives d - a = -145.
    TARGET_OFFSET = -145.0   # => p ~= 0.70
    START_ABILITY = 1200.0

    def __init__(self, items: list[Item], seed: int = 7):
        self.items: dict[str, Item] = {i.item_id: i for i in items}
        self.students: dict[str, dict[str, ConceptState]] = {}
        self.history: list[dict] = []
        self.rng = random.Random(seed)

    # ---- student state --------------------------------------------------
    def _state(self, student: str, concept: str) -> ConceptState:
        book = self.students.setdefault(student, {})
        if concept not in book:
            book[concept] = ConceptState(concept)
        return book[concept]

    # ---- selection ------------------------------------------------------
    @staticmethod
    def _expected(ability: float, difficulty: float) -> float:
        """Logistic probability the student answers correctly. The Elo core."""
        return 1.0 / (1.0 + 10 ** ((difficulty - ability) / 400.0))

    def next_item(self, student: str, concept: str | None = None,
                  now: datetime | None = None) -> Item | None:
        """
        Choose the next question.

        Priority: anything *due* for review first (spaced repetition beats new
        material), then the unseen question closest to the desired difficulty.
        """
        now = now or datetime.now()

        pool = [i for i in self.items.values() if concept is None or i.concept == concept]
        pool = [i for i in pool if i.seen < 3]
        if not pool:
            return None

        due: list[tuple[Item, ConceptState]] = []
        fresh: list[tuple[float, Item]] = []

        for it in pool:
            st = self._state(student, it.concept)
            if st.due and datetime.fromisoformat(st.due) <= now and it.seen > 0:
                due.append((it, st))
            elif it.seen == 0:
                target = st.ability + self.TARGET_OFFSET
                fresh.append((abs(it.difficulty - target), it))

        if due:
            # among due items, the one furthest below current ability is safest
            # to consolidate first
            due.sort(key=lambda x: x[0].difficulty)
            return due[0][0]

        if fresh:
            fresh.sort(key=lambda x: x[0])
            return fresh[0][1]
        return None

    # ---- update ---------------------------------------------------------
    def record(self, student: str, item: Item, correct: bool,
               now: datetime | None = None) -> dict:
        """
        Apply an answer: update ability, reschedule, and return the delta so
        the caller can show the student what changed.
        """
        now = now or datetime.now()
        st = self._state(student, item.concept)

        before = st.ability
        expected = self._expected(st.ability, item.difficulty)
        outcome = 1.0 if correct else 0.0
        st.ability += self.K * (outcome - expected)
        st.attempts += 1
        st.correct += int(correct)

        item.seen += 1
        item.correct += int(correct)

        # --- spaced repetition -------------------------------------------
        if correct:
            # How far above the aimed-at level was this item? A hard win earns
            # a longer interval than an easy one.
            target = before + self.TARGET_OFFSET
            overshoot = max(0.0, item.difficulty - target)
            hard_bonus = 1.0 + min(1.0, overshoot / 400.0)
            if st.interval_days == 0:
                st.interval_days = 1.0 * hard_bonus
            elif st.interval_days < 3:
                st.interval_days = 3.0 * hard_bonus
            else:
                st.interval_days = st.interval_days * st.ease * hard_bonus
            st.ease = min(3.0, st.ease + 0.05)
        else:
            st.interval_days = 0.5          # back tomorrow
            st.ease = max(1.3, st.ease - 0.2)

        st.due = (now + timedelta(days=st.interval_days)).isoformat()

        # --- difficulty calibration --------------------------------------
        # Items answered correctly by almost everyone, or by almost no one,
        # carry little information. Nudge their difficulty toward the observed
        # outcome so the pool self-calibrates over time.
        if item.seen >= 2:
            observed = item.correct / item.seen
            if observed > 0.9:
                item.difficulty += 8
            elif observed < 0.25:
                item.difficulty -= 8

        rec = {
            "student": student,
            "item_id": item.item_id,
            "concept": item.concept,
            "correct": correct,
            "ability_before": round(before, 1),
            "ability_after": round(st.ability, 1),
            "delta": round(st.ability - before, 1),
            "expected_prob": round(expected, 3),
            "next_interval_days": round(st.interval_days, 2),
            "citation": item.citation(),
        }
        self.history.append(rec)
        return rec

    # ---- reporting ------------------------------------------------------
    def report(self, student: str) -> dict:
        book = self.students.get(student, {})
        if not book:
            return {"student": student, "concepts": [], "overall_mastery": 0.0}
        rows = []
        for st in sorted(book.values(), key=lambda s: s.mastery()):
            rows.append({
                "concept": st.concept,
                "ability": round(st.ability, 1),
                "mastery": round(st.mastery(), 3),
                "attempts": st.attempts,
                "accuracy": round(st.correct / st.attempts, 3) if st.attempts else 0.0,
                "next_review_in_days": round(st.interval_days, 2),
            })
        all_concepts = {i.concept for i in self.items.values()}
        attempted = [r for r in rows if r["attempts"] > 0]
        return {
            "student": student,
            "concepts": rows,
            "concepts_attempted": len(attempted),
            "concepts_available": len(all_concepts),
            "overall_mastery": round(sum(r["mastery"] for r in rows) / len(rows), 3)
                               if rows else 0.0,
            "mastery_over_attempted": round(
                sum(r["mastery"] for r in attempted) / len(attempted), 3)
                if attempted else 0.0,
            "weakest": min(attempted, key=lambda r: r["mastery"])["concept"]
                        if attempted else None,
        }

    def save(self, path: str) -> None:
        json.dump({
            "items": [asdict(i) for i in self.items.values()],
            "history": self.history,
        }, open(path, "w"), indent=2)


# ============================================================================
# Item generation from the knowledge base
# ============================================================================

def items_from_knowledge_base(kb, max_items_per_chunk: int = 2) -> list[Item]:
    """
    Generate assessment items from the cited knowledge base.

    Difficulty is seeded from measurable properties of the source sentence
    rather than assigned arbitrarily:
      - a definition ("X is Y") is easier than an inference or a comparison
      - a longer answer with more distinct terms is harder to recall
      - clauses joined by "while"/"whereas" imply a comparison, which is harder

    Seeding from text properties means the first calibration pass is already
    roughly sensible instead of starting every item at the same difficulty.
    """
    import knowledge_base as K

    items: list[Item] = []
    if not isinstance(kb, K.KnowledgeBase):
        kb = K.KnowledgeBase.load(kb) if isinstance(kb, str) else kb

    defi = []
    others = []
    for chunk in kb.chunks:
        for s in K.sentences(chunk.text):
            (defi if K.DEFINITION_LIKE.search(s) else others).append((s, chunk))

    def make(sent: str, chunk, idx: int, base: float) -> Item:
        ok = K.tokens(sent)
        difficulty = base + min(220.0, len(ok) * 9.0)
        if any(w in sent.lower() for w in ("while", "whereas", "compared", "versus", "unlike")):
            difficulty += 90          # comparisons are harder to reproduce
        concept = chunk.concept or _concept_of(chunk, sent)
        return Item(
            item_id=f"q{idx:03d}",
            question=_question_for(sent, chunk),
            answer=sent,
            concept=concept,
            difficulty=round(difficulty, 1),
            source=chunk.source,
            locator=chunk.locate(sent),
        )

    # Base difficulties are set so the pool spans roughly 980-1400. The
    # desired-difficulty target for a new student is ability + TARGET_OFFSET,
    # i.e. 1200 - 145 = 1055, and a pool that does not reach down that far
    # cannot serve the target band at all — the closest item wins and the
    # realised success rate drifts back toward 50%.
    i = 0
    for sent, chunk in defi[: max_items_per_chunk * len(kb.chunks)]:
        items.append(make(sent, chunk, i, 980.0))    # definitions are easier
        i += 1
    for sent, chunk in others[: max_items_per_chunk * len(kb.chunks)]:
        items.append(make(sent, chunk, i, 1140.0))   # recall of prose is harder
        i += 1

    return items


def _concept_of(chunk, sentence: str = "") -> str:
    """
    Derive a stable concept label for a chunk.

    This is computed over the WHOLE CHUNK, not per sentence. An earlier version
    labelled each sentence by its longest word, which meant a 31-question
    assessment produced 26 distinct "concepts" — so every question arrived on a
    fresh concept with a fresh novice rating, and ability never accumulated.
    Mastery stayed flat at 2.6% no matter how well the student did.

    Labelling per chunk gives one concept per coherent passage, which is the
    unit a student actually revises and the unit a mastery score should
    describe. The label is the two most frequent content words, in the order
    they first appear, which yields readable names like "learning rate" and
    "gradient descent" instead of stems like "accumulat".
    """
    import knowledge_base as K
    from collections import Counter as _C
    import re as _re

    source_text = chunk.text if hasattr(chunk, "text") else str(chunk)

    words = [w.lower() for w in _re.findall(r"[A-Za-z][A-Za-z-]{3,}", source_text)]
    words = [w for w in words if w not in K.STOPWORDS]
    if not words:
        return "general"

    freq = _C(words)
    top = [w for w, _ in freq.most_common(2)]
    seen = []
    for w in words:                      # preserve natural reading order
        if w in top and w not in seen:
            seen.append(w)

    # Only join two words into a phrase when they actually occur near each
    # other. Frequency alone produced labels like "overfitting model" and
    # "bias error" — two common words that never appear as a phrase, which
    # reads as noise in a mastery report.
    if len(seen) == 2:
        a, b = seen
        window = f"{a} {b}"
        near = any(
            window in " ".join(words[i:i + 4])
            for i in range(len(words) - 1)
        )
        if near:
            return window
    return seen[0][:24] if seen else top[0][:24]


def _question_for(sentence: str, chunk) -> str:
    """Build a cloze question from the sentence, blanking its key term."""
    import knowledge_base as K
    import re

    toks = K.tokens(sentence)
    if len(toks) < 4:
        return f"Recall: {sentence}"
    target = max(toks, key=len)
    blanked = re.sub(rf"\b{re.escape(target)}\w*\b", "______", sentence, count=1, flags=re.I)
    if blanked == sentence:
        return f"Recall: {sentence}"
    return blanked


# ============================================================================
# Demo
# ============================================================================

def run_demo() -> None:
    print("=" * 74)
    print("StudySnap — adaptive assessment engine (Track D)")
    print("=" * 74)

    import knowledge_base as K
    import pathlib

    kb = K.KnowledgeBase("demo")
    pathlib.Path("/tmp/_lect.txt").write_text(K.LECTURE, encoding="utf-8")
    pathlib.Path("/tmp/_slides.md").write_text(K.SLIDES, encoding="utf-8")
    kb.add_transcript("/tmp/_lect.txt", "lecture_04_gradient_descent.mp4")
    kb.add_pages(K.TEXTBOOK_PAGES, "ml_textbook.pdf")
    kb.add_notes_markdown("/tmp/_slides.md")

    items = items_from_knowledge_base(kb)
    print(f"\nGenerated {len(items)} assessment items from {len(kb.chunks)} chunks")
    print(f"  difficulty range: {min(i.difficulty for i in items):.0f}"
          f" - {max(i.difficulty for i in items):.0f}")

    eng = AdaptiveEngine(items, seed=11)

    # Simulate a student whose true ability rises as they practise.
    print("\n" + "=" * 74)
    print("SIMULATED SESSION — one student, 30 questions")
    print("=" * 74)
    print(f"{'#':>2}  {'concept':<20} {'diff':>6} {'P(correct)':>10} {'result':>7} {'ability':>8} {'Δ':>6}")
    print("-" * 74)

    # Start the simulated student at the engine's own prior, otherwise the
    # printed P(correct) describes a student the engine is not modelling.
    true_ability = AdaptiveEngine.START_ABILITY
    for n in range(1, 31):
        it = eng.next_item("student_A")
        if it is None:
            break
        st = eng._state("student_A", it.concept)
        p = eng._expected(true_ability, it.difficulty)
        # simulate: as the student practises, latent ability grows
        correct = random.Random(n).random() < p
        rec = eng.record("student_A", it, correct)
        true_ability += 14 if correct else 4
        print(f"{n:>2}  {it.concept[:20]:<20} {it.difficulty:>6.0f} "
              f"{rec['expected_prob']:>10.2f} {'✓' if correct else '✗':>7} "
              f"{rec['ability_after']:>8.1f} {rec['delta']:>+6.1f}")

    hits = [h for h in eng.history if h["student"] == "student_A"]
    if hits:
        acc = sum(1 for h in hits if h["correct"]) / len(hits)
        print(f"\nrealised success rate: {acc:.0%}   (target band 70-85%)")
        print(f"ability moved {hits[0]['ability_before']:.0f} -> {hits[-1]['ability_after']:.0f}")

    rep = eng.report("student_A")
    print("\n" + "=" * 74)
    print("MASTERY REPORT")
    print("=" * 74)
    print(f"concepts attempted: {rep['concepts_attempted']}/{rep['concepts_available']}")
    print(f"mastery over attempted concepts: {rep['mastery_over_attempted']:.1%}")
    print(f"weakest area: {rep['weakest']}\n")
    print(f"{'concept':<26} {'mastery':>8} {'ability':>9} {'acc':>6} {'next review':>12}")
    print("-" * 74)
    for r in sorted(rep["concepts"], key=lambda x: -x["mastery"]):
        print(f"{r['concept'][:26]:<26} {r['mastery']:>7.1%} {r['ability']:>9.1f} "
              f"{r['accuracy']:>6.1%} {r['next_review_in_days']:>10.1f}d")

    print("\n" + "=" * 74)
    print("SAMPLE TRACEABLE ITEM — a wrong answer leads back to the source")
    print("=" * 74)
    sample = max(items, key=lambda i: i.seen)
    print(f"  concept    : {sample.concept}")
    print(f"  question   : {sample.question}")
    print(f"  answer     : {sample.answer[:110]}")
    print(f"  difficulty : {sample.difficulty:.0f}")
    print(f"  source     : {sample.citation()}")

    eng.save("adaptive_state.json")
    print("\nSaved adaptive_state.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="StudySnap adaptive assessment engine")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--kb", help="Path to a saved knowledge_base.json")
    args = ap.parse_args()
    if args.demo or not args.kb:
        run_demo()
        return
    kb = __import__("knowledge_base").KnowledgeBase.load(args.kb)
    items = items_from_knowledge_base(kb)
    print(json.dumps({"items_generated": len(items),
                      "concepts": sorted({i.concept for i in items})}, indent=2))


if __name__ == "__main__":
    main()

"""
knowledge_base.py — source-cited knowledge base for StudySnap.

Track D of the Multimodal AI Hackathon 2026 requires that lecture videos,
textbooks and slides be unified into a *source-cited* knowledge base. This
module is that knowledge base.

The design principle is that **every retrieved fact carries its provenance**,
precisely enough to act on. A citation is only useful if it lands in the right
place, so provenance is tracked at segment level: each chunk knows the
timestamp or page of every line inside it, not just where the chunk began.
Citing "lecture.mp4 @ 00:00" for something said at 02:10 sends the student to
the wrong part of the video, which is worse than citing nothing.

    ingest -> chunk w/ per-segment provenance -> BM25 -> cited answer
                                                              |
                                          answer + [source @ 02:10]

Run:  python3 knowledge_base.py --demo
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path


# ============================================================================
# Core data model
# ============================================================================

@dataclass
class Segment:
    """One addressable unit inside a chunk — a timestamp or a page."""
    locator: str
    text: str


@dataclass
class Chunk:
    """A retrievable unit of text that knows exactly where it came from."""
    chunk_id: str
    text: str
    source: str
    kind: str                      # lecture | textbook | slides | board
    locator: str                   # locator of the chunk's first segment
    concept: str = ""
    segments: list[Segment] = field(default_factory=list)

    def citation(self) -> str:
        return f"{self.source} @ {self.locator}"

    def locate(self, sentence: str) -> str:
        """
        Find the precise locator for a specific sentence.

        This is what makes citations trustworthy: rather than reporting where
        the chunk starts, we report where the *answer* is.
        """
        probe = re.sub(r"\W+", " ", sentence.lower()).strip()[:60]
        if not probe:
            return self.locator
        for seg in self.segments:
            if probe[:40] and probe[:40] in re.sub(r"\W+", " ", seg.text.lower()):
                return seg.locator
        # fall back to token overlap when the sentence was assembled across lines
        pt = set(probe.split())
        best, best_ov = self.locator, 0
        for seg in self.segments:
            ov = len(pt & set(re.sub(r"\W+", " ", seg.text.lower()).split()))
            if ov > best_ov:
                best_ov, best = ov, seg.locator
        return best

    def short(self, n: int = 92) -> str:
        t = re.sub(r"\s+", " ", self.text).strip()
        return t[:n] + ("..." if len(t) > n else "")


@dataclass
class Answer:
    text: str
    citations: list[str]
    chunks: list[Chunk]
    confidence: float
    grounded: bool

    def render(self) -> str:
        if not self.grounded:
            return "I could not find this in the provided material."
        out = [self.text, "", "Sources:"]
        seen = set()
        for c in self.citations:
            if c not in seen:
                seen.add(c)
                out.append(f"  - {c}")
        return "\n".join(out)


# ============================================================================
# Text utilities
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "it", "its", "this", "that", "these", "those", "we", "you", "they",
    "he", "she", "i", "me", "my", "our", "your", "their", "there", "here",
    "what", "which", "who", "when", "where", "why", "how", "can", "could",
    "will", "would", "should", "may", "might", "must", "do", "does", "did",
    "have", "has", "had", "not", "no", "also", "such", "into", "over", "under",
    "between", "about", "each", "other", "more", "most", "some", "any", "all",
    "both", "very", "just", "only", "own", "same", "too", "s", "difference",
    "between", "explain",
}


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ising", "izing", "ised", "ized", "ing", "ies", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-3] + "y" if suf == "ies" else w[: -len(suf)]
    return w


def tokens(text: str, stem: bool = True) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out = [w for w in raw if w not in STOPWORDS and len(w) > 1]
    return [_stem(w) for w in out] if stem else out


# A sentence that defines or explains a term is a better assessment item and a
# more useful answer than a sentence that merely mentions it.
DEFINITION_LIKE = re.compile(
    r"\b(is|are|means|refers to|occurs when|describes|consists of|"
    r"is defined as|is called|is the)\b", re.I)


def sentences(text: str, min_len: int = 25) -> list[str]:
    """
    Split into sentences, respecting line boundaries before collapsing space.

    Splitting on terminal punctuation alone glues a document heading onto the
    sentence below it ("Evaluation metrics For classification we report..."),
    because a heading has no full stop. Splitting per line first keeps headings
    separate, and the min_len filter then discards them as non-answers.
    """
    out: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", line):
            part = part.strip()
            if len(part) >= min_len:
                out.append(part)
    return out


# ============================================================================
# Chunking — per-segment provenance preserved
# ============================================================================

TIMESTAMP_RE = re.compile(r"^\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.*)$")


def chunk_transcript(text: str, source: str, kind: str = "lecture",
                     target_words: int = 90, overlap: int = 25) -> list[Chunk]:
    """
    Chunk a timestamped transcript into overlapping windows, keeping the
    timestamp of every line so citations can point at the exact moment.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    stamped: list[Segment] = []
    for ln in lines:
        m = TIMESTAMP_RE.match(ln)
        if m:
            stamped.append(Segment(m.group(1), m.group(2).strip()))
        elif stamped:
            stamped[-1].text += " " + ln

    if not stamped:
        return []

    chunks: list[Chunk] = []
    i = n = 0
    while i < len(stamped):
        window: list[Segment] = []
        words = 0
        j = i
        while j < len(stamped) and words < target_words:
            window.append(stamped[j])
            words += len(stamped[j].text.split())
            j += 1
        if not window:
            break
        chunks.append(Chunk(
            chunk_id=f"{source}#c{n}",
            text=" ".join(s.text for s in window),
            source=source,
            kind=kind,
            locator=window[0].locator,
            segments=list(window),
        ))
        n += 1
        if j >= len(stamped):
            break
        back = w = 0
        while j - back - 1 > i and w < overlap:
            w += len(stamped[j - back - 1].text.split())
            back += 1
        i = max(i + 1, j - back)

    return chunks


MARKDOWN_RE = re.compile(r"(^|\s)[#*_`>]+\s*|\*\*|__", re.M)


def clean_markdown(text: str) -> str:
    """
    Strip presentation syntax before chunking, preserving line structure.

    Without this, a slide titled "# Evaluation metrics" puts a literal hash into
    the middle of an answer. Line breaks are kept so headings remain separable
    from the prose beneath them.
    """
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r"^\s*#{1,6}\s*", "", ln)
        ln = re.sub(r"\*\*|__|`", "", ln)
        ln = re.sub(r"^\s*[*\-]\s+", "", ln)
        lines.append(ln)
    return "\n".join(lines)


def chunk_pages(pages: list[str], source: str, kind: str = "textbook",
                target_words: int = 110) -> list[Chunk]:
    """Chunk page-delimited material, keeping the page number of every sentence."""
    pages = [clean_markdown(p) for p in pages]
    chunks: list[Chunk] = []
    n = 0
    for page_no, page in enumerate(pages, 1):
        locator = f"p. {page_no}"
        sents = sentences(page) or ([page.strip()] if page.strip() else [])
        buf: list[Segment] = []
        words = 0
        for s in sents:
            buf.append(Segment(locator, s))
            words += len(s.split())
            if words >= target_words:
                chunks.append(Chunk(
                    chunk_id=f"{source}#c{n}",
                    text=" ".join(x.text for x in buf),
                    source=source, kind=kind, locator=locator, segments=list(buf),
                ))
                n += 1
                buf, words = [], 0
        if buf:
            chunks.append(Chunk(
                chunk_id=f"{source}#c{n}",
                text=" ".join(x.text for x in buf),
                source=source, kind=kind, locator=locator, segments=list(buf),
            ))
            n += 1
    return chunks


def chunk_board(blocks: list, source: str = "board_photo.jpg") -> list[Chunk]:
    """Ingest structured board notes, using headings as concept labels."""
    chunks: list[Chunk] = []
    concept = "board notes"
    buf: list[str] = []
    n = 0

    def flush():
        nonlocal buf, n, concept
        if not buf:
            return
        chunks.append(Chunk(
            chunk_id=f"{source}#c{n}",
            text=" ".join(buf),
            source=source, kind="board", locator=concept, concept=concept,
            segments=[Segment(concept, t) for t in buf],
        ))
        n += 1
        buf = []

    for b in blocks:
        kind = getattr(b, "kind", None) or b.get("kind")
        text = getattr(b, "text", None) or b.get("text")
        if kind == "heading":
            flush()
            concept = text
        else:
            buf.append(text)
    flush()
    return chunks


# ============================================================================
# Retrieval — BM25
# ============================================================================

class BM25Index:
    """Okapi BM25. No model weights, instant on CPU, fully inspectable."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_tokens = [tokens(c.text) for c in chunks]
        self.doc_len = [len(d) for d in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf = [Counter(d) for d in self.doc_tokens]
        self.df: dict[str, int] = defaultdict(int)
        for d in self.doc_tokens:
            for w in set(d):
                self.df[w] += 1
        self.N = len(chunks)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, idx: int) -> float:
        q = tokens(query)
        if not q or self.avg_len == 0:
            return 0.0
        dl = self.doc_len[idx] or 1
        s = 0.0
        for term in q:
            f = self.tf[idx].get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
            s += self.idf(term) * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 4, kind: str | None = None) -> list[tuple[Chunk, float]]:
        scored = []
        for i, c in enumerate(self.chunks):
            if kind and c.kind != kind:
                continue
            s = self.score(query, i)
            if s > 0:
                scored.append((c, s))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ============================================================================
# Knowledge base
# ============================================================================

class KnowledgeBase:
    def __init__(self, name: str = "course"):
        self.name = name
        self.chunks: list[Chunk] = []
        self._index: BM25Index | None = None

    # ---- ingestion -----------------------------------------------------
    def add_transcript(self, path: str, source: str | None = None,
                       kind: str = "lecture") -> int:
        text = Path(path).read_text(encoding="utf-8")
        new = chunk_transcript(text, source or Path(path).name, kind)
        self.chunks.extend(new)
        self._index = None
        return len(new)

    def add_pages(self, pages: list[str], source: str, kind: str = "textbook") -> int:
        new = chunk_pages(pages, source, kind)
        self.chunks.extend(new)
        self._index = None
        return len(new)

    def add_pdf(self, path: str, kind: str = "textbook") -> int:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit("PDF ingestion needs pypdf:  pip install pypdf")
        reader = PdfReader(path)
        return self.add_pages([p.extract_text() or "" for p in reader.pages],
                              Path(path).name, kind)

    def add_notes_markdown(self, path: str, kind: str = "slides") -> int:
        text = Path(path).read_text(encoding="utf-8")
        slides = [s for s in re.split(r"\n(?=#{1,3}\s|\f)", text) if s.strip()]
        return self.add_pages(slides, Path(path).name, kind)

    def add_board(self, blocks: list, source: str = "board_photo.jpg") -> int:
        new = chunk_board(blocks, source)
        self.chunks.extend(new)
        self._index = None
        return len(new)

    # ---- retrieval -----------------------------------------------------
    @property
    def index(self) -> BM25Index:
        if self._index is None:
            self._index = BM25Index(self.chunks)
        return self._index

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        return self.index.search(query, k=k)

    def _score_sentence(self, sent: str, q_terms: set[str], idx: BM25Index) -> float:
        """
        Rank candidate sentences as answers.

        Scoring is IDF-weighted *coverage of the question's distinct terms*,
        and deliberately nothing else. Two tempting additions both made results
        worse and were removed:

        - A verbosity reward (favouring longer sentences). This flipped a
          correct answer into a wrong one: "If the learning rate is too large
          the algorithm diverges" has higher coverage than the Adam sentence,
          but the Adam sentence is longer, so the length term won. Longer is
          not more answer-like.
        - A density measure (overlap divided by length). This promoted
          fragments like "Use precision, recall or AUC instead" purely because
          they are short and repeat query words.

        The only length term kept is a penalty for genuinely fragmentary
        sentences, which are rarely self-contained answers.
        """
        st = tokens(sent)
        if not st:
            return -1.0
        total = sum(idx.idf(t) for t in q_terms) or 1.0
        matched = sum(idx.idf(t) for t in q_terms if t in set(st))
        coverage = matched / total

        fragment_penalty = 1.0 if len(st) >= 5 else 0.6
        return coverage * fragment_penalty

    def ask(self, query: str, k: int = 4, min_score: float = 0.6) -> Answer:
        """
        Answer from the material, with precise citations.

        The answer is extractive, so nothing can be hallucinated. If nothing
        clears the relevance threshold we return grounded=False rather than
        inventing something — for a study tool, a confident wrong answer is the
        worst possible failure.
        """
        hits = self.search(query, k=k)
        if not hits or hits[0][1] < min_score:
            return Answer("", [], [], 0.0, grounded=False)

        q_terms = set(tokens(query))
        idx = self.index

        # Gather every candidate sentence, de-duplicated: chunks overlap by
        # design, so a sentence near a boundary appears in two chunks and would
        # otherwise be scored and possibly cited twice.
        candidates: list[tuple[float, str, Chunk]] = []
        seen: set[str] = set()
        for chunk, _ in hits:
            for s in sentences(chunk.text):
                key = re.sub(r"\W+", "", s.lower())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((self._score_sentence(s, q_terms, idx), s, chunk))
        candidates.sort(key=lambda x: -x[0])

        parts: list[str] = []
        if not candidates or candidates[0][0] <= 0:
            best_sent, best_chunk = hits[0][0].text[:300], hits[0][0]
            parts = [best_sent]
        else:
            best_score, best_sent, best_chunk = candidates[0]
            # Assemble a second sentence when it is nearly as relevant. Many
            # real questions span two concepts — "difference between precision
            # and recall" has no single-sentence answer — and one sentence is
            # often a listing rather than an explanation. The cap on combined
            # length stops this from degenerating into a paragraph dump.
            parts = [best_sent]
            words = len(best_sent.split())
            for sc, s, ch in candidates[1:]:
                if sc < 0.5 * best_score or words + len(s.split()) > 70:
                    continue
                parts.append(s)
                break
            if len(parts) > 1:
                best_sent = " ".join(parts)

        # Cite the answer itself first, then every other source consulted.
        # When the answer spans two sentences we locate each one, because they
        # can come from different timestamps or pages.
        anchors = []
        for part in parts:
            loc = best_chunk.locate(part)
            entry = f"{best_chunk.source} @ {loc}"
            if entry not in anchors:
                anchors.append(entry)

        others = [c.citation() for c, _ in hits if c is not best_chunk]
        top = hits[0][1]
        second = hits[1][1] if len(hits) > 1 else 0.0
        conf = min(1.0, (top - second) / top + 0.35) if top else 0.0

        return Answer(best_sent, anchors + others, [c for c, _ in hits],
                      round(conf, 3), True)

    def stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "by_kind": dict(Counter(c.kind for c in self.chunks)),
            "by_source": dict(Counter(c.source for c in self.chunks)),
            "vocabulary": len(self.index.df),
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(
            {"name": self.name,
             "chunks": [{**asdict(c), "segments": [asdict(s) for s in c.segments]}
                        for c in self.chunks]}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "KnowledgeBase":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        kb = cls(d.get("name", "course"))
        for c in d["chunks"]:
            segs = [Segment(**s) for s in c.pop("segments", [])]
            kb.chunks.append(Chunk(**c, segments=segs))
        return kb


# ============================================================================
# Demo
# ============================================================================

LECTURE = """[00:00] Today we cover gradient descent and its variants.
[00:45] Gradient descent is an iterative optimisation algorithm that minimises a loss function by repeatedly stepping in the direction of steepest descent.
[02:10] The learning rate controls the size of each step. If the learning rate is too large the algorithm diverges instead of converging.
[03:40] If it is too small, convergence takes a very long time and you may run out of patience before the loss stops improving.
[05:15] Momentum accumulates past gradients to smooth the update direction. Momentum helps escape shallow local minima where plain gradient descent stalls.
[07:30] Stochastic gradient descent updates the weights after every single training sample. This makes each step noisy but converges far faster on large datasets.
[09:00] Mini-batch gradient descent is the middle ground: update after a batch of samples, typically between thirty-two and five hundred and twelve.
[11:20] Adam combines momentum with an adaptive per-parameter learning rate. Adam is the default choice for most deep learning work today."""

TEXTBOOK_PAGES = [
    """Supervised learning is the task of learning a mapping from inputs to outputs
given a set of labelled examples. The training set provides the labels.
Common supervised tasks include classification, where the output is a discrete
category, and regression, where the output is a continuous value.""",
    """Overfitting occurs when a model memorises the training examples rather than
learning the underlying pattern. An overfitted model achieves low training
error and high validation error. Regularisation, dropout and early stopping are
the standard techniques for reducing overfitting. Underfitting is the opposite
condition, where the model is too simple to represent the true structure.""",
    """The bias-variance trade-off describes the tension between two sources of
error. Bias is the error from approximating a complex problem with a simpler
model. Variance is the error from sensitivity to the particular training
sample. Total error is the sum of bias squared, variance and irreducible noise.""",
]

SLIDES = """# Evaluation metrics
For classification we report accuracy, precision, recall and the F1 score.
Precision answers what fraction of predicted positives were correct.
Recall answers what fraction of the true positives were found.
The F1 score is the harmonic mean of precision and recall.

# Imbalanced data
Accuracy is misleading when the classes are imbalanced.
A classifier that always predicts the majority class can score 99 percent
accuracy while being completely useless.
Use precision, recall or AUC instead.

# Confusion matrix
The confusion matrix tabulates true positives, true negatives, false positives
and false negatives. Every classification metric above is derived from it."""


def run_demo() -> None:
    print("=" * 74)
    print("StudySnap — source-cited knowledge base (Track D core)")
    print("=" * 74)

    kb = KnowledgeBase("BCA AI-ML-DS coursework")
    Path("/tmp/_lecture.txt").write_text(LECTURE, encoding="utf-8")
    Path("/tmp/_slides.md").write_text(SLIDES, encoding="utf-8")
    kb.add_transcript("/tmp/_lecture.txt", "lecture_04_gradient_descent.mp4")
    kb.add_pages(TEXTBOOK_PAGES, "ml_textbook.pdf")
    kb.add_notes_markdown("/tmp/_slides.md")

    st = kb.stats()
    print(f"\nINGESTED {st['total_chunks']} chunks from {len(st['by_source'])} sources")
    for src, n in st["by_source"].items():
        print(f"   {src:38s} {n:3d} chunks")
    print(f"   vocabulary: {st['vocabulary']} terms")

    print("\n" + "=" * 74)
    print("CITED QUESTION ANSWERING")
    print("=" * 74)

    for q in [
        "why does a large learning rate cause problems?",
        "what is the difference between precision and recall?",
        "how does Adam differ from plain gradient descent?",
        "what is overfitting?",
        "what is the capital of France?",
    ]:
        ans = kb.ask(q, k=3)
        print(f"\nQ: {q}")
        print(f"   confidence {ans.confidence}  grounded {ans.grounded}")
        print("   " + ans.render().replace("\n", "\n   "))

    kb.save("knowledge_base.json")
    print("\n\nSaved knowledge_base.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="StudySnap source-cited knowledge base")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--transcript")
    ap.add_argument("--pdf", action="append", default=[])
    ap.add_argument("--slides", action="append", default=[])
    ap.add_argument("--ask")
    ap.add_argument("--save", default="knowledge_base.json")
    args = ap.parse_args()

    if args.demo or not any([args.transcript, args.pdf, args.slides]):
        run_demo()
        return

    kb = KnowledgeBase()
    if args.transcript:
        print(f"added {kb.add_transcript(args.transcript)} chunks")
    for p in args.pdf:
        print(f"added {kb.add_pdf(p)} chunks from {p}")
    for s in args.slides:
        print(f"added {kb.add_notes_markdown(s)} chunks from {s}")
    print(json.dumps(kb.stats(), indent=2))
    if args.ask:
        print("\n" + kb.ask(args.ask, k=3).render())
    kb.save(args.save)


if __name__ == "__main__":
    main()

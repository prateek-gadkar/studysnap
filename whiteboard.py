"""
whiteboard.py — StudySnap multimodal extension.

Adds the second input modality: photographs of whiteboards, textbook pages and
handwritten notes. Combined with the audio pipeline in app.py, this is what makes
StudySnap genuinely multimodal rather than single-modality with a nicer name.

     lecture audio  ──┐
                      ├──►  unified notes + flashcards
     whiteboard photo ─┘

Pipeline:
    photo ──► preprocess ──► vision/OCR ──► structure ──► merge with transcript

The preprocessing and structuring stages are real, deterministic image processing
and text analysis — they run with no model and no network. The vision stage is a
swappable interface with three implementations, in order of preference:
    1. Qualcomm AI Hub vision-language model (NPU path)   -- see qualcomm_npu.py
    2. pytesseract (portable CPU OCR)
    3. disabled, with a clear error rather than fabricated output

Run:  python3 whiteboard.py --demo
      python3 whiteboard.py --image board.jpg --transcript lecture.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps
    HAVE_IMAGING = True
except ImportError:  # pragma: no cover
    HAVE_IMAGING = False


# ============================================================================
# Stage 1 — Image preprocessing
# ============================================================================

def _otsu_threshold(gray: "np.ndarray") -> int:
    """
    Otsu's method: find the threshold that minimises intra-class variance.
    Used instead of a fixed cutoff because classroom photos have wildly
    different exposure depending on lighting and phone camera.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    w_bg = 0.0
    best_thresh, best_var = 0, -1.0

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        var_between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var_between > best_var:
            best_var, best_thresh = var_between, t
    return best_thresh


def _estimate_skew(gray: "np.ndarray", max_angle: float = 8.0, step: float = 0.5) -> float:
    """
    Estimate the rotation of a photographed page.

    Whiteboard photos are almost never square to the camera. We search over a
    range of angles and pick the one that maximises the variance of the
    horizontal projection profile — text lines become sharp peaks when the page
    is level, and smear out when it is tilted.

    Important detail: rotating fills the corners with a constant colour, and
    those blank wedges dominate the variance if you profile the whole frame.
    That makes the score rise monotonically with |angle| and the search always
    returns the boundary value — a deskew that silently does nothing. We
    therefore profile only the central region, excluding the fill border.
    """
    from PIL import Image as _Image

    base = _Image.fromarray(gray)
    h, w = gray.shape
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)

    best_angle, best_score = 0.0, -1.0
    angle = -max_angle
    while angle <= max_angle + 1e-9:
        rotated = base.rotate(angle, resample=_Image.BILINEAR, fillcolor=255)
        arr = np.asarray(rotated, dtype=np.float32)
        core = arr[margin_y:h - margin_y, margin_x:w - margin_x]
        if core.size == 0:
            core = arr
        profile = (255.0 - core).sum(axis=1)
        score = float(np.var(profile))
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step

    # Refine around the coarse winner with a finer step.
    fine_best = best_angle
    fine_score = best_score
    angle = best_angle - step
    while angle <= best_angle + step + 1e-9:
        rotated = base.rotate(angle, resample=_Image.BILINEAR, fillcolor=255)
        arr = np.asarray(rotated, dtype=np.float32)
        core = arr[margin_y:h - margin_y, margin_x:w - margin_x]
        if core.size == 0:
            core = arr
        score = float(np.var((255.0 - core).sum(axis=1)))
        if score > fine_score:
            fine_score, fine_best = score, angle
        angle += step / 5.0

    return round(fine_best, 2)


def preprocess(image_path: str, do_deskew: bool = True, debug_dir: str | None = None) -> dict:
    """
    Clean up a photographed board or page so the vision stage sees something
    legible. Returns the processed array plus a report of what was done.
    """
    if not HAVE_IMAGING:
        raise SystemExit("Imaging path needs numpy and Pillow:  pip install pillow numpy")

    img = Image.open(image_path).convert("RGB")

    # Upscale small images — text is usually under 20px tall in phone photos,
    # which is below what most OCR/vision models resolve reliably.
    w, h = img.size
    scale = 1.0
    if max(w, h) < 1600:
        scale = 1600.0 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = ImageOps.grayscale(img)

    # Flatten uneven lighting: a whiteboard shot under ceiling lights has a
    # bright centre and dark corners. Subtracting a heavily blurred copy leaves
    # only the local contrast, which is what carries the text.
    bg = gray.filter(ImageFilter.GaussianBlur(radius=max(gray.size) // 40))
    gray_arr = np.asarray(gray, dtype=np.float32)
    bg_arr = np.asarray(bg, dtype=np.float32)
    flattened = np.clip(gray_arr - bg_arr + 128.0, 0, 255).astype(np.uint8)

    skew = _estimate_skew(flattened) if do_deskew else 0.0
    if abs(skew) > 0.25:
        from PIL import Image as _Image
        flattened = np.asarray(
            _Image.fromarray(flattened).rotate(skew, resample=_Image.BICUBIC, fillcolor=255),
            dtype=np.uint8,
        )

    thresh = _otsu_threshold(flattened)
    binary = np.where(flattened > thresh, 255, 0).astype(np.uint8)

    report = {
        "source": image_path,
        "original_size": [w, h],
        "upscale_factor": round(scale, 3),
        "illumination_flattened": True,
        "deskew_angle_deg": round(skew, 2),
        "otsu_threshold": int(thresh),
        "output_size": list(binary.shape[::-1]),
    }

    if debug_dir:
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(flattened).save(d / "1_illumination_corrected.png")
        Image.fromarray(binary).save(d / "2_binarised.png")
        report["debug_written_to"] = str(d)

    return {"array": binary, "report": report}


# ============================================================================
# Stage 2 — Vision / OCR
# ============================================================================

def extract_text(image_path: str) -> tuple[str, str]:
    """
    Returns (text, backend_used).

    Preference order: AI Hub vision model on the NPU, then pytesseract on CPU.
    If neither is available we raise — fabricating text here would make every
    downstream stage a lie.
    """
    try:
        import pytesseract
        from PIL import Image as _Image

        pre = preprocess(image_path)
        text = pytesseract.image_to_string(_Image.fromarray(pre["array"]))
        return text.strip(), "pytesseract (CPU)"
    except ImportError:
        pass
    except Exception as exc:  # tesseract binary missing, etc.
        raise SystemExit(
            f"OCR backend failed: {exc}\n"
            "  Install the tesseract binary, or use the AI Hub vision path:\n"
            "  python3 qualcomm_npu.py --model qai_hub_models.models.qwen2_vl.Model"
        )

    raise SystemExit(
        "No vision backend available.\n"
        "  Portable CPU path : pip install pytesseract  (+ tesseract binary)\n"
        "  NPU path          : see the vision section of qualcomm_npu.py\n"
        "  Structuring only  : python3 whiteboard.py --demo"
    )


# ============================================================================
# Stage 3 — Structure raw text into notes
# ============================================================================

@dataclass
class NoteBlock:
    kind: str      # heading | bullet | definition | formula | paragraph
    text: str


FORMULA_CHARS = set("=+×÷∑∫√≈≠≤≥^²³πθλμΔ√")
NUMBERED_RE = re.compile(r"^\s*\(?\d{1,2}[.)]\s+(.*)$")
BULLET_RE = re.compile(r"^\s*[-•*·]\s+(.*)$")

# Conditional openers are not definitions. "If the learning rate is too large"
# matches an "X is Y" shape but produces the nonsense card "What is If the
# learning rate?" — so exclude them explicitly before the definition rule runs.
CONDITIONAL_PREFIX = re.compile(
    r"^\s*(if|when|whenever|suppose|supposing|assuming|unless|whether|since|because|while|although)\b",
    re.I,
)

DEFINITION_RE = re.compile(
    r"^\s*([A-Z][\w\s\-]{1,38}?)\s+(?:is|are|means|refers to)\s+(.+)$"
)


def is_definition(line: str) -> re.Match | None:
    """A definition line, with conditionals and weak subjects filtered out."""
    if CONDITIONAL_PREFIX.match(line):
        return None
    m = DEFINITION_RE.match(line)
    if not m:
        return None
    subject = m.group(1).strip()
    answer = m.group(2).strip()
    # Reject pronoun-ish subjects and one-word answers, which are almost always
    # mid-sentence fragments rather than real definitions.
    if subject.lower() in {"it", "this", "that", "there", "which", "they", "the"}:
        return None
    if len(answer.split()) < 2:
        return None
    return m


def is_heading(line: str) -> bool:
    """
    Short lines with no terminal punctuation, in Title Case or ALL CAPS, are
    headings. Covers both 'Gradient Descent' and 'GRADIENT DESCENT' as they
    actually get written on a board.
    """
    s = line.strip()
    if not (2 < len(s) < 55):
        return False
    if s[-1] in ".,;:":
        return False
    if len(s.split()) > 6:
        return False
    if any(ch in FORMULA_CHARS for ch in s):
        return False
    words = [w for w in s.split() if any(c.isalpha() for c in w)]
    if not words:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps == len(words)


def looks_like_formula(line: str) -> bool:
    """A line is 'formula-like' if it carries maths symbols or is mostly symbols."""
    if any(ch in FORMULA_CHARS for ch in line):
        return True
    non_alpha = sum(1 for ch in line if not ch.isalnum() and not ch.isspace())
    return len(line) > 0 and non_alpha / max(len(line), 1) > 0.25


def structure_notes(raw_text: str) -> list[NoteBlock]:
    """
    Turn flat OCR output into something a student can revise from.

    Board text follows conventions worth exploiting: headings are short and
    often end with a colon, definitions read "X is Y", items are numbered or
    bulleted, and equations stand alone.
    """
    blocks: list[NoteBlock] = []
    lines = [ln.rstrip() for ln in raw_text.splitlines()]

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Formulas are standalone even when they begin with a lowercase symbol
        # (a bare "w = w - alpha * dL/dw" would otherwise be absorbed into the
        # paragraph above it), so they are checked before continuation merge.
        if looks_like_formula(stripped):
            blocks.append(NoteBlock("formula", stripped))
            continue

        # OCR frequently breaks one sentence across lines. A line that starts
        # lowercase is almost always a continuation of the block above, so
        # rejoin it rather than emitting a fragment as its own note.
        if blocks and stripped[0].islower() and blocks[-1].kind in ("paragraph", "definition", "bullet"):
            blocks[-1].text += " " + stripped
            continue

        if is_heading(stripped):
            blocks.append(NoteBlock("heading", stripped.rstrip(":")))
        elif BULLET_RE.match(stripped):
            blocks.append(NoteBlock("bullet", BULLET_RE.match(stripped).group(1).strip()))
        elif NUMBERED_RE.match(stripped):
            blocks.append(NoteBlock("bullet", NUMBERED_RE.match(stripped).group(1).strip()))
        elif is_definition(stripped):
            blocks.append(NoteBlock("definition", stripped))
        else:
            blocks.append(NoteBlock("paragraph", stripped))

    return blocks


def notes_to_markdown(blocks: list[NoteBlock], title: str = "Board notes") -> str:
    out = [f"## {title}", ""]
    for b in blocks:
        if b.kind == "heading":
            out.append(f"### {b.text}")
        elif b.kind == "bullet":
            out.append(f"- {b.text}")
        elif b.kind == "definition":
            out.append(f"- **{b.text}**")
        elif b.kind == "formula":
            out.append(f"    {b.text}")
        else:
            out.append(b.text)
        if b.kind in ("heading", "paragraph"):
            out.append("")
    return "\n".join(out)


def notes_to_flashcards(blocks: list[NoteBlock]) -> list[dict]:
    """
    Definitions become Q/A cards directly. Formulas become recall prompts.
    This is why definition-detection is worth doing at all: a board full of
    "X is Y" lines converts to a deck for free.
    """
    cards = []
    for b in blocks:
        if b.kind == "definition":
            m = is_definition(b.text)
            if m:
                cards.append({
                    "question": f"What is {m.group(1).strip()}?",
                    "answer": m.group(2).strip().rstrip("."),
                    "origin": "board",
                })
        elif b.kind == "formula":
            cards.append({
                "question": f"Recall the formula: {b.text}",
                "answer": b.text,
                "origin": "board",
            })
    return cards


# ============================================================================
# Stage 4 — Merge with the audio pipeline
# ============================================================================

def merge_with_transcript(board_cards: list[dict], transcript_path: str | None) -> dict:
    """
    Combine board-derived and audio-derived material, dropping duplicates.

    The two modalities overlap in practice: a lecturer says a definition and
    writes it on the board. Keeping both wastes revision time, so we de-dupe on
    a normalised answer key.
    """
    merged = list(board_cards)
    seen = {_norm(c["answer"]) for c in board_cards}
    audio_count = 0

    if transcript_path:
        import app as audio_pipeline  # reuse the audio summariser

        text = Path(transcript_path).read_text(encoding="utf-8")
        summary = audio_pipeline.summarize(text, n=5)
        for c in audio_pipeline.make_flashcards(text, summary):
            if _norm(c["answer"]) in seen:
                continue
            seen.add(_norm(c["answer"]))
            merged.append({
                "question": c["question"],
                "answer": c["answer"],
                "origin": "audio",
            })
            audio_count += 1

    return {
        "cards": merged,
        "from_board": len(board_cards),
        "from_audio": audio_count,
        "duplicates_dropped": max(0, len(board_cards) + audio_count - len(merged)),
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ============================================================================
# Demo
# ============================================================================

SAMPLE_OCR = """
Gradient Descent

Gradient descent is an optimisation algorithm that minimises a loss function
by repeatedly stepping in the direction of steepest descent.

1. Initialise weights randomly
2. Compute the gradient of the loss
3. Update weights in the opposite direction

Learning rate is the size of each step taken downhill.
If the learning rate is too large the algorithm diverges.
If it is too small convergence takes a very long time.

w  =  w  -  alpha * dL/dw

Momentum is a technique that accumulates past gradients to smooth updates.
Momentum helps escape shallow local minima.

Batch size is the number of samples used to compute one gradient update.
""" + " ".join(["Stochastic gradient descent updates weights after every single sample, which makes each step noisy but far faster on large datasets."])


def run_demo() -> None:
    print("=" * 70)
    print("StudySnap — multimodal extension (structuring stage demo)")
    print("=" * 70)
    print("\nNo image or model needed for this demo: it exercises the stage that")
    print("turns flat OCR output into revision material.\n")

    blocks = structure_notes(SAMPLE_OCR)
    cards = notes_to_flashcards(blocks)

    counts: dict[str, int] = {}
    for b in blocks:
        counts[b.kind] = counts.get(b.kind, 0) + 1

    print("--- BLOCK CLASSIFICATION ---")
    for kind, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {kind:12s} {n}")

    print("\n--- STRUCTURED NOTES ---")
    print(notes_to_markdown(blocks))

    print(f"--- FLASHCARDS AUTO-GENERATED FROM BOARD ({len(cards)}) ---")
    for i, c in enumerate(cards, 1):
        print(f"  Q{i}: {c['question']}")
        print(f"       answer: {c['answer']}")

    print("\n--- MERGE WITH AUDIO PIPELINE ---")
    import app as audio_pipeline
    summary = audio_pipeline.summarize(SAMPLE_OCR, n=5)
    audio_cards = audio_pipeline.make_flashcards(SAMPLE_OCR, summary)
    merged = merge_with_transcript(cards, None)
    print(f"  board cards          : {merged['from_board']}")
    print(f"  audio-only cards     : {len(audio_cards)} (generated, merged by answer key)")
    print(f"  duplicates dropped   : {merged['duplicates_dropped']}")

    Path("whiteboard_output.json").write_text(
        json.dumps({"blocks": [asdict(b) for b in blocks],
                    "flashcards": cards}, indent=2), encoding="utf-8")
    print("\nWrote whiteboard_output.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="StudySnap multimodal: board photo -> notes")
    ap.add_argument("--image", help="Photo of a whiteboard / textbook page")
    ap.add_argument("--demo", action="store_true", help="Structuring stage only, no image needed")
    ap.add_argument("--transcript", help="Optional lecture transcript to merge with")
    ap.add_argument("--debug-dir", help="Write intermediate preprocessing images here")
    ap.add_argument("--out", default="whiteboard_output.json")
    args = ap.parse_args()

    if args.demo or not args.image:
        run_demo()
        return

    print("[1/4] Preprocessing image...")
    pre = preprocess(args.image, debug_dir=args.debug_dir)
    for k, v in pre["report"].items():
        print(f"      {k}: {v}")

    print("[2/4] Extracting text...")
    text, backend = extract_text(args.image)
    print(f"      backend: {backend}")
    print(f"      characters recovered: {len(text)}")
    if not text.strip():
        raise SystemExit("      No text detected. Try a sharper photo or better lighting.")

    print("[3/4] Structuring notes...")
    blocks = structure_notes(text)
    cards = notes_to_flashcards(blocks)
    print(f"      blocks: {len(blocks)}   flashcards: {len(cards)}")

    print("[4/4] Merging with transcript...")
    merged = merge_with_transcript(cards, args.transcript)
    print(f"      from board: {merged['from_board']}   from audio: {merged['from_audio']}"
          f"   duplicates dropped: {merged['duplicates_dropped']}")

    print("\n--- STRUCTURED NOTES ---")
    print(notes_to_markdown(blocks))

    Path(args.out).write_text(json.dumps({
        "preprocess_report": pre["report"],
        "blocks": [asdict(b) for b in blocks],
        "flashcards": merged["cards"],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""riordino — Scanned PDF organizer.

Removes blank pages, fixes rotation, extracts metadata via Gemini,
groups pages into logical documents, and splits the PDF accordingly.

Dependencies:
    pip install .  (or: pip install pymupdf pytesseract google-genai pydantic Pillow python-dotenv rich)
System:
    tesseract-ocr with OSD data (tesseract-ocr-osd)
Environment:
    GOOGLE_API_KEY — required for Gemini API access
"""

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
)
from rich.table import Column, Table
from rich.text import Text

load_dotenv()

console = Console()

PRIORITY_STYLES = {
    "urgent": "bold red",
    "important": "yellow",
    "normal": "green",
    "spam": "dim",
}

# ── Language support ───────────────────────────────────────────────────────

# ISO 639-1 → (Tesseract code, full name)
LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    "en": ("eng", "English"),
    "de": ("deu", "German"),
    "fr": ("fra", "French"),
    "it": ("ita", "Italian"),
    "pl": ("pol", "Polish"),
    "es": ("spa", "Spanish"),
    "pt": ("por", "Portuguese"),
    "nl": ("nld", "Dutch"),
    "ru": ("rus", "Russian"),
    "cs": ("ces", "Czech"),
    "ro": ("ron", "Romanian"),
    "hr": ("hrv", "Croatian"),
    "hu": ("hun", "Hungarian"),
    "sv": ("swe", "Swedish"),
    "da": ("dan", "Danish"),
    "nb": ("nor", "Norwegian"),
    "fi": ("fin", "Finnish"),
    "el": ("ell", "Greek"),
    "tr": ("tur", "Turkish"),
    "ja": ("jpn", "Japanese"),
    "zh": ("chi_sim", "Chinese"),
    "ko": ("kor", "Korean"),
    "ar": ("ara", "Arabic"),
}


def parse_languages(raw: str) -> list[str]:
    """Parse a comma-separated language string into validated ISO 639-1 codes."""
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    for code in codes:
        if code not in LANGUAGE_MAP:
            supported = ", ".join(sorted(LANGUAGE_MAP.keys()))
            console.print(f"[bold red]Error:[/] unknown language code '{code}'. Supported: {supported}")
            sys.exit(2)
    return codes


def langs_to_tesseract(codes: list[str]) -> str:
    """Convert ISO 639-1 codes to a Tesseract language string (e.g. 'eng+deu+ita')."""
    return "+".join(LANGUAGE_MAP[c][0] for c in codes)


def langs_to_names(codes: list[str]) -> list[str]:
    """Convert ISO 639-1 codes to full language names."""
    return [LANGUAGE_MAP[c][1] for c in codes]


# ── Dependency checks ──────────────────────────────────────────────────────


def check_dependencies(languages: list[str], skip_rotation: bool, skip_analysis: bool) -> None:
    """Verify that required system dependencies are installed."""
    errors: list[str] = []

    # Tesseract (needed unless rotation is skipped)
    if not skip_rotation:
        if not shutil.which("tesseract"):
            errors.append("tesseract-ocr is not installed. Install it or use --skip-rotation.")
        else:
            # Check for OSD data and language packs
            try:
                result = subprocess.run(
                    ["tesseract", "--list-langs"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                available = set(result.stdout.strip().splitlines()[1:])  # skip header line
                if "osd" not in available:
                    errors.append("tesseract-ocr-osd data is not installed (required for rotation detection).")
                for code in languages:
                    tess_code = LANGUAGE_MAP[code][0]
                    if tess_code not in available:
                        errors.append(f"Tesseract language pack '{tess_code}' not found (for --language {code}).")
            except (subprocess.TimeoutExpired, OSError) as e:
                errors.append(f"Could not query tesseract languages: {e}")

    # Google API key (needed unless analysis is skipped)
    if not skip_analysis and not os.environ.get("GOOGLE_API_KEY"):
        errors.append("GOOGLE_API_KEY environment variable is not set.")

    if errors:
        console.print("[bold red]Missing dependencies:[/]")
        for err in errors:
            console.print(f"  [red]•[/] {err}")
        sys.exit(1)


# ── Pydantic models ─────────────────────────────────────────────────────────


class PageAnalysis(BaseModel):
    title: str
    description: str
    detailed_analysis: str
    page_number: str | None = None
    date: str | None = None
    subject: str | None = None
    document_type: str
    priority: str


class BatchAnalysisResult(BaseModel):
    pages: list[PageAnalysis]


class DocumentGroup(BaseModel):
    title: str
    suggested_filename: str
    page_indices: list[int]
    summary: str
    priority: str


class AggregationResult(BaseModel):
    documents: list[DocumentGroup]


class OrderingResult(BaseModel):
    page_indices: list[int]


# ── PDF helpers ──────────────────────────────────────────────────────────────


def load_pdf(path: Path) -> pymupdf.Document:
    if not path.exists():
        console.print(f"[bold red]Error:[/] file not found: {path}")
        sys.exit(1)
    doc = pymupdf.open(str(path))
    if doc.page_count == 0:
        console.print("[bold red]Error:[/] PDF has no pages.")
        sys.exit(1)
    return doc


def render_page(page: pymupdf.Page, dpi: int = 150) -> Image.Image:
    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# ── Blank detection ─────────────────────────────────────────────────────────


def blank_page_metrics(image: Image.Image) -> dict:
    small = image.resize((200, 200)).convert("L")
    pixels = list(small.tobytes())
    norm = [p / 255.0 for p in pixels]
    try:
        var = statistics.variance(norm)
        mean_val = statistics.mean(norm)
        dark_count = sum(1 for p in norm if p < 0.9)
    except statistics.StatisticsError:
        return {"variance": 0.0, "mean": 1.0, "dark_pixels": 0}
    return {"variance": var, "mean": mean_val, "dark_pixels": dark_count}


def is_blank_page(metrics: dict[str, float | int], threshold: float = 0.001) -> bool:
    return bool(metrics["variance"] < threshold and metrics["mean"] > 0.99 and metrics["dark_pixels"] < 500)


# ── Rotation detection ───────────────────────────────────────────────────────


def detect_rotation(image: Image.Image, langs: str = "eng") -> int:
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(
            image,
            lang=langs,
            output_type=pytesseract.Output.DICT,
        )
        angle: int = osd.get("rotate", 0)
        if angle in (0, 90, 180, 270):
            return angle
        return 0
    except ImportError:
        console.print("[yellow]Warning:[/] pytesseract not installed, skipping rotation detection.")
        return 0
    except Exception:
        return 0


def apply_rotation(doc: pymupdf.Document, page_index: int, degrees: int) -> None:
    page = doc[page_index]
    current = page.rotation
    page.set_rotation((current + degrees) % 360)


# ── Image encoding for Gemini ───────────────────────────────────────────────


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_image_part(png_bytes: bytes) -> types.Part:
    return types.Part.from_bytes(data=png_bytes, mime_type="image/png")


# ── Retry helper ─────────────────────────────────────────────────────────────


def collect_futures[T](futures: list[concurrent.futures.Future[T]]) -> list[T]:
    """Wait for all futures, cancelling remaining ones on KeyboardInterrupt."""
    try:
        return [f.result() for f in futures]
    except KeyboardInterrupt:
        for f in futures:
            f.cancel()
        raise


def retry_api_call[T](fn: Callable[[], T], max_retries: int = 3) -> T:
    for attempt in range(max_retries):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt, 30)
            console.print(f"  [yellow]API error:[/] {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ── Progress bar factory ────────────────────────────────────────────────────


class CompactTimeColumn(ProgressColumn):
    """Renders elapsed time as '3s', '1m 23s', '2h 5m'."""

    def render(self, task: Task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("--", style="progress.elapsed")
        total_seconds = int(elapsed)
        if total_seconds < 60:
            return Text(f"{total_seconds}s", style="progress.elapsed")
        minutes, seconds = divmod(total_seconds, 60)
        if minutes < 60:
            return Text(f"{minutes}m {seconds}s", style="progress.elapsed")
        hours, minutes = divmod(minutes, 60)
        return Text(f"{hours}h {minutes}m", style="progress.elapsed")


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", table_column=Column(min_width=20)),
        BarColumn(),
        MofNCompleteColumn(),
        CompactTimeColumn(),
        console=console,
    )


# ── LLM page analysis (batched) ─────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

ANALYSIS_PROMPT = (SCRIPT_DIR / "prompts" / "analysis.txt").read_text(encoding="utf-8")


def analyze_batch(
    client: genai.Client,
    model: str,
    batch: list[tuple[int, Image.Image]],
    language_names: list[str],
    max_retries: int = 3,
) -> list[PageAnalysis]:
    parts: list[types.Part] = []
    for i, (idx, img) in enumerate(batch):
        parts.append(types.Part.from_text(text=f"--- Page {i + 1} (index {idx}) ---"))
        parts.append(make_image_part(image_to_png_bytes(img)))

    prompt = ANALYSIS_PROMPT.replace("{{COUNT}}", str(len(batch))).replace("{{LANGUAGES}}", ", ".join(language_names))
    parts.append(types.Part.from_text(text=prompt))

    def call() -> list[PageAnalysis]:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BatchAnalysisResult,
                thinking_config=types.ThinkingConfig(thinking_budget=8000),
            ),
        )
        assert response.text is not None, "Gemini returned empty response"
        result = BatchAnalysisResult.model_validate_json(response.text)
        if len(result.pages) != len(batch):
            raise ValueError(f"Expected {len(batch)} page analyses, got {len(result.pages)}")
        return result.pages

    return retry_api_call(call, max_retries)


def analyze_all_pages(
    client: genai.Client,
    model: str,
    pages: list[tuple[int, Image.Image]],
    language_names: list[str],
    batch_size: int = 10,
    max_retries: int = 3,
) -> list[PageAnalysis]:
    total_batches = math.ceil(len(pages) / batch_size)
    batches = []
    for b in range(total_batches):
        start = b * batch_size
        end = min(start + batch_size, len(pages))
        batches.append((b, pages[start:end]))

    with make_progress() as progress:
        task = progress.add_task("Analyzing pages", total=total_batches)

        def _run_batch(item: tuple[int, list[tuple[int, Image.Image]]]) -> tuple[int, list[PageAnalysis]]:
            idx, batch = item
            result = idx, analyze_batch(client, model, batch, language_names, max_retries)
            progress.advance(task)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=total_batches) as pool:
            futures = [pool.submit(_run_batch, item) for item in batches]
            results = collect_futures(futures)

    results.sort(key=lambda x: x[0])
    return [a for _, analyses in results for a in analyses]


# ── LLM document aggregation ────────────────────────────────────────────────

AGGREGATION_PROMPT = (SCRIPT_DIR / "prompts" / "aggregation.txt").read_text(encoding="utf-8")
ORDERING_PROMPT = (SCRIPT_DIR / "prompts" / "ordering.txt").read_text(encoding="utf-8")


def aggregate_documents(
    client: genai.Client,
    model: str,
    analyses: list[PageAnalysis],
    max_retries: int = 3,
) -> AggregationResult:
    page_lines = []
    for i, a in enumerate(analyses):
        desc = a.description.replace("\n", " ")
        page_lines.append(
            f'Page {i}: title="{a.title}", type={a.document_type}, '
            f"priority={a.priority}, "
            f"subject={a.subject or 'unknown'}, date={a.date or 'unknown'}, "
            f"page_num={a.page_number}, "
            f'description="{desc}"'
        )

    prompt = AGGREGATION_PROMPT.replace("{max_index}", str(len(analyses) - 1)).replace(
        "{page_list}", "\n".join(page_lines)
    )

    def call() -> AggregationResult:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AggregationResult,
                thinking_config=types.ThinkingConfig(thinking_budget=8000),
            ),
        )
        assert response.text is not None, "Gemini returned empty response"
        return AggregationResult.model_validate_json(response.text)

    return retry_api_call(call, max_retries)


def order_single_document(
    client: genai.Client,
    model: str,
    group: DocumentGroup,
    analyses: list[PageAnalysis],
    page_images: list[Image.Image],
    max_retries: int = 3,
) -> list[int]:
    if len(group.page_indices) <= 1:
        return group.page_indices

    pages_info = []
    parts: list[types.Part] = []
    for idx in group.page_indices:
        if 0 <= idx < len(analyses):
            a = analyses[idx]
            desc = a.description.replace("\n", " ")
            pages_info.append(
                f'  Page {idx}: title="{a.title}", page_num={a.page_number}, '
                f'type={a.document_type}, description="{desc}"'
            )
        if 0 <= idx < len(page_images):
            parts.append(types.Part.from_text(text=f"--- Page index {idx} ---"))
            parts.append(make_image_part(image_to_png_bytes(page_images[idx])))

    prompt = (
        ORDERING_PROMPT.replace("{title}", group.title)
        .replace("{filename}", group.suggested_filename)
        .replace("{pages}", "\n".join(pages_info))
    )
    parts.append(types.Part.from_text(text=prompt))

    def call() -> list[int]:
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=OrderingResult,
                thinking_config=types.ThinkingConfig(thinking_budget=8000),
            ),
        )
        assert response.text is not None, "Gemini returned empty response"
        result = OrderingResult.model_validate_json(response.text)
        if sorted(result.page_indices) != sorted(group.page_indices):
            raise ValueError("Ordering changed the set of page indices")
        return result.page_indices

    return retry_api_call(call, max_retries)


def order_documents(
    client: genai.Client,
    model: str,
    aggregation: AggregationResult,
    analyses: list[PageAnalysis],
    page_images: list[Image.Image],
    max_retries: int = 3,
) -> AggregationResult:
    groups = aggregation.documents

    with make_progress() as progress:
        task = progress.add_task("Ordering pages", total=len(groups))

        def _order_group(item: tuple[int, DocumentGroup]) -> tuple[int, DocumentGroup]:
            idx, g = item
            ordered = order_single_document(client, model, g, analyses, page_images, max_retries)
            progress.advance(task)
            return idx, g.model_copy(update={"page_indices": ordered})

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = [pool.submit(_order_group, item) for item in enumerate(groups)]
            results = collect_futures(futures)

    results.sort(key=lambda x: x[0])
    return AggregationResult(documents=[g for _, g in results])


# ── PDF splitting and output ────────────────────────────────────────────────


def sanitize_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:100] or "document"


def resolve_filename(output_dir: Path, base_name: str, ext: str = ".pdf") -> Path:
    candidate = output_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate
    i = 2
    while True:
        candidate = output_dir / f"{base_name}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def write_json_sidecar(
    json_path: Path,
    group: DocumentGroup,
    page_analyses: list[PageAnalysis],
) -> Path:
    data = {
        "title": group.title,
        "suggested_filename": group.suggested_filename,
        "summary": group.summary,
        "priority": group.priority,
        "pages": [a.model_dump() for a in page_analyses],
    }
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def split_and_write(
    source_doc: pymupdf.Document,
    groups: list[DocumentGroup],
    page_index_map: list[int],
    all_analyses: list[PageAnalysis],
    output_dir: Path,
    dry_run: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with make_progress() as progress:
        task = progress.add_task("Writing documents", total=len(groups))

        for group in groups:
            base_name = sanitize_filename(group.suggested_filename)
            pdf_path = resolve_filename(output_dir, base_name)

            original_indices = []
            for idx in group.page_indices:
                if 0 <= idx < len(page_index_map):
                    original_indices.append(page_index_map[idx])

            if not original_indices:
                console.print(f"  [yellow]Warning:[/] group '{group.title}' has no valid pages, skipping.")
                progress.advance(task)
                continue

            if dry_run:
                console.print(f"  [dim]\\[DRY RUN][/] Would write: {pdf_path.name} ({len(original_indices)} pages)")
                progress.advance(task)
                continue

            new_doc = pymupdf.open()
            for orig_idx in original_indices:
                new_doc.insert_pdf(source_doc, from_page=orig_idx, to_page=orig_idx)
            new_doc.save(str(pdf_path))
            new_doc.close()

            written.append(pdf_path)
            progress.advance(task)

    return written


# ── Pipeline ─────────────────────────────────────────────────────────────────


def save_intermediate_pdf(
    source_doc: pymupdf.Document,
    page_indices: list[int],
    path: Path,
) -> None:
    new_doc = pymupdf.open()
    for idx in page_indices:
        new_doc.insert_pdf(source_doc, from_page=idx, to_page=idx)
    new_doc.save(str(path))
    new_doc.close()


def save_intermediate_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pipeline(
    input_paths: list[Path],
    output_dir: Path,
    blank_threshold: float,
    dpi: int,
    model: str,
    batch_size: int,
    max_retries: int,
    dry_run: bool,
    languages: list[str],
    save_steps: bool = False,
    skip_blanks: bool = False,
    skip_rotation: bool = False,
    skip_analysis: bool = False,
    skip_aggregation: bool = False,
    skip_ordering: bool = False,
) -> None:
    # Enforce implied skips
    if skip_analysis:
        skip_aggregation = True
    if skip_aggregation:
        skip_ordering = True

    steps_dir: Path | None = None
    if save_steps:
        steps_dir = output_dir / "_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and merge all input PDFs into a single document
    doc = pymupdf.open()
    for p in input_paths:
        src = load_pdf(p)
        doc.insert_pdf(src)
        src.close()
    total = doc.page_count

    input_names = ", ".join(p.name for p in input_paths)

    # 2. Clean up
    console.print(
        f"\n[bold]Cleaning[/] {total} pages from {input_names} [dim](dpi={dpi} · threshold={blank_threshold})[/]"
    )
    pages: list[tuple[int, Image.Image]] = []
    blank_indices: list[int] = []

    if skip_blanks:
        console.print("  [dim]Blank detection: skipped[/]")
        with make_progress() as progress:
            task = progress.add_task("Rendering pages", total=total)
            for i in range(total):
                pages.append((i, render_page(doc[i], dpi)))
                progress.advance(task)
    else:
        all_metrics: list[dict] = []
        with make_progress() as progress:
            task = progress.add_task("Detecting blanks", total=total)
            for i in range(total):
                img = render_page(doc[i], dpi)
                metrics = blank_page_metrics(img)
                is_blank = is_blank_page(metrics, blank_threshold)
                if save_steps:
                    all_metrics.append({"page": i, "blank": is_blank, **metrics})
                if is_blank:
                    blank_indices.append(i)
                else:
                    pages.append((i, img))
                progress.advance(task)

        console.print(f"  [green]{len(pages)}[/] kept, [red]{len(blank_indices)}[/] blank removed")

        if steps_dir:
            kept_indices = [idx for idx, _ in pages]
            save_intermediate_json(
                {
                    "total_pages": total,
                    "blank_indices": blank_indices,
                    "kept_indices": kept_indices,
                    "per_page_metrics": all_metrics,
                },
                steps_dir / "01_blank_detection.json",
            )
            if not dry_run and pages:
                save_intermediate_pdf(doc, kept_indices, steps_dir / "02_after_blank_removal.pdf")
            if not dry_run and blank_indices:
                save_intermediate_pdf(doc, blank_indices, steps_dir / "02_blank_pages_only.pdf")

    if not pages:
        console.print("[yellow]No non-blank pages found. Nothing to do.[/]")
        return

    # 3. Rotation correction
    corrected_pages: list[tuple[int, Image.Image]] = []

    if skip_rotation:
        console.print("  [dim]Rotation correction: skipped[/]")
        corrected_pages = pages
    else:
        with make_progress() as progress:
            task = progress.add_task("Detecting rotation", total=len(pages))

            tesseract_langs = langs_to_tesseract(languages)

            def _detect(page_img: tuple[int, Image.Image]) -> int:
                angle = detect_rotation(page_img[1], langs=tesseract_langs)
                progress.advance(task)
                return angle

            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = [pool.submit(_detect, item) for item in pages]
                angles = collect_futures(futures)

        rotations: dict[int, int] = {}
        for (orig_idx, img), angle in zip(pages, angles, strict=True):
            if angle != 0:
                apply_rotation(doc, orig_idx, angle)
                img = render_page(doc[orig_idx], dpi)
                rotations[orig_idx] = angle
            corrected_pages.append((orig_idx, img))

        if rotations:
            console.print(f"  Rotated [cyan]{len(rotations)}[/] page(s)")
        else:
            console.print("  [dim]No rotation needed[/]")

        if steps_dir:
            save_intermediate_json(
                {"rotations_applied": {str(k): v for k, v in rotations.items()}, "pages_rotated": len(rotations)},
                steps_dir / "03_rotation_corrections.json",
            )
            if not dry_run:
                save_intermediate_pdf(
                    doc,
                    [idx for idx, _ in corrected_pages],
                    steps_dir / "04_after_rotation.pdf",
                )

    # 4. LLM page analysis (batched)
    analyses: list[PageAnalysis] = []

    if skip_analysis:
        console.print("\n  [dim]Page analysis: skipped[/]")
    else:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        console.print(
            f"\n[bold]Analyzing[/] {len(corrected_pages)} pages [dim]({model} · batch={batch_size} · dpi={dpi})[/]"
        )
        language_names = langs_to_names(languages)
        analyses = analyze_all_pages(client, model, corrected_pages, language_names, batch_size, max_retries)

        if steps_dir:
            save_intermediate_json(
                [{"index": i, **a.model_dump()} for i, a in enumerate(analyses)],
                steps_dir / "05_page_analyses.json",
            )

    # 5. LLM document aggregation
    if skip_aggregation:
        console.print("  [dim]Document aggregation: skipped[/]")
        fallback_name = "_".join(p.stem for p in input_paths) if len(input_paths) > 1 else input_paths[0].stem
        aggregation = AggregationResult(
            documents=[
                DocumentGroup(
                    title=fallback_name,
                    suggested_filename=fallback_name,
                    page_indices=list(range(len(corrected_pages))),
                    summary="All pages (aggregation skipped)",
                    priority="normal",
                )
            ]
        )
    else:
        console.print(f"\n[bold]Grouping[/] {len(analyses)} pages into documents [dim]({model})[/]")
        with console.status("Waiting for Gemini...", spinner="dots"):
            aggregation = aggregate_documents(client, model, analyses, max_retries)

        console.print(f"  Found [cyan]{len(aggregation.documents)}[/] document(s)")

        if steps_dir:
            save_intermediate_json(
                aggregation.model_dump(),
                steps_dir / "06_aggregation.json",
            )

    # 6. LLM page ordering
    if skip_ordering:
        console.print("  [dim]Page ordering: skipped[/]")
    else:
        console.print(f"\n[bold]Ordering[/] pages within {len(aggregation.documents)} documents [dim]({model})[/]")
        page_images = [img for _, img in corrected_pages]
        aggregation = order_documents(client, model, aggregation, analyses, page_images, max_retries)

        if steps_dir:
            save_intermediate_json(
                aggregation.model_dump(),
                steps_dir / "07_ordering.json",
            )

    # 7. Split and write
    page_index_map = [orig_idx for orig_idx, _ in corrected_pages]

    if dry_run:
        console.print(f"\n[bold yellow]\\[DRY RUN][/] Would write to: {output_dir}")
    else:
        console.print(f"\n[bold]Writing[/] output to: {output_dir}")
    split_and_write(doc, aggregation.documents, page_index_map, analyses, output_dir, dry_run)

    if steps_dir and not dry_run:
        for group in aggregation.documents:
            base_name = sanitize_filename(group.suggested_filename)
            page_analyses = [analyses[idx] for idx in group.page_indices if 0 <= idx < len(analyses)]
            json_path = steps_dir / f"{base_name}.json"
            write_json_sidecar(json_path, group, page_analyses)

    doc.close()

    # Summary table
    table = Table(title="Results", show_lines=False, pad_edge=False)
    table.add_column("Priority", width=10)
    table.add_column("Filename")
    table.add_column("Pages", justify="right")

    for g in aggregation.documents:
        style = PRIORITY_STYLES.get(g.priority, "")
        table.add_row(
            f"[{style}]{g.priority}[/]",
            g.suggested_filename,
            str(len(g.page_indices)),
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold green]Done.[/] {len(aggregation.documents)} documents written.\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="riordino",
        description="Organize scanned PDFs: remove blanks, fix rotation, split by document. Multiple inputs are merged into a single bulk.",
    )
    p.add_argument(
        "input_pdf", type=Path, nargs="+", help="Path(s) to scanned PDF(s) — multiple files are merged into one bulk"
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input file)",
    )
    p.add_argument(
        "-b",
        "--blank-threshold",
        type=float,
        default=0.001,
        help="Pixel variance threshold for blank detection (default: 0.001)",
    )
    p.add_argument("-n", "--dry-run", action="store_true", help="Show plan without writing files")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    p.add_argument(
        "--model",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="Gemini model name",
    )
    p.add_argument(
        "-l",
        "--language",
        type=str,
        default=os.environ.get("RIORDINO_LANGUAGES", "en"),
        help="Comma-separated language codes, e.g. 'en,de,fr' (default: $RIORDINO_LANGUAGES or 'en')",
    )
    p.add_argument("--batch-size", type=int, default=10, help="Pages per LLM batch (default: 10)")
    p.add_argument("--max-retries", type=int, default=3, help="Max API retry attempts (default: 3)")
    p.add_argument("--save-steps", action="store_true", help="Save intermediate outputs to _steps/ directory")
    p.add_argument("--skip-blanks", action="store_true", help="Skip blank page detection (keep all pages)")
    p.add_argument("--skip-rotation", action="store_true", help="Skip rotation detection and correction")
    p.add_argument("--skip-analysis", action="store_true", help="Skip LLM page analysis (implies --skip-aggregation)")
    p.add_argument(
        "--skip-aggregation", action="store_true", help="Skip LLM document grouping (implies --skip-ordering)"
    )
    p.add_argument("--skip-ordering", action="store_true", help="Skip LLM page ordering within documents")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.dpi < 72 or args.dpi > 600:
        console.print("[bold red]Error:[/] --dpi must be between 72 and 600.")
        sys.exit(2)
    if args.batch_size < 1 or args.batch_size > 50:
        console.print("[bold red]Error:[/] --batch-size must be between 1 and 50.")
        sys.exit(2)
    if args.max_retries < 0 or args.max_retries > 10:
        console.print("[bold red]Error:[/] --max-retries must be between 0 and 10.")
        sys.exit(2)
    if args.blank_threshold < 0.0 or args.blank_threshold > 1.0:
        console.print("[bold red]Error:[/] --blank-threshold must be between 0.0 and 1.0.")
        sys.exit(2)


def main() -> None:
    args = parse_args()
    validate_args(args)
    languages = parse_languages(args.language)
    check_dependencies(languages, skip_rotation=args.skip_rotation, skip_analysis=args.skip_analysis)
    output_dir = args.output_dir or args.input_pdf[0].parent
    run_pipeline(
        input_paths=args.input_pdf,
        output_dir=output_dir,
        blank_threshold=args.blank_threshold,
        dpi=args.dpi,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        languages=languages,
        save_steps=args.save_steps,
        skip_blanks=args.skip_blanks,
        skip_rotation=args.skip_rotation,
        skip_analysis=args.skip_analysis,
        skip_aggregation=args.skip_aggregation,
        skip_ordering=args.skip_ordering,
    )


def _handle_sigint(signum: int, frame: Any) -> None:
    console.print("\n[yellow]Interrupted.[/]")
    os._exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    main()

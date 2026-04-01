#!/usr/bin/env python3
"""riordino — Scanned PDF organizer."""

import concurrent.futures
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Any

import click
import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageStat
from pydantic import BaseModel, ConfigDict, ValidationError
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
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()

console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent

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

ANALYSIS_PROMPT = (SCRIPT_DIR / "prompts" / "analysis.txt").read_text(encoding="utf-8")
AGGREGATION_PROMPT = (SCRIPT_DIR / "prompts" / "aggregation.txt").read_text(encoding="utf-8")
ORDERING_PROMPT = (SCRIPT_DIR / "prompts" / "ordering.txt").read_text(encoding="utf-8")


class RiordinoError(Exception):
    """Base application error."""


class CliError(RiordinoError):
    """User-facing CLI or configuration error."""


class DependencyError(RiordinoError):
    """Missing runtime dependency."""


class ModelResponseError(RiordinoError):
    """LLM returned invalid or inconsistent structured data."""


class Priority(StrEnum):
    URGENT = "urgent"
    IMPORTANT = "important"
    NORMAL = "normal"
    SPAM = "spam"


PRIORITY_STYLES: dict[Priority, str] = {
    Priority.URGENT: "bold red",
    Priority.IMPORTANT: "yellow",
    Priority.NORMAL: "green",
    Priority.SPAM: "dim",
}


class PipelineOptions(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_paths: list[Path]
    output_dir: Path
    blank_threshold: float
    dpi: int
    model: str
    batch_size: int
    max_retries: int
    dry_run: bool
    languages: list[str]
    save_steps: bool = False
    skip_blanks: bool = False
    skip_rotation: bool = False
    skip_analysis: bool = False
    skip_aggregation: bool = False
    skip_ordering: bool = False

    @classmethod
    def from_cli(
        cls,
        *,
        input_paths: list[Path],
        output_dir: Path | None,
        blank_threshold: float,
        dpi: int,
        model: str,
        batch_size: int,
        max_retries: int,
        dry_run: bool,
        language: str,
        save_steps: bool,
        skip_blanks: bool,
        skip_rotation: bool,
        skip_analysis: bool,
        skip_aggregation: bool,
        skip_ordering: bool,
    ) -> PipelineOptions:
        if not input_paths:
            raise CliError("at least one input PDF is required.")
        resolved_skip_aggregation = skip_aggregation or skip_analysis
        resolved_skip_ordering = skip_ordering or resolved_skip_aggregation
        return cls(
            input_paths=input_paths,
            output_dir=output_dir or input_paths[0].parent,
            blank_threshold=blank_threshold,
            dpi=dpi,
            model=model,
            batch_size=batch_size,
            max_retries=max_retries,
            dry_run=dry_run,
            languages=parse_languages(language),
            save_steps=save_steps,
            skip_blanks=skip_blanks,
            skip_rotation=skip_rotation,
            skip_analysis=skip_analysis,
            skip_aggregation=resolved_skip_aggregation,
            skip_ordering=resolved_skip_ordering,
        )


@dataclass(frozen=True)
class BlankPageMetrics:
    variance: float
    mean: float
    dark_pixels: int


@dataclass(frozen=True)
class RenderedPage:
    original_index: int
    image: Image.Image


@dataclass(frozen=True)
class WrittenDocument:
    path: Path
    group: DocumentGroup
    page_analyses: list[PageAnalysis]


@dataclass(frozen=True)
class CleanupResult:
    pages: list[RenderedPage]
    blank_indices: list[int]
    blank_metrics: list[dict[str, float | int | bool]]


@dataclass(frozen=True)
class RotationResult:
    pages: list[RenderedPage]
    rotations: dict[int, int]


@dataclass(frozen=True)
class PreparedPagesState:
    pages: list[RenderedPage]


@dataclass(frozen=True)
class AnalyzedPagesState:
    pages: list[RenderedPage]
    analyses: list[PageAnalysis]


@dataclass(frozen=True)
class GroupedDocumentsState:
    pages: list[RenderedPage]
    analyses: list[PageAnalysis]
    aggregation: AggregationResult


@dataclass(frozen=True)
class OrderedDocumentsState:
    pages: list[RenderedPage]
    analyses: list[PageAnalysis]
    aggregation: AggregationResult


@dataclass
class PipelineContext:
    options: PipelineOptions
    source_doc: pymupdf.Document
    steps_dir: Path | None


class PageAnalysis(BaseModel):
    title: str
    description: str
    detailed_analysis: str
    page_number: str | None = None
    date: str | None = None
    subject: str | None = None
    document_type: str
    priority: Priority


class BatchAnalysisResult(BaseModel):
    pages: list[PageAnalysis]


class DocumentGroup(BaseModel):
    title: str
    suggested_filename: str
    page_indices: list[int]
    summary: str
    priority: Priority


class AggregationResult(BaseModel):
    documents: list[DocumentGroup]


class OrderingResult(BaseModel):
    page_indices: list[int]


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


def parse_languages(raw: str) -> list[str]:
    codes = [code.strip().lower() for code in raw.split(",") if code.strip()]
    if not codes:
        raise CliError("at least one language code is required.")
    unknown = sorted(code for code in codes if code not in LANGUAGE_MAP)
    if unknown:
        supported = ", ".join(sorted(LANGUAGE_MAP))
        raise CliError(f"unknown language code(s): {', '.join(unknown)}. Supported: {supported}")
    return codes


def langs_to_tesseract(codes: list[str]) -> str:
    return "+".join(LANGUAGE_MAP[code][0] for code in codes)


def langs_to_names(codes: list[str]) -> list[str]:
    return [LANGUAGE_MAP[code][1] for code in codes]


def check_dependencies(options: PipelineOptions) -> None:
    errors: list[str] = []
    if not options.skip_rotation:
        if not shutil.which("tesseract"):
            errors.append("tesseract-ocr is not installed. Install it or use --skip-rotation.")
        else:
            try:
                result = subprocess.run(
                    ["tesseract", "--list-langs"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                errors.append(f"Could not query tesseract languages: {exc}")
            else:
                available = set(result.stdout.strip().splitlines()[1:])
                if "osd" not in available:
                    errors.append("tesseract-ocr-osd data is not installed (required for rotation detection).")
                for code in options.languages:
                    tess_code = LANGUAGE_MAP[code][0]
                    if tess_code not in available:
                        errors.append(f"Tesseract language pack '{tess_code}' not found (for --language {code}).")
    if not options.skip_analysis and not os.environ.get("GOOGLE_API_KEY"):
        errors.append("GOOGLE_API_KEY environment variable is not set.")
    if errors:
        raise DependencyError("\n".join(errors))


def load_pdf(path: Path) -> pymupdf.Document:
    if not path.exists():
        raise CliError(f"file not found: {path}")
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise CliError(f"could not open PDF: {path}") from exc
    if doc.page_count == 0:
        doc.close()
        raise CliError(f"PDF has no pages: {path}")
    return doc


def merge_input_pdfs(input_paths: list[Path]) -> pymupdf.Document:
    merged = pymupdf.open()
    try:
        for path in input_paths:
            src = load_pdf(path)
            try:
                merged.insert_pdf(src)
            finally:
                src.close()
    except Exception:
        merged.close()
        raise
    if merged.page_count == 0:
        merged.close()
        raise CliError("no input pages found.")
    return merged


def render_page(page: pymupdf.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def blank_page_metrics(image: Image.Image) -> BlankPageMetrics:
    small = image.resize((200, 200)).convert("L")
    stat = ImageStat.Stat(small)
    mean = stat.mean[0] / 255.0
    variance = stat.var[0] / (255.0**2)
    histogram = small.histogram()
    dark_pixels = sum(histogram[:230])
    return BlankPageMetrics(variance=variance, mean=mean, dark_pixels=dark_pixels)


def is_blank_page(metrics: BlankPageMetrics, threshold: float) -> bool:
    return metrics.variance < threshold and metrics.mean > 0.99 and metrics.dark_pixels < 500


def max_workers_for(items: int, cap: int) -> int:
    return max(1, min(items, cap))


def detect_rotation(image: Image.Image, langs: str) -> int:
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(image, lang=langs, output_type=pytesseract.Output.DICT)
    except ImportError as exc:
        raise DependencyError("pytesseract is not installed. Install it or use --skip-rotation.") from exc
    except Exception:
        return 0
    angle = int(osd.get("rotate", 0))
    return angle if angle in (0, 90, 180, 270) else 0


def apply_rotation(doc: pymupdf.Document, page_index: int, degrees: int) -> None:
    page = doc[page_index]
    page.set_rotation((page.rotation + degrees) % 360)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def make_image_part(image: Image.Image) -> types.Part:
    return types.Part.from_bytes(data=image_to_png_bytes(image), mime_type="image/png")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(data, indent=2, ensure_ascii=False))
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def save_pdf_subset(source_doc: pymupdf.Document, page_indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_doc = pymupdf.open()
    try:
        for idx in page_indices:
            new_doc.insert_pdf(source_doc, from_page=idx, to_page=idx)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        new_doc.save(str(tmp_path))
    finally:
        new_doc.close()
    tmp_path.replace(path)


def write_json_sidecar(json_path: Path, group: DocumentGroup, page_analyses: list[PageAnalysis]) -> None:
    save_json(
        {
            "title": group.title,
            "suggested_filename": group.suggested_filename,
            "summary": group.summary,
            "priority": group.priority.value,
            "pages": [analysis.model_dump() for analysis in page_analyses],
        },
        json_path,
    )


def sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^\w\s-]", "", name)
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean[:100] or "document"


def resolve_filename(output_dir: Path, base_name: str, ext: str = ".pdf") -> Path:
    candidate = output_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        candidate = output_dir / f"{base_name}_{suffix}{ext}"
        if not candidate.exists():
            return candidate
        suffix += 1


def is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, ValidationError, ModelResponseError, ValueError, AssertionError)):
        return False
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return exc.code in {408, 409, 429}
    if isinstance(exc, genai_errors.APIError):
        return exc.code in {408, 409, 429, 500, 502, 503, 504}
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def retry_api_call[T](fn: Callable[[], T], max_retries: int) -> T:
    def _before_sleep(state: Any) -> None:
        exc = state.outcome.exception() if state.outcome else None
        if exc is None:
            return
        wait = state.next_action.sleep if state.next_action else 0
        console.print(f"  [yellow]API error:[/] {exc}. Retrying in {wait:.0f}s...")

    attempts = max_retries + 1
    for attempt in Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception(is_transient_api_error),
        before_sleep=_before_sleep,
        reraise=True,
    ):
        with attempt:
            return fn()
    raise RuntimeError("unreachable")


class GeminiService:
    def __init__(self, client: genai.Client, model: str, max_retries: int):
        self.client = client
        self.model = model
        self.max_retries = max_retries

    def _parse_response[T: BaseModel](self, response: types.GenerateContentResponse, schema: type[T]) -> T:
        if not response.text:
            raise ModelResponseError("Gemini returned empty response.")
        try:
            return schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise ModelResponseError(f"Gemini returned invalid JSON for {schema.__name__}.") from exc

    def _generate_text[T: BaseModel](self, prompt: str, schema: type[T]) -> T:
        def call() -> T:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=8000),
                ),
            )
            return self._parse_response(response, schema)

        return retry_api_call(call, self.max_retries)

    def _generate_multimodal[T: BaseModel](self, parts: list[types.Part], schema: type[T]) -> T:
        contents = types.Content(role="user", parts=parts)

        def call() -> T:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=8000),
                ),
            )
            return self._parse_response(response, schema)

        return retry_api_call(call, self.max_retries)

    def analyze_batch(self, batch: list[RenderedPage], language_names: list[str]) -> list[PageAnalysis]:
        parts: list[types.Part] = []
        for position, page in enumerate(batch, start=1):
            parts.append(types.Part.from_text(text=f"--- Page {position} (index {page.original_index}) ---"))
            parts.append(make_image_part(page.image))
        prompt = ANALYSIS_PROMPT.replace("{{COUNT}}", str(len(batch))).replace(
            "{{LANGUAGES}}", ", ".join(language_names)
        )
        parts.append(types.Part.from_text(text=prompt))
        result = self._generate_multimodal(parts, BatchAnalysisResult)
        if len(result.pages) != len(batch):
            raise ModelResponseError(f"expected {len(batch)} page analyses, got {len(result.pages)}")
        return result.pages

    def aggregate(self, analyses: list[PageAnalysis]) -> AggregationResult:
        page_lines = []
        for index, analysis in enumerate(analyses):
            description = analysis.description.replace("\n", " ")
            page_lines.append(
                f'Page {index}: title="{analysis.title}", type={analysis.document_type}, '
                f"priority={analysis.priority.value}, "
                f"subject={analysis.subject or 'unknown'}, date={analysis.date or 'unknown'}, "
                f"page_num={analysis.page_number}, "
                f'description="{description}"'
            )
        prompt = AGGREGATION_PROMPT.replace("{max_index}", str(len(analyses) - 1)).replace(
            "{page_list}", "\n".join(page_lines)
        )
        return self._generate_text(prompt, AggregationResult)

    def order_group(
        self, group: DocumentGroup, analyses: list[PageAnalysis], page_images: list[Image.Image]
    ) -> list[int]:
        if len(group.page_indices) <= 1:
            return group.page_indices
        parts: list[types.Part] = []
        page_lines: list[str] = []
        for page_index in group.page_indices:
            if 0 <= page_index < len(analyses):
                analysis = analyses[page_index]
                description = analysis.description.replace("\n", " ")
                page_lines.append(
                    f'  Page {page_index}: title="{analysis.title}", page_num={analysis.page_number}, '
                    f'type={analysis.document_type}, description="{description}"'
                )
            if 0 <= page_index < len(page_images):
                parts.append(types.Part.from_text(text=f"--- Page index {page_index} ---"))
                parts.append(make_image_part(page_images[page_index]))
        prompt = (
            ORDERING_PROMPT.replace("{title}", group.title)
            .replace("{filename}", group.suggested_filename)
            .replace("{pages}", "\n".join(page_lines))
        )
        parts.append(types.Part.from_text(text=prompt))
        result = self._generate_multimodal(parts, OrderingResult)
        if sorted(result.page_indices) != sorted(group.page_indices):
            raise ModelResponseError("ordering changed the set of page indices")
        return result.page_indices


def build_context(options: PipelineOptions) -> PipelineContext:
    source_doc = merge_input_pdfs(options.input_paths)
    steps_dir = None
    if options.save_steps:
        steps_dir = options.output_dir / "_steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
    return PipelineContext(options=options, source_doc=source_doc, steps_dir=steps_dir)


def remove_blank_pages(context: PipelineContext) -> CleanupResult:
    doc = context.source_doc
    total = doc.page_count
    pages: list[RenderedPage] = []
    blank_indices: list[int] = []
    blank_metrics: list[dict[str, float | int | bool]] = []
    if context.options.skip_blanks:
        console.print("  [dim]Blank detection: skipped[/]")
        with make_progress() as progress:
            task = progress.add_task("Rendering pages", total=total)
            for index in range(total):
                pages.append(RenderedPage(index, render_page(doc[index], context.options.dpi)))
                progress.advance(task)
        return CleanupResult(pages=pages, blank_indices=blank_indices, blank_metrics=blank_metrics)

    with make_progress() as progress:
        task = progress.add_task("Detecting blanks", total=total)
        for index in range(total):
            image = render_page(doc[index], context.options.dpi)
            metrics = blank_page_metrics(image)
            blank = is_blank_page(metrics, context.options.blank_threshold)
            if context.options.save_steps:
                blank_metrics.append(
                    {
                        "page": index,
                        "blank": blank,
                        "variance": metrics.variance,
                        "mean": metrics.mean,
                        "dark_pixels": metrics.dark_pixels,
                    }
                )
            if blank:
                blank_indices.append(index)
            else:
                pages.append(RenderedPage(index, image))
            progress.advance(task)
    console.print(f"  [green]{len(pages)}[/] kept, [red]{len(blank_indices)}[/] blank removed")
    if context.steps_dir:
        kept_indices = [page.original_index for page in pages]
        save_json(
            {
                "total_pages": total,
                "blank_indices": blank_indices,
                "kept_indices": kept_indices,
                "per_page_metrics": blank_metrics,
            },
            context.steps_dir / "01_blank_detection.json",
        )
        if not context.options.dry_run and kept_indices:
            save_pdf_subset(doc, kept_indices, context.steps_dir / "02_after_blank_removal.pdf")
        if not context.options.dry_run and blank_indices:
            save_pdf_subset(doc, blank_indices, context.steps_dir / "02_blank_pages_only.pdf")
    return CleanupResult(pages=pages, blank_indices=blank_indices, blank_metrics=blank_metrics)


def correct_rotations(context: PipelineContext, pages: list[RenderedPage]) -> RotationResult:
    if context.options.skip_rotation:
        console.print("  [dim]Rotation correction: skipped[/]")
        return RotationResult(pages=pages, rotations={})

    tesseract_langs = langs_to_tesseract(context.options.languages)
    max_workers = max_workers_for(len(pages), os.cpu_count() or 1)
    rotations: dict[int, int] = {}
    corrected_by_index: dict[int, RenderedPage] = {}

    with make_progress() as progress:
        task = progress.add_task("Detecting rotation", total=len(pages))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(detect_rotation, page.image, tesseract_langs): page for page in pages}
            for future in concurrent.futures.as_completed(future_map):
                page = future_map[future]
                angle = future.result()
                image = page.image
                if angle:
                    apply_rotation(context.source_doc, page.original_index, angle)
                    image = render_page(context.source_doc[page.original_index], context.options.dpi)
                    rotations[page.original_index] = angle
                corrected_by_index[page.original_index] = RenderedPage(page.original_index, image)
                progress.advance(task)

    corrected_pages = [corrected_by_index[page.original_index] for page in pages]
    if rotations:
        console.print(f"  Rotated [cyan]{len(rotations)}[/] page(s)")
    else:
        console.print("  [dim]No rotation needed[/]")
    if context.steps_dir:
        save_json(
            {
                "rotations_applied": {str(index): angle for index, angle in rotations.items()},
                "pages_rotated": len(rotations),
            },
            context.steps_dir / "03_rotation_corrections.json",
        )
        if not context.options.dry_run:
            save_pdf_subset(
                context.source_doc,
                [page.original_index for page in corrected_pages],
                context.steps_dir / "04_after_rotation.pdf",
            )
    return RotationResult(pages=corrected_pages, rotations=rotations)


def analyze_pages(service: GeminiService, context: PipelineContext, pages: list[RenderedPage]) -> list[PageAnalysis]:
    if context.options.skip_analysis:
        console.print("\n  [dim]Page analysis: skipped[/]")
        return []
    console.print(
        f"\n[bold]Analyzing[/] {len(pages)} pages [dim]({context.options.model} · batch={context.options.batch_size} · dpi={context.options.dpi})[/]"
    )
    language_names = langs_to_names(context.options.languages)
    batches = [
        pages[start : start + context.options.batch_size] for start in range(0, len(pages), context.options.batch_size)
    ]
    results: dict[int, list[PageAnalysis]] = {}
    max_workers = max_workers_for(len(batches), 4)
    with make_progress() as progress:
        task = progress.add_task("Analyzing pages", total=len(batches))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(service.analyze_batch, batch, language_names): index for index, batch in enumerate(batches)
            }
            for future in concurrent.futures.as_completed(future_map):
                batch_index = future_map[future]
                results[batch_index] = future.result()
                progress.advance(task)
    analyses = [analysis for index in range(len(batches)) for analysis in results[index]]
    if context.steps_dir:
        save_json(
            [{"index": index, **analysis.model_dump()} for index, analysis in enumerate(analyses)],
            context.steps_dir / "05_page_analyses.json",
        )
    return analyses


def analyze_stage(
    service: GeminiService | None, context: PipelineContext, pages: list[RenderedPage]
) -> PreparedPagesState | AnalyzedPagesState:
    if service is None:
        console.print("\n  [dim]Page analysis: skipped[/]")
        return PreparedPagesState(pages=pages)
    return AnalyzedPagesState(pages=pages, analyses=analyze_pages(service, context, pages))


def aggregate_pages(
    service: GeminiService | None, context: PipelineContext, analyses: list[PageAnalysis], page_count: int
) -> AggregationResult:
    if context.options.skip_aggregation:
        console.print("  [dim]Document aggregation: skipped[/]")
        fallback_name = (
            "_".join(path.stem for path in context.options.input_paths)
            if len(context.options.input_paths) > 1
            else context.options.input_paths[0].stem
        )
        return AggregationResult(
            documents=[
                DocumentGroup(
                    title=fallback_name,
                    suggested_filename=fallback_name,
                    page_indices=list(range(page_count)),
                    summary="All pages (aggregation skipped)",
                    priority=Priority.NORMAL,
                )
            ]
        )
    if service is None:
        raise RuntimeError("Gemini service is required for aggregation.")
    console.print(f"\n[bold]Grouping[/] {len(analyses)} pages into documents [dim]({context.options.model})[/]")
    with console.status("Waiting for Gemini...", spinner="dots"):
        aggregation = service.aggregate(analyses)
    console.print(f"  Found [cyan]{len(aggregation.documents)}[/] document(s)")
    if context.steps_dir:
        save_json(aggregation.model_dump(), context.steps_dir / "06_aggregation.json")
    return aggregation


def aggregate_stage(
    service: GeminiService | None, context: PipelineContext, state: PreparedPagesState | AnalyzedPagesState
) -> GroupedDocumentsState:
    match state:
        case AnalyzedPagesState(pages=pages, analyses=analyses):
            aggregation = aggregate_pages(service, context, analyses, len(pages))
            return GroupedDocumentsState(pages=pages, analyses=analyses, aggregation=aggregation)
        case PreparedPagesState(pages=pages):
            aggregation = aggregate_pages(service, context, [], len(pages))
            return GroupedDocumentsState(pages=pages, analyses=[], aggregation=aggregation)


def order_groups(
    service: GeminiService | None,
    context: PipelineContext,
    aggregation: AggregationResult,
    analyses: list[PageAnalysis],
    pages: list[RenderedPage],
) -> AggregationResult:
    if context.options.skip_ordering:
        console.print("  [dim]Page ordering: skipped[/]")
        return aggregation
    if service is None:
        raise RuntimeError("Gemini service is required for ordering.")
    console.print(
        f"\n[bold]Ordering[/] pages within {len(aggregation.documents)} documents [dim]({context.options.model})[/]"
    )
    page_images = [page.image for page in pages]
    max_workers = max_workers_for(len(aggregation.documents), 4)
    ordered_groups: dict[int, DocumentGroup] = {}
    with make_progress() as progress:
        task = progress.add_task("Ordering pages", total=len(aggregation.documents))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(service.order_group, group, analyses, page_images): index
                for index, group in enumerate(aggregation.documents)
            }
            for future in concurrent.futures.as_completed(future_map):
                index = future_map[future]
                group = aggregation.documents[index]
                ordered_groups[index] = group.model_copy(update={"page_indices": future.result()})
                progress.advance(task)
    ordered = AggregationResult(documents=[ordered_groups[index] for index in range(len(aggregation.documents))])
    if context.steps_dir:
        save_json(ordered.model_dump(), context.steps_dir / "07_ordering.json")
    return ordered


def order_stage(
    service: GeminiService | None, context: PipelineContext, state: GroupedDocumentsState
) -> OrderedDocumentsState:
    aggregation = order_groups(service, context, state.aggregation, state.analyses, state.pages)
    return OrderedDocumentsState(pages=state.pages, analyses=state.analyses, aggregation=aggregation)


def write_outputs(
    context: PipelineContext,
    groups: list[DocumentGroup],
    page_index_map: list[int],
    analyses: list[PageAnalysis],
) -> list[WrittenDocument]:
    context.options.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[WrittenDocument] = []
    with make_progress() as progress:
        task = progress.add_task("Writing documents", total=len(groups))
        for group in groups:
            base_name = sanitize_filename(group.suggested_filename)
            pdf_path = resolve_filename(context.options.output_dir, base_name)
            original_indices = [page_index_map[idx] for idx in group.page_indices if 0 <= idx < len(page_index_map)]
            if not original_indices:
                console.print(f"  [yellow]Warning:[/] group '{group.title}' has no valid pages, skipping.")
                progress.advance(task)
                continue
            if context.options.dry_run:
                console.print(f"  [dim]\\[DRY RUN][/] Would write: {pdf_path.name} ({len(original_indices)} pages)")
                progress.advance(task)
                continue
            save_pdf_subset(context.source_doc, original_indices, pdf_path)
            page_analyses = [analyses[idx] for idx in group.page_indices if 0 <= idx < len(analyses)]
            written.append(WrittenDocument(path=pdf_path, group=group, page_analyses=page_analyses))
            progress.advance(task)
    return written


def print_summary(groups: list[DocumentGroup], written_count: int, dry_run: bool) -> None:
    table = Table(title="Results", show_lines=False, pad_edge=False)
    table.add_column("Priority", width=10)
    table.add_column("Filename")
    table.add_column("Pages", justify="right")
    for group in groups:
        style = PRIORITY_STYLES.get(group.priority, "")
        table.add_row(f"[{style}]{group.priority.value}[/]", group.suggested_filename, str(len(group.page_indices)))
    console.print()
    console.print(table)
    verb = "planned" if dry_run else "written"
    console.print(f"\n[bold green]Done.[/] {written_count if not dry_run else len(groups)} documents {verb}.\n")


def run_pipeline(options: PipelineOptions) -> None:
    context = build_context(options)
    try:
        total = context.source_doc.page_count
        input_names = ", ".join(path.name for path in options.input_paths)
        console.print(
            f"\n[bold]Cleaning[/] {total} pages from {input_names} [dim](dpi={options.dpi} · threshold={options.blank_threshold})[/]"
        )
        cleanup = remove_blank_pages(context)
        if not cleanup.pages:
            console.print("[yellow]No non-blank pages found. Nothing to do.[/]")
            return
        rotation = correct_rotations(context, cleanup.pages)
        service = None
        if not options.skip_analysis:
            service = GeminiService(
                genai.Client(api_key=os.environ["GOOGLE_API_KEY"]), options.model, options.max_retries
            )
        analyzed_state = analyze_stage(service, context, rotation.pages)
        grouped_state = aggregate_stage(service, context, analyzed_state)
        ordered_state = order_stage(service, context, grouped_state)
        page_index_map = [page.original_index for page in ordered_state.pages]
        if options.dry_run:
            console.print(f"\n[bold yellow]\\[DRY RUN][/] Would write to: {options.output_dir}")
        else:
            console.print(f"\n[bold]Writing[/] output to: {options.output_dir}")
        written = write_outputs(context, ordered_state.aggregation.documents, page_index_map, ordered_state.analyses)
        if context.steps_dir and not options.dry_run:
            for item in written:
                write_json_sidecar(context.steps_dir / f"{item.path.stem}.json", item.group, item.page_analyses)
        print_summary(ordered_state.aggregation.documents, len(written), options.dry_run)
    finally:
        context.source_doc.close()


def format_error(exc: RiordinoError) -> str:
    if isinstance(exc, DependencyError):
        lines = exc.args[0].splitlines() if exc.args else []
        return "[bold red]Missing dependencies:[/]\n" + "\n".join(f"  [red]•[/] {line}" for line in lines)
    return f"[bold red]Error:[/] {exc}"


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Organize scanned PDFs: remove blanks, fix rotation, split by document. Multiple inputs are merged into a single bulk.",
)
@click.argument("input_pdf", nargs=-1, type=click.Path(path_type=Path))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: same as input file)",
)
@click.option(
    "-b",
    "--blank-threshold",
    type=click.FloatRange(0.0, 1.0),
    default=0.001,
    show_default=True,
    help="Pixel variance threshold for blank detection.",
)
@click.option("-n", "--dry-run", is_flag=True, help="Show plan without writing files")
@click.option("--dpi", type=click.IntRange(72, 600), default=150, show_default=True, help="Render DPI")
@click.option("--model", type=str, default="gemini-3.1-flash-lite-preview", show_default=True, help="Gemini model name")
@click.option(
    "-l",
    "--language",
    type=str,
    default=lambda: os.environ.get("RIORDINO_LANGUAGES", "en"),
    show_default="$RIORDINO_LANGUAGES or en",
    help="Comma-separated language codes, e.g. 'en,de,fr'.",
)
@click.option("--batch-size", type=click.IntRange(1, 50), default=10, show_default=True, help="Pages per LLM batch")
@click.option("--max-retries", type=click.IntRange(0, 10), default=3, show_default=True, help="Max API retry attempts")
@click.option("--save-steps", is_flag=True, help="Save intermediate outputs to _steps/ directory")
@click.option("--skip-blanks", is_flag=True, help="Skip blank page detection (keep all pages)")
@click.option("--skip-rotation", is_flag=True, help="Skip rotation detection and correction")
@click.option("--skip-analysis", is_flag=True, help="Skip LLM page analysis (implies --skip-aggregation)")
@click.option("--skip-aggregation", is_flag=True, help="Skip LLM document grouping (implies --skip-ordering)")
@click.option("--skip-ordering", is_flag=True, help="Skip LLM page ordering within documents")
def main(
    input_pdf: tuple[Path, ...],
    output_dir: Path | None,
    blank_threshold: float,
    dry_run: bool,
    dpi: int,
    model: str,
    language: str,
    batch_size: int,
    max_retries: int,
    save_steps: bool,
    skip_blanks: bool,
    skip_rotation: bool,
    skip_analysis: bool,
    skip_aggregation: bool,
    skip_ordering: bool,
) -> None:
    try:
        options = PipelineOptions.from_cli(
            input_paths=list(input_pdf),
            output_dir=output_dir,
            blank_threshold=blank_threshold,
            dpi=dpi,
            model=model,
            batch_size=batch_size,
            max_retries=max_retries,
            dry_run=dry_run,
            language=language,
            save_steps=save_steps,
            skip_blanks=skip_blanks,
            skip_rotation=skip_rotation,
            skip_analysis=skip_analysis,
            skip_aggregation=skip_aggregation,
            skip_ordering=skip_ordering,
        )
        check_dependencies(options)
        run_pipeline(options)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        raise SystemExit(130) from None
    except RiordinoError as exc:
        console.print(format_error(exc))
        raise SystemExit(1) from exc


def _handle_sigint(signum: int, frame: Any) -> None:
    del signum, frame
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_sigint)
    main()

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def load_riordino_module():
    for name in [key for key in list(sys.modules) if key.startswith("riordino_test")]:
        sys.modules.pop(name, None)

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None

    class FakeBaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        @classmethod
        def model_validate_json(cls, raw: str):
            return cls(**json.loads(raw))

        def model_dump(self):
            return self.__dict__.copy()

        def model_copy(self, update=None):
            data = self.model_dump()
            data.update(update or {})
            return type(self)(**data)

    class FakeValidationError(Exception):
        pass

    pydantic = types.ModuleType("pydantic")
    pydantic.BaseModel = FakeBaseModel
    pydantic.ConfigDict = lambda **kwargs: kwargs
    pydantic.ValidationError = FakeValidationError

    pymupdf = types.ModuleType("pymupdf")
    pymupdf.Document = type("Document", (), {})
    pymupdf.Page = type("Page", (), {})
    pymupdf.Matrix = lambda x, y: (x, y)
    pymupdf.open = lambda *args, **kwargs: None

    image_module = types.ModuleType("PIL.Image")
    image_module.Image = type("FakeImage", (), {})
    image_module.frombytes = staticmethod(lambda *args, **kwargs: image_module.Image())

    image_stat_module = types.ModuleType("PIL.ImageStat")
    image_stat_module.Stat = lambda image: None

    pil = types.ModuleType("PIL")
    pil.Image = image_module
    pil.ImageStat = image_stat_module

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    google_types = types.ModuleType("google.genai.types")

    class FakePart:
        @staticmethod
        def from_text(text):
            return ("text", text)

        @staticmethod
        def from_bytes(data, mime_type):
            return ("bytes", mime_type, data)

    class FakeContent:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeGenerateContentResponse:
        def __init__(self, text=None):
            self.text = text

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = types.SimpleNamespace(generate_content=lambda **kw: None)

    google_types.Part = FakePart
    google_types.Content = FakeContent
    google_types.GenerateContentConfig = FakeGenerateContentConfig
    google_types.GenerateContentResponse = FakeGenerateContentResponse
    google_types.ThinkingConfig = FakeThinkingConfig
    genai.Client = FakeClient
    genai.types = google_types
    google.genai = genai

    rich_console = types.ModuleType("rich.console")
    rich_progress = types.ModuleType("rich.progress")
    rich_table = types.ModuleType("rich.table")
    rich_text = types.ModuleType("rich.text")
    click = types.ModuleType("click")

    class FakeStatus:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeConsole:
        def print(self, *args, **kwargs):
            return None

        def status(self, *args, **kwargs):
            return FakeStatus()

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            self.total = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_task(self, description, total):
            self.total = total
            return 1

        def advance(self, task):
            return None

    class FakeProgressColumn:
        pass

    class FakeTask:
        finished = False
        finished_time = None
        elapsed = None

    class FakeColumn:
        def __init__(self, *args, **kwargs):
            pass

    class FakeTable:
        def __init__(self, *args, **kwargs):
            pass

        def add_column(self, *args, **kwargs):
            pass

        def add_row(self, *args, **kwargs):
            pass

    class FakeText(str):
        def __new__(cls, text, style=None):
            return str.__new__(cls, text)

    rich_console.Console = FakeConsole
    rich_progress.Progress = FakeProgress
    rich_progress.ProgressColumn = FakeProgressColumn
    rich_progress.Task = FakeTask
    rich_progress.SpinnerColumn = lambda *args, **kwargs: None
    rich_progress.TextColumn = lambda *args, **kwargs: None
    rich_progress.BarColumn = lambda *args, **kwargs: None
    rich_progress.MofNCompleteColumn = lambda *args, **kwargs: None
    rich_table.Column = FakeColumn
    rich_table.Table = FakeTable
    rich_text.Text = FakeText

    tenacity = types.ModuleType("tenacity")

    class FakeAttempt:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeRetrying:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            yield FakeAttempt()

    tenacity.Retrying = FakeRetrying
    tenacity.retry_if_exception = lambda predicate: predicate
    tenacity.stop_after_attempt = lambda attempts: attempts
    tenacity.wait_exponential = lambda **kwargs: kwargs

    def identity_decorator(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    click.command = identity_decorator
    click.argument = identity_decorator
    click.option = identity_decorator
    click.IntRange = lambda *args, **kwargs: None
    click.FloatRange = lambda *args, **kwargs: None
    click.Path = lambda *args, **kwargs: None

    sys.modules.update(
        {
            "click": click,
            "dotenv": dotenv,
            "google": google,
            "google.genai": genai,
            "google.genai.types": google_types,
            "PIL": pil,
            "PIL.Image": image_module,
            "PIL.ImageStat": image_stat_module,
            "pydantic": pydantic,
            "pymupdf": pymupdf,
            "rich.console": rich_console,
            "rich.progress": rich_progress,
            "rich.table": rich_table,
            "rich.text": rich_text,
            "tenacity": tenacity,
        }
    )

    spec = importlib.util.spec_from_file_location("riordino_test", Path(__file__).resolve().parents[1] / "riordino.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def riordino():
    return load_riordino_module()


def test_parse_languages_rejects_unknown_codes(riordino):
    with pytest.raises(riordino.CliError):
        riordino.parse_languages("en,xx")


def test_pipeline_options_from_cli_applies_implied_skips(riordino, tmp_path):
    options = riordino.PipelineOptions.from_cli(
        input_paths=[tmp_path / "scan.pdf"],
        output_dir=None,
        blank_threshold=0.001,
        dpi=150,
        model="model",
        batch_size=10,
        max_retries=3,
        dry_run=False,
        language="en,it",
        save_steps=False,
        skip_blanks=False,
        skip_rotation=False,
        skip_analysis=True,
        skip_aggregation=False,
        skip_ordering=False,
    )
    assert options.output_dir == tmp_path
    assert options.skip_analysis is True
    assert options.skip_aggregation is True
    assert options.skip_ordering is True
    assert options.languages == ["en", "it"]


def test_resolve_filename_adds_numeric_suffix(riordino, tmp_path):
    (tmp_path / "doc.pdf").write_text("existing", encoding="utf-8")
    (tmp_path / "doc_2.pdf").write_text("existing", encoding="utf-8")
    assert riordino.resolve_filename(tmp_path, "doc").name == "doc_3.pdf"


def test_is_transient_api_error_distinguishes_validation_from_retriable(riordino):
    class RateLimitedError(Exception):
        status_code = 429

    assert riordino.is_transient_api_error(RateLimitedError()) is True
    assert riordino.is_transient_api_error(riordino.ModelResponseError("bad schema")) is False
    assert riordino.is_transient_api_error(ValueError("bad value")) is False


def test_write_outputs_does_not_write_sidecar_without_steps(riordino, tmp_path, monkeypatch):
    written_pdf_paths = []

    def fake_save_pdf_subset(source_doc, page_indices, path):
        del source_doc, page_indices
        written_pdf_paths.append(path)
        path.write_text("pdf", encoding="utf-8")

    monkeypatch.setattr(riordino, "save_pdf_subset", fake_save_pdf_subset)

    options = riordino.PipelineOptions(
        input_paths=[tmp_path / "scan.pdf"],
        output_dir=tmp_path,
        blank_threshold=0.001,
        dpi=150,
        model="model",
        batch_size=10,
        max_retries=3,
        dry_run=False,
        languages=["en"],
    )
    context = riordino.PipelineContext(options=options, source_doc=object(), steps_dir=None)
    group = riordino.DocumentGroup(
        title="Title",
        suggested_filename="Title",
        page_indices=[0],
        summary="Summary",
        priority="normal",
    )
    analysis = riordino.PageAnalysis(
        title="Page title",
        description="Page description",
        detailed_analysis="Detailed",
        document_type="letter",
        priority="normal",
    )

    written = riordino.write_outputs(context, [group], [7], [analysis])

    assert [path.name for path in written_pdf_paths] == ["Title.pdf"]
    assert written[0].path == tmp_path / "Title.pdf"
    assert not (tmp_path / "Title.json").exists()


def test_write_json_sidecar_targets_steps_directory(riordino, tmp_path):
    steps_dir = tmp_path / "_steps"
    steps_dir.mkdir()
    group = riordino.DocumentGroup(
        title="Title",
        suggested_filename="Title",
        page_indices=[0],
        summary="Summary",
        priority="normal",
    )
    analysis = riordino.PageAnalysis(
        title="Page title",
        description="Page description",
        detailed_analysis="Detailed",
        document_type="letter",
        priority="normal",
    )

    riordino.write_json_sidecar(steps_dir / "Title.json", group, [analysis])

    assert (steps_dir / "Title.json").exists()

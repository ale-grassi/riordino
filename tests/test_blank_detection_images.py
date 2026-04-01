import importlib.util
import sys
import types
from pathlib import Path

from PIL import Image, ImageDraw


def load_local_riordino_module():
    click = types.ModuleType("click")
    dotenv = types.ModuleType("dotenv")
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai_errors = types.ModuleType("google.genai.errors")
    google_types = types.ModuleType("google.genai.types")
    pymupdf = types.ModuleType("pymupdf")
    rich_console = types.ModuleType("rich.console")
    rich_progress = types.ModuleType("rich.progress")
    rich_table = types.ModuleType("rich.table")
    rich_text = types.ModuleType("rich.text")
    tenacity = types.ModuleType("tenacity")

    dotenv.load_dotenv = lambda: None
    click.command = lambda *args, **kwargs: lambda fn: fn
    click.argument = lambda *args, **kwargs: lambda fn: fn
    click.option = lambda *args, **kwargs: lambda fn: fn
    click.IntRange = lambda *args, **kwargs: None
    click.FloatRange = lambda *args, **kwargs: None
    click.Path = lambda *args, **kwargs: None

    pymupdf.Document = type("Document", (), {})
    pymupdf.Page = type("Page", (), {})
    pymupdf.Matrix = lambda x, y: (x, y)
    pymupdf.open = lambda *args, **kwargs: None

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

    class FakeAPIError(Exception):
        def __init__(self, code, response_json=None, response=None):
            super().__init__(code)
            self.code = code
            self.response_json = response_json
            self.response = response

    class FakeClientError(FakeAPIError):
        pass

    class FakeServerError(FakeAPIError):
        pass

    google_types.Part = FakePart
    google_types.Content = FakeContent
    google_types.GenerateContentConfig = FakeGenerateContentConfig
    google_types.ThinkingConfig = FakeThinkingConfig
    google_types.GenerateContentResponse = FakeGenerateContentResponse
    genai.Client = FakeClient
    genai.types = google_types
    genai.errors = genai_errors
    genai_errors.APIError = FakeAPIError
    genai_errors.ClientError = FakeClientError
    genai_errors.ServerError = FakeServerError
    google.genai = genai

    class FakeConsole:
        def print(self, *args, **kwargs):
            return None

        def status(self, *args, **kwargs):
            return types.SimpleNamespace(__enter__=lambda self: self, __exit__=lambda self, exc_type, exc, tb: False)

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            pass

    rich_console.Console = FakeConsole
    rich_progress.Progress = FakeProgress
    rich_progress.ProgressColumn = type("FakeProgressColumn", (), {})
    rich_progress.Task = type("FakeTask", (), {"finished": False, "finished_time": None, "elapsed": None})
    rich_progress.SpinnerColumn = lambda *args, **kwargs: None
    rich_progress.TextColumn = lambda *args, **kwargs: None
    rich_progress.BarColumn = lambda *args, **kwargs: None
    rich_progress.MofNCompleteColumn = lambda *args, **kwargs: None
    rich_table.Column = lambda *args, **kwargs: None
    rich_table.Table = type("FakeTable", (), {})
    rich_text.Text = str

    class FakeRetrying:
        def __init__(self, *args, **kwargs):
            pass

        def __iter__(self):
            return iter(())

    tenacity.Retrying = FakeRetrying
    tenacity.retry_if_exception = lambda predicate: predicate
    tenacity.stop_after_attempt = lambda attempts: attempts
    tenacity.wait_exponential = lambda **kwargs: kwargs

    sys.modules.update(
        {
            "click": click,
            "dotenv": dotenv,
            "google": google,
            "google.genai": genai,
            "google.genai.errors": genai_errors,
            "google.genai.types": google_types,
            "pymupdf": pymupdf,
            "rich.console": rich_console,
            "rich.progress": rich_progress,
            "rich.table": rich_table,
            "rich.text": rich_text,
            "tenacity": tenacity,
        }
    )

    spec = importlib.util.spec_from_file_location("riordino_local", Path(__file__).resolve().parents[1] / "riordino.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


riordino = load_local_riordino_module()


def test_blank_page_detects_clean_white_page():
    image = Image.new("RGB", (800, 1000), "white")

    metrics = riordino.blank_page_metrics(image)

    assert riordino.is_blank_page(metrics, threshold=0.001) is True


def test_blank_page_detects_light_scanner_noise_as_blank():
    image = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(image)
    for x, y in [(20, 20), (120, 80), (240, 160), (360, 240), (480, 320), (600, 400)]:
        draw.rectangle((x, y, x + 3, y + 3), fill=(235, 235, 235))

    metrics = riordino.blank_page_metrics(image)

    assert riordino.is_blank_page(metrics, threshold=0.001) is True


def test_blank_page_detects_dark_content_as_non_blank():
    image = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(image)
    for top in range(100, 700, 40):
        draw.rectangle((80, top, 720, top + 10), fill="black")

    metrics = riordino.blank_page_metrics(image)

    assert riordino.is_blank_page(metrics, threshold=0.001) is False

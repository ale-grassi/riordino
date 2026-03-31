# riordino

**Intelligent scanned PDF organizer** — Splits a bulk-scanned PDF into separate, well-named documents using AI.

`riordino` takes one or more PDFs (the kind you get from scanning an entire stack of mixed paperwork at once) and treats them as a single bulk to analyze.

## How It Works

```
┌────────────────┐
│  Input PDF(s)  │  (bulk scans with mixed documents)
└──────┬─────────┘
       ▼
 1. Load & merge input PDFs, render pages
       ▼
 2. Detect and remove blank pages
       ▼
 3. Detect and correct rotation (Tesseract OSD)
       ▼
 4. Analyze each page with Gemini (batched)
    → title, date, type, subject, priority, description
       ▼
 5. Aggregate pages into logical document groups
       ▼
 6. Determine correct page order within each group
       ▼
 7. Split PDF and write output files
       ▼
┌──────────────────────────────────────────────┐
│  Output: separate PDFs + JSON metadata each  │
└──────────────────────────────────────────────┘
```

## Features

- **Blank page removal** — detects and strips scanner-introduced blank pages using pixel variance analysis
- **Automatic rotation correction** — uses Tesseract OSD to detect and fix page orientation
- **AI-powered analysis** — Google Gemini extracts structured metadata from each page (title, date, type, priority, and more)
- **Smart document grouping** — pages are aggregated into logical documents based on content, subject, dates, and page numbering
- **Intelligent page ordering** — reconstructs the correct reading order within each document
- **Descriptive filenames** — output files are named with dates, subjects, and descriptions (e.g., `2024-03_ccss_certificate.pdf`)
- **JSON sidecar metadata** — each output PDF has a companion JSON file with full extracted metadata
- **Configurable language support** — `--language` flag constrains Tesseract and Gemini to specific languages (23 languages supported)
- **Selective pipeline control** — skip individual steps with `--skip-blanks`, `--skip-rotation`, `--skip-analysis`, `--skip-aggregation`, `--skip-ordering`
- **Dependency checks** — verifies Tesseract, language packs, and API keys before running
- **Dry-run mode** — preview the plan without writing any files
- **Step-by-step debugging** — optionally save all intermediate outputs for inspection


## Prerequisites

- **Python 3.14+**
- **Tesseract OCR** with OSD data
- **Google API key** with access to the Gemini API

### Installing Tesseract

```bash
# Arch Linux
sudo pacman -S tesseract tesseract-data-osd

# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-osd

# macOS
brew install tesseract
```

Install language data packages for each language you plan to use (riordino checks for these at startup):

```bash
# Arch Linux (example: German, French, Italian, Polish)
sudo pacman -S tesseract-data-deu tesseract-data-fra tesseract-data-ita tesseract-data-pol

# Ubuntu / Debian
sudo apt install tesseract-ocr-deu tesseract-ocr-fra tesseract-ocr-ita tesseract-ocr-pol
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/riordino.git
cd riordino

python -m venv .venv
source .venv/bin/activate

pip install .
```

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_api_key_here
RIORDINO_LANGUAGES=en,de,fr,it,pl
```

`RIORDINO_LANGUAGES` sets the default for `--language`. If omitted, defaults to `en`.

### Docker

```bash
docker build -t riordino .

# Install additional language packs at build time:
docker build -t riordino --build-arg LANGS="deu fra ita" .
```

```bash
docker run --rm \
  -e GOOGLE_API_KEY \
  -v "$PWD":/data \
  riordino /data/scan.pdf -o /data/output/
```

### Development setup

```bash
pip install -e '.[dev]'  # installs ruff, mypy, pytest
```

## Usage

```bash
# Single PDF
python riordino.py scan.pdf

# Multiple PDFs (merged into one bulk for analysis)
python riordino.py scan1.pdf scan2.pdf scan3.pdf
```

This processes the input PDF(s) and writes the split documents to the same directory as the first input file.

## CLI Reference

| Option | Default | Description |
|---|---|---|
| `input_pdf` | *(required)* | Path(s) to scanned PDF file(s) — multiple files are merged into one bulk |
| `-o`, `--output-dir` | Same as input file | Directory for output files |
| `-b`, `--blank-threshold` | `0.001` | Pixel variance threshold for blank page detection (0.0–1.0) |
| `-n`, `--dry-run` | off | Show the processing plan without writing any files |
| `--dpi` | `150` | DPI for page rendering (72–600) |
| `--model` | `gemini-3.1-flash-lite-preview` | Gemini model to use |
| `-l`, `--language` | `$RIORDINO_LANGUAGES` or `en` | Comma-separated ISO 639-1 language codes |
| `--batch-size` | `10` | Number of pages per LLM analysis batch (1–50) |
| `--max-retries` | `3` | Maximum API retry attempts on failure (0–10) |
| `--save-steps` | off | Save intermediate outputs to an `_steps/` subdirectory |
| `--skip-blanks` | off | Skip blank page detection (keep all pages) |
| `--skip-rotation` | off | Skip rotation detection and correction |
| `--skip-analysis` | off | Skip LLM page analysis (implies `--skip-aggregation`) |
| `--skip-aggregation` | off | Skip LLM document grouping (implies `--skip-ordering`) |
| `--skip-ordering` | off | Skip LLM page ordering within documents |

## Next Steps

- [x] Publish to PyPI for `pip install riordino`
- [ ] Add a `--verbose` / `--quiet` flag for log level control
- [ ] Explore local/open-source LLM backends as an alternative to Gemini
- [ ] Support OCR-based text extraction as a fallback when Gemini is unavailable
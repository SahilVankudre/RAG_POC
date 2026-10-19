# Setup Guide

How to install RAG POC on a new machine, check that it works, and configure it. To use it once
it's installed, see [`RUN_MANUAL.md`](RUN_MANUAL.md).

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12** | Pinned in `.python-version`. uv can install it for you |
| **Git** | To clone the repository |
| **uv** (recommended) or `pip` | uv installs from the lock file, so everyone gets identical versions |
| **Internet access on the first run** | Downloads the dataset (~327 MB) and the embedding model (~90 MB) once; both are cached |
| **Disk space** | About 1.7 GB: virtual environment ~1.2 GB (mostly CPU PyTorch), Hugging Face cache ~400 MB, generated `data/` ~40 MB |
| **GPU** | Not required. PyTorch is pinned to the CPU-only build |

Supported on Windows, macOS and Linux.

## 2. Install uv

If you don't have uv yet:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, run `pip install uv`. Open a new terminal afterwards and check that
`uv --version` works.

## 3. Get the code

```bash
git clone https://github.com/SahilVankudre/RAG_POC.git RAG-POC
cd RAG-POC
```

Run every later command from this repository root. The CLI writes its files to `data/` relative
to the current directory.

## 4. Install dependencies

### Option A: uv (recommended)

```bash
uv sync
```

This command:

- creates `.venv/` and installs Python 3.12 into it if needed
- installs the exact versions from `uv.lock`, including the development tools (pytest, ruff,
  import-linter)
- installs the project itself, which provides the `rag-poc` command

With uv, prefix commands with `uv run` (for example `uv run rag-poc --help`). There is no need
to activate the environment.

### Option B: venv + pip

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

- `requirements.txt` is exported from `uv.lock`. It adds the PyTorch CPU wheel index, so pip
  installs the small CPU-only torch instead of the multi-GB CUDA build.
- It ends with `-e .`, which installs this project in editable mode and provides the `rag-poc`
  command.
- With the environment activated, run `rag-poc …` directly, without `uv run`.

If you install `requirements.txt` with `uv pip` instead of pip, add
`--index-strategy unsafe-best-match`:

```bash
uv pip install -r requirements.txt --index-strategy unsafe-best-match
```

By default uv only takes a package from the first index that has it, and the PyTorch index
also hosts older copies of some PyPI packages.

## 5. Verify the installation

```bash
uv run rag-poc --help        # lists the commands
uv run rag-poc show-config   # prints the resolved configuration and file paths
uv run pytest                # unit tests: expect "38 passed, 2 deselected" in about 10 s
```

The unit tests need no network and no model. To also check the real embedding model (this
downloads it on first use):

```bash
uv run pytest -m integration
```

Next, build the index and run a search as described in [`RUN_MANUAL.md`](RUN_MANUAL.md).

## 6. Configuration

Every setting has a default that suits the standard 10K demo. You can override any of them
with an environment variable prefixed `RAG_`, or with a `.env` file in the repository root.
`.env` is gitignored.

Precedence, from lowest to highest: **defaults < `.env` < environment variables < CLI
`--log-level`**.

| Variable | Default | Purpose |
|---|---|---|
| `RAG_DATA_DIR` | `data` | Root folder for the sample, prepared data, embeddings and index |
| `RAG_HF_DATASET_REPO` | `McAuley-Lab/Amazon-Reviews-2023` | Hugging Face dataset repository |
| `RAG_AMAZON_CATEGORY` | `All_Beauty` | Amazon Reviews 2023 category to load |
| `RAG_HF_CACHE_DIR` | Hugging Face default | Where the raw dataset download is cached |
| `RAG_SAMPLE_SIZE` | `10000` | Number of reviews to sample (10K–50K recommended) |
| `RAG_RANDOM_SEED` | `42` | Seed for reproducible sampling |
| `RAG_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers model |
| `RAG_EMBEDDING_BATCH_SIZE` | `64` | Texts encoded per batch |
| `RAG_EMBEDDING_QUERY_PREFIX` | model default | Instruction added to queries only. Set automatically for BGE models; `""` disables it |
| `RAG_DEFAULT_TOP_K` | `5` | Results returned per search |
| `RAG_SEARCH_MIN_WORDS` | `3` | Default minimum review length, in words, for search results |
| `RAG_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

Example `.env`:

```ini
RAG_SAMPLE_SIZE=50000
RAG_LOG_LEVEL=WARNING
```

Setting a variable for a single command:

```powershell
# Windows PowerShell
$env:RAG_SAMPLE_SIZE = "50000"; uv run rag-poc build-index
Remove-Item Env:RAG_SAMPLE_SIZE          # undo afterwards
```

```bash
# macOS / Linux / Git Bash
RAG_SAMPLE_SIZE=50000 uv run rag-poc build-index
```

Invalid values (for example `RAG_SAMPLE_SIZE=0` or an unknown log level) stop the program at
startup with a clear `Invalid configuration` error.

**Optional:** setting `HF_TOKEN` (a Hugging Face access token) raises download rate limits. It
is not required, because the dataset is public.

## 7. Where files are stored

| Location | Contents | Safe to delete? |
|---|---|---|
| `.venv/` | Virtual environment | Yes. Recreate it with `uv sync` |
| `data/` | Sample, prepared feedback, embeddings, FAISS index | Yes. Recreate it with `rag-poc build-index` |
| `~/.cache/huggingface/` (Windows: `%USERPROFILE%\.cache\huggingface\`) | Raw dataset JSONL and model weights | Yes, but they will be downloaded again |

Nothing in `data/`, `.venv/` or the cache is committed to Git.

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `uv: command not found` | Open a new terminal after installing uv, or add it to `PATH` |
| pip downloads a multi-GB torch with CUDA | You installed torch without `requirements.txt`. Use `pip install -r requirements.txt`, which points pip at the CPU wheel index |
| `Error: Could not download raw/review_categories/All_Beauty.jsonl …` | No internet access, or Hugging Face rate limiting. Retry, or set `HF_TOKEN` |
| `Error: No review sample at … Run rag-poc sample first.` (or the same for prepared feedback, embeddings or index) | A stage was skipped. Run `uv run rag-poc build-index`, which runs every stage in order |
| `Error: The index was built with <model A>, but queries would be embedded with <model B>` | You changed `RAG_EMBEDDING_MODEL_NAME`. Run `uv run rag-poc build-index` to rebuild with the new model |
| The first `search` takes about 10 s | Expected: torch and the model load once. Use the interactive session (`rag-poc search` with no query) so later queries take milliseconds |
| Warning about symlinks in the Hugging Face cache on Windows | Harmless (the cache copies files instead), and the CLI suppresses it |
| `Invalid configuration: …` | A `RAG_*` variable or `.env` entry has an invalid value. The message names the field |

## 9. Development tools

The development tools are installed by `uv sync` (they are in the `dev` dependency group):

```bash
uv run pytest                        # unit tests
uv run pytest -m integration         # tests with the real model
uv run ruff check .                  # lint
uv run ruff format .                 # format (line length 100)
uv run lint-imports                  # enforce module boundaries
```

After adding or upgrading a dependency (`uv add <package>`), regenerate `requirements.txt` so
the pip install route stays in sync:

```bash
{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; uv export --no-hashes --quiet; } > requirements.txt
```

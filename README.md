# RAG POC — Semantic Search over Customer Feedback

The retrieval foundation of a Retrieval-Augmented Generation (RAG) system. Amazon customer
reviews are sampled, converted into a source-agnostic `FeedbackItem`, cleaned, embedded with
Sentence Transformers and indexed in FAISS. A natural-language question then returns the most
semantically relevant reviews. No LLM is involved.

```
Amazon Reviews 2023 (All_Beauty) → 10K seeded sample → FeedbackItem → cleaning
  → all-MiniLM-L6-v2 embeddings (384-d) → FAISS index → top-5 semantic search
```

```
$ rag-poc search "Problems with product durability" --max-rating 2 -k 3

1. score 0.484  amz_fb85addb4b952b05 | product B08GB1VQZM | 2/5 | 2021-01-31
Possible dud
Sadly this product has not really worked for me. Hope it works for others.

2. score 0.483  amz_987b82b9c21e219e | product B098SC44N6 | 1/5 | 2022-01-25
Product arrived defective
Received an old product, both the glue and glass coating were dried out and brushes broken.

3. score 0.478  amz_a0ad451b1d1c89b9 | product B019GBG0IE | 1/5 | 2018-08-16
Falls apart
Falls apart. Poor quality pieces.
```

Search matches **meaning, not keywords**: "Falls apart" shares no words with the query.

## Documentation

| Document | Contents |
|---|---|
| [`SETUP.md`](SETUP.md) | Prerequisites, installation (uv or pip), verification, configuration, troubleshooting |
| [`RUN_MANUAL.md`](RUN_MANUAL.md) | Every command step by step, search options, output files, rebuilding, scaling up |

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU is needed. See
[`SETUP.md`](SETUP.md) for the pip alternative.

```bash
git clone https://github.com/SahilVankudre/RAG_POC.git RAG-POC && cd RAG-POC
uv sync                          # install dependencies into .venv
uv run rag-poc build-index       # download, sample, clean, embed, index (first run: ~2–3 min)
uv run rag-poc search            # interactive search; try: Problems with product durability
```

## Features

- **Reproducible dataset.** A seeded 10K sample of the Amazon Reviews 2023 `All_Beauty`
  category, downloaded once and stored locally as Parquet.
- **Source-agnostic schema.** Each review becomes a validated `FeedbackItem` with a stable id.
  Supporting Yelp, surveys or support tickets only needs a new adapter.
- **Light, traceable cleaning.** Removes markup, normalizes whitespace, combines useful titles
  with the body, and drops same-author duplicates. Originals are kept, and a data-quality report
  accounts for every record.
- **Fast CPU embeddings.** `all-MiniLM-L6-v2` (384-d, L2-normalized), batched by length for
  roughly 4× throughput.
- **Exact vector search.** FAISS `IndexFlatIP` (cosine similarity), saved to disk with a
  position-to-review map and a manifest that is checked on load.
- **Search filters.** Rating bounds (`--max-rating 2` for complaints) and a minimum review
  length, applied exactly.
- **Incremental builds.** Each stage is skipped when its inputs are unchanged. An up-to-date
  rebuild takes about 1 second.

## How it works

| Stage | Command | Output (under `data/`) |
|---|---|---|
| 1. Sample | `rag-poc sample` | `raw/all_beauty_10k_seed42.parquet`: 10,000 reviews with text, seed 42 |
| 2. Prepare | `rag-poc prepare` | `processed/feedback_items.parquet` + `quality_report.json` |
| 3. Embed | `rag-poc embed` | `index/embeddings.npy`: 9,999 × 384 float32 vectors + manifest |
| 4. Index | `rag-poc index` | `index/faiss.index`, `id_map.parquet`, `index_manifest.json` |
| Search | `rag-poc search` | Embeds the query, ranks reviews by cosine similarity, applies filters |

`rag-poc build-index` runs stages 1–4 in order. Details are in [`RUN_MANUAL.md`](RUN_MANUAL.md).

## Data model: `FeedbackItem`

| Field | Type | Amazon source |
|---|---|---|
| `feedback_id` | `str` | `amz_` + hash of `user_id`, `asin`, `timestamp` (row position if any is missing) |
| `source` | `str` | `"amazon"` |
| `entity_id` | `str` | `asin` (fallback `parent_asin`) |
| `text` | `str` | `text` (must not be blank) |
| `title` | `str \| None` | `title` |
| `rating` | `float \| None` | `rating` (1–5) |
| `timestamp` | `datetime \| None` | `timestamp` (epoch ms → UTC) |
| `metadata` | `dict` | `parent_asin`, `user_id`, `verified_purchase`, `helpful_vote`, `source_row`, `category` |

## Project structure

```
RAG-POC/
├─ pyproject.toml           # dependencies, `rag-poc` CLI entry point, tool configuration
├─ uv.lock                  # exact, reproducible dependency versions
├─ requirements.txt         # pip-compatible export of uv.lock (CPU-only PyTorch)
├─ README.md · SETUP.md · RUN_MANUAL.md
├─ data/                    # runtime artifacts, gitignored, created by the CLI
└─ src/backend/
   ├─ src/rag_poc/
   │  ├─ bootstrap/         # composition root: settings + container that wires all adapters
   │  ├─ lib/               # shared types (FeedbackItem, SearchResult, reports), errors, logging
   │  ├─ ingestion/         # Hugging Face download, sampling, Amazon → FeedbackItem adapter
   │  ├─ preprocessing/     # cleaning steps, embedding-text composition, quality report
   │  ├─ embedding/         # Embedder port, Sentence Transformers adapter, .npy store
   │  ├─ retrieval/         # VectorIndex port, FAISS adapter, index + id-map store
   │  ├─ application/       # use cases: sample, prepare, embed, index, pipeline, search
   │  └─ entrypoints/       # Typer CLI and terminal rendering
   └─ tests/
      ├─ unit/              # fast, no model, no network
      └─ integration/       # real model (pytest -m integration)
```

**Architecture.** The code is a modular monolith with a single composition root:

- Feature modules (`ingestion`, `preprocessing`, `embedding`, `retrieval`) depend only on `lib`
  and never on each other.
- Use cases in `application` work against ports (Python `Protocol`s).
- Only `bootstrap/container.py` creates concrete adapters, and it creates them lazily, so
  commands that don't need the model never import torch.
- `import-linter` enforces these boundaries (`uv run lint-imports`).

## Tech stack

Python 3.12 · uv · pandas + pyarrow · pydantic / pydantic-settings · huggingface_hub ·
sentence-transformers + PyTorch (CPU) · faiss-cpu · Typer + Rich · pytest · ruff · import-linter

## Development

```bash
uv run pytest                    # unit tests (~10 s)
uv run pytest -m integration     # slower tests that load the real model
uv run ruff check . && uv run ruff format .
uv run lint-imports              # verify architecture boundaries
```

After changing dependencies, regenerate `requirements.txt`:

```bash
{ echo "--extra-index-url https://download.pytorch.org/whl/cpu"; uv export --no-hashes --quiet; } > requirements.txt
```

## Search quality

Embedding similarity captures **topic**, not opinion. "The smell is too strong" also matches
"the smell is amazing". For complaint-style questions, add `--max-rating 2`. This was the most
effective fix measured, for example durability went from 3/5 to 5/6 relevant results. See
[`RUN_MANUAL.md`](RUN_MANUAL.md#search-tips) for more.

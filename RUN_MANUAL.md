# Run Manual

How to run RAG POC: building the search index, searching, and what each command does and
produces. Install the project first, as described in [`SETUP.md`](SETUP.md).

Run all commands from the repository root. With uv, prefix each command with `uv run`. With an
activated virtual environment, call `rag-poc` directly. `python -m rag_poc …` is equivalent.

## Contents

1. [Command overview](#1-command-overview)
2. [Fastest path: build and search](#2-fastest-path-build-and-search)
3. [Running the stages one at a time](#3-running-the-stages-one-at-a-time)
4. [Searching](#4-searching)
5. [Search tips](#search-tips)
6. [Output files](#6-output-files)
7. [Rebuilding, resetting and reuse](#7-rebuilding-resetting-and-reuse)
8. [Scaling up and switching models](#8-scaling-up-and-switching-models)
9. [Errors and exit codes](#9-errors-and-exit-codes)

---

## 1. Command overview

| Command | What it does | Typical time |
|---|---|---|
| `rag-poc --help` | Lists the commands | instant |
| `rag-poc show-config` | Prints the resolved settings and file paths | instant |
| `rag-poc explore [--examples N]` | Profiles the **full** raw dataset (701K reviews) | a few seconds (+ ~327 MB download on first run) |
| `rag-poc sample [--force] [--examples N]` | Draws the seeded sample and saves it as Parquet | a few seconds, then reused |
| `rag-poc prepare` | Converts to `FeedbackItem`, cleans, writes the quality report | a few seconds |
| `rag-poc embed [--force]` | Embeds every prepared review | ~1–1.5 min for 10K on a laptop CPU |
| `rag-poc index [--force]` | Builds the FAISS index from stored embeddings | < 1 s |
| `rag-poc build-index [--force]` | Runs sample → prepare → embed → index | ~2–3 min on first run, ~1 s afterwards |
| `rag-poc search [QUERY] [options]` | One-shot search, or an interactive session without a query | ~10 s to load, then ~12 ms per query |

Global option (put it **before** the command): `--log-level DEBUG|INFO|WARNING|ERROR`, for
example `uv run rag-poc --log-level WARNING build-index`.

---

## 2. Fastest path: build and search

```bash
uv run rag-poc build-index
uv run rag-poc search
```

On the first run, `build-index` downloads the dataset and model, then shows a progress bar while
embedding, and finishes with a summary:

```
Build summary
┌────────────┬──────────────────────────────────────────┬────────────────┐
│ Stage      │ Result                                   │ Status         │
├────────────┼──────────────────────────────────────────┼────────────────┤
│ 1. Sample  │ 10,000 reviews                           │ built          │
│            │ (all_beauty_10k_seed42.parquet)          │                │
│ 2. Prepare │ 9,999 prepared, 1 excluded               │ reconciled     │
│ 3. Embed   │ 9,999 x 384                              │ built in 90.7s │
│            │ (sentence-transformers/all-MiniLM-L6-v2) │                │
│ 4. Index   │ IndexFlatIP, 9,999 vectors               │ built in 0.3s  │
└────────────┴──────────────────────────────────────────┴────────────────┘
Done in 93.5s. Index: …\RAG-POC\data\index\faiss.index
```

Running it again finishes in about a second, because every stage shows `reused (up to date)`.

`search` then loads the index and model once and prompts for queries:

```
Semantic search over 9,999 reviews (sentence-transformers/all-MiniLM-L6-v2). Filters: at least 3 words.
Type a question in plain English; exit or Ctrl+C to quit.

query> Problems with product durability
```

---

## 3. Running the stages one at a time

`build-index` runs these four stages for you. Run them separately to inspect each step's output.

### 3.0 Explore the raw dataset (optional)

```bash
uv run rag-poc explore --examples 3
```

The first time, this downloads `raw/review_categories/All_Beauty.jsonl` (~327 MB) from the
[Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset
into the Hugging Face cache. It then prints the record count, raw fields, column types and
missing values, rating distribution, date range, median text length and sample reviews.

Nothing is written to `data/`.

### 3.1 Create the sample

```bash
uv run rag-poc sample            # reuses the existing sample if present
uv run rag-poc sample --force    # redraws it (same seed → identical file)
```

- Drops reviews with empty text, then draws `RAG_SAMPLE_SIZE` reviews (10,000) at random with
  seed `RAG_RANDOM_SEED` (42).
- Keeps each review's original line number in a `source_row` column.
- **Writes:** `data/raw/all_beauty_10k_seed42.parquet`. The file name includes the size and
  seed, so changing either creates a new file.
- **Prints:** a profile of the sample (7,576 distinct products, all five ratings).

### 3.2 Prepare (adapt and clean)

```bash
uv run rag-poc prepare
```

1. **Adapt.** Each Amazon row becomes a validated `FeedbackItem` with a stable id
   (`amz_` + hash of user, product and timestamp). Unusable rows are rejected with a reason:
   `missing_text`, `missing_entity_id`, `duplicate_feedback_id` or `invalid_record`.
2. **Clean** (deliberately light, because the model benefits from natural language):
   - removes literal `<br />` and similar tags, unescapes HTML entities (`&#34;` → `"`), and
     strips `[[VIDEOID:…]]` markers
   - collapses whitespace and line breaks
   - keeps casing, punctuation, stopwords and emoji
   - uses an informative title as the text when the body was only a video marker
   - drops repeated reviews by the same author; identical text from different authors is kept
   - keeps originals in `metadata.original_text` / `metadata.original_title`
3. **Compose the embedding text.** The title is added before the body unless it is
   auto-generated ("Five Stars") or the body already starts with it.

- **Writes:** `data/processed/feedback_items.parquet` and `data/processed/quality_report.json`.
- **Prints:** the record flow and the text changes:

```
Record flow
┌────────────────────────────────────────────────┬─────────┐
│ Stage                                          │ Records │
├────────────────────────────────────────────────┼─────────┤
│ Sampled reviews                                │  10,000 │
│   - rejected by adapter: duplicate_feedback_id │      -1 │
│ Prepared for embedding                         │   9,999 │
└────────────────────────────────────────────────┴─────────┘
Text changes (records affected)
┌────────────────────────────────────┬─────────┐
│ Change                             │ Records │
├────────────────────────────────────┼─────────┤
│ html entities unescaped            │      93 │
│ html tags removed                  │     697 │
│ title used as text                 │       5 │
│ video markers removed              │      97 │
│ whitespace normalized              │   2,000 │
│ title combined into embedding text │   8,086 │
└────────────────────────────────────┴─────────┘
Kept but notable: 379 very short texts, 154 long texts the model will truncate.
Accounting: reconciled (1 excluded in total)
```

"Reconciled" means sampled = prepared + every exclusion, so no record went missing.

### 3.3 Embed

```bash
uv run rag-poc embed             # reuses stored embeddings if nothing changed
uv run rag-poc embed --force     # recompute
```

- Encodes each embedding text with `sentence-transformers/all-MiniLM-L6-v2` (~90 MB, downloaded
  on first use) in batches of `RAG_EMBEDDING_BATCH_SIZE` (64), with a progress bar.
- Batches are grouped by text length for speed. Rows keep their original order.
- **Writes:**
  - `data/index/embeddings.npy`: float32 matrix of shape (9,999, 384); every row has length 1
  - `data/index/embeddings_manifest.json`: model, the `feedback_id` for each row, and a content
    fingerprint
- **Reuse:** if the fingerprint (model + ids + texts) is unchanged, the stage is skipped and the
  model is never loaded.

### 3.4 Build the index

```bash
uv run rag-poc index             # reuses the index if built from the same embeddings
uv run rag-poc index --force     # rebuild
```

- Loads the stored vectors into a FAISS `IndexFlatIP`, an exact inner-product index. Because the
  vectors have length 1, the inner product equals cosine similarity.
- Needs **no model** and does **no re-embedding**.
- **Writes** three files to `data/index/`: `faiss.index` (vectors), `id_map.parquet` (vector
  position → full review), and `index_manifest.json` (model, dimension, count, embeddings
  fingerprint; written last).
- **Refuses** to run when the embeddings are stale relative to the prepared data. Run `embed`
  first.

---

## 4. Searching

### One-shot query

```bash
uv run rag-poc search "customers having charging problems"
uv run rag-poc search "Problems with product durability" --max-rating 2 --top-k 10
```

### Interactive session

```bash
uv run rag-poc search
uv run rag-poc search --max-rating 2      # filters apply to every query in the session
```

The session loads the index and model once (about 10 s), then answers each query in about
12 ms. To leave, type `exit`, `quit` or `q`, or press Ctrl+C. An error in one query (for example
an empty query) is shown, and the session continues.

### Options

| Option | Default | Effect |
|---|---|---|
| `--top-k N`, `-k N` | 5 (`RAG_DEFAULT_TOP_K`) | Number of results |
| `--max-rating R` | none | Only reviews rated ≤ R. Use `2` for complaints |
| `--min-rating R` | none | Only reviews rated ≥ R. Use `4` for praise |
| `--min-words N` | 3 (`RAG_SEARCH_MIN_WORDS`) | Skip reviews with fewer than N words. Use `0` to allow all |

Filtered search is still **exact**: if too few of the nearest reviews pass the filters, the
search fetches more candidates until it has `top_k` passing results or has checked every review.

### Reading a result

```
3. score 0.478  amz_a0ad451b1d1c89b9 | product B019GBG0IE | 1/5 | 2018-08-16
Falls apart
Falls apart. Poor quality pieces.
```

| Part | Meaning |
|---|---|
| `3.` | Rank |
| `score 0.478` | Cosine similarity to the query: 1 = same meaning, ~0 = unrelated. Compare scores within one query, not across queries |
| `amz_a0ad451b1d1c89b9` | Stable `feedback_id` |
| `product B019GBG0IE` | `entity_id` (Amazon ASIN) |
| `1/5 · 2018-08-16` | Rating and review date |
| Second and third lines | Title (italic) and review text, shortened to 400 characters |

---

## Search tips

The model matches **topic**, not **opinion** or negation. These measurements are from the 10K
sample, counting relevant results in the top 5–6:

| Query | No rating filter | With `--max-rating 2` |
|---|---|---|
| Problems with product durability | 3/5 | **5/6** |
| the smell is too strong | 2/5 | **4/6** |
| caused a rash or allergic reaction | 2/5 (a negation ranked first) | **5/6** |
| customers having charging problems | 4/5 | **5/5** |

- **Looking for problems?** Add `--max-rating 2`. It removes praise and negated matches ("does
  *not* cause a rash").
- **Looking for praise?** Add `--min-rating 4`.
- **Seeing very short reviews?** Raise `--min-words`.

---

## 6. Output files

All files live under `RAG_DATA_DIR` (default `data/`). The folder is gitignored and safe to
delete.

| File | Created by | Contents |
|---|---|---|
| `raw/all_beauty_10k_seed42.parquet` | `sample` | The sampled raw Amazon rows + `source_row` |
| `processed/feedback_items.parquet` | `prepare` | Cleaned `FeedbackItem`s (`metadata` as JSON) + `embedding_text` |
| `processed/quality_report.json` | `prepare` | Record counts, exclusion reasons, text changes, `is_reconciled` |
| `index/embeddings.npy` | `embed` | float32 vectors, one row per review |
| `index/embeddings_manifest.json` | `embed` | Model, dimension, row → `feedback_id` list, fingerprint |
| `index/faiss.index` | `index` | FAISS `IndexFlatIP` |
| `index/id_map.parquet` | `index` | Vector position → full review |
| `index/index_manifest.json` | `index` | Model, dimension, count, embeddings fingerprint |

To explore the data yourself:

```python
import pandas as pd

items = pd.read_parquet("data/processed/feedback_items.parquet")
items[["feedback_id", "rating", "embedding_text"]].head()
```

---

## 7. Rebuilding, resetting and reuse

| Situation | What happens / what to run |
|---|---|
| Run `build-index` again with nothing changed | Every stage is reused (~1 s) |
| Changed the cleaning code | `build-index`: prepare re-runs, the fingerprint changes, so embed and index rebuild automatically |
| Changed `RAG_EMBEDDING_MODEL_NAME` | `build-index`: re-embeds and re-indexes. Search refuses an index built with another model |
| Want fresh embeddings and index anyway | `build-index --force` |
| Want to redraw the sample | `sample --force`, then `build-index` |
| Start completely from scratch | Delete the `data/` folder, then `build-index` (the cached download is reused) |

---

## 8. Scaling up and switching models

**Larger sample (50K reviews):**

```powershell
$env:RAG_SAMPLE_SIZE = "50000"; uv run rag-poc build-index      # PowerShell
```

```bash
RAG_SAMPLE_SIZE=50000 uv run rag-poc build-index                 # bash
```

This was measured at 49,940 prepared reviews, ~4 minutes of embedding and a 2 s index build.
Search stays interactive. The 50K sample is saved under its own file name, so the 10K sample is
kept. The prepared data, embeddings and index files are shared, though, so switching between
the two sizes re-embeds and re-indexes each time. To keep both, give each its own
`RAG_DATA_DIR`.

**Alternative model (`BAAI/bge-small-en-v1.5`, also 384-d):**

```bash
RAG_EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5 uv run rag-poc build-index
RAG_EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5 uv run rag-poc search
```

Its query instruction is applied automatically. It is more precise on opinion queries ("too
strong" 5/5 vs 2/5) but embeds about 2× slower. Keep the variable set for `search` too, or put
it in `.env`.

---

## 9. Errors and exit codes

Expected problems print a red `Error: …` message that names the next step, and the command
exits with code **1** (no traceback). For example:

```
Error: No prepared feedback at …\data\processed\feedback_items.parquet. Run `rag-poc prepare` first.
```

A full Python traceback means an unexpected bug. Please report it with the command you ran.
See the troubleshooting table in [`SETUP.md`](SETUP.md#8-troubleshooting) for common fixes.

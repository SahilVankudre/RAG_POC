# RAG POC — Semantic Search and RAG over Customer Feedback

A Retrieval-Augmented Generation (RAG) system: ask a question about customer feedback and get an
answer that cites the reviews behind it. Amazon customer reviews are sampled, converted into a
source-agnostic `FeedbackItem`, cleaned, embedded with Sentence Transformers and indexed in FAISS.
A question retrieves the most relevant reviews, an LLM answers from them, and the answer is checked
before it's shown. When the reviews can't support an answer, it says so rather than guessing.

```
Amazon Reviews 2023 (All_Beauty) → 10K seeded sample → FeedbackItem → cleaning
  → all-MiniLM-L6-v2 embeddings (384-d) → FAISS index → top-k semantic search
  → review context with ids → LLM → checked answer + cited reviews
```

```
$ rag-poc ask "What are the common quality complaints?" --max-rating 2 -k 4

Answer: The quality complaints in these reviews are general poor quality [R1][R4], items arriving
broken or breaking during first use [R2], colors appearing less rich than shown [R3], and hair not
matching its description [R4].

Main observations:
- General poor quality, including poor quality for the price [R1][R4]
- Items arrived broken and more broke during first use [R2]
- Colors not as rich as they appear [R3]
- Hair did not fit the description [R4]

Supporting reviews (4)
R1  Review amz_24640f0bda48c1f8  ·  Rating: 2/5  ·  product B0711T3LDQ
   "Quality no good."
R4  Review amz_eb4568b97fe2bad0  ·  Rating: 1/5  ·  product B07ZL5LZDD
   "Extremely poor quality for the price was packaged inside of a box with clumps of hair ..."
R2  Review amz_fd43a84f737bf0c3  ·  Rating: 1/5  ·  product B00DD4KY6K
   "Some came broken, more broke during first use. Awful customer service."
R3  Review amz_130ed1fd0245611a  ·  Rating: 2/5  ·  product B08PKFC4FV
   "The colors are not as rich as it appears"
```

`rag-poc search` runs retrieval alone, with no LLM:

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
| [`RUN_MANUAL.md`](RUN_MANUAL.md) | Every command step by step, search and ask options, evaluation, output files, rebuilding, scaling up |

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). No GPU is needed. See
[`SETUP.md`](SETUP.md) for the pip alternative.

```bash
git clone https://github.com/SahilVankudre/RAG_POC.git RAG-POC && cd RAG-POC
uv sync                          # install dependencies into .venv
uv run rag-poc build-index       # download, sample, clean, embed, index (first run: ~2–3 min)
uv run rag-poc search            # interactive search; try: Problems with product durability
```

`search` needs no LLM. To ask questions, configure one first:

```bash
cp .env.example .env             # then put your API key in .env
uv run rag-poc llm-check         # one real call, to confirm the setup works
uv run rag-poc ask "What are the common quality complaints?" --max-rating 2 --show-context
```

`--show-context` also prints exactly what the LLM received, so the output shows the question, the
retrieved reviews with similarity scores, the context, the answer and the source ids.

## LLM setup

Any API that speaks the OpenAI chat completions format works, so changing provider is a
configuration change. [`.env.example`](.env.example) has a ready preset for each of these:

| Provider | `RAG_LLM_BASE_URL` | `RAG_LLM_MODEL_NAME` | API key |
|---|---|---|---|
| OpenRouter | `https://openrouter.ai/api/v1` | `deepseek/deepseek-v4.1-flash-1017` | Required |
| DeepSeek | `https://api.deepseek.com` | `deepseek-flash` | Required |
| OpenAI | `https://api.openai.com/v1` | A model you can access | Required |
| Local Ollama | `http://localhost:11434/v1` | A pulled model that fits in free memory | Not needed |

Keep your key in `.env`, which is gitignored. Details are in
[`SETUP.md`](SETUP.md#llm-configuration).

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
- **Search filters.** Rating bounds (`--max-rating 2` for complaints), a minimum review length and
  a single product (`--product`), applied exactly.
- **Cited answers.** Each claim cites the reviews behind it, using short labels that code maps back
  to real review ids. The output separates reviews retrieved from reviews actually relied on.
- **Refuses to guess.** Off-topic questions are refused without calling the LLM, and the LLM declines
  when the reviews don't answer the question. Made-up citations are removed and uncited answers are
  withheld.
- **Replaceable LLM.** One client serves OpenRouter, DeepSeek, OpenAI and local Ollama.
- **Evaluation.** `rag-poc evaluate` runs 20 representative questions and saves full results for
  manual review.
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
| Ask | `rag-poc ask` | Retrieves reviews, asks the LLM to answer from them, checks the citations |
| Evaluate | `rag-poc evaluate` | `evaluation/run_<timestamp>.json`: every answer with its reviews |

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
├─ .env.example             # LLM presets; copy to .env and add your key
├─ data/                    # runtime artifacts, gitignored, created by the CLI
├─ evaluation/              # test_questions.json: the 20 evaluation questions
└─ src/rag_poc/
   ├─ bootstrap/            # composition root: settings + container that wires all adapters
   ├─ lib/                  # shared types (FeedbackItem, SearchResult, reports), errors, logging
   ├─ ingestion/            # Hugging Face download, sampling, Amazon → FeedbackItem adapter
   ├─ preprocessing/        # cleaning steps, embedding-text composition, quality report
   ├─ embedding/            # Embedder port, Sentence Transformers adapter, .npy store
   ├─ retrieval/            # VectorIndex port, FAISS adapter, index store, SemanticRetriever
   ├─ llm/                  # TextGenerator port, LLMService for OpenAI-format APIs
   ├─ rag/                  # prompt, review context, interpretation of LLM replies
   ├─ evaluation/           # question set and run storage
   ├─ application/          # use cases: sample, prepare, embed, index, pipeline, RAG, evaluation
   └─ entrypoints/          # Typer CLI and terminal rendering
```

**Architecture.** The code is a modular monolith with a single composition root:

- Feature modules (`ingestion`, `preprocessing`, `embedding`, `retrieval`, `llm`, `rag`,
  `evaluation`) depend only on `lib` and never on each other, so the LLM can be replaced without
  touching retrieval.
- Use cases in `application` work against ports (Python `Protocol`s).
- Only `bootstrap/container.py` creates concrete adapters, and it creates them lazily, so
  commands that don't need the model never import torch.
- `import-linter` enforces these boundaries (`uv run lint-imports`).

## Tech stack

Python 3.12 · uv · pandas + pyarrow · pydantic / pydantic-settings · huggingface_hub ·
sentence-transformers + PyTorch (CPU) · faiss-cpu · openai SDK · Typer + Rich · ruff ·
import-linter

## Development

```bash
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

## Answer quality

Evaluated on 20 questions from the `All_Beauty` 10K sample with
`deepseek/deepseek-v4.1-flash-1017`. 15 questions expect an answer and 5 expect a refusal. Each
answer was read against the full text of the reviews it cites. That was one reviewer's judgement,
and 15 answers are enough to find problems, not to give a reliable accuracy figure.

| Measure | Result |
|---|---|
| Answered or refused as expected | 20/20 in each of 3 runs |
| Answers fully supported by their reviews | 12 of 15 |
| Citations that support their claim as written | 119 of 122 |
| Made-up citations | 0 |
| Retrieved reviews relevant to the question | 8.2 of 10 on average |
| Review-based prompt injection attempts followed (5 attacks, 4 runs each) | 0 of 20 |

Know these limits when reading answers:

- **"Not found" means "not in the top 10".** Answers see only the most similar reviews, so they can
  miss evidence that exists. A delivery question reported only positive feedback, although the
  sample holds complaints such as "the item took too long to arrive".
- **Use `--max-rating 2` for problems.** It lifted relevant reviews for quality problems from 5 of
  10 to 9 of 10, and it finds the delivery complaints above.
- **Answers describe the retrieved reviews,** not how common something is among all customers.
- **A valid citation proves the review exists,** not that it supports the claim. Code checks the
  first, and only a person can check the second.

The most useful next steps are hybrid retrieval (keyword search plus vectors) and retrieving low and
high ratings separately for opinion questions.

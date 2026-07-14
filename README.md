# vlm-project

> **Abstract.** `vlm-project` turns driving footage into a **searchable database of scene
> descriptions**. Point it at a set of clips — the nuScenes dataset, a folder of images, or video —
> and it uses a vision-language model to caption a representative frame of each clip (or several, if
> you enable clustering), storing every description in a local SQLite database with full-text search.
> One command ingests, another searches (*"show me scenes with pedestrians"*). It runs fully offline
> on CPU inside a container, and the database is a single portable file you can query, copy, or
> archive. Three things are configurable: the **dataset** (loader), how **keyframes** are chosen
> (single frame vs. clustering), and the **VLM backend** (local BLIP by default, or an API model).

![Real nuScenes v1.0-mini front-camera keyframes with the BLIP captions this tool produced](docs/assets/nuscenes_captions.png)

*The image above is a genuine artifact of a real run — the actual `CAM_FRONT` keyframes of all 10
nuScenes v1.0-mini scenes with the exact BLIP captions this pipeline wrote to the database. Regenerate
it any time with `python scripts/make_readme_assets.py`.*

## What you get, on real data

A single `ingest` over nuScenes v1.0-mini produces 10 rows like these (verbatim from a real run):

| scene | BLIP description |
|---|---|
| scene-0553 | a group of people walking across a crosswalk |
| scene-1077 | a car driving down the road at night |
| scene-0103 | the cars are parked on the street |
| scene-0916 | trees on the side of the road |
| scene-0796 | the road is empty |
| scene-1100 | a car is parked on the side of the road at night |

With `--selector clusters`, a single ~20 s clip is split into a few temporally-distinct sub-scenes,
each captioned from its representative frame:

![Clustering scene-0061 into three sub-scenes](docs/assets/clustering_example.png)

```
        your clips                         vlm-project ingest
   ┌───────────────────┐              ┌──────────────────────────────┐
   │ nuScenes          │              │ 1. load clip                 │
   │ image folder      │ ───────────▶ │ 2. select keyframe(s)        │
   │ video (.mp4)      │              │      single ▸ or ▸ clusters  │
   └───────────────────┘              │ 3. caption with VLM          │
        (configurable                 │      BLIP (CPU) ▸ or ▸ API    │
         loader)                      └──────────────┬───────────────┘
                                                     ▼
                                         ┌───────────────────────┐
                                         │      scenes.db        │
                                         │  SQLite + full-text   │
                                         └───────────┬───────────┘
                                                     │
                            vlm-project query "pedestrian"   │   copy / ship the .db,
                                     ▼                       ▼   or export-json
                             matching scenes            keep for later
```

---

## Installation & quickstart

### Option A — Docker (standalone, recommended)

Nothing to install but Docker. The image bundles the app, CPU-only PyTorch, and the BLIP weights
(baked at build time), so it runs fully offline. The dataset and outputs are mounted at run time.

```bash
# Build once (~a few minutes; downloads CPU torch + ~0.5 GB BLIP weights into the image).
docker build -t vlm-project .

# Verify the image is healthy (deps + baked weights + FTS5).
docker run --rm vlm-project doctor

# Ingest: caption a representative frame of every nuScenes scene into a searchable DB.
docker run --rm \
  -v /path/to/nuscenes:/data:ro \
  -v "$PWD/out:/out" \
  -e DATAROOT=/data -e DB=/out/scenes.db \
  vlm-project ingest -v

# Search the descriptions.
docker run --rm -v "$PWD/out:/out" -e DB=/out/scenes.db \
  vlm-project query "pedestrian crossing"

# Export to a flat JSON file if you prefer.
docker run --rm -v "$PWD/out:/out" -e DB=/out/scenes.db \
  vlm-project export-json --out /out/descriptions.json
```

`out/scenes.db` is an ordinary SQLite file — open it in any SQLite browser, copy it, or ship it.
Each command runs a job and exits; there is no server or daemon.

### Option B — local Python (for development)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Full unit suite — no dataset, no model download, runs in seconds.
pytest

# Once nuScenes v1.0-mini is under ./data:
vlm-project ingest --dataroot ./data --db ./out/scenes.db -v
vlm-project query "truck" --db ./out/scenes.db
```

### Getting the dataset

Download **nuScenes v1.0-mini** (~4 GB) and extract it so the root directly contains
`samples/ sweeps/ maps/ v1.0-mini/`. That root is your `DATAROOT`. It is **not** bundled in the image
or repo. The production acceptance script (below) can fetch it for you.

## Configuration

Set by env var (the container's contract) or the matching CLI flag (which wins):

| Concern | Env var | CLI flag | Default | Values |
|---|---|---|---|---|
| Dataset loader | `LOADER` | `--loader` | `nuscenes` | `nuscenes`, `imagefolder`, `video` |
| Dataset path | `DATAROOT` | `--dataroot` | `./data` | any path |
| nuScenes camera | `CAMERA` | `--camera` | `CAM_FRONT` | any channel |
| Keyframe selection | `SELECTOR` | `--selector` | `single` | `single`, `uniform`, `clusters` |
| Single strategy | `SINGLE_STRATEGY` | `--single-strategy` | `middle` | `middle`, `sharpest` |
| Uniform samples | `UNIFORM_SAMPLES` | `--uniform-samples` | `3` | frames/clip for `uniform` |
| Cluster cut point | `CLUSTER_THRESHOLD` | `--cluster-threshold` | `0.05` | cosine distance |
| VLM backend | `VLM_BACKEND` | `--backend` | `blip` | `blip`, `fake`, `anthropic`, `openai`, `moondream` |
| Database path | `DB` | `--db` | `./out/scenes.db` | any path |
| Ingest stage dir | `STAGE_DIR` | `--stage-dir` | *(set in image)* | local scratch path |

```bash
# Multi-scene: segment each clip into a few sub-scenes (scene-0061#0, #1, ...)
docker run --rm -v /path/to/nuscenes:/data:ro -v "$PWD/out:/out" \
  -e DATAROOT=/data vlm-project ingest --selector clusters -v

# Any image folder — the container is dataset-agnostic.
docker run --rm -v /path/to/pics:/data:ro -v "$PWD/out:/out" \
  -e LOADER=imagefolder -e DATAROOT=/data vlm-project ingest -v
```

> **Tuning clustering.** The cut point is an absolute cosine distance between consecutive frames'
> BLIP embeddings. Those embeddings are highly concentrated (a hard scene cut measures ~0.08 and
> within-scene frames ~0.00), so the default (`0.05`) is deliberately small. Raise it to merge more
> aggressively, lower it to split more finely; it is dataset-dependent.

> **Choosing a selector (and the CPU cost of clustering).** Three options trade cost for coverage:
> `single` describes one representative frame per clip — cheapest, no image decode. `uniform` emits N
> evenly-spaced frames (`--uniform-samples`, default 3 = start/middle/end) for temporal coverage at
> **zero embedding cost** — pure index math. `clusters` gives content-aware segmentation but embeds
> *every* frame (~40/clip) through BLIP's vision encoder; on a CPU-only container that embedding pass
> can rival the cost of just captioning every frame, so its saving over captioning-all is real only
> when clips are long and segments few. Reach for `uniform` when you want a handful of frames per clip
> and don't need learned scene-cut detection; reach for `clusters` when the cut points matter.

## How it works

An `ingest` run flows left to right through four swappable stages, wired together by `pipeline.run()`:
the **loader** turns a data source into clips of frames → the **selector** picks which frame(s) of each
clip to describe → the **backend** (the VLM) turns each chosen frame into a sentence → the **store**
writes one row per scene into a searchable SQLite file. A later `query` reads *only* the store — no
model, no dataset loaded — which is exactly why the heavy `ingest` and the light `query` are separate
commands.

| Stage | What it does |
|---|---|
| **Loader** (`dataset/`) | Turns a source into `ClipItem`s. nuScenes → one clip per scene (its `next`-linked keyframes); image folder → one clip per image; video → sampled frames. |
| **Selector** (`selectors/`) | Picks which frame(s) to describe. `single` → one representative frame per clip. `uniform` → N evenly-spaced frames, zero embedding cost. `clusters` → embed frames with BLIP's own vision encoder, cut temporally-contiguous segments where consecutive-frame cosine distance spikes, caption only each segment's medoid (VLM runs ~3×/clip, not ~40×). |
| **VLM backend** (`vlm/`) | `describe(image) → str`. Default `blip` (CPU, offline, ~0.5 GB). `fake` for tests. `moondream` → instruction-following VLM (~1.5B, opt-in via the `moondream` extra; better comprehension, heavier). `anthropic`/`openai` opt-in via the `api` extra. Only `blip`/`fake` can embed, so `clusters` requires one of them. |
| **Store** (`store.py`) | SQLite with an FTS5 full-text index; `ingest` writes rows, `query` searches them. With `STAGE_DIR` set (default in the image), ingest builds the DB on local scratch and moves the finished file to `--db` in one step, keeping per-row commits off slow/locking bind mounts. |
| **Pipeline** (`pipeline.py`) | `run(loader, selector, backend, store)` — dependency-injected so it's testable with fakes; per-scene errors are logged and skipped. |

Concrete classes are assembled from config in exactly one place, `factory.py`.

## AI & model choices

Two independent AI decisions live here: **which model writes the captions**, and **how we prove the
captions are any good**.

### The captioning models (backends)

Every backend honours one contract — `describe(image) → sentence` — so switching between them is a
config flag (`--backend`), never a code change. Only the local models expose image embeddings, which
the `clusters` selector needs.

| Backend | Size / where it runs | Main advantage | Trade-off | Embeds? |
|---|---|---|---|---|
| **`blip`** *(default)* | ~0.5 GB · local CPU · offline | small and stable; **baked into the image** (no runtime download); doubles as the clustering embedder, so clustering needs no second model | a plain *captioner* — rigid phrasing, tends to name only the single most salient object | ✅ |
| **`moondream`** | ~1.5 B · local CPU · offline *(opt-in)* | *instruction-following* → richer, more scene-aware descriptions, still fully local and offline | larger download, noticeably slower on CPU | ❌ |
| **`anthropic` / `openai`** | cloud API *(opt-in)* | highest caption quality, nothing to run locally | per-call cost, needs network + an API key | ❌ |
| **`fake`** | none | instant and deterministic — runs the whole pipeline with no model or network; powers the fast test suite | not a real caption | ✅ |

*Guiding principle:* default to the **smallest model that is good enough and runs offline**, and make
higher quality a one-flag opt-in rather than a cost everyone pays by default.

### How we know the captions are any good — two independent measures

A generative caption has no single "correct" answer, so we measure quality two ways, each answering a
different question:

- **Grounding — *is this caption right for this frame?*** (`vlm-project verify-accuracy`). nuScenes
  ships no gold captions, so BLEU/CIDEr can't be used on it — but it ships ground-truth 3D boxes, so we
  compare each caption to the objects **actually visible in that frame**. **Precision** asks "did it
  invent an object that isn't there?" — BLIP scores a perfect **1.00**, i.e. it never hallucinates.
  Recall is intentionally low (BLIP names one salient object, not every box), so it is *reported, not
  gated*. This measures **our pipeline on driving data**.
- **BLEU sanity — *has the model itself regressed?*** (`vlm-project bleu-sanity`). A separate axis: a
  library or weights bump could quietly degrade BLIP in general, and grounding wouldn't catch it. So we
  score BLIP on a few **COCO** images — general-domain photos that *do* have gold captions — with
  **BLEU-4**. It is a tripwire: a healthy model scores well above the floor, a broken one scores ~0.
  This measures **the model in general**, independent of driving.

The two catch different failures — grounding catches "the captions don't fit these frames," BLEU
sanity catches "the model stopped working" — and both run automatically in the real-data CI stage.

## Why these choices

A short rationale for the remaining load-bearing decisions (model choices are covered just above). The
full section-by-section design rationale lives in [`docs/DESIGN.md`](docs/DESIGN.md).

**Why SQLite + FTS5.** The deliverable is "a searchable database of descriptions." SQLite makes that a
single portable file — copy it, ship it, archive it, open it in any SQLite browser — with **zero
infrastructure**: no server to run, secure, or scale. Its built-in **FTS5** gives real ranked
full-text search (`MATCH … ORDER BY rank`) natively, so `query` needs only the Python standard
library — no ML stack, no service. A server DB (Postgres) would add operational overhead with no
benefit for a single-file, offline, batch deliverable; a vector DB would add a dependency to solve
semantic search, which is a *possible extension*, not what this tool targets (keyword search over
captions). The store seam leaves that upgrade open. We use a **standalone** FTS table (not
external-content) maintained in `upsert`, because delete-then-insert re-ingest stays idempotent
without the rowid-churn footguns of trigger-driven external-content indexes.

**Why this clustering.** A driving clip is a *time series*, so segments must be **temporally
contiguous** — generic k-means could group non-adjacent frames into one "scene," which is wrong for
video. So we walk frames in order and cut a new segment where the cosine distance between consecutive
frame embeddings spikes past a threshold. The embeddings are **BLIP's own pooled vision features**
(no extra model), and each segment is represented by its **medoid** — an *actual* frame that can be
captioned, unlike a synthetic centroid vector. The default threshold (`0.05`) came from measuring real
distances (a hard scene cut ≈ 0.08, within-scene ≈ 0.00), not a guess, and is exposed as
`--cluster-threshold`. The honest trade: embedding every frame costs CPU, which is why `uniform`
exists as the zero-embedding alternative when you don't need *learned* cut points.

**Why the three selectors.** They trade cost for coverage so you pick per workload: `single` (one
representative frame, cheapest, the core use case), `uniform` (a few evenly-spaced frames, temporal
coverage at zero embedding cost), and `clusters` (content-aware sub-scenes, at the cost of an
embedding pass). Splitting "which frames?" from "load" and "describe" means the same three options
work for nuScenes clips, image folders, and video alike.

## Testing & CI/CD

Layered, fast to slow — unit → integration → real-data soundness → real-data correctness → model sanity:

```bash
pytest                 # unit suite: no dataset, no model, no network (fast)
pytest -m integration  # opt-in: loads real BLIP and captions one image (~0.5 GB download)
ruff check src tests   # lint
```

- **Unit** — dependency injection lets the whole pipeline run against a `FakeBackend` and a fake
  `NuScenes` stand-in; selectors run on scripted embeddings; the store has an FTS5 round-trip test;
  the acceptance checks are themselves unit-tested. No 4 GB dataset, no weights.
- **Integration** — one opt-in test loads real BLIP end to end.
- **Production acceptance** — the real-data gate (below).

### Production acceptance test

One command builds/installs the app, verifies dependencies, obtains the **real** nuScenes v1.0-mini
dataset, ingests real clips end to end (single-frame **and** clustering), asserts the results are
sound, and runs a search — exit code `0` means production-ready.

```bash
# Linux / macOS / CI
./scripts/test_vlm_app_production.sh                 # docker mode (needs a running daemon)
ACCEPTANCE_MODE=local ./scripts/test_vlm_app_production.sh   # no Docker, uses the venv

# Windows
./scripts/test_vlm_app_production.ps1 -Mode local
```

The correctness assertions (`vlm-project verify-db`, backed by `src/vlm_project/acceptance.py`) go
beyond "it ran":

- **scene coverage** — the produced rows cover exactly the dataset's scenes (cross-checked against
  `v1.0-mini/scene.json`).
- **well-formed descriptions** — every caption is non-empty, multi-word, printable.
- **image files exist** — every `image_path` resolves on disk.
- **FTS round-trip** — a distinctive word from each caption finds its own row via search.
- **driving vocabulary** — a healthy fraction of captions mention driving-relevant terms (road,
  street, car, pedestrian, …), a soundness signal that BLIP actually described driving scenes.
- **timings** — inference time is positive and bounded.

### Correctness: object-grounding vs. ground truth (`verify-accuracy`)

Soundness ("is it well-formed?") is not correctness ("is it *right*?"). nuScenes has no gold captions,
so BLEU/CIDEr don't apply — but it has ground-truth 3D boxes, so we measure whether each caption is
**grounded** in the objects actually visible in its CAM_FRONT frame
([`src/vlm_project/grounding.py`](src/vlm_project/grounding.py)):

```bash
vlm-project verify-accuracy --db out/scenes.db --dataroot ./data
```

- **precision** — object families the caption names that are truly in frame (hallucination guard).
- **recall / dominant-family recall** — reported; dominant-recall is gated loosely, plain recall never
  is (a terse captioner names only the most salient object).
- **Gate** = an absolute precision floor **plus regression vs. a committed baseline**
  (`src/vlm_project/baselines/accuracy_baseline.json`, shipped in the package). With pinned weights
  BLIP is deterministic, so this reliably catches a dependency/model bump that degrades grounding.
  Runs as the final production-acceptance stage and uploads `accuracy.json` as a CI artifact.

*Current baseline (BLIP-base, single-frame): precision 1.00, recall 0.17, dominant-recall 0.50 — BLIP
is precise (no hallucinations) but low-salience, which is exactly the signal that motivates the
Moondream / API upgrade paths.*

### Model sanity: BLIP on COCO

A separate axis — has the *model itself* regressed on general-domain images? A small, fixed COCO
caption set is fetched from the **official** source (not vendored) and BLIP is scored with BLEU-4
([`src/vlm_project/bleu.py`](src/vlm_project/bleu.py), gate in
[`src/vlm_project/sanity.py`](src/vlm_project/sanity.py)). This is the **final stage of the production
acceptance flow**, so it runs for real in the `nuscenes-e2e` CI job — and it **skips gracefully when
COCO can't be fetched** (offline), never blocking acceptance for lack of network.

Run it standalone too:

```bash
python scripts/fetch_coco_sanity.py                 # one-time fetch (needs network)
vlm-project bleu-sanity --data tests/data/coco_sanity      # BLEU-4 vs a conservative floor
# or via pytest (self-skips until fetched):
pytest -m integration tests/test_coco_sanity.py
```

### GitHub Actions (`.github/workflows/ci.yml`)

| Job | Trigger | What it proves |
|---|---|---|
| `lint-and-unit` | every push / PR | ruff + full unit suite, installed without the heavy ML deps → fast. |
| `docker-smoke` | every push / PR | image builds, `doctor` is healthy, a fake-backend ingest+query works **without** the dataset. |
| `nuscenes-e2e` | manual (`workflow_dispatch`) | full production acceptance on the **real** 4 GB dataset (cached): ingest → soundness (`verify-db`) → **correctness** (`verify-accuracy`, object-grounding gate) → **model-sanity** (BLIP BLEU-4 on an official COCO micro-set). Uploads the `.db` files and `accuracy.json` as artifacts. |

## Deployment

It is a **CLI batch job**, and that job is the deployment unit:

1. **Local** — run the container as above; the output is a portable `scenes.db`.
2. **Scale-out (same image, no code change)** — run `ingest` as a batch job on any scheduler (Cloud
   Run Job, AWS Batch, Kubernetes `Job`), reading the dataset from object storage and writing the DB
   back. It's stateless and CPU-only, so it shards horizontally by scene and needs no GPU. Set
   `-e VLM_BACKEND=anthropic` (with the `api` extra and a key) when higher-quality captions are worth
   the cost.

   > **How sharding would actually work.** Today each `ingest` process builds the full `NuScenes`
   > metadata index and iterates *all* scenes — fine for v1.0-mini, but on the full dataset every
   > worker would parse the whole index to process its slice. A true scale-out separates concerns: a
   > one-time **orchestration step** parses the metadata once and emits a **manifest** (per-scene
   > image paths / tokens); stateless workers each take a manifest shard, caption it, and write a
   > partition of the DB (or JSONL), which a final step merges. The loader/selector/backend seams
   > already support this — it's an added orchestration layer, not a rewrite.
3. **A live search service**, if ever wanted, is a thin read-only API over the produced DB — a small
   later add, not a rewrite, precisely because ingest and query are already separate.

## Assumptions

- You provide nuScenes v1.0-mini yourself (account/EULA) and mount it; it is not redistributed here.
- Default output is **one representative `CAM_FRONT` frame per scene**; camera and selector are
  configurable. `--selector uniform` gives a few evenly-spaced frames per clip at no embedding cost,
  and `--selector clusters` opts into content-aware sub-scenes.
- A description is **one short sentence**; BLIP-base (~0.5 GB) is the default for its stability and
  small CPU footprint, and its caption quality is accepted as good enough for this use case (a
  small/basic VLM is sufficient here). `moondream` (an instruction-following ~1.5B VLM) is
  available when better scene comprehension is worth a larger download and slower CPU inference; the
  cloud `anthropic`/`openai` backends are there for top quality at per-call cost.
- **Clustering pays a CPU embedding cost**: it runs every frame through BLIP's vision encoder, which
  on a CPU-only container can approach the cost of captioning every frame. `uniform` is the cheaper
  path when learned scene-cut detection isn't needed. Clustering is also single-camera and its
  threshold is a heuristic; multi-camera fusion and learned scene-boundary detection are out of scope,
  but the selector interface leaves room for them.
- Backends that can't embed (`moondream`, `anthropic`, `openai`) pair with `single`/`uniform`; the
  factory rejects `clusters` for them with a clear message rather than failing deep in the run.

## Project layout

```
src/vlm_project/
  config.py    models.py    store.py     pipeline.py   factory.py   cli.py
  acceptance.py  doctor.py
  dataset/   nuscenes_loader · imagefolder_loader · video_loader
  selectors/ single · uniform · clusters
  vlm/       blip · fake · moondream · api
scripts/   production_acceptance.py · test_vlm_app_production.{sh,ps1} · make_readme_assets.py · bake_weights.py
tests/     loaders · selectors · store · pipeline · cli · api · acceptance · integration
.github/workflows/ci.yml
Dockerfile · docker-compose.yml · Makefile · pyproject.toml · requirements.lock
```

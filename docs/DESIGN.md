# Design Rationale — Scene Description via VLM

This document explains the design of `vlm-project` section by section: what each part does and why it
is built the way it is, including the trade-offs weighed and the alternatives considered. It
complements the README (which covers usage) by focusing on the *why*.

---

## 1. Overall shape

### 1.1 A CLI batch job, not a service

The deployment unit is a container that runs one command (`ingest`, `query`, `export-json`, …) and
exits — there is no daemon or API server. The problem is a batch transform (turn clips into
descriptions) plus lookups, and a batch job is the simplest thing that fits: it is trivially
schedulable (cron, AWS Batch, Kubernetes `Job`) and has no idle cost or uptime surface to secure.
Interactive search is still fast, because `query` is a sub-second CLI over the SQLite full-text index
and the database is a portable file. A live search service, if ever needed, is a thin read-only layer
over that database; it is left as a later addition rather than built now, and stays cheap to add
precisely because ingest and query are already separate.

### 1.2 Separate `ingest` and `query` commands

Writing the database and reading it are distinct entrypoints because they have different lifecycles
(write once or occasionally, read often), different dependencies (ingest needs torch/transformers;
query needs only the standard-library `sqlite3`), and different scaling needs. The separation lets
`query` run in a tiny environment with no ML stack, and lets `ingest` shard horizontally while the
database remains a single read target.

---

## 2. Architecture & seams

### 2.1 Four explicit stages behind a pipeline

The pipeline is `run(loader, selector, backend, store)` in `pipeline.py`, with each collaborator behind
a small `Protocol` (`dataset/base.py`, `selectors/base.py`, `vlm/base.py`). The three axes that need to
be configurable — dataset, keyframe choice, and model — map one-to-one onto three swappable interfaces,
and passing them in as arguments (dependency injection) makes the whole pipeline testable with fakes:
no dataset, no model, no network. Each interface is only one or two methods, so the abstraction cost is
negligible. `Protocol` (structural typing) is used rather than abstract base classes so a fake backend
or a scripted embedder is valid without inheriting anything — ideal for tests — while
`@runtime_checkable` still lets the factory verify capabilities such as "can this backend embed?".

### 2.2 A single wiring point: `factory.py`

All mapping from config strings to classes (`backend == "blip"`, `selector == "clusters"`, …) lives in
one module, and heavy imports are deferred into the branch that needs them, so importing the CLI does
not import torch. Concentrating the wiring keeps the pipeline, CLI, and tests free of branching and
gives one obvious place to enforce cross-cutting rules such as the clustering/embedding capability
check. A plugin or entry-point registry would be the right move once third parties add backends
out-of-tree; with a handful of in-tree implementations, an explicit factory is more readable and
greppable.

### 2.3 Configuration: environment first, CLI override second

`Config.from_env()` reads environment variables — the container's natural contract — and any explicitly
passed CLI flag takes precedence, giving the order **CLI flag > environment variable > built-in
default**. Containers are configured by environment while humans at a terminal prefer flags; supporting
both collapses to a single `Config` dataclass with one precedence rule. An environment-only design
would be hostile to local development, and a flag-only design would be hostile to the container
contract.

---

## 3. Data model

### 3.1 A small dataclass contract

`Frame`, `ClipItem`, `SceneItem`, and `SceneDescription` are the single serialization source: the
SQLite rows and the JSON export both derive from `SceneDescription`, so JSON, the database, and the
tests all agree on one shape and cannot drift. The input types are frozen (hashable and safe to pass
around); the terminal record is mutable only where convenient. Plain dictionaries were avoided because
dataclasses give named, typed fields and one `to_dict()` boundary, whereas dicts scatter key strings
across the codebase and invite typos.

### 3.2 `scene_id` primary key with a `#N` segment suffix

Single-frame selection uses the clip id verbatim (`scene-0061`); multi-frame selectors append a segment
index (`scene-0061#0`). A stable, human-readable primary key makes re-ingest idempotent (upsert by
`scene_id`) and keeps the default "one row per scene" literally one row while extending cleanly to
sub-scenes. The `#` suffix is an opaque uniqueness/display convenience, never parsed for meaning
downstream; `clip_id` is stored separately for grouping.

---

## 4. Dataset loaders

### 4.1 nuScenes: one clip per scene

`NuScenesLoader` walks each scene's `next`-linked sample chain, collecting one camera image per
keyframe, and hands the full frame list to the selector — matching nuScenes' own structure and leaving
"which frame(s)" entirely to the selector, so the loader stays a pure adapter. The devkit `NuScenes`
object is injected rather than constructed internally, so tests drive the loader with a tiny fake
exposing only the three methods used (`scene`, `get`, `get_sample_data`); the loader is exercised on
kilobytes, with no 4 GB dataset and no devkit dependency in the unit tests.

One known trade-off: each process builds the full metadata index and iterates all scenes. That is
trivial for v1.0-mini but would be inefficient at full-dataset scale, where every worker would parse
the whole index to process its shard. A true scale-out separates a one-time orchestration step that
emits a per-scene manifest from stateless workers that each caption a shard; the interfaces already
allow this, so it is documented as the scaling path (see the README Deployment section) rather than
built, since distributed orchestration for a 10-scene dataset would be speculative.

### 4.2 Dataset-agnostic loaders

Every image is a one-frame clip, and every video becomes a clip of stride-sampled frames written to
scratch JPEGs. Normalizing all sources to "clips of frames with image paths" means the selector,
backend, and store never need to know where frames came from, and video input reuses the entire
keyframe/clustering machinery unchanged. Writing sampled video frames to disk keeps that single
invariant (downstream deals only in image paths), and the scratch directory is disposable; an
in-memory path is a straightforward optimization behind the same interface if profiling ever justifies
it.

### 4.3 `CAM_FRONT` by default

The front camera is the canonical "what is this driving scene" view and the one most captions describe
well, so it is the default; `--camera` selects any other channel. Multi-camera fusion is a genuinely
different problem with different caption semantics, so it is left out of scope with the interface open
rather than half-built.

---

## 5. Selectors

Selecting *which* frame(s) of a clip to describe is a concern distinct from loading the clip, so it is
its own stage. Three selectors trade cost for coverage.

### 5.1 `single` (default)

`single` picks one representative frame. The default strategy `middle` takes the temporal midpoint with
zero image decoding — a robust, free default for a ~20-second clip. The optional `sharpest` strategy
chooses the least-blurred frame by variance-of-the-Laplacian, avoiding a motion-blurred keyframe when
quality matters. That focus measure is implemented directly with numpy array shifts rather than pulling
in OpenCV, keeping the core install light; it is a standard, well-understood metric and only a few
lines.

### 5.2 `uniform`

`uniform` emits N evenly-spaced frames by index arithmetic alone (`--uniform-samples`, default 3 =
start/middle/end); `n=1` collapses to the middle frame with no suffix. It fills the gap between `single`
(one frame, cheap) and `clusters` (temporal coverage but a full embedding pass): it gives temporal
coverage for free — no model, no image decoding, deterministic. Frame positions are spaced across
`[0, n-1]` with round-half-to-even, clamped to the available frame count and de-duplicated, so a short
clip asked for more samples than it has yields distinct frames rather than duplicates. This behavior is
unit-tested.

### 5.3 `clusters`

`clusters` embeds every frame (using the injected embedder — BLIP's vision encoder in production),
starts a new temporally-contiguous segment wherever the cosine distance between consecutive frames
exceeds a threshold (default `0.05`), and captions only each segment's medoid. This yields content-aware
sub-scenes while running the VLM roughly 3×/clip instead of ~40×. Segmentation is kept contiguous in
time because a clip is a time series — generic k-means could group non-adjacent frames into one "scene,"
which is wrong for video. The default threshold comes from measuring real distances (a hard scene cut ≈
0.08, within-scene frames ≈ 0.00), not a guess, and is exposed as `--cluster-threshold` since it is
dataset-dependent. The representative is the medoid — an actual frame that can be captioned — rather
than a synthetic centroid vector. Reusing the caption model as the embedder avoids loading a second
model; a dedicated embedding model would be higher quality, and the `Embedder` interface allows
swapping one in without touching the selector. The embedding pass does cost CPU, which is exactly why
`uniform` exists as the cheaper alternative when learned cut points are not needed.

---

## 6. VLM backends

All backends share one contract — `describe(image) → str` — so the choice is config, not code, and the
pipeline never changes when models are swapped. At a glance:

| Backend | Footprint / locus | Chief advantage | Cost | Embeds? |
|---|---|---|---|---|
| `blip` *(default)* | ~0.5 GB, local CPU, offline | small, stable, bakeable into the image; reused as the clustering embedder | rigid captioner, low salience | yes |
| `moondream` | ~1.5 B, local CPU, offline | instruction-following, richer scene comprehension | larger download, slower CPU | no |
| `anthropic` / `openai` | cloud API | best caption quality, no local compute | per-call fee, network, key | no |
| `fake` | none | deterministic and instant — the test workhorse | not a real caption | yes |

How captions are *measured* — object-grounding (correctness) and COCO BLEU (model-sanity) — is covered
in §10.5–10.6.

### 6.1 BLIP-base as the default

`Salesforce/blip-image-captioning-base` (~0.5 GB) runs offline on CPU, is small enough to bake into the
image, and produces adequate one-sentence captions. It is loaded lazily so imports and tests stay cheap,
with `num_beams=3` and `max_new_tokens=30` as a modest quality/latency setting (inference time is
recorded per row, so regressions are visible). BLIP is a captioner rather than an instruction-following
model, so its phrasing is somewhat rigid; it stays the default for stability and footprint, with
`moondream` available when richer comprehension is worth the cost.

### 6.2 `fake`

A deterministic, dependency-free backend whose `describe` is a function of image size and whose `embed`
is a stable hash of pixels. It lets the entire pipeline — CLI, factory, selector, store — run in
milliseconds with no model and no network, which is what makes the unit suite fast and hermetic. It
intentionally does not exercise the real model; that is the job of the opt-in `integration` tests and
the real-data CI job, kept separate so the fast suite stays fast.

### 6.3 `anthropic` / `openai` cloud backends

Optional `describe`-only backends behind the `api` extra, with an injectable client so they are
unit-tested against a mock (no key, no network). They are a drop-in quality upgrade when per-call cost
is acceptable, and change nothing else because they honor the same `describe` contract. They do not
produce embeddings, so the factory rejects `clusters` for them up front with a clear message.

### 6.4 `moondream` local backend

Optional local Moondream2 (~1.5 B) behind a `moondream` extra: an instruction-following model tuned for
CPU/edge that gives more scene-aware descriptions while staying local and offline, with BLIP remaining
the default. It is opt-in because it is a larger download and slower on CPU. Its weights and remote
modeling code are pinned to a specific Hub revision so behavior stays reproducible across image
rebuilds — a supply-chain hygiene point. Like the cloud backends it only captions, so `clusters` is
rejected for it by the same factory guard.

### 6.5 Embedding as an optional capability

`describe` is the core contract; producing embeddings is a separate `Embedder` protocol, and the
factory checks `isinstance(backend, Embedder)` before building `clusters`. Most backends only caption,
so requiring every backend to implement `embed` would be dishonest for the cloud and Moondream
backends. The capability check is explicit, testable, and happens at wiring time — before any work — so
an incompatible combination fails immediately with a helpful message rather than deep in a run.

---

## 7. Store

### 7.1 SQLite + FTS5, a single portable file

The deliverable is a searchable database of descriptions, so the store is a single `.db` file with a
`descriptions` table and an FTS5 full-text index queried with `MATCH … ORDER BY rank`. SQLite is
zero-infrastructure and portable — the whole database is one file to copy, ship, or archive — and FTS5
provides ranked full-text search natively, so `query` needs only the standard library. A server
database (Postgres) would add operational overhead with no benefit for a single-file, offline, batch
deliverable; a vector database would add a dependency to solve semantic search, which is a plausible
future extension but not the current goal (keyword search over captions), and the store interface
leaves room for it. The FTS table is standalone rather than external-content: external-content FTS5
couples the index to base-table rowids via triggers, with churn and rebuild hazards on
delete-then-insert, whereas a standalone table maintained in `upsert` keeps re-ingest idempotent and
simple.

### 7.2 Idempotent `upsert`

Each write deletes any existing row and FTS entry for that `scene_id` and then inserts, so re-running
ingest over the same data yields the same database (no duplicates) and the index never drifts from the
base table. Explicit delete-then-insert on both tables is slightly heavier than a single UPSERT, but
with a standalone FTS table the two must be kept in lock-step, and this is the clearest way to guarantee
that; ingest volume is small, so correctness is preferred over the micro-optimization.

### 7.3 Staged writes

When `STAGE_DIR` is set (the default in the Docker image), ingest builds the database on
container-local scratch and moves the finished file to its destination in a single operation on clean
completion; a failed run never publishes a partial over an existing good database. This exists because
many small per-row commits directly onto a bind-mounted host volume (notably on Windows under WSL2 /
Docker Desktop) risk severe slowdown, locking, and corruption — building locally and publishing once
avoids all of it. The publish is done safely across filesystems: the staging directory and the
destination are usually different filesystems, so the file is copied into the destination directory and
then `os.replace`d onto the final name, which is atomic within that filesystem, so a reader never sees a
half-written database. Staging is opt-in because local development, where the database is not on a bind
mount, gains nothing from the extra copy; read commands never stage. The trade-off is that a crash
mid-ingest discards the in-progress staged database and keeps the previous good one — the safe outcome,
though not resumable. Streaming durability (append to JSONL, then rebuild) was considered and deferred
as more machinery than this workload warrants.

---

## 8. CLI

One binary exposes `ingest`, `query`, `export-json`, `doctor`, `verify-db`, `verify-accuracy`, and
`bleu-sanity`, with `--db` and `--verbose` shared and per-command `--help`. `-v` is accepted before or
after the subcommand — natural usage that argparse does not provide for free — via a small amount of
explicit handling that is pinned by a regression test.

---

## 9. Packaging & deployment

### 9.1 CPU image with baked weights

The image installs CPU-only torch from its dedicated index, installs pinned runtime dependencies from
`requirements.lock`, and bakes the BLIP weights at build time; the dataset and outputs are never baked.
The result is a self-contained, offline, reproducible image that runs anywhere with no GPU and no
runtime model download. Baking the ~0.5 GB weights buys offline, deterministic startup (no first-run
download, no production network dependency); the genuinely large thing — the dataset — stays out of the
image and is mounted at run time, which is both correct (licensing, size) and clean (immutable image,
external state). The target is CPU inference on a small model, so there is no GPU path; nothing
precludes adding a GPU variant later.

### 9.2 Optional extras

Heavy or rarely-used dependencies live behind extras (`api`, `video`, `moondream`, `dev`) so the core
install stays light — you do not need OpenCV to caption images or the Anthropic SDK to run BLIP. Each
extra maps to one lazily-imported code path, and the factory raises a clear error if a selected
backend or loader's extra is missing.

### 9.3 `requirements.lock` alongside `pyproject.toml`

`pyproject.toml` keeps dependency ranges for library-style installs, while `requirements.lock` pins
exact runtime versions for reproducible image builds. CI builds the image from the lock, so any drift
between the two surfaces as a build failure. Pinning also has to account for transitive constraints:
`nuscenes-devkit` governs the Shapely and matplotlib versions, so the lock pins a devkit release whose
transitive dependencies ship prebuilt wheels for the image's Python — otherwise those packages would be
compiled from source in the toolchain-free slim image.

---

## 10. Testing, CI/CD, and acceptance

### 10.1 Fast hermetic unit suite

The default `pytest` run uses no dataset, model, or network — the fake backend and a fake devkit
stand-in let the whole pipeline run in seconds — while `pytest -m integration` opts into the real BLIP
weights. Fakes verify wiring and logic; they do not exercise the real model, which is deliberately the
job of the integration tests and the real-data CI job, kept separate so the fast suite stays fast and
deterministic.

### 10.2 CI layers

Three layers run cheapest-first: `lint-and-unit` (ruff + the unit suite, no heavy deps) and
`docker-smoke` (build the image and run `doctor` plus a fake-backend ingest/query, no dataset) on every
push; and `nuscenes-e2e` (the full real-data acceptance) manually via `workflow_dispatch`. Every-push
feedback stays fast and free while still proving the shipping image builds and runs end-to-end; the
4 GB real-data run is deliberate and on-demand, since downloading it on every push would be wasteful.
The real-data job uploads the produced databases and `accuracy.json` as artifacts for inspection.

### 10.3 Soundness checks (`verify-db`)

`acceptance.py` is pure standard library so it runs on the host, in the container, or in CI without
torch/nuscenes. It asserts that a produced database is *sound*: row coverage matches the dataset's
`scene.json`, every description is well-formed, timings are positive and bounded, image paths exist, the
full-text index round-trips, and a healthy fraction of captions contain plausible driving vocabulary.
Because a generative caption has no single correct answer, these are structural and heuristic checks
rather than exact matches. The vocabulary check is a set-level soundness signal (a fraction of scenes
must mention driving terms), not a per-scene grade — it catches gross failures such as empty or garbage
output. The round-trip check searches with the limit set to the row count, verifying that the index
*contains* each row rather than that it ranks in some top-N.

### 10.4 `doctor`

`doctor` verifies the runtime environment: Python version, core and ML imports, a working FTS5, devkit
presence, and whether the BLIP weights are cached (a warning, not a failure). It turns "why did it fail"
into a one-command diagnosis in any environment and distinguishes critical failures (non-zero exit) from
a cold weight cache. It complements the tests, which validate code, by validating the environment — the
thing that actually varies in deployment.

### 10.5 Correctness: object-grounding (`verify-accuracy`)

Soundness ("is it well-formed?") is not correctness ("is it right?"). nuScenes ships no gold captions,
so BLEU/CIDEr/SPICE do not apply, but it ships ground-truth 3D boxes — so for each caption the object
families it names are compared to the ones actually visible in that CAM_FRONT frame (boxes projected via
the devkit). The gate is an absolute precision floor (no hallucinated objects) plus regression against a
committed, package-shipped baseline; it runs as an acceptance stage and emits `accuracy.json`.

Precision is gated but recall is not. A terse captioner names only the most salient object, so recall
against every visible box is intrinsically low and noisy, and a hard recall floor would be an arbitrary
value judgment; precision (does it invent objects?) is the trustworthy signal, dominant-family recall is
gated only loosely, and plain recall is reported. The gate is reliable despite using a generative model
because, with weights and dependencies pinned, BLIP's beam search is deterministic (observed in practice
as byte-identical captions across independent runs), so regression-vs-baseline dependably flags a
dependency or model change that degrades grounding; grounding metrics are also timing-independent,
unlike the soundness timing check. A distant, un-annotated object could in principle count as a false
positive, but precision is empirically a clean 1.00 and the baseline tolerance absorbs edge cases. The
baseline ships inside the package so `verify-accuracy` finds it wherever the image runs, and each
rebuilt image carries the last-known-good numbers.

### 10.6 Model sanity: COCO BLEU (`bleu-sanity`)

This is a separate axis from grounding: has the model itself regressed on general-domain images? A
torch/transformers/weights bump could quietly degrade BLIP in a way that grounding on driving scenes
would not catch. To measure that, a small fixed COCO caption set is fetched from the official source
(never vendored) and BLIP is scored with a dependency-free BLEU-4 against a conservative floor; the
stage self-skips when COCO cannot be fetched. COCO is used because it has real gold reference captions,
which nuScenes lacks. The data is fetched at run time rather than vendored for licensing and size
reasons — and because fabricating pairs would be fake benchmark data — mirroring how the 4 GB nuScenes
set is handled. BLEU is hand-rolled (~40 lines, unit-tested on synthetic strings) rather than pulling in
`sacrebleu`/`nltk` and the COCO eval toolkit (whose SPICE metric needs Java), keeping the core install
light. The floor is a broken-model tripwire, not a quality target — a working BLIP scores well above it,
a broken one scores near zero — and the real operating point is recorded from CI runs. The scoring core
(`run_bleu_sanity`) is shared by the CLI command, the pytest test, and the acceptance stage, so there is
a single implementation.

---

## 11. Known limitations and future work

- **Caption quality is bounded by the chosen model.** BLIP is adequate rather than excellent; the
  grounding metrics quantify this — it is precise (no hallucinated objects) but low-salience (it names
  only the most prominent object, so recall is low) — which is the motivation for the richer
  Moondream/API backends and the `uniform`/`clusters` selectors.
- **The model-sanity floor is a tripwire, not a quality bar.** It catches a broken model, not subtle
  drift; its real operating point comes from CI runs.
- **Clustering is single-camera and its threshold is a heuristic.** Multi-camera fusion and learned
  scene-boundary detection are out of scope, with the interface left open.
- **Scale-out is documented, not implemented.** Each worker currently parses the full metadata index;
  the manifest/orchestration split is the intended path, reasonable to defer at mini-dataset scale.
- **Search is lexical (FTS), not semantic.** "pedestrian" will not match "person walking" unless the
  caption says so; a vector index is the natural upgrade and the store interface allows it.
- **Staged writes trade a crash's in-progress work for safety.** A crash mid-ingest keeps the previous
  good database and discards the partial — intentional, but not resumable.

The default path — nuScenes + `single`/`middle` + BLIP, non-staged locally — is the simplest route;
the additional selectors, backends, staging, and quality gates are all opt-in or additive.

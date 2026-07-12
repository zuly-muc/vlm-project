# vlm-project

> **Abstract.** `vlm-project` turns driving footage into a **searchable database of scene
> descriptions**. Point it at a set of clips — the nuScenes dataset, a folder of images, or video —
> and it uses a vision-language model to caption a representative frame of each clip (or several, if
> you enable clustering), storing every description in a local SQLite database with full-text search.
> One command ingests, another searches (*"show me scenes with pedestrians"*). It runs fully offline
> on CPU inside a container, and the database is a single portable file you can query, copy, or
> archive. Three things are configurable: the **dataset** (loader), how **keyframes** are chosen
> (single frame vs. clustering), and the **VLM backend** (local BLIP by default, or an API model).

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

## Quickstart (Docker — nothing to install but Docker)

```bash
# Build once. This bakes the ~0.5 GB BLIP weights into the image,
# so runs need no network and CPU-only torch keeps the image lean.
docker build -t vlm-project .

# 1) Ingest: caption a representative frame of every nuScenes scene into the DB.
#    Mount your dataset read-only and an output dir for the DB.
docker run --rm \
  -v /path/to/nuscenes:/data:ro \
  -v "$PWD/out:/out" \
  -e DATAROOT=/data -e DB=/out/scenes.db \
  vlm-project ingest -v

# 2) Query: full-text search the descriptions.
docker run --rm -v "$PWD/out:/out" -e DB=/out/scenes.db \
  vlm-project query "pedestrian crossing"

# 3) Export everything to JSON if you prefer a flat file.
docker run --rm -v "$PWD/out:/out" -e DB=/out/scenes.db \
  vlm-project export-json --out /out/descriptions.json
```

The output `out/scenes.db` is a normal SQLite file — open it in any SQLite browser, copy it, or ship
it. There is no server and no daemon: each command runs a job and exits.

## Quickstart (local, for development)

```bash
pip install -e ".[dev]"

# Full unit suite — no dataset, no model download, runs in seconds.
pytest

# Once nuScenes v1.0-mini is under ./data:
vlm-project ingest --dataroot ./data --db ./out/scenes.db -v
vlm-project query "truck" --db ./out/scenes.db
```

## Getting the dataset

Download **nuScenes v1.0-mini** (~4 GB) from <https://www.nuscenes.org/download> (free account /
EULA) and extract it so the root directly contains `samples/ sweeps/ maps/ v1.0-mini/`. That root is
your `DATAROOT`. The dataset is **not** included in this repo or the image — it is mounted at runtime.

## Configuration

Everything is set by env var (the container's contract) or the matching CLI flag (which wins):

| Concern | Env var | CLI flag | Default | Values |
|---|---|---|---|---|
| Dataset loader | `LOADER` | `--loader` | `nuscenes` | `nuscenes`, `imagefolder`, `video` |
| Dataset path | `DATAROOT` | `--dataroot` | `./data` | any path |
| nuScenes camera | `CAMERA` | `--camera` | `CAM_FRONT` | any channel |
| Keyframe selection | `SELECTOR` | `--selector` | `single` | `single`, `clusters` |
| Single strategy | `SINGLE_STRATEGY` | `--single-strategy` | `middle` | `middle`, `sharpest` |
| Cluster cut point | `CLUSTER_THRESHOLD` | `--cluster-threshold` | `0.05` | cosine distance |
| VLM backend | `VLM_BACKEND` | `--backend` | `blip` | `blip`, `fake`, `anthropic`, `openai` |
| Database path | `DB` | `--db` | `./out/scenes.db` | any path |

### Handy variations

```bash
# Multi-scene: segment each clip into a few sub-scenes (scene-0061#0, #1, ...)
docker run --rm -v /path/to/nuscenes:/data:ro -v "$PWD/out:/out" \
  -e DATAROOT=/data vlm-project ingest --selector clusters -v

# Any image folder — the container is dataset-agnostic.
docker run --rm -v /path/to/pics:/data:ro -v "$PWD/out:/out" \
  -e LOADER=imagefolder -e DATAROOT=/data vlm-project ingest -v

# A better single-frame pick (least motion blur).
vlm-project ingest --single-strategy sharpest --dataroot ./data
```

> **Tuning clustering.** The cut point is an absolute cosine distance between
> consecutive frames' BLIP embeddings. Those embeddings are highly concentrated
> — in a synthetic clip, a hard scene cut measures ~0.08 and within-scene frames
> ~0.00 — so the default (`0.05`) is deliberately small. Raise it to merge more
> aggressively, lower it to split more finely; it is dataset-dependent.

## How it works

| Stage | What it does |
|---|---|
| **Loader** (`dataset/`) | Turns a source into `ClipItem`s. nuScenes → one clip per scene (its `next`-linked keyframes); image folder → one clip per image; video → sampled frames. |
| **Selector** (`selectors/`) | Picks which frame(s) to describe. `single` → one representative frame per clip. `clusters` → embed frames with BLIP's own vision encoder, cut temporally-contiguous segments where consecutive-frame cosine distance spikes, caption only each segment's medoid (VLM runs ~3×/clip, not ~40×). |
| **VLM backend** (`vlm/`) | `describe(image) → str`. Default `blip` (CPU, offline). `fake` for tests. `anthropic`/`openai` opt-in via the `api` extra. |
| **Store** (`store.py`) | SQLite with an FTS5 full-text index; `ingest` writes rows, `query` searches them. |
| **Pipeline** (`pipeline.py`) | `run(loader, selector, backend, store)` — dependency-injected so it's testable with fakes; per-scene errors are logged and skipped. |

Concrete classes are assembled from config in exactly one place, `factory.py`.

## Testing

```bash
pytest                 # unit suite: no dataset, no model, no network (fast)
pytest -m integration  # opt-in: loads real BLIP and captions one image (~0.5 GB download)
ruff check src tests   # lint
```

- **Dependency injection** lets the whole pipeline run against a `FakeBackend` and a fake `NuScenes`
  stand-in (a short `next`-linked keyframe chain built in `tests/conftest.py`) — no 4 GB dataset, no
  weights. Tiny JPEGs are generated on the fly with Pillow.
- **Selectors** are tested with scripted embeddings, so clustering logic is verified deterministically
  without BLIP.
- **Store** has an FTS5 round-trip test and an idempotent-upsert test.
- **API backends** are tested with a mocked client — never a real call.

## Deployment

It is a **CLI batch job**, and that job is the deployment unit:

1. **Local** — run the container as above; the output is a portable `scenes.db`.
2. **Scale-out (same image, no code change)** — run `ingest` as a batch job on any scheduler (Cloud
   Run Job, AWS Batch, Kubernetes `Job`), reading the dataset from object storage and writing the DB
   back. It's stateless and CPU-only, so it shards horizontally by scene and needs no GPU. Set
   `-e VLM_BACKEND=anthropic` (with the `api` extra and a key) when higher-quality captions are worth
   the cost.
3. **A live search service**, if ever wanted, is a thin read-only API over the produced DB — a small
   later add, not a rewrite, precisely because ingest and query are already separate.

## Assumptions

- You provide nuScenes v1.0-mini yourself (account/EULA) and mount it; it is not redistributed here.
- Default output is **one representative `CAM_FRONT` frame per scene**; camera and selector are
  configurable, and `--selector clusters` opts into a few sub-scenes per clip.
- A description is **one short sentence**; BLIP's caption quality is accepted as good enough for this
  use case (per the task's "a small/basic VLM is fine" note).
- Clustering is single-camera and its threshold is a heuristic; multi-camera fusion and learned
  scene-boundary detection are out of scope, but the selector interface leaves room for them.

## Project layout

```
src/vlm_project/
  config.py      models.py      store.py      pipeline.py     factory.py     cli.py
  dataset/   nuscenes_loader · imagefolder_loader · video_loader
  selectors/ single · clusters
  vlm/       blip · fake · api
tests/         loaders · selectors · store · pipeline · cli · api · integration
Dockerfile · docker-compose.yml · Makefile · pyproject.toml · requirements.lock
```

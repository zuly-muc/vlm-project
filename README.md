# vlm-project

> **Abstract.** `vlm-project` turns footage into a **searchable database of scene descriptions**. Point
> it at a source (the nuScenes driving dataset, a folder of images, or video files) and it captions a
> representative frame of each clip with a vision-language model, storing every description in a local
> SQLite database with full-text search. One command ingests, another searches. It runs fully offline
> on CPU inside a container, and the database is a single portable file you can query, copy, or archive.

![A nuScenes frame captioned by BLIP](docs/assets/hero.png)

*An example result: a nuScenes v1.0-mini front-camera frame, described by the local BLIP model.*

## What it is

- **Ingests** the nuScenes dataset, a folder of images, or video files (`.mp4/.mov/.avi/.mkv`), with no code change.
- **Selects** the frame(s) to describe per clip: one representative frame, a few evenly-spaced, or content-aware sub-scenes.
- **Captions** each frame with a local vision-language model: **BLIP** by default, or optional **Moondream2**. Both run offline on CPU.
- **Stores** every caption in one **SQLite file with FTS5 full-text search**, queryable by the CLI or any SQLite tool and portable as a single file.
- **Runs offline on CPU** as a batch job, no server. Two commands: `ingest` and `query` (plus `export-json` and `doctor`).

```
        your input                          vlm-project ingest
   ┌───────────────────┐              ┌──────────────────────────────┐
   │ nuScenes          │              │ 1. load clip                 │
   │ image folder      │ ───────────▶ │ 2. select keyframe(s)        │
   │ video (.mp4)      │              │      single ▸ uniform ▸ clusters
   └───────────────────┘              │ 3. caption with a local VLM  │
        (the loader)                  │      BLIP ▸ or ▸ Moondream    │
                                      └──────────────┬───────────────┘
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

## Install and use

**Prerequisites:** Docker. The image is **~4 GB** (it bundles CPU PyTorch and the baked BLIP weights so
it runs offline) and is **CPU-only**, no GPU. There is no prebuilt image; you build it from this repo.

**Build:**

```bash
docker build -t vlm-project .          # a few minutes, one time
docker run --rm vlm-project doctor     # verify deps + weights + FTS5 are healthy
```

**Try it in 30 seconds** (no dataset): point it at a folder of your own images.

```bash
mkdir -p pics out            # drop a few .jpg / .png files into ./pics
docker run --rm -v "$PWD/pics:/data:ro" -v "$PWD/out:/out" \
  -e LOADER=imagefolder -e DATAROOT=/data -e DB=/out/demo.db vlm-project ingest -v
docker run --rm -v "$PWD/out:/out" -e DB=/out/demo.db vlm-project query car
```

Results land in `./out/demo.db` on your machine, an ordinary SQLite file.

> **Windows (PowerShell):** use `${PWD}` and Windows source paths (for example `-v ${PWD}\pics:/data:ro`), or run inside WSL.

**Full run on nuScenes:** download **nuScenes v1.0-mini** (~4 GB, free account/EULA) and extract it so
the root holds `samples/ sweeps/ maps/ v1.0-mini/`, then:

```bash
docker run --rm -v /path/to/nuscenes:/data:ro -v "$PWD/out:/out" \
  -e DATAROOT=/data -e DB=/out/scenes.db vlm-project ingest -v
docker run --rm -v "$PWD/out:/out" -e DB=/out/scenes.db vlm-project query "pedestrian crossing"
```

BLIP captions a frame in roughly **2 to 9 seconds on CPU** (the 10-scene mini set is about 15 minutes).

**Local install (development):**

```bash
pip install -e ".[dev]"     # add ".[video]" for local video decoding
pytest                      # fast unit suite: no dataset, model, or network
vlm-project ingest --dataroot ./data --db out/scenes.db -v
```

**Configuration** is set by an environment variable, or the matching CLI flag which wins:

| Concern | Env var | CLI flag | Default | Values |
|---|---|---|---|---|
| Dataset loader | `LOADER` | `--loader` | `nuscenes` | `nuscenes`, `imagefolder`, `video` |
| Dataset path | `DATAROOT` | `--dataroot` | `./data` | any path |
| nuScenes camera | `CAMERA` | `--camera` | `CAM_FRONT` | any channel |
| Keyframe selection | `SELECTOR` | `--selector` | `single` | `single`, `uniform`, `clusters` |
| Uniform samples | `UNIFORM_SAMPLES` | `--uniform-samples` | `3` | frames/clip for `uniform` |
| Cluster cut point | `CLUSTER_THRESHOLD` | `--cluster-threshold` | `0.05` | cosine distance |
| VLM backend | `VLM_BACKEND` | `--backend` | `blip` | `blip`, `moondream`, `fake` |
| Database path | `DB` | `--db` | `./out/scenes.db` | any path |

## Deploy

vlm-project deploys as a **stateless batch container**: push the image to a registry, then run it on
anything that runs containers (a VM, a CI runner, Kubernetes, AWS Batch, Google Cloud Run, ECS, and so
on) with the dataset mounted in and an output volume mounted out, passing config as env vars. The
container itself never changes; only the platform's wrapper around it does.

See **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)** for the full walkthrough: pushing the image, a
portable Kubernetes `Job`, a managed Google Cloud Run Jobs example, and sharding for scale.

## Design

The pipeline is four swappable stages (`loader → selector → backend → store`) wired by `pipeline.run()`
and assembled from config in one place (`factory.py`), so the dataset, keyframe strategy, and model are
configuration choices rather than code changes. The store is a single SQLite file with an FTS5 index
(`ingest` writes, `query` reads). Caption quality is checked two ways on real data: object-grounding
against nuScenes ground-truth boxes (`verify-accuracy`), and a COCO BLEU-4 model-sanity tripwire
(`bleu-sanity`).

The design decisions and trade-offs are in **[`docs/DESIGN.md`](docs/DESIGN.md)**; the test strategy and
coverage matrix are in **[`docs/TEST_PLAN.md`](docs/TEST_PLAN.md)**.

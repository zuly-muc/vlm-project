# Configuration

`vlm-project` is configured entirely through **environment variables**. Most have a matching **CLI flag**
that takes precedence when both are set; a few (the model checkpoints, noted below) are environment-only.
Precedence is: **CLI flag > environment variable > built-in default**. The container's contract is the
environment variables; the flags are for local runs and overrides.

## Reference

### Input: where scenes come from

| Env var | CLI flag | Default | What it does |
|---|---|---|---|
| `LOADER` | `--loader` | `nuscenes` | Chooses how the input is read: `nuscenes` (a driving dataset), `imagefolder` (a directory of images), or `video` (a video file, sampled into frames). |
| `DATAROOT` | `--dataroot` | `./data` | Where the input lives: the dataset root, the image directory, or the video file path, depending on `LOADER`. |
| `NUSCENES_VERSION` | `--nuscenes-version` | `v1.0-mini` | Which nuScenes split to read (for example `v1.0-mini` for the small sample, `v1.0-trainval` for the full set). Ignored by the other loaders. (The `verify-db` / `verify-accuracy` commands take the same value as `--version`.) |
| `CAMERA` | `--camera` | `CAM_FRONT` | Which nuScenes camera to caption (the front camera by default). Ignored by the other loaders. |

### Selection: which frame(s) to describe

| Env var | CLI flag | Default | What it does |
|---|---|---|---|
| `SELECTOR` | `--selector` | `single` | How many frames per clip to caption: `single` (one representative frame), `uniform` (a few evenly-spaced frames), or `clusters` (one frame per detected sub-scene). |
| `SINGLE_STRATEGY` | `--single-strategy` | `middle` | For `single` only, which frame to pick: `middle` (the midpoint, no image decoding) or `sharpest` (the least-blurry frame, avoids motion blur). |
| `UNIFORM_SAMPLES` | `--uniform-samples` | `3` | For `uniform` only, how many evenly-spaced frames to take per clip (`1` collapses to the middle frame). |
| `CLUSTER_THRESHOLD` | `--cluster-threshold` | `0.05` | For `clusters` only, how different two frames must look to start a new sub-scene. Lower = more, shorter scenes; higher = fewer, longer ones. |

### Model: which backend captions

| Env var | CLI flag | Default | What it does |
|---|---|---|---|
| `VLM_BACKEND` | `--backend` | `blip` | Which model writes the captions: `blip` (fast local default), `moondream` (heavier local model, richer captions), or `fake` (canned text, for tests). |
| `BLIP_MODEL` | (none) | `Salesforce/blip-image-captioning-base` | Which BLIP checkpoint to load. Baked into the image, so it is rarely changed and has no CLI flag. |
| `MOONDREAM_MODEL` | (none) | `vikhyatk/moondream2` | Which Moondream checkpoint to load when `VLM_BACKEND=moondream`. Environment-only, same reasoning as above. |

### Output: where results land

| Env var | CLI flag | Default | What it does |
|---|---|---|---|
| `DB` | `--db` | `./out/scenes.db` | Path of the SQLite database to write (on `ingest`) or read (on `query` / `export-json`). |

## Notes

- **`clusters` needs an embedding-capable backend.** BLIP doubles as the embedder; pairing `clusters`
  with `moondream` (no embedder) fails fast with a clear message. Pair `moondream` with `single` or
  `uniform`.
- **`CLUSTER_THRESHOLD` is small on purpose** because BLIP's pooled vision embeddings are highly
  concentrated (within-scene frames measure ~0.00, a hard scene cut ~0.08). Tune it per dataset.
- **The image runs as a non-root user** (uid 10001). When writing to a host-owned output directory, pass
  `--user "$(id -u):$(id -g)"` so the container can write to it.

## Examples

```bash
# nuScenes, sharpest single frame, explicit DB path (flags override env)
vlm-project ingest --loader nuscenes --dataroot ./data \
  --selector single --single-strategy sharpest --db out/scenes.db -v

# a folder of images, five evenly-spaced frames per clip
LOADER=imagefolder DATAROOT=./pics SELECTOR=uniform UNIFORM_SAMPLES=5 \
  vlm-project ingest -v

# video with content-aware sub-scenes, looser cut point
vlm-project ingest --loader video --dataroot clip.mp4 \
  --selector clusters --cluster-threshold 0.1 --db out/clip.db -v
```

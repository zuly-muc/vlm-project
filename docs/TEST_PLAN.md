# Test plan

How `vlm-project` is tested, what each layer covers, and how "success" is measured. The goal is that a
regression in any load-bearing part (a loader, a selector, a backend, the store, the correctness of the
captions) is caught by something that runs automatically.

## Layers, fast to slow

| Layer | When it runs | Needs | What it proves |
|---|---|---|---|
| **Unit** | every push (`lint-and-unit`) | nothing (fakes) | logic and wiring are correct |
| **Integration** | opt-in (`pytest -m integration`) | real BLIP weights | the real model path works |
| **docker-smoke** | every push (`docker-smoke`) | Docker | the shipping image builds and runs end to end, no dataset |
| **Real-data acceptance** | manual (`nuscenes-e2e` / `test_vlm_app_production.sh`) | the 4 GB dataset | results are sound, correct, and the model is healthy |

## How "success" is measured at each layer

- **Unit:** deterministic assertions on exact behavior, with no model or network (a `FakeBackend`, a
  fake `NuScenes` stand-in, scripted embeddings). Example: `uniform` over 10 frames selects indices
  `[0, 4, 9]`; `clusters` cuts a segment where cosine distance spikes. Success = the code does exactly
  what is asserted.
- **docker-smoke:** the real commands run inside the built image and exit `0` (build works, deps are
  present, ingest and query work). It uses the `fake` backend so it is fast and needs no dataset.
- **Real-data acceptance:** three independent bars on real output:
  - **Soundness** (`verify-db`): row coverage vs `scene.json`, well-formed captions, image files exist,
    FTS round-trip, plausible driving vocabulary, bounded timings.
  - **Correctness** (`verify-accuracy`): object-grounding vs ground-truth boxes (see below).
  - **Model sanity** (`bleu-sanity`): BLIP BLEU-4 on real COCO images vs a floor.

## Coverage matrix

| Component | Unit (every push) | docker-smoke (every push) | Real-data acceptance (manual) |
|---|---|---|---|
| **Loader: nuscenes** | yes (fake devkit) | no | yes (ingest + soundness + grounding) |
| **Loader: imagefolder** | yes | yes (real ingest + query) | no |
| **Loader: video** | yes (decodes committed clip) | yes (real OpenCV decode in image) | no |
| **Selector: single** | yes | yes (on the video clip) | yes (soundness + grounding) |
| **Selector: uniform** | yes | yes (on the video clip) | no |
| **Selector: clusters** | yes | yes (on the video clip) | yes (soundness + grounding, precision-floor only) |
| **Backend: blip** | integration test | no (fake used for speed) | yes (the real captions) |
| **Backend: fake** | yes | yes | no |
| **Backend: moondream** | yes (wiring, mocked) | no | no |
| **Store (SQLite + FTS5)** | yes (round-trip, idempotent, staged writes) | yes (writes + query) | yes |
| **Grounding (`verify-accuracy`)** | yes (scorer + gate, fake provider) | no | yes (single + clusters) |
| **Model sanity (`bleu-sanity`)** | yes (scorer, fake backend) | no | yes (real COCO BLEU-4) |

Notes:

- **Selectors** are the axis worth calling out: all three are unit-tested for exact selection logic,
  and all three run a real ingest on every push in `docker-smoke` (the committed `highway.mp4` has ~90
  frames, so it is a genuine multi-frame clip where `uniform` and `clusters` produce multiple rows).
  On real nuScenes data, `single` and `clusters` are additionally graded for correctness.
- **The video fixture** (`tests/data/video/highway.mp4`) is real, committed motorway footage, so both
  the unit test and the docker smoke decode genuine content, not a synthetic file.

## How correctness is measured (`verify-accuracy`)

nuScenes has no gold captions, so instead of BLEU/CIDEr we compare the object *families* a caption
names to the ground-truth 3D boxes actually visible in that frame:

- **Precision** (gated): did the caption name an object that is not in the frame? BLIP's precision on
  single frames is a perfect **1.00** (it never hallucinates). The absolute floor is `0.90`.
- **Recall / dominant-family recall** (reported, gated loosely or not at all): a terse captioner names
  only the most salient object, so full recall is intrinsically low and is not a hard gate.
- **Regression vs. baseline:** for the single-frame config a baseline
  (`src/vlm_project/baselines/accuracy_baseline.json`) is committed and shipped in the package, and the
  gate fails if metrics regress against it. `clusters` has no committed baseline, so it is gated on the
  precision floor only (`--no-baseline`, with a slightly lower `--min-precision 0.8` for margin).

This is trustworthy because, with weights and dependencies pinned, BLIP's beam search is deterministic
(two independent real runs produced byte-identical captions), so the metrics are reproducible and a
regression reliably signals a dependency or model change, not noise.

## How model health is measured (`bleu-sanity`)

A separate axis from grounding: has the model itself regressed on general-domain images? A small COCO
caption set is fetched from the official source (never vendored) and BLIP is scored with a
dependency-free BLEU-4 against a conservative floor. It is a tripwire (a healthy model scores well
above the floor, a broken one near zero), and it self-skips when COCO cannot be fetched.

## Running it yourself

```bash
pytest                                            # unit suite (seconds, no model/data)
pytest -m integration                             # opt-in: real BLIP + COCO sanity
ruff check src tests                              # lint

./scripts/test_vlm_app_production.sh              # full real-data acceptance (Docker)
ACCEPTANCE_MODE=local ./scripts/test_vlm_app_production.sh   # same, no Docker

# individual gates against a produced DB
vlm-project verify-db       --db out/scenes.db --dataroot ./data
vlm-project verify-accuracy --db out/scenes.db --dataroot ./data
```

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

## Per-test reference

Every test, by the layer that runs it, with the one thing it proves. The first two layers run on every
push; the integration and real-data layers are opt-in or manual.

### Unit (every push, `pytest -m "not integration"`)

**Keyframe selection (`test_selectors.py`)**
- `test_single_middle_picks_midpoint`: single/middle picks the temporal midpoint frame.
- `test_single_empty_clip`: single returns nothing for a clip with no frames.
- `test_clusters_split_on_distance`: clusters starts a new segment where frame-to-frame distance spikes.
- `test_clusters_single_segment_when_uniform`: clusters yields one segment when all frames look alike.
- `test_uniform_picks_evenly_spaced_endpoints`: uniform picks evenly-spaced indices including endpoints.
- `test_uniform_n1_is_middle_with_no_suffix`: uniform n=1 collapses to the middle frame, no segment suffix.
- `test_uniform_clamps_and_dedupes_on_short_clip`: uniform clamps N to the frame count and dedupes indices.
- `test_uniform_empty_clip`: uniform returns nothing for an empty clip.
- `test_uniform_rejects_zero`: uniform rejects n=0 as invalid.

**Store: SQLite + FTS5 (`test_store_search.py`)**
- `test_search_finds_the_right_row`: full-text search returns the row whose caption matches.
- `test_upsert_is_idempotent`: re-upserting the same scene_id replaces rather than duplicates.
- `test_row_shape_matches_scene_description`: a stored/returned row matches the SceneDescription shape.
- `test_staged_publishes_only_on_finalize`: staged writes reach the final path only on finalize.
- `test_staged_context_manager_finalizes_on_clean_exit`: context-manager exit publishes on a clean run.
- `test_staged_failure_does_not_clobber_existing_db`: a failed run leaves any existing final DB intact.

**CLI commands (`test_cli.py`)**
- `test_ingest_then_query`: end-to-end ingest then query through the CLI.
- `test_verbose_flag_accepted_after_subcommand`: -v works before or after the subcommand.
- `test_ingest_uniform_selector_flags`: --selector uniform and --uniform-samples plumb through.
- `test_ingest_with_stage_dir_publishes_db`: --stage-dir builds then publishes the database.
- `test_export_json`: export-json dumps all rows to JSON.

**Soundness checks, `verify-db` (`test_acceptance.py`)**
- `test_passes_on_sound_db`: a well-formed database passes every check.
- `test_fails_on_missing_scene`: coverage check fails when a scene is missing.
- `test_fails_on_empty_description`: well-formed check fails on an empty caption.
- `test_fails_on_missing_image_file`: file-existence check fails when an image is gone.
- `test_fails_on_low_driving_vocabulary`: vocabulary check fails when too few captions mention driving terms.
- `test_fts_roundtrip_not_truncated_by_common_word`: FTS round-trip finds every row even for a common word.
- `test_scene_names_from_dataset`: reads scene names straight from scene.json.

**Correctness scorer + gate, `verify-accuracy` (`test_grounding.py`)**
- `test_family_and_caption_mapping`: maps caption words and GT boxes into object families.
- `test_evaluate_precision_recall_and_dominant`: computes precision, recall, and dominant-recall correctly.
- `test_rows_without_sample_token_are_skipped`: rows lacking a sample token are skipped, not errored.
- `test_gate_passes_when_precise_and_no_regression`: gate passes when precise and not regressed vs baseline.
- `test_gate_fails_on_precision_floor`: gate fails below the precision floor.
- `test_gate_fails_on_dominant_recall_regression`: gate fails on a dominant-recall regression vs baseline.
- `test_default_baseline_is_shipped_and_wellformed`: the shipped baseline file exists and is well-formed.

**BLEU implementation (`test_bleu.py`)**
- `test_identical_hypothesis_scores_one`: an identical hypothesis scores 1.0.
- `test_disjoint_hypothesis_scores_near_zero`: a fully disjoint hypothesis scores near zero.
- `test_partial_overlap_is_between`: partial overlap scores between 0 and 1.
- `test_brevity_penalty_punishes_short_hypotheses`: the brevity penalty lowers too-short hypotheses.
- `test_multiple_references_take_the_best`: with several references, the best match counts.

**Model-sanity report, `bleu-sanity` (`test_sanity.py`)**
- `test_passes_when_captions_match_references`: sanity passes when captions match references above the floor.
- `test_fails_when_model_output_is_unrelated`: sanity fails when output is unrelated (near-zero BLEU).

**Video loader (`test_dataset_video.py`)**
- `test_fixture_exists`: the committed highway.mp4 fixture is present.
- `test_samples_keyframes_and_writes_them`: decodes and samples keyframes, writing them out.
- `test_stride_controls_keyframe_count`: the sampling stride controls how many keyframes are produced.
- `test_directory_input_discovers_the_video`: pointing at a directory discovers the video file.

**Image-folder loader (`test_dataset_imagefolder.py`)**
- `test_one_clip_per_image_sorted`: one clip per image, in sorted order.
- `test_recursive_discovery`: discovers images in nested subdirectories.

**nuScenes loader, fake devkit (`test_dataset_nuscenes.py`)**
- `test_iter_clips_yields_one_clip_per_scene`: yields one clip per scene.
- `test_walks_the_full_keyframe_chain`: walks the full next-linked keyframe chain.

**Pipeline wiring (`test_pipeline.py`)**
- `test_single_mode_one_row_per_clip`: single mode writes exactly one row per clip.
- `test_missing_image_is_skipped_not_fatal`: a missing image is skipped, not fatal to the run.

**Moondream backend wiring (`test_vlm_moondream.py`)**
- `test_construction_is_lazy_no_weights_loaded`: constructing the backend loads no weights (lazy).
- `test_describe_uses_encode_then_answer`: describe uses the encode-then-answer path.
- `test_clusters_rejects_moondream_backend`: pairing clusters with moondream fails fast with a clear message.

### docker-smoke (every push)

Builds the shipping image and runs the real commands inside it, no dataset needed:
- **doctor**: dependency health inside the image (imports, SQLite FTS5, cached BLIP weights).
- **fake-backend ingest + query**: a real ingest and query over generated images.
- **video decode + all selectors**: decodes the committed highway.mp4 and runs single, uniform, and
  clusters, then queries the result.

### Integration (opt-in, `pytest -m integration`)

Real-model checks that run in the manual tier rather than on every push:
- `test_blip_integration.py::test_blip_captions_and_embeds`: the real BLIP model captions and embeds an image.
- `test_coco_sanity.py::test_blip_bleu_on_coco_micro_set`: real BLIP scores acceptable BLEU-4 on a COCO micro-set.

### Real-data acceptance (manual, `nuscenes-e2e`)

Full acceptance on the real nuScenes v1.0-mini dataset with the real BLIP model, via the CLI gates:
- **`verify-db`**: the soundness checks above, run on the real produced database.
- **`verify-accuracy`**: object-grounding correctness on single and clusters (see next section).
- **`bleu-sanity`**: BLIP BLEU-4 on a small official COCO set against a floor.

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

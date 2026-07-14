# Design notes

Why `vlm-project` is built the way it is. The README covers how to use it; this covers the decisions
behind it, one junction at a time.

## A batch job, not a service

The tool runs one command and exits: `ingest` writes captions into the database, `query` reads them
back. There is no server to keep running. The task is "process a batch of clips, then look things up",
which a batch job fits exactly, and search is still instant because it is just a query over a local
index. Keeping `ingest` and `query` separate means `query` needs none of the heavy ML libraries, and
`ingest` can be run many times in parallel over different slices of the data.

## The pipeline: four swappable parts

A run flows through four stages, wired together in `pipeline.run()`:

```
loader  ->  selector  ->  backend  ->  store
(read)      (pick        (caption)     (save +
             frames)                    index)
```

The three things anyone actually wants to change (which **dataset**, which **frames**, which **model**)
are exactly the first three stages, so they are chosen by configuration, not by editing code. Each
stage is a tiny interface, which also lets the whole pipeline run in tests with stand-ins (no dataset,
no model, no network). One file, `factory.py`, turns the config strings into real objects, so nothing
else in the codebase has to know about "which backend" or "which loader".

## Inputs: the loader

Every input is turned into the same shape, a **clip of frames**, so nothing downstream cares where the
frames came from:

- **nuScenes**: one clip per scene (its sequence of keyframes), reading the front camera by default.
- **image folder**: one clip per image.
- **video**: frames sampled at a fixed interval, decoded with OpenCV.

The nuScenes toolkit is passed in rather than created inside the loader, so tests can hand it a tiny
fake instead of the real 4 GB dataset.

## Which frames to describe: the selector

Captioning is the slow part, so a clip's ~40 frames are narrowed down first:

- **single** (default): one representative frame, the middle one, or the sharpest if you want to avoid
  motion blur.
- **uniform**: a few evenly-spaced frames, chosen purely by position, so it is basically free.
- **clusters**: splits the clip where the scene changes and captions one frame per part. It walks the
  frames in order and starts a new part when consecutive frames look different enough. It is the most
  expensive option because it has to look at every frame, which is exactly why `uniform` exists as the
  cheaper middle ground.

## The model: the backend

Captioning is kept **entirely local, on purpose**. The default is **BLIP** (about 0.5 GB, runs on CPU,
baked into the image so it works offline). **Moondream2** is an optional heavier model that gives
richer descriptions. A **fake** backend returns canned text so the test suite runs in milliseconds.

Cloud APIs (OpenAI, Anthropic, and the like) were deliberately left out to keep the tool simple and
self-contained: no API keys, no per-call cost, no network at run time, and identical results every run.
The model is just one swappable stage, so a networked backend could be added later without touching
anything else; it is simply not shipped.

## Storage: one searchable file

The result is a single **SQLite file with full-text search**. That is the simplest thing that is both
portable (copy it, ship it, open it in any SQLite tool) and needs no server to run. A full database
server or a vector database would add moving parts we do not need for keyword search over short
captions; the door is left open to add semantic search later.

Writes are made safe on the container: `ingest` builds the database on the container's own disk and then
moves the finished file onto the mounted output volume in one step. So a slow or crashed run never
leaves a half-written file where someone might read it, and it sidesteps a real problem with writing
many small updates directly onto a mounted Windows folder.

## Packaging

The image bundles the app, a CPU-only build of PyTorch, and the model weights, so it runs offline with
no GPU and no first-run download. The dataset and the outputs are never baked in; they are mounted at
run time, which keeps the image small and immutable. Dependency versions are pinned so a rebuild
produces the same image.

## Are the captions any good?

A generated caption has no single "correct" answer, so quality is checked two independent ways on real
data:

- **Grounding** (`verify-accuracy`): compare the objects a caption mentions to what is actually in the
  frame, using nuScenes' ground-truth boxes. The gate is on **precision**, "did it name something that
  is not there?", which BLIP gets perfect (1.00). Recall is not gated, because a one-line caption
  naturally mentions only the most obvious thing, so scoring it against every object in frame would be
  unfair.
- **Model sanity** (`bleu-sanity`): score BLIP on a handful of ordinary COCO photos that *do* have
  reference captions. This is a tripwire that catches the model quietly getting worse after a library or
  weights change, which the driving-only grounding check would miss.

Both are trustworthy because the model is deterministic: with versions pinned, it produces byte-for-byte
identical captions run to run, so a change in the numbers means a real regression, not noise.

## Testing, briefly

Unit tests use the fakes and run in seconds on every push; a real-model test and the full run on the
real dataset are opt-in and manual. The full strategy and coverage are in
[`TEST_PLAN.md`](TEST_PLAN.md).

## Known limits

- Caption quality is only as good as BLIP: precise but terse. Moondream and the `uniform`/`clusters`
  selectors exist for when you want more.
- Search is keyword-based, not semantic: "pedestrian" will not match "person walking" unless the caption
  actually says so.
- Running at full-dataset scale means sharding across many jobs. The pieces allow it, but the
  orchestration is described, not built.
- A crash mid-`ingest` keeps the previous good database and discards the partial one. Safe, but not
  resumable.

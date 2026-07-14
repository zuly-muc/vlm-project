# Deployment

`vlm-project` is a **stateless batch container**: it runs one command (`ingest`), writes a SQLite file,
and exits. Deploying it means running that container wherever you run containers. Nothing stays running,
and the container is identical everywhere; only the wrapper around it changes.

## 1. Build and push the image

```bash
docker build -t vlm-project .
docker tag vlm-project <registry>/vlm-project:latest
docker push <registry>/vlm-project:latest
```

## 2. The run contract

Wherever it runs, the container needs the same three things, exactly as in a local `docker run`:

- the **dataset** on a readable path: `-e DATAROOT=/data` plus a mount at `/data`;
- an **output location** for the `.db`: `-e DB=/out/scenes.db` plus a writable mount at `/out`;
- **config** as environment variables (`LOADER`, `SELECTOR`, `VLM_BACKEND`, and so on; see
  [`CONFIGURATION.md`](CONFIGURATION.md)).

The dataset is read from a filesystem path, so if it lives in object storage, mount the bucket or sync it
to a local volume first. The image runs as a non-root user; when writing to a host-owned output
directory, pass `--user "$(id -u):$(id -g)"` so it can write the database.

## 3. Running it on a scheduler

Any batch scheduler runs the same container the same way: give it the image, the two mounts, and the env
vars above. A Kubernetes `Job`, an AWS Batch job, and a Google Cloud Run Job are all the same shape.

One managed example, Google Cloud Run Jobs (no cluster to operate), mounting a bucket for the dataset and
one for the output:

```bash
gcloud run jobs create vlm-ingest \
  --image <registry>/vlm-project:latest \
  --region us-central1 --cpu 2 --memory 4Gi --max-retries 1 --task-timeout 3h \
  --args ingest,-v \
  --set-env-vars DATAROOT=/data,DB=/out/scenes.db \
  --add-volume       name=data,type=cloud-storage,bucket=$BUCKET \
  --add-volume-mount volume=data,mount-path=/data \
  --add-volume       name=out,type=cloud-storage,bucket=$OUT_BUCKET \
  --add-volume-mount volume=out,mount-path=/out

gcloud run jobs execute vlm-ingest --region us-central1
```

The finished `scenes.db` lands in your output bucket.

## 4. Scaling

One run is single-process and CPU-bound (about 2 to 9 seconds per BLIP caption), so the full nuScenes is
a many-hours job on one core. Because every stage is stateless, the work fans out naturally: run several
jobs over different slices of the input, each writing its own database, and combine the results.

Scaling further is future work: adding shard flags (`--shard-index` / `--shard-count`) to split the
dataset automatically, and a `merge` command to combine the per-shard databases, would turn this into a
single supported workflow. Today each worker is simply pointed at a different dataset root.

Captioning is always local, so a worker needs no network or API keys at run time, only CPU and the
mounted dataset.

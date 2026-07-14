# Deployment

`vlm-project` is a **stateless batch container**: it runs one command (`ingest`), writes a SQLite file,
and exits. Deploying it means running that container on any host or scheduler. Nothing stays running,
and the container is identical everywhere; only the platform's wrapper around it changes.

## 1. Push the image to a registry

Build once, then push to a registry your target can pull from (Docker Hub, GitHub GHCR, AWS ECR, Google
Artifact Registry, and so on):

```bash
docker build -t vlm-project .
docker tag vlm-project <registry>/vlm-project:latest
docker push <registry>/vlm-project:latest
```

## 2. What every deployment needs

Whatever the platform, the container needs the same three things, exactly as in a local `docker run`:

- the **dataset** on a path it can read: `-e DATAROOT=/data` plus a mount at `/data`;
- an **output location** for the `.db`: `-e DB=/out/scenes.db` plus a writable mount at `/out`;
- **config** as environment variables (`LOADER`, `SELECTOR`, `VLM_BACKEND`, and so on; see the README's
  configuration table).

The dataset is read from a **filesystem path**, not from object storage directly, so it must be on a
volume the container can see. If it lives in S3 or GCS, either mount the bucket (a CSI driver, gcsfuse,
EFS) or add an init step that syncs it to a local volume before `ingest` runs.

## 3. Portable example: Kubernetes Job

Runs unchanged on any Kubernetes (EKS, GKE, AKS, or on-prem):

```yaml
apiVersion: batch/v1
kind: Job
metadata: { name: vlm-ingest }
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: ingest
          image: <registry>/vlm-project:latest
          args: ["ingest", "-v"]
          env:
            - { name: DATAROOT, value: /data }
            - { name: DB,       value: /out/scenes.db }
          resources:
            requests: { cpu: "2", memory: 4Gi }
          volumeMounts:
            - { name: data, mountPath: /data, readOnly: true }
            - { name: out,  mountPath: /out }
      volumes:
        - { name: data, persistentVolumeClaim: { claimName: nuscenes-pvc } }
        - { name: out,  persistentVolumeClaim: { claimName: scenes-out-pvc } }
```

`kubectl apply -f job.yaml`, then collect `scenes.db` from the `scenes-out-pvc` volume when it finishes.

## 4. Managed example: Google Cloud Run Jobs

A fully managed run with no cluster to operate. Push to Artifact Registry, then create and execute a Job
that mounts a GCS bucket holding the extracted dataset (and one for the output). Replace `$PROJECT`,
`$BUCKET`, and `$OUT_BUCKET`:

```bash
# one-time: push to Artifact Registry
gcloud artifacts repositories create vlm --repository-format=docker --location=us-central1
docker tag vlm-project us-central1-docker.pkg.dev/$PROJECT/vlm/vlm-project:latest
docker push us-central1-docker.pkg.dev/$PROJECT/vlm/vlm-project:latest

# create the job, mounting the dataset bucket at /data and an output bucket at /out
gcloud run jobs create vlm-ingest \
  --image us-central1-docker.pkg.dev/$PROJECT/vlm/vlm-project:latest \
  --region us-central1 --cpu 2 --memory 4Gi --max-retries 1 --task-timeout 3h \
  --args ingest,-v \
  --set-env-vars DATAROOT=/data,DB=/out/scenes.db \
  --add-volume       name=nusc,type=cloud-storage,bucket=$BUCKET \
  --add-volume-mount volume=nusc,mount-path=/data \
  --add-volume       name=results,type=cloud-storage,bucket=$OUT_BUCKET \
  --add-volume-mount volume=results,mount-path=/out

gcloud run jobs execute vlm-ingest --region us-central1
```

The finished `scenes.db` lands in your output bucket. AWS Batch follows the same shape: an ECR image, a
job definition carrying the two mounts and the env vars, then `aws batch submit-job`.

## 5. Scale by sharding

One run is single-process and CPU-bound (about 2 to 9 seconds per BLIP caption), so the full nuScenes is
a many-hours job on one core. Because it is stateless, fan it out: run N jobs, each over a slice of the
input, each writing its own `.db`, then merge the files. The stages are stateless, so this fans out
cleanly; today each run reads the whole dataset, so slicing is done by pointing workers at different
inputs.

Captioning is always local (BLIP by default, Moondream optionally), so a worker needs no network or API
keys at run time, only CPU and the mounted dataset.

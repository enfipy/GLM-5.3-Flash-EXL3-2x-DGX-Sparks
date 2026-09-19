# Nomad deployment of this recipe on 2x DGX Spark (enfis1 + enfis2)

This directory runs the upstream recipe as one Nomad job instead of
`start.sh`: the head rank on enfis1 (Ubuntu) and the worker rank on enfis2
(Enfios), tensor parallel 2 over the ConnectX-7 RoCE link, serving
`http://100.76.243.97:8888/v1` (tailnet only) as model id `glm-5.3-flash`.
Every start warms the model up by itself (see "Automatic warmup").

## From a fresh head

1. Clone this fork (branch `nomad-2x-sparks`) anywhere, e.g. `~/models/glm53f`.
2. Put the private control-host access outside the tree:
   `~/.config/glm53-nomad/control/control-ssh` (SSH wrapper to the control
   host) and `nomad-remote` (exports `NOMAD_ADDR`, `NOMAD_CACERT`,
   `NOMAD_TLS_SERVER_NAME`, `NOMAD_TOKEN` and execs `nomad "$@"` there).
3. Review `nomad/env` (weights root, worker paths, node IDs, API, image,
   generator flags); put machine-specific overrides in `nomad/env.local`.
4. Download the weights into the root from `env`:
   `HF_HOME=$GLM53_WEIGHTS/hf ./download.sh` (target checkpoint + DFlash2
   drafter, about 166 GB). Optional ABLIT transplant tensors go to
   `$GLM53_WEIGHTS/ablit-transplant/` (`ablit/fetch_transplant.py`).
5. `nomad/setup.sh`: checks the layout, places the cooperative kernel runtime
   next to the weights, loads the image from `$GLM53_WEIGHTS/images/` or tells
   you how to build it, and registers the host volumes on both nodes.
6. `nomad/model.sh start`, then `nomad/model.sh status` until it shows
   `API READY` and the head's `warmup=dead` (warmup finished).

The worker (enfis2) needs the same weights under `GLM53_WORKER_MODELS`, the
image loaded, and `/dev/infiniband` (`modprobe ib_uverbs` after a reboot until
the Enfios image loads it itself).

## All commands

Everything below runs on the head (enfis1) from the checkout root unless
stated otherwise. Paths come from `nomad/env` (override in `nomad/env.local`).

### Private control wrapper (once)

```bash
mkdir -p ~/.config/glm53-nomad/control && chmod 700 ~/.config/glm53-nomad
cat > ~/.config/glm53-nomad/control/control-ssh <<'EOF'
#!/bin/sh
exec ssh -o BatchMode=yes root@<control-host> "$@"
EOF
cat > ~/.config/glm53-nomad/control/nomad-remote <<'EOF'
export NOMAD_ADDR=https://<control-host>:4646 NOMAD_CACERT=/etc/cloud-ctrl/tls/nomad-ca.crt
export NOMAD_TLS_SERVER_NAME=<control-host> NOMAD_TOKEN=<management token>
exec /var/lib/cloud-node/current/nomad "$@"
EOF
chmod 700 ~/.config/glm53-nomad/control/control-ssh
```

`model.sh` and `setup.sh` run every Nomad command as
`control-ssh bash -s -- <args> < nomad-remote`, so nothing private is stored
in the tree. To run a raw command the same way:

```bash
C=~/.config/glm53-nomad/control
$C/control-ssh bash -s -- job status glm53 < $C/nomad-remote
$C/control-ssh bash -s -- volume status -type host < $C/nomad-remote
$C/control-ssh bash -s -- alloc status <alloc-id> < $C/nomad-remote
```

### Weights and image

```bash
HF_HOME=$GLM53_WEIGHTS/hf ./download.sh                     # target + DFlash2 drafter into hf/hub (about 166 GB)
mkdir -p $GLM53_WEIGHTS/ablit-transplant && ln -sfn $GLM53_WEIGHTS/ablit-transplant ablit/transplant
python3 ablit/fetch_transplant.py                           # optional ABLIT tensors (L15..L45 .bin) into that directory
nomad/setup.sh                                              # layout check, cooperative runtime, image load, volumes (idempotent)

# Build the thin-decode image from this checkout (BuildKit is required, the
# legacy builder skips the Dockerfile heredocs). Do it on a node that is not
# serving the model: the build starves the host for about an hour.
docker buildx build --load --build-arg GLM53_RECIPE_STAMP=local -t $GLM53_IMAGE .
docker save -o $GLM53_WEIGHTS/images/$(echo $GLM53_IMAGE | tr '/:' '--').tar $GLM53_IMAGE   # what setup.sh loads

# Worker (enfis2, as root): same image and weights, InfiniBand device.
scp $GLM53_WEIGHTS/images/<tag>.tar root@enfis2:/state/models/ && ssh root@enfis2 docker load -i /state/models/<tag>.tar
rsync -a $GLM53_WEIGHTS/hf/ root@enfis2:$GLM53_WORKER_MODELS/                      # once; 166 GB over the ConnectX link
$C/control-ssh bash -s -- job run nomad/volumes/sync-worker-ablit.nomad.hcl < $C/nomad-remote   # copies the ABLIT tensors into the worker cache volume
ssh root@enfis2 modprobe ib_uverbs                                                  # after every enfis2 reboot until the image does it
```

### Run

```bash
nomad/model.sh start                  # regenerate the job from this checkout and submit it (new version every time)
nomad/model.sh start --vision         # multimodal (drops --language-model-only)
nomad/model.sh start --no-ablit       # ABLIT transplant off in both ranks
nomad/model.sh start --no-gen         # submit nomad/glm53.nomad.hcl as is, without regenerating it
nomad/model.sh status                 # job status, per-task state of the latest version, API health
nomad/model.sh logs                   # last 100 lines of the head's vLLM log
nomad/model.sh logs worker -n 500     # worker rank
nomad/model.sh logs warmup            # the poststart warmup task's output
nomad/model.sh warmup                 # rerun the boot-shape warmup by hand (the job already does it after start)
nomad/model.sh bench --concurrency 1,2,4 --repeats 3       # sparkDash-protocol prose decode benchmark
nomad/model.sh bench --concurrency 4 --max-tokens 400 --mode varied --out /tmp/bench.json
nomad/model.sh stop                   # stop the job (allocations end, images stay for a week)
```

### Job file generator

`model.sh start` runs `gen_hcl.py $GLM53_GEN_FLAGS --image $GLM53_IMAGE`
before submitting. Run it directly to inspect the result or to change the
kernel set:

```bash
python3 nomad/gen_hcl.py                          # stock exl3 kernels, published GHCR image tag kept
python3 nomad/gen_hcl.py --coop                   # cooperative MoE decode kernel (nomad/cooperative_moe)
python3 nomad/gen_hcl.py --fast                   # thin-decode pipeline (GLM53_EXL3_MOE_FAST=1, needs a locally built image)
python3 nomad/gen_hcl.py --instanttensor          # direct-I/O weight loading, about 45 s instead of 300 s
python3 nomad/gen_hcl.py --image <ref>            # container image for both ranks and the warmup task
python3 nomad/gen_hcl.py --template <hcl> --out <hcl>   # defaults: nomad/glm53.nomad.hcl in place
```

The output is deterministic (same tree, same file), so `git diff` after a
`git pull` shows exactly what changed in the served overlay.

### Diagnostics

```bash
python3 nomad/acc_runs.py                          # per-run DFlash acceptance from /metrics
python3 nomad/watch-memory.py                      # head memory monitor while loading
curl -s http://100.76.243.97:8888/health           # 200 when the API is up
curl -s http://100.76.243.97:8888/v1/models        # model id glm-5.3-flash
```

## Automatic warmup

The head group has a second Docker task, `warmup`, with
`lifecycle { hook = "poststart" sidecar = false }`. Nomad starts it right
after the vLLM task; it waits for `/health` (polling every 10 s, so it just
idles through the 4 minute load), runs upstream
`scripts/boot-shape-warmup.sh` once (`GLM53_WARMUP_MAX_CONCURRENCY=4`,
`GLM53_WARMUP_DFLASH_K=7`, 24 requests) and exits 0. The task is generated by
`gen_hcl.py` with the script embedded base64 (a `template` stanza would
otherwise try to interpolate the script's `${...}`), uses the same image as
the ranks and 500 MHz / 1 GiB. It is non-fatal: a warmup failure never
restarts or stops the model, it only shows in `model.sh logs warmup`.
`model.sh status` shows the head as `vllm=running warmup=running` while the
shapes compile and `warmup=dead` once the model is fully warm. Restarting the
job runs it again, so `model.sh warmup` is only needed to force a rerun.

## Files

- `env` — deployment settings (weights root, worker paths, node IDs, API,
  image, generator flags, control wrapper path); `env.local` for overrides.
- `model.sh` — start / status / logs / warmup / bench / stop, all through the
  control host.
- `setup.sh` — idempotent head preparation and host volume registration for
  both ranks (`nomad volume register`, reusing the existing volume ID when the
  path is unchanged).
- `glm53.nomad.hcl` — the job. Both ranks run the same image and the same
  embedded launcher (`/local/start-optimized.py`), which unpacks the recipe
  overlay into `/opt/glm53`, applies the patches in upstream
  `GLM53_OVERLAY_ORDER`, checks the DFlash2 weights and execs `vllm serve`.
  Serving geometry follows the upstream README prose tables: 850k context,
  4 sequences, 7168 batched tokens, 14 GiB FP8 KV pool (883,552 tokens),
  DFlash2 k=7 with adaptive-k `2,4,7`, dense/KDA FP8, ABLIT transplant.
  Variables: `vision` (bool), `ablit` (bool), `submit` (timestamp, makes
  every start a new version so `failed` allocations under the no-reschedule
  policy are replaced).
- `gen_hcl.py` — regenerates the embedded overlay from `../overlay`,
  `../files/chat_template.jinja`, the order in `../start.sh`, and the warmup
  task from `../scripts/boot-shape-warmup.sh`. Run it after every `git pull`
  (`model.sh start` does).
- `cooperative_moe/` — the opt-in cooperative decode kernel built in the
  served image (`cooperative_moe.so`, digest in `SHA256SUMS`), its adapter
  `runtime.py` and profile generator re-pinned to that digest, and the
  generated overlay `exl3-cooperative.py`. Both hosts keep a copy at
  `/models/cooperative_moe` inside the models volume. Measured +10-15 % on
  good prose runs, +2 min startup (native prepare). The packaged GPU gate
  passed on both nodes; the maintainer's frozen numerical study does not
  cover this rebuild.
- `volumes/sync-worker-ablit.nomad.hcl` — batch job that copies the ABLIT
  transplant tensors into the worker's cache volume (the worker has no direct
  path to the head's files).
- `decode_bench.py` — sparkDash Decode protocol (prose, 400 tokens,
  temperature 0, thinking off; per-stream and aggregate tok/s).
- `acc_runs.py` — per-run DFlash acceptance from `/metrics`.
- `watch-memory.py` — temporary head memory monitor.

Host volumes (registered by `setup.sh`): head `glm53-models` (`hf/`, read
only), `glm53-ablit-transplant`, `glm53-head-cache` (`cache/`, Triton /
TileLang / inductor caches so restarts do not recompile); worker
`glm53-models`, `glm53-cache`, `glm53-ablit-transplant`.

## Host prerequisites

- Both nodes hold the image locally; the job references it by tag. Nomad's
  Docker driver garbage-collects unused images 3 minutes after the last task
  that used them unless the client sets `gc { image_delay = "168h" }`; with
  the default a stopped job loses the local tag and the next start fails with
  `pull access denied for glm53-flash-sm121`. Recover with `docker images`
  (the layers usually survive untagged) and
  `docker tag <id> $GLM53_IMAGE`, or `docker load` the saved tar.
- enfis2 (Enfios) needs `ib_uverbs` loaded so `/dev/infiniband` exists
  (`modprobe ib_uverbs`); the image loads it at boot from the next release.
- The Enfios thermal guard caps enfis2's GPU at 2200 MHz and drops to
  1800 MHz on a 6 C rise within 10 s; this bounds the pair and adds
  run-to-run spread.

## Measured 2026-09-18 (this kit, worker capped, ABLIT on)

| | x1 stream | x2 aggregate | x4 aggregate | submit to healthy |
|---|---:|---:|---:|---:|
| stock kernels | 27-29 | 41-44 | 60-63 | 10 min 50 s |
| cooperative MoE | 30-34 (some runs 21-24) | 46-47 | 51-68 | 13 min 01 s |

Upstream's published table (two stock Ubuntu Sparks, ABLIT off): x1 36-37,
x2 51.1, x4 75.3, and 40 / 78 with the thin-decode kernels.

## Thin-decode image (2026-09-18)

The published GHCR image predates `overlay/patch_exl3_decode_pipeline.py`, so
`GLM53_EXL3_MOE_FAST=1` fails closed on it. `glm53-flash-sm121:thin-ca85576`
was built from this tree with BuildKit (`docker buildx build --load
--build-arg GLM53_RECIPE_STAMP=local-ca85576-thin`; the legacy builder skips
the Dockerfile's heredoc steps and produces an image without the `exl3`
quantization registration). The image lives only on the two nodes; a
loadable copy is kept at
`$GLM53_WEIGHTS/images/glm53-flash-sm121-thin-ca85576.tar`
(`docker load -i`). Same protocol, three runs each, worker capped, ABLIT on:

| Config | x1 stream | x2 aggregate | x4 aggregate |
|---|---:|---:|---:|
| stock + thin-decode | 30.1, 32.3, 18.4 | 41.9, 33.2, 45.0 | 57.1, 50.3, 52.3 |
| cooperative + thin-decode (current) | 32.5, 23.7, 32.6 | 35.6, 46.0, 35.6 | 55.3, 70.2, 55.0 |

On this pair the thin-decode path is within the run-to-run noise (2-5
guard throttles per window); cooperative + thin-decode gave the best peaks and
stays deployed. `--no-ablit` (ABLIT verified off in both containers) measured
x1 32.0 / 25.3 / 32.9, x2 41.5 / 46.4 / 43.3, x4 65.9 / 59.5 / 59.9: the same
band, so abliteration is not what separates this kit from the published table;
the worker clock policy is.

## Load time

The published ~60 s load is InstantTensor (`--load-format instanttensor`).
`gen_hcl.py --instanttensor` enables it with
`INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.9` on both ranks; the loader sizes its
staging buffer from the CUDA free-memory reading, and a stale reading right
after a failed container once made it refuse a 1.27 GB buffer. With it and the
persistent head cache: weights 44.9 s, model loading 52.8 s, submit to healthy
4 min 06 s, warmup 65 s (was 306 s / 10.5-13 min / ~100 s), and x1 33.2 / 34.8,
x4 aggregate 68.4 / 65.9 right after. This is the deployed configuration. The
head keeps Triton/TileLang/inductor caches on the persistent
`glm53-head-cache` volume (`$GLM53_WEIGHTS/cache`) so restarts do not
recompile them.

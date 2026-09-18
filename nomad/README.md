# Nomad deployment of this recipe on 2x DGX Spark (enfis1 + enfis2)

This directory runs the upstream recipe as one Nomad job instead of
`start.sh`: the head rank on enfis1 (Ubuntu) and the worker rank on enfis2
(Enfios), tensor parallel 2 over the ConnectX-7 RoCE link, serving
`http://100.76.243.97:8888/v1` (tailnet only) as model id `glm-5.3-flash`.

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
6. `nomad/model.sh start`, then `nomad/model.sh status` until `API READY`,
   then `nomad/model.sh warmup` once.

The worker (enfis2) needs the same weights under `GLM53_WORKER_MODELS`, the
image loaded, and `/dev/infiniband` (`modprobe ib_uverbs` after a reboot until
the Enfios image loads it itself).

```bash
nomad/model.sh start [--vision] [--no-ablit] [--no-gen]   # regenerates the job from this checkout, then submits
nomad/model.sh status             # allocations + API health
nomad/model.sh warmup             # upstream boot-shape warmup, once after READY
nomad/model.sh bench --concurrency 1,2,4 --repeats 3      # sparkDash-protocol prose decode benchmark
nomad/model.sh stop
```

## Files

- `glm53.nomad.hcl` — the job. Both ranks run the same image and the same
  embedded launcher (`/local/start-optimized.py`), which unpacks the recipe
  overlay into `/opt/glm53`, applies the patches in upstream
  `GLM53_OVERLAY_ORDER`, checks the DFlash2 weights and execs `vllm serve`.
  Serving geometry follows the upstream README prose tables: 850k context,
  4 sequences, 7168 batched tokens, 14 GiB FP8 KV pool (883,552 tokens),
  DFlash2 k=7 with adaptive-k `2,4,7`, dense/KDA FP8, ABLIT transplant.
- `gen_hcl.py` — regenerates the embedded overlay from `../overlay`,
  `../files/chat_template.jinja` and the order in `../start.sh`. Run it after
  every `git pull`, `--coop` selects the cooperative MoE overlay, `--image`
  switches the container image on both ranks.
- `cooperative_moe/` — the opt-in cooperative decode kernel built in the
  served image (`cooperative_moe.so`, digest in `SHA256SUMS`), its adapter
  `runtime.py` and profile generator re-pinned to that digest, and the
  generated overlay `exl3-cooperative.py`. Both hosts keep a copy at
  `/models/cooperative_moe` inside the models volume. Measured +10-15 % on
  good prose runs, +2 min startup (native prepare). The packaged GPU gate
  passed on both nodes; the maintainer's frozen numerical study does not
  cover this rebuild.
- Every `model.sh start` passes `-var submit=<timestamp>`, so each start is a
  new job version and both groups get fresh allocations even when the previous
  ones are `failed` under the job's no-reschedule policy (typical after a node
  reboot). The head keeps its caches on the allocation's ephemeral disk; the
  `glm53-cache` host volume is only mounted on the worker.
- `volumes/` — dynamic host volume registrations (`nomad volume register`):
  enfis1 models at `/home/god/models/weights/glm53f/hf`, ablit transplant
  tensors at `/home/god/models/weights/glm53f/ablit-transplant`; enfis2 under
  `/state/models/`.
- `decode_bench.py` — sparkDash Decode protocol (prose, 400 tokens,
  temperature 0, thinking off; per-stream and aggregate tok/s).
- `acc_runs.py` — per-run DFlash acceptance from `/metrics`.
- `watch-memory.py` — temporary head memory monitor.

## Host prerequisites

- Both nodes hold the image locally; the job references it by tag so a
  missing image is pulled from GHCR (about 30 min here). Nomad's Docker
  driver keeps images for a week after their last task on these workers.
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
`/home/god/models/weights/glm53f/images/glm53-flash-sm121-thin-ca85576.tar`
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
x4 aggregate 68.4 / 65.9 right after. This is the deployed configuration. The head keeps Triton/TileLang/inductor caches on the
persistent `glm53-head-cache` volume (`/home/god/models/weights/glm53f/cache`)
so restarts do not recompile them.

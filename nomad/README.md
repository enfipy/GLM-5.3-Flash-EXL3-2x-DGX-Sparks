# Nomad deployment of this recipe on 2x DGX Spark (enfis1 + enfis2)

This directory runs the upstream recipe as one Nomad job instead of
`start.sh`: the head rank on enfis1 (Ubuntu) and the worker rank on enfis2
(Enfios), tensor parallel 2 over the ConnectX-7 RoCE link, serving
`http://100.76.243.97:8888/v1` (tailnet only) as model id `glm-5.3-flash`.

```bash
nomad/model.sh start [--vision]   # submit the job (text-only by default)
nomad/model.sh status             # allocations + API health
nomad/model.sh warmup             # upstream boot-shape warmup, run once after READY
nomad/model.sh stop
```

`model.sh` needs the private control-host wrapper outside the tree:
`~/.config/glm53-nomad/control/control-ssh` (SSH to the control host) and
`nomad-remote` (Nomad address, CA and token on the control host). Override
the location with `GLM53_NOMAD_CONTROL`.

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
- `volumes/` — dynamic host volume registrations (`nomad volume create`):
  enfis1 models at `/home/god/models/weights/glm53f/hf`, ablit transplant
  tensors at `/home/god/models/weights/glm53f/ablit-transplant`; enfis2 under
  `/state/models/`. The head cache volume is `/dev/shm/glm53-cache`
  (recreated by `model.sh start`).
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
x2 51.1, x4 75.3; the later thin-decode kernels (`GLM53_EXL3_MOE_FAST=1`,
40 / 78 tok/s) need an image built from the current Dockerfile.

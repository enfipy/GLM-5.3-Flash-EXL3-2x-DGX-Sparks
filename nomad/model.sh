#!/usr/bin/env bash
# GLM-5.3-Flash on two DGX Sparks as one Nomad job. See nomad/README.md.
set -euo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env
. "$dir/env"
[ -f "$dir/env.local" ] && . "$dir/env.local"
control="$GLM53_NOMAD_CONTROL/control-ssh"
[ -x "$control" ] || { echo "missing $control (private control-host wrapper, see README)" >&2; exit 1; }
nomad() { "$control" bash -s -- "$@" < "$GLM53_NOMAD_CONTROL/nomad-remote"; }
case "${1:-start}" in
 start)
  vision=false; ablit=true; gen=true
  shift
  for arg in "$@"; do
    case "$arg" in
      --vision) vision=true ;;
      --no-ablit) ablit=false ;;
      --no-gen) gen=false ;;
      *) echo "Usage: $0 start [--vision] [--no-ablit] [--no-gen]" >&2; exit 2 ;;
    esac
  done
  if ! systemctl is-active --quiet cloud-ctrl-nomad; then
    echo "Cannot start: cloud-ctrl-nomad is not running on this host." >&2
    echo "Inspect: journalctl -u cloud-ctrl-nomad -b --no-pager -n 30" >&2
    exit 1
  fi
  docker info >/dev/null 2>&1 || { echo "Cannot start: Docker is unavailable." >&2; exit 1; }
  docker image inspect "$GLM53_IMAGE" >/dev/null 2>&1 || { echo "Image $GLM53_IMAGE is missing here; run $dir/setup.sh" >&2; exit 1; }
  # Regenerate the job from this checkout so a git pull is always reflected.
  # shellcheck disable=SC2086
  [ "$gen" = true ] && python3 "$dir/gen_hcl.py" $GLM53_GEN_FLAGS --image "$GLM53_IMAGE"
  "$control" 'umask 077; cat > /tmp/glm53-default.nomad.hcl' < "$dir/glm53.nomad.hcl"
  submit="$(date -u +%Y%m%dT%H%M%SZ)"
  nomad job validate -var="vision=$vision" -var="ablit=$ablit" -var="submit=$submit" /tmp/glm53-default.nomad.hcl
  nomad job run -detach -var="vision=$vision" -var="ablit=$ablit" -var="submit=$submit" /tmp/glm53-default.nomad.hcl
  # Optional: keep a local Pi client's model entry in sync with the vision flag.
  if [ -f "$HOME/.pi/agent/models.json" ]; then python3 - "$vision" <<'PICONFIG'
import json,sys,pathlib
p=pathlib.Path.home()/".pi/agent/models.json"
d=json.loads(p.read_text())
for m in d.get("providers",{}).get("vllm-local",{}).get("models",[]):
 if m.get("id")=="glm-5.3-flash":
  m["contextWindow"]=262144
  m["maxTokens"]=32768
  m["input"]=["text","image"] if sys.argv[1]=="true" else ["text"]
tmp=p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d,indent=2)+"\n")
tmp.chmod(p.stat().st_mode & 0o777)
tmp.replace(p)
PICONFIG
  fi
  echo "Submitted glm53 (ablit=$ablit vision=$vision). Check readiness with: $0 status, then run: $0 warmup"
  ;;
 stop) nomad job stop -yes glm53 ;;
 warmup)
  # Upstream boot-shape warmup: precompiles DFlash2 / sampler / prefill-chunk
  # kernels so the first real requests do not pay JIT spikes. Run once after
  # "status" reports API READY (about a minute with a warm cache volume).
  curl -fsS --max-time 3 "$GLM53_API/health" >/dev/null || { echo "API not ready" >&2; exit 1; }
  GLM53_WARMUP_MAX_CONCURRENCY=4 GLM53_WARMUP_DFLASH_K=7 \
    bash "$dir/../scripts/boot-shape-warmup.sh" "$GLM53_API" glm-5.3-flash
  ;;
 status)
  nomad job status glm53
  if curl -fsS --max-time 3 "$GLM53_API/health" >/dev/null; then echo "API READY"; else echo "API NOT READY"; fi
  ;;
 bench)
  shift
  python3 "$dir/decode_bench.py" --url "$GLM53_API" "$@"
  ;;
 *) echo "Usage: $0 {start [--vision] [--no-ablit] [--no-gen]|stop|status|warmup|bench [args]}" >&2; exit 2 ;;
esac

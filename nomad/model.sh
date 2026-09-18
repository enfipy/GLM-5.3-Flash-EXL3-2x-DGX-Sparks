#!/usr/bin/env bash
set -euo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Private control-host access (SSH wrapper + Nomad helper) lives outside the
# repository: ~/.config/glm53-nomad/control/{control-ssh,nomad-remote}.
control_dir="${GLM53_NOMAD_CONTROL:-$HOME/.config/glm53-nomad/control}"
control="$control_dir/control-ssh"
[ -x "$control" ] || { echo "missing $control (private control-host wrapper)" >&2; exit 1; }
nomad() { "$control" bash -s -- "$@" < "$control_dir/nomad-remote"; }
case "${1:-start}" in
 start)
  vision=false; ablit=true
  shift
  for arg in "$@"; do
    case "$arg" in
      --vision) vision=true ;;
      --no-ablit) ablit=false ;;
      *) echo "Usage: $0 start [--vision] [--no-ablit]" >&2; exit 2 ;;
    esac
  done
  if ! systemctl is-active --quiet cloud-ctrl-nomad; then
    echo "Cannot start: cloud-ctrl-nomad is not running on enfis1." >&2
    echo "Inspect: journalctl -u cloud-ctrl-nomad -b --no-pager -n 30" >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "Cannot start: Docker is unavailable." >&2
    exit 1
  fi
  "$control" 'umask 077; cat > /tmp/glm53-default.nomad.hcl' < "$dir/glm53.nomad.hcl"
  submit="$(date -u +%Y%m%dT%H%M%SZ)"
  nomad job validate -var="vision=$vision" -var="ablit=$ablit" -var="submit=$submit" /tmp/glm53-default.nomad.hcl
  nomad job run -detach -var="vision=$vision" -var="ablit=$ablit" -var="submit=$submit" /tmp/glm53-default.nomad.hcl
  python3 - "$vision" <<'PICONFIG'
import json,sys,pathlib
p=pathlib.Path.home()/".pi/agent/models.json"
d=json.loads(p.read_text())
for m in d["providers"]["vllm-local"]["models"]:
 if m["id"]=="glm-5.3-flash":
  m["contextWindow"]=262144
  m["maxTokens"]=32768
  m["input"]=["text","image"] if sys.argv[1]=="true" else ["text"]
tmp=p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(d,indent=2)+"\n")
tmp.chmod(p.stat().st_mode & 0o777)
tmp.replace(p)
PICONFIG
  echo "Submitted glm53. Check readiness with: $0 status"
  ;;
 stop) nomad job stop -yes glm53 ;;
 warmup)
  # Upstream boot-shape warmup: precompiles DFlash2 / sampler / prefill-chunk
  # kernels so the first real requests do not pay multi-second JIT spikes.
  # Run once after "status" reports API READY (takes a few minutes).
  curl -fsS --max-time 3 http://100.76.243.97:8888/health >/dev/null || { echo "API not ready" >&2; exit 1; }
  GLM53_WARMUP_MAX_CONCURRENCY=4 GLM53_WARMUP_DFLASH_K=7 \
    bash "$dir/../scripts/boot-shape-warmup.sh" http://100.76.243.97:8888 glm-5.3-flash
  ;;
 status)
  nomad job status glm53
  if curl -fsS --max-time 3 http://100.76.243.97:8888/health >/dev/null; then
    echo "API READY"
  else
    echo "API NOT READY"
  fi
  ;;
 *) echo "Usage: $0 {start [--vision] [--no-ablit]|stop|status|warmup}" >&2; exit 2 ;;
esac

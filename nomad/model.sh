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
usage() { echo "Usage: $0 {start [--vision] [--no-ablit] [--no-gen]|status|logs [head|worker|warmup] [-n N]|warmup|bench [args]|stop}" >&2; exit 2; }
# Allocation IDs of the latest job version, one "id group" pair per line.
latest_allocs() {
  nomad job allocs -json glm53 2>/dev/null | python3 -c '
import json, sys
al = json.load(sys.stdin)
if al:
    v = max(a["JobVersion"] for a in al)
    for a in al:
        if a["JobVersion"] == v:
            ts = a.get("TaskStates") or {}
            print(a["ID"], a["TaskGroup"], a["ClientStatus"], " ".join("%s=%s" % (k, t["State"]) for k, t in sorted(ts.items())))'
}
case "${1:-start}" in
 start)
  vision=false; ablit=true; gen=true
  shift
  for arg in "$@"; do
    case "$arg" in
      --vision) vision=true ;;
      --no-ablit) ablit=false ;;
      --no-gen) gen=false ;;
      *) usage ;;
    esac
  done
  if ! systemctl is-active --quiet cloud-ctrl-nomad; then
    echo "Cannot start: cloud-ctrl-nomad is not running on this host." >&2
    echo "Inspect: journalctl -u cloud-ctrl-nomad -b --no-pager -n 30" >&2
    exit 1
  fi
  docker info >/dev/null 2>&1 || { echo "Cannot start: Docker is unavailable." >&2; exit 1; }
  if ! docker image inspect "$GLM53_IMAGE" >/dev/null 2>&1; then
    # Nomad's Docker driver drops unused images 3 minutes after the last task
    # unless the client sets gc { image_delay }; reload the saved copy if there is one.
    tar="$GLM53_WEIGHTS/images/$(printf '%s' "$GLM53_IMAGE" | tr '/:' '--').tar"
    [ -f "$tar" ] || { echo "Image $GLM53_IMAGE is missing here and no $tar; run $dir/setup.sh" >&2; exit 1; }
    echo "Image $GLM53_IMAGE was garbage-collected; reloading it from $tar (a few minutes)" >&2
    docker load -i "$tar" >&2
  fi
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
  echo "Submitted glm53 (ablit=$ablit vision=$vision). The head's poststart task warms the model up once the API is healthy."
  echo "Follow it with: $0 status  (warmup=dead means done) and: $0 logs warmup"
  ;;
 stop) nomad job stop -yes glm53 ;;
 warmup)
  # Upstream boot-shape warmup, the same thing the job's poststart "warmup"
  # task runs after every start. Rerun it by hand after the API is READY if
  # you want fresh kernels compiled again (about a minute with a warm cache).
  curl -fsS --max-time 3 "$GLM53_API/health" >/dev/null || { echo "API not ready" >&2; exit 1; }
  GLM53_WARMUP_MAX_CONCURRENCY=4 GLM53_WARMUP_DFLASH_K=7 \
    bash "$dir/../scripts/boot-shape-warmup.sh" "$GLM53_API" glm-5.3-flash
  ;;
 status)
  nomad job status glm53
  echo
  echo "Latest version tasks (vllm=running + warmup=dead is fully warmed up):"
  latest_allocs | while read -r id group state tasks; do echo "  ${id:0:8} $group $state $tasks"; done
  if curl -fsS --max-time 3 "$GLM53_API/health" >/dev/null; then echo "API READY"; else echo "API NOT READY"; fi
  ;;
 logs)
  # logs [head|worker|warmup] [-n N]: vLLM stderr of a rank, or the warmup task's output.
  shift
  which="${1:-head}"; [ $# -gt 0 ] && shift
  n=100; if [ "${1:-}" = "-n" ]; then n="$2"; shift 2; fi
  case "$which" in
    head|warmup) group=head ;;
    worker) group=worker ;;
    *) usage ;;
  esac
  id="$(latest_allocs | awk -v g="$group" '$2==g{print $1; exit}')"
  [ -n "$id" ] || { echo "no $group allocation in the latest job version" >&2; exit 1; }
  if [ "$which" = warmup ]; then
    nomad alloc logs -stdout -tail -n "$n" "$id" warmup
    nomad alloc logs -stderr -tail -n "$n" "$id" warmup
  else
    nomad alloc logs -stderr -tail -n "$n" "$id" vllm
  fi
  ;;
 bench)
  shift
  python3 "$dir/decode_bench.py" --url "$GLM53_API" "$@"
  ;;
 *) usage ;;
esac

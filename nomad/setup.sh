#!/usr/bin/env bash
# One-time (and idempotent) preparation of the head for nomad/model.sh:
#   1. check the weights layout under $GLM53_WEIGHTS
#   2. place the cooperative kernel runtime next to the weights
#   3. load the container image from $GLM53_WEIGHTS/images if it is missing
#   4. register the Nomad host volumes for both ranks from nomad/env
# Needs the private control wrapper in $GLM53_NOMAD_CONTROL. Safe to rerun.
set -euo pipefail
dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$dir/env"
[ -f "$dir/env.local" ] && . "$dir/env.local"
control="$GLM53_NOMAD_CONTROL/control-ssh"
[ -x "$control" ] || { echo "missing $control: create $GLM53_NOMAD_CONTROL/{control-ssh,nomad-remote} (see README)" >&2; exit 1; }
nomad() { "$control" bash -s -- "$@" < "$GLM53_NOMAD_CONTROL/nomad-remote"; }

echo "== weights ($GLM53_WEIGHTS)"
target="$GLM53_WEIGHTS/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots"
draft="$GLM53_WEIGHTS/hf/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots"
ok=true
[ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ] || { echo "  missing target checkpoint: HF_HOME=$GLM53_WEIGHTS/hf $dir/../download.sh"; ok=false; }
[ -d "$draft" ] && [ -n "$(ls -A "$draft" 2>/dev/null)" ] || { echo "  missing DFlash2 drafter: HF_HOME=$GLM53_WEIGHTS/hf $dir/../download.sh"; ok=false; }
[ -f "$GLM53_WEIGHTS/ablit-transplant/L15.bin" ] || echo "  note: no ablit-transplant tensors (start with --no-ablit, or fetch them with ablit/fetch_transplant.py into $GLM53_WEIGHTS/ablit-transplant)"
mkdir -p "$GLM53_WEIGHTS/cache" "$GLM53_WEIGHTS/images"
$ok || { echo "weights incomplete" >&2; exit 1; }
echo "  ok"

echo "== cooperative kernel runtime"
mkdir -p "$GLM53_WEIGHTS/hf/cooperative_moe"
install -m 644 "$dir/cooperative_moe/cooperative_moe.so" "$dir/cooperative_moe/runtime.py" "$dir/cooperative_moe/SHA256SUMS" "$GLM53_WEIGHTS/hf/cooperative_moe/"
(cd "$GLM53_WEIGHTS/hf/cooperative_moe" && sha256sum -c --ignore-missing SHA256SUMS | sed 's/^/  /')

echo "== image $GLM53_IMAGE"
if docker image inspect "$GLM53_IMAGE" >/dev/null 2>&1; then
  echo "  present"
else
  tar="$GLM53_WEIGHTS/images/$(printf '%s' "$GLM53_IMAGE" | tr '/:' '--').tar"
  if [ -f "$tar" ]; then docker load -i "$tar"; else
    echo "  not present and no $tar; build it: docker buildx build --load --build-arg GLM53_RECIPE_STAMP=local -t $GLM53_IMAGE $dir/.. (then docker save -o $tar $GLM53_IMAGE)" >&2; exit 1
  fi
fi

echo "== host volumes"
# Idempotent: reuse the existing volume ID for the same name on the same node.
existing="$(nomad volume status -type host -verbose 2>/dev/null || true)"
register() { # name node path access_mode
  local name=$1 node=$2 path=$3 mode=$4 id
  id="$(printf '%s\n' "$existing" | awk -v n="$name" -v d="$node" '$2==n && $5==d {print $1}' | head -1)"
  if [ -n "$id" ] && [ "$(nomad volume status -type host "$id" 2>/dev/null | awk -F' = ' '/^Host Path/ {print $2}')" = "$path" ]; then
    echo "  $name on $node: unchanged ($path)"; return 0
  fi
  { [ -n "$id" ] && printf 'id = "%s"\n' "$id"; printf 'type = "host"\nname = "%s"\nnode_id = "%s"\nhost_path = "%s"\ncapability {\n  access_mode = "%s"\n  attachment_mode = "file-system"\n}\n' "$name" "$node" "$path" "$mode"; } \
    | "$control" 'umask 077; cat > /tmp/glm53-volume.hcl' 
  # A volume claimed by a running allocation cannot be updated; that is fine
  # when its path is unchanged, so report and continue.
  nomad volume register /tmp/glm53-volume.hcl 2>&1 | sed 's/^/  /' || true
}
register glm53-models           "$GLM53_HEAD_NODE"   "$GLM53_WEIGHTS/hf"               single-node-reader-only
register glm53-ablit-transplant "$GLM53_HEAD_NODE"   "$GLM53_WEIGHTS/ablit-transplant" single-node-reader-only
register glm53-head-cache       "$GLM53_HEAD_NODE"   "$GLM53_WEIGHTS/cache"            single-node-writer
register glm53-models           "$GLM53_WORKER_NODE" "$GLM53_WORKER_MODELS"            single-node-reader-only
register glm53-cache            "$GLM53_WORKER_NODE" "$GLM53_WORKER_CACHE"             single-node-writer
register glm53-ablit-transplant "$GLM53_WORKER_NODE" "$GLM53_WORKER_CACHE/ablit-transplant" single-node-reader-only
"$control" 'rm -f /tmp/glm53-volume.hcl'
nomad volume status -type host | sed 's/^/  /'
echo "== done; start with: $dir/model.sh start"

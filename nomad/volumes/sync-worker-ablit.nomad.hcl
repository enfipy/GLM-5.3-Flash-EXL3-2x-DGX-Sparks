job "glm53-ablit-sync" {
  datacenters = ["dc1"]
  type = "batch"

  group "sync" {
    count = 1
    constraint {
      attribute = "${node.unique.id}"
      value = "9b5e3f32-d773-5b0e-449f-2d51ffa40067"
    }
    restart {
      attempts = 0
      mode = "fail"
    }
    reschedule {
      attempts = 0
      unlimited = false
    }
    volume "cache" {
      type = "host"
      source = "glm53-cache"
    }

    task "sync" {
      driver = "docker"
      template {
        destination = "local/sync.py"
        change_mode = "noop"
        data = <<PY
import hashlib
import json
import pathlib
import urllib.request

base = "http://10.100.200.1:18080"
dest = pathlib.Path("/cache/ablit-transplant")
dest.mkdir(parents=True, exist_ok=True)
manifest = json.load(urllib.request.urlopen(base + "/MANIFEST.json"))
for layer, info in sorted(manifest["layers"].items(), key=lambda item: int(item[0])):
    name = f"L{layer}.bin"
    target = dest / name
    if target.is_file() and target.stat().st_size == info["nbytes"]:
        if hashlib.sha256(target.read_bytes()).hexdigest() == info["sha256"]:
            print(f"{name}: already valid", flush=True)
            continue
    tmp = target.with_suffix(".bin.tmp")
    print(f"{name}: downloading {info['nbytes']} bytes", flush=True)
    urllib.request.urlretrieve(base + "/" + name, tmp)
    if tmp.stat().st_size != info["nbytes"]:
        raise RuntimeError(f"{name}: wrong size {tmp.stat().st_size}")
    if hashlib.sha256(tmp.read_bytes()).hexdigest() != info["sha256"]:
        raise RuntimeError(f"{name}: sha256 mismatch")
    tmp.replace(target)
(dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"synced {len(manifest['layers'])} transplant tensors", flush=True)
PY
      }
      config {
        image = "sha256:ff09065cda8b201513662d8f0a1c89b82e40b513c8a5e051d996d5e919482594"
        command = "python3"
        entrypoint = []
        args = ["/local/sync.py"]
        network_mode = "host"
      }
      volume_mount {
        volume = "cache"
        destination = "/cache"
      }
      resources {
        cpu = 1000
        memory = 1024
      }
      logs {
        max_files = 2
        max_file_size = 10
      }
    }
  }
}

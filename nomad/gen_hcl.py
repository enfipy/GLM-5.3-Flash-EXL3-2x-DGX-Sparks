#!/usr/bin/env python3
"""Re-embed this repository's runtime overlay into nomad/glm53.nomad.hcl.

The Nomad job cannot mount host files from the recipe checkout on both ranks,
so the launcher template in the job carries a base64 tar of everything the
upstream launcher mounts into /opt/glm53 (the overlay patches in
GLM53_OVERLAY_ORDER, exl3.py, ablit_runtime.py and files/chat_template.jinja)
and applies them in the same order at container start.

Run after a `git pull` that touches overlay/, files/ or start.sh, then submit
with ./model.sh start.

  python3 nomad/gen_hcl.py                 # stock exl3.py
  python3 nomad/gen_hcl.py --coop          # nomad/cooperative_moe/exl3-cooperative.py
  python3 nomad/gen_hcl.py --image IMAGE   # switch the container image on both ranks
  python3 nomad/gen_hcl.py --fast          # thin-decode kernels (image built from this Dockerfile)
  python3 nomad/gen_hcl.py --instanttensor # direct-I/O weight loading (about 60 s instead of 300 s)
"""
import argparse, base64, gzip, io, pathlib, re, subprocess, sys, tarfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

def overlay_order(start_sh: pathlib.Path) -> list[str]:
    text = start_sh.read_text()
    m = re.search(r"GLM53_OVERLAY_ORDER=\((.*?)\)", text, re.S)
    if not m:
        sys.exit("GLM53_OVERLAY_ORDER not found in start.sh")
    return [line.strip() for line in m.group(1).splitlines() if line.strip().endswith(".py")]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=pathlib.Path, default=HERE / "glm53.nomad.hcl")
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "glm53.nomad.hcl")
    ap.add_argument("--coop", action="store_true", help="embed nomad/cooperative_moe/exl3-cooperative.py as exl3.py")
    ap.add_argument("--image", help="container image reference for both ranks")
    ap.add_argument("--fast", action="store_true", help="GLM53_EXL3_MOE_FAST=1 (needs an image built with overlay/patch_exl3_decode_pipeline.py)")
    ap.add_argument("--instanttensor", action="store_true", help="--load-format instanttensor (direct-I/O weight loading; image must ship the instanttensor wheel)")
    a = ap.parse_args()

    order = overlay_order(ROOT / "start.sh")
    members: list[tuple[str, pathlib.Path]] = [(p, ROOT / "overlay" / p) for p in order]
    exl3 = HERE / "cooperative_moe" / "exl3-cooperative.py" if a.coop else ROOT / "overlay" / "exl3.py"
    members += [("exl3.py", exl3), ("ablit_runtime.py", ROOT / "overlay" / "ablit_runtime.py"),
                ("chat_template.jinja", ROOT / "files" / "chat_template.jinja")]
    for name, path in members:
        if not path.is_file():
            sys.exit(f"missing {path}")

    # Deterministic: fixed tar metadata and a zero gzip timestamp, so the same
    # inputs always produce the same job file.
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz, tarfile.open(fileobj=gz, mode="w") as tar:
        for name, path in members:
            info = tar.gettarinfo(str(path), arcname=name)
            info.uid = info.gid = 0; info.uname = info.gname = ""; info.mtime = 0
            with open(path, "rb") as f:
                tar.addfile(info, f)
    payload = base64.b64encode(buf.getvalue()).decode()

    text = a.template.read_text()
    text, n = re.subn(r"payload = '[A-Za-z0-9+/=]+'", "payload = '" + payload + "'", text)
    if n != 2:
        sys.exit(f"expected two payload strings in the template, found {n}")
    patches = ", ".join(repr(p[:-3]) for p in order)
    text, n = re.subn(r"for patch in \[[^\]]*\]:", "for patch in [" + patches + "]:", text)
    if n != 2:
        sys.exit(f"expected two patch lists in the template, found {n}")
    rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    text = re.sub(r"upstream [0-9a-f]{7,40}", "upstream " + rev, text)
    fat = "0" if a.coop else "1"
    text, n = re.subn(r'EXL3_FAT_KERNEL = "[01]"', f'EXL3_FAT_KERNEL = "{fat}"', text)
    if n != 2:
        sys.exit("expected two EXL3_FAT_KERNEL settings")
    if a.image:
        text, n = re.subn(r'image = "[^"]+"', f'image = "{a.image}"', text)
        if n != 2:
            sys.exit("expected two image settings")
    text, n = re.subn(r'GLM53_EXL3_MOE_FAST = "[01]"', f'GLM53_EXL3_MOE_FAST = "{1 if a.fast else 0}"', text)
    if n != 2:
        sys.exit("expected two GLM53_EXL3_MOE_FAST settings")
    # InstantTensor sizes its staging buffer from the CUDA free-memory reading at
    # load time; keep the loader flag and its budget fraction together.
    text = text.replace('"--enable-prefix-caching", "--load-format", "instanttensor",', '"--enable-prefix-caching",')
    text = re.sub(r'        INSTANTTENSOR_MAX_FREE_MEM_USAGE = "[0-9.]+"\n', '', text)
    if a.instanttensor:
        text, n = re.subn(r'"--enable-prefix-caching",', '"--enable-prefix-caching", "--load-format", "instanttensor",', text)
        if n != 2:
            sys.exit("expected two --enable-prefix-caching args")
        text, n = re.subn(r'(        DEFAULT_MAX_NEW_TOKENS = "65536"\n)', r'\1        INSTANTTENSOR_MAX_FREE_MEM_USAGE = "0.9"\n', text)
        if n != 2:
            sys.exit("expected two DEFAULT_MAX_NEW_TOKENS settings")
    a.out.write_text(text)
    print(f"wrote {a.out} ({len(text)} bytes): {len(members)} payload members, exl3={'cooperative' if a.coop else 'stock'}, upstream {rev}", file=sys.stderr)

if __name__ == "__main__":
    main()

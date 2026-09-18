"""Create an exclusive opt-in overlay from the pinned source and native artifacts.

Does not download, install, change defaults, or restart anything. Rebuilt binaries
with a different hash require review and numerical validation before repinning.
The DS4.1 cooperative .so must not be substituted: it is locked to DeepSeek-V4.1.
"""

import argparse
import hashlib
from pathlib import Path, PurePosixPath

# Reviewed repin (thin-decode split from #182): overlay/exl3.py gained the opt-in
# GLM53_EXL3_MOE_FAST dispatch and its fail-closed load gates. With the flag unset
# the module builds the same pointer tables as before (the gate/up SUH alias is
# created only in fast mode, and it is content-identical by the load-time per-expert
# torch.equal proof), so a generated profile behaves as it did. Refusal on any
# further drift is unchanged.
STOCK_SHA = "7677ab42f4a20698371b5c22d27ecb1b0b416a6860d00a137e13f25e9fd0ed40"
BINARY_SHA = "ade0e5607fccec977b95d83ac2ce323d91fe3d56c4bed4f498f486131ab66cef"
ADAPTER_SHA = "3af923cc26e556402d95d64f2d98a63b49d57aae5f749d464ff88f93c6d43689"


def checked(path, digest):
    data = path.read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if digest.startswith("UNVALIDATED") or got != digest:
        raise ValueError(f"Unvalidated source/binary hash: {path}")
    return data


def make_profile(stock, artifacts, runtime_directory, output):
    base = checked(Path(stock), STOCK_SHA)
    artifacts = Path(artifacts)
    checked(artifacts / "cooperative_moe.so", BINARY_SHA)
    checked(artifacts / "runtime.py", ADAPTER_SHA)
    runtime_root = PurePosixPath(runtime_directory)
    if not runtime_root.is_absolute() or ".." in runtime_root.parts:
        raise ValueError(
            "Use an absolute container runtime directory without parent traversal"
        )
    footer = (
        "\n# Explicit fixed cooperative MoE opt-in; unsupported calls stay stock.\n"
        "import runpy as _coop_runpy\nimport sys as _coop_sys\n"
        f'_coop_setup = _coop_runpy.run_path({str(runtime_root / "runtime.py")!r})\n'
        f'_coop_setup["install"](_coop_sys.modules[__name__], library_root={str(runtime_root)!r}, enabled=True)\n'
    )
    with Path(output).open("xb") as handle:
        handle.write(base + footer.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--runtime-directory",
        required=True,
        help="Container path containing the verified binary and adapter on BOTH ranks",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_profile(args.stock, args.artifacts, args.runtime_directory, args.output)
    print(f"Wrote opt-in overlay: {args.output}; no service changes made")

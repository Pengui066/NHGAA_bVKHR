#!/usr/bin/env python

"""Compile every GLSL shader stage under share/shaders into SPIR-V.

The renderer loads pre-built .spv modules at startup and never compiles GLSL
itself (src/vkpp/shader_module.cc), while *.spv is gitignored -- so a fresh
clone ships no shader binaries at all and the program cannot start until they
exist. This is the cross-platform route; the Linux route is `make shaders`,
which regenerates the per-directory Makefiles through utils/glslc.py.

Usage:
    python utils/build_shaders.py                     # write share/shaders/**/*.spv
    python utils/build_shaders.py --check             # compile to a temp dir only
    python utils/build_shaders.py --glslc C:/path/glslc.exe

glslc comes with the Vulkan SDK (its Bin/ dir is usually already on PATH).
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

STAGES = { ".vert", ".frag", ".comp", ".geom", ".tesc", ".tese" }

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHADERS = ROOT / "share" / "shaders"


def compile_one(glslc, source, destination):
    command = [glslc, "-O", "-g", "-c", "-o", str(destination), str(source)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        print(f"[FAIL] {source.relative_to(ROOT)}: {message}", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compile share/shaders/**/*.{vert,frag,comp,geom,tesc,tese} to .spv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--glslc", default="glslc",
                        help="glslc executable (default: from PATH, i.e. the Vulkan SDK)")
    parser.add_argument("--check", action="store_true",
                        help="compile into a temporary directory, do not touch share/shaders")
    args = parser.parse_args()

    if not SHADERS.is_dir():
        print(f"no shader directory at {SHADERS}", file=sys.stderr)
        return 1

    sources = sorted(path for path in SHADERS.rglob("*") if path.suffix in STAGES)
    if not sources:
        print(f"no shader stages found under {SHADERS}", file=sys.stderr)
        return 1

    try:
        subprocess.run([args.glslc, "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        print(f"couldn't run '{args.glslc}' -- install the Vulkan SDK or pass --glslc",
              file=sys.stderr)
        return 1

    staging = pathlib.Path(tempfile.mkdtemp(prefix="vkhr-shaders-")) if args.check else None

    compiled = 0
    failed = 0
    for source in sources:
        if staging:
            destination = staging / source.relative_to(SHADERS)
            destination = destination.with_suffix(source.suffix + ".spv")
            destination.parent.mkdir(parents=True, exist_ok=True)
        else:
            destination = source.with_suffix(source.suffix + ".spv")

        if compile_one(args.glslc, source, destination):
            compiled += 1
        else:
            failed += 1

    where = f"a temporary directory ({staging})" if staging else "share/shaders"
    print(f"compiled {compiled}/{len(sources)} shaders into {where}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

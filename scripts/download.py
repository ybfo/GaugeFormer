"""Download and verify selected main-manuscript assets from GitHub Releases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def unpack(archive, root):
    root = root.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = (root / member.name).resolve()
            if not path.is_relative_to(root) or not member.isfile():
                raise ValueError(f"Unsafe archive member: {member.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                try:
                    with tar.extractfile(member) as source:
                        shutil.copyfileobj(source, output, length=8 * 1024**2)
                except BaseException:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if path.exists():
                    if digest(path) != digest(temporary):
                        raise FileExistsError(f"Existing file differs: {path}")
                else:
                    temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--group",
        choices=("data", "sources", "initialization", "checkpoints", "results", "all"),
        default="data",
    )
    p.add_argument(
        "--method",
        choices=("moirai", "timer_xl", "gtm", "unitime", "cpiri", "tefn", "tggc"),
    )
    p.add_argument("--seed", type=int, choices=(17, 29, 43, 71, 101))
    p.add_argument(
        "--list", action="store_true", help="List selected assets without downloading"
    )
    args = p.parse_args()
    manifest = json.loads((ROOT / "configs/assets.json").read_text())
    assets = [
        a
        for a in manifest["assets"]
        if (args.group == "all" or a["group"] == args.group)
        and (not args.method or a.get("method", args.method) == args.method)
        and (not args.seed or a.get("seed", args.seed) == args.seed)
    ]
    if not assets:
        raise ValueError("No assets match")
    print(f'{len(assets)} assets, {sum(a["bytes"] for a in assets)/1024**3:.2f} GiB')
    for a in assets:
        print(a["name"], flush=True)
        if args.list:
            continue
        with tempfile.TemporaryDirectory(prefix="gaugeformer-") as directory:
            download = subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    manifest["tag"],
                    "--repo",
                    manifest["repository"],
                    "--pattern",
                    a["name"],
                    "--dir",
                    directory,
                ],
                capture_output=True,
                text=True,
            )
            if download.returncode:
                raise RuntimeError(
                    f"GitHub download failed for {a['name']}; check gh authentication "
                    "and connectivity to release-assets.githubusercontent.com."
                )
            archive = Path(directory) / a["name"]
            if archive.stat().st_size != a["bytes"] or digest(archive) != a["sha256"]:
                raise RuntimeError(f"Asset integrity check failed: {archive.name}")
            if a["group"] == "results":
                destination = ROOT / "results/reference" / a["name"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and digest(destination) != a["sha256"]:
                    raise FileExistsError(destination)
                shutil.copyfile(archive, destination)
            else:
                unpack(archive, ROOT)
    print("Done.")


if __name__ == "__main__":
    main()

"""Fetch bounded, revision-pinned public energy labels for our own research."""
from pathlib import Path
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "microsoft/msr-acc-tae25"
DIRECTORY = ROOT / ".research/data/msr-acc-tae25"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{REPOSITORY}"
    with urllib.request.urlopen(api, timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    record = {"repository": REPOSITORY, "revision": revision,
              "license": metadata.get("cardData", {}).get("license"), "files": []}
    for name, limit in (("README.md", 100_000), ("msr-acc-tae25-all.parquet", 128_000_000)):
        destination = DIRECTORY / name
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{name}"
        if not destination.exists():
            with urllib.request.urlopen(url, timeout=60) as response, destination.with_suffix(destination.suffix + ".partial").open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise RuntimeError(f"Public data download exceeds reserved bound: {name}")
                    output.write(chunk)
            destination.with_suffix(destination.suffix + ".partial").replace(destination)
        if destination.stat().st_size > limit:
            raise RuntimeError(f"Unexpected cached file size: {destination}")
        digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
        record["files"].append({"name": name, "url": url, "bytes": destination.stat().st_size, "sha256": digest})
    (DIRECTORY / "source.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()

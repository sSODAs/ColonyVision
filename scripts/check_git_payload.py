"""Audit a Git repository before commits and GitHub pushes.

Standard-library only. Prevents including private photos, external datasets,
downloaded/pretrained weights, output artifacts and large opaque files.
It also checks historical commit paths: .gitignore cannot remove past blobs.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PREFIXES = (
    "raw/", "annotations/", "outputs/", "weights/", "runs/", "results/",
    "predictions/", "checkpoints/", "artifacts/", "data/", "datasets/",
)
PRIVATE_DIR_NAMES = {
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".ipynb_checkpoints", ".cache", ".uv", ".vscode",
    ".idea", "node_modules",
}
PRIVATE_SUFFIXES = {
    ".pt", ".pth", ".ptl", ".ckpt", ".safetensors", ".onnx", ".engine",
    ".tflite", ".pb", ".h5", ".hdf5", ".keras", ".mlmodel",
    ".caffemodel", ".weights", ".torchscript", ".pkl", ".pickle",
    ".joblib", ".npy", ".npz", ".bin", ".gguf", ".model",
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".gif", ".dcm", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3",
    ".csv", ".tsv", ".parquet", ".feather", ".arrow", ".sqlite",
    ".sqlite3", ".db", ".zip", ".7z", ".rar", ".tar", ".gz", ".zst",
    ".pem", ".key",
}
PRIVATE_NAMES = {"thumbs.db", ".ds_store", "desktop.ini", "id_rsa", "id_ed25519"}
MAX_FILE_BYTES = 5 * 1024 * 1024

def git(*args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: "
                           + proc.stderr.decode("utf-8", "replace"))
    return proc.stdout

def prohibited(path: str) -> str | None:
    p = path.replace("\\", "/").removeprefix("./")
    low = p.lower()
    components = low.split("/")
    if low.startswith(PRIVATE_PREFIXES):
        return "private/runtime directory"
    if components[0] == "dataset" and low != "dataset/data.yaml":
        return "dataset file (only template data.yaml can be published)"
    if any(part in PRIVATE_DIR_NAMES for part in components[:-1]):
        return "cache/local environment directory"
    filename = components[-1]
    if filename in PRIVATE_NAMES or filename == ".env" or filename.startswith(".env."):
        if filename == ".env.example":
            return None
        return "credential or machine-specific file"
    if any(filename.endswith(ext) for ext in PRIVATE_SUFFIXES):
        return "model weight, external data or binary artifact"
    if filename.endswith((".log", ".tmp", ".part")):
        return "generated runtime artifact"
    return None

def main() -> int:
    errors: list[str] = []
    tracked = git("ls-files", "--cached", "-z").decode("utf-8", "replace").split("\0")
    tracked = [p for p in tracked if p]
    for p in tracked:
        reason = prohibited(p)
        if reason:
            errors.append(f"TRACKED {p}: {reason}")

    # Inspect *all* available commit history, not only the latest tree.
    history = git("log", "--all", "--format=", "--name-only").decode(
        "utf-8", "replace"
    ).splitlines()
    for p in sorted(set(name.strip() for name in history if name.strip())):
        reason = prohibited(p)
        if reason:
            errors.append(f"HISTORY {p}: {reason} (requires history cleanup)")

    # Catch unrecognized binary extensions by checking object sizes in index.
    for record in git("ls-files", "--stage", "-z").decode(
        "utf-8", "replace"
    ).split("\0"):
        if not record or "\t" not in record:
            continue
        index_info, path = record.split("\t", 1)
        fields = index_info.split()
        if len(fields) < 3 or fields[2] != "0":
            continue
        size_out = git("cat-file", "-s", fields[1])
        size = int(size_out.strip())
        if size > MAX_FILE_BYTES:
            errors.append(f"TRACKED {path}: {size:,} bytes (>5 MiB safety limit)")

    if errors:
        print("[BLOCKED] Git payload includes private or generated content:", file=sys.stderr)
        for entry in sorted(set(errors)):
            print(" - " + entry, file=sys.stderr)
        print("Do not use git add -f or --no-verify to bypass this check.", file=sys.stderr)
        return 1

    print(f"[OK] Git payload safe: {len(tracked)} tracked source/config files; "
          "no restricted paths found in Git history; no large blobs staged.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

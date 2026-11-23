"""Audit the final ColonyVision Git payload.

Published payload:
- final source/configuration,
- frozen Test50 metadata and exactly 50 Test50 photographs,
- exactly one final student model weight.

The Test50 photographs and model weight must be Git LFS pointers.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_MODEL = "weights/colonyvision_student_v1.pt"
TEST50_RAW_PREFIX = "raw/test50/"
EXPECTED_TEST50_IMAGES = 50

ALLOWED_CSV = {
    "evaluation/test50/selection.csv",
    "evaluation/test50/ground_truth_points.csv",
    "evaluation/test50/ground_truth_summary.csv",
    "evaluation/test50/reference_metrics.csv",
}

BLOCKED_PREFIXES = (
    "dataset/",
    "annotations/",
    "output/",
    "outputs/",
    ".venv/",
    "venv/",
    "env/",
    "runs/",
    "results/",
    "predictions/",
    "checkpoints/",
    "artifacts/",
)

BLOCKED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".cache",
    ".uv",
    ".vscode",
    ".idea",
}

MODEL_SUFFIXES = {
    ".pt", ".pth", ".ptl", ".ckpt", ".safetensors", ".onnx", ".engine",
    ".tflite", ".pb", ".h5", ".hdf5", ".keras", ".mlmodel",
    ".caffemodel", ".weights", ".torchscript", ".pkl", ".pickle",
    ".joblib", ".npy", ".npz", ".bin", ".gguf", ".model",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

OTHER_MEDIA_SUFFIXES = {
    ".gif", ".dcm", ".mp4", ".mov", ".avi", ".mkv", ".wav", ".mp3",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".zst",
}

MAX_NON_LFS_BYTES = 5 * 1024 * 1024


def git(*args: str, check: bool = True) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed: "
            + proc.stderr.decode("utf-8", "replace")
        )
    return proc.stdout


def normalized(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def is_test50_image(path: str) -> bool:
    low = normalized(path).lower()
    return (
        low.startswith(TEST50_RAW_PREFIX)
        and Path(low).suffix in IMAGE_SUFFIXES
    )


def prohibited(path: str) -> str | None:
    p = normalized(path)
    low = p.lower()
    parts = low.split("/")
    suffix = Path(low).suffix

    # The only publishable raw files are the frozen Test50 images.
    if low.startswith("raw/") and not is_test50_image(low):
        return "raw file outside frozen Test50"

    if low.startswith(BLOCKED_PREFIXES):
        return "private/generated directory"

    if any(part in BLOCKED_DIR_NAMES for part in parts[:-1]):
        return "cache or machine-specific directory"

    if "teacher" in low:
        return "teacher artifact/tooling"

    if suffix in MODEL_SUFFIXES and low != ALLOWED_MODEL:
        return "model/checkpoint other than final student weight"

    if suffix in IMAGE_SUFFIXES and not is_test50_image(low):
        return "image outside frozen Test50"

    if suffix in OTHER_MEDIA_SUFFIXES:
        return "binary media/archive"

    if suffix == ".csv" and low not in ALLOWED_CSV:
        return "CSV outside frozen Test50 metadata"

    name = parts[-1]
    if (
        name in {"thumbs.db", ".ds_store", "desktop.ini", "id_rsa", "id_ed25519"}
        or name == ".env"
        or (name.startswith(".env.") and name != ".env.example")
        or suffix in {".pem", ".key"}
    ):
        return "secret or machine-specific file"

    return None


def is_lfs_pointer_in_index(path: str) -> bool:
    stage = git("ls-files", "--stage", "--", path, check=False).decode(
        "utf-8", "replace"
    ).strip()
    if not stage or "\t" not in stage:
        return False
    index_info, _ = stage.split("\t", 1)
    fields = index_info.split()
    if len(fields) < 2:
        return False
    blob = git("cat-file", "-p", fields[1], check=False)
    blob = blob.lstrip(b"\xef\xbb\xbf")
    return blob.startswith(b"version https://git-lfs.github.com/spec/v1")


def lfs_configured(path: str) -> bool:
    attr = git("check-attr", "filter", "--", path).decode("utf-8", "replace")
    return "filter: lfs" in attr


def main() -> int:
    errors: list[str] = []

    tracked = [
        p
        for p in git("ls-files", "--cached", "-z")
        .decode("utf-8", "replace")
        .split("\0")
        if p
    ]

    for path in tracked:
        reason = prohibited(path)
        if reason:
            errors.append(f"TRACKED {path}: {reason}")

    if ALLOWED_MODEL not in tracked:
        errors.append(f"MISSING {ALLOWED_MODEL}: final student weight is not staged/tracked")
    else:
        if not lfs_configured(ALLOWED_MODEL):
            errors.append(f"{ALLOWED_MODEL}: Git LFS filter is not configured")
        if not is_lfs_pointer_in_index(ALLOWED_MODEL):
            errors.append(f"{ALLOWED_MODEL}: index does not contain a Git LFS pointer")

    test_images = sorted(p for p in tracked if is_test50_image(p))
    if len(test_images) != EXPECTED_TEST50_IMAGES:
        errors.append(
            f"Test50 images: expected {EXPECTED_TEST50_IMAGES}, tracked {len(test_images)}"
        )

    for path in test_images:
        if not lfs_configured(path):
            errors.append(f"{path}: Git LFS filter is not configured")
        elif not is_lfs_pointer_in_index(path):
            errors.append(f"{path}: index does not contain a Git LFS pointer")

    # Catch accidental large non-LFS blobs in the index.
    for record in git("ls-files", "--stage", "-z").decode(
        "utf-8", "replace"
    ).split("\0"):
        if not record or "\t" not in record:
            continue
        index_info, path = record.split("\t", 1)
        fields = index_info.split()
        if len(fields) < 3 or fields[2] != "0":
            continue
        if normalized(path) == ALLOWED_MODEL or is_test50_image(path):
            continue
        size = int(git("cat-file", "-s", fields[1]).strip())
        if size > MAX_NON_LFS_BYTES:
            errors.append(
                f"TRACKED {path}: {size:,} bytes (>5 MiB, not allowed outside LFS)"
            )

    if errors:
        print("[BLOCKED] Git payload is not ready:", file=sys.stderr)
        for entry in sorted(set(errors)):
            print(" - " + entry, file=sys.stderr)
        return 1

    print(
        f"[OK] Git payload ready: {len(tracked)} tracked files; "
        f"{len(test_images)} Test50 images + final student weight protected by Git LFS."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

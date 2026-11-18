"""Run installed local WeCFU Ultra for a folder via its public local HTTP API.
Does not edit originals or promote automatic predictions into YOLO ground truth.
Prerequisite: WeCFU local server is running with Ultra ready.
"""
import argparse
import csv
import hashlib
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def request(url, endpoint, method="GET", body=None, *, timeout=900):
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + endpoint, data=payload, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            blob = resp.read()
            return json.loads(blob) if blob else {}
    except urllib.error.HTTPError as exc:
        msg = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "raw" / "pilot_review")
    ap.add_argument("--output", type=Path, default=ROOT / "annotations" / "pseudo_labels" / "wecfu_ultra")
    ap.add_argument("--batch", default="colonyvision_ultra_pilot")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--limit", type=int, default=0, help="0 means all; 1 for a first test")
    args = ap.parse_args()
    source = args.source.resolve()
    files = sorted(f for f in source.iterdir() if f.is_file() and f.suffix.lower() in EXTS)
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        raise SystemExit("No image files found")
    config = request(args.url, "/api/config")
    sam = config.get("models", {}).get("sam2", {})
    if not sam.get("available"):
        raise SystemExit("Ultra/SAM is not available: " + str(sam))
    if config.get("web_mode"):
        raise SystemExit("Only the local WeCFU server supports path ingestion")
    batch = urllib.parse.quote(args.batch, safe="")
    images_endpoint = f"/api/batch/{batch}/images"
    try:
        existing = {i["name"] for i in request(args.url, images_endpoint)["images"]}
    except RuntimeError as exc:
        if not str(exc).startswith("HTTP 404:"):
            raise
        existing = set()
    missing = [str(f) for f in files if f.name not in existing]
    if missing:
        added = request(args.url, "/api/ingest", "POST",
                        {"paths": missing, "batch": args.batch, "view": "unknown"})
        print("Imported:", added.get("linked"), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    failed = 0
    for pos, path in enumerate(files, 1):
        name = urllib.parse.quote(path.name, safe="")
        ep = f"/api/batch/{batch}/image/{name}"
        print(f"[{pos}/{len(files)}] {path.name}", flush=True)
        try:
            try:
                before = request(args.url, ep + "/detections")
            except RuntimeError as exc:
                if not str(exc).startswith("HTTP 404:"):
                    raise
                before = request(args.url, ep + "/initialize", "POST", {"mode": "auto"})
            if before.get("counted") or before.get("reviewed") or before.get("detections"):
                print("  SKIP existing marks/count; no overwrite", flush=True)
                records.append((path.name, "skipped_existing", len(before.get("detections", [])), ""))
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            candidate = request(args.url, ep + "/model-preview", "POST", {
                "engine": "sam2_local_v1", "expected_revision": before["revision"],
                "expected_image_sha256": digest, "model_params": {},
                "only_unrecognized": True,
            })
            if not candidate.get("candidate_id"):
                raise RuntimeError("Missing candidate_id from SAM preview")
            result = request(args.url, ep + "/model-preview/apply", "POST", {
                "candidate_id": candidate["candidate_id"],
                "expected_revision": before["revision"],
                "expected_image_sha256": digest,
                "request_id": uuid.uuid4().hex,
                "replace_all": True, "only_unrecognized": True,
            })
            if not result.get("counted") or result.get("count_engine") != "sam2_local_v1":
                raise RuntimeError("SAM result was not confirmed saved")
            count = len(result.get("detections", []))
            records.append((path.name, "pseudo_unreviewed", count, ""))
            print("  SAM markers:", count, flush=True)
        except Exception as exc:
            failed += 1
            records.append((path.name, "failed_or_uncertain", "", str(exc)[:300]))
            print("  ERROR:", exc, file=sys.stderr, flush=True)
            print("  Stopping to avoid silently replaying an uncertain result.", file=sys.stderr)
            break
        finally:
            report = args.output / "ultra_run_report.csv"
            with report.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["image", "status", "candidate_count", "error"])
                w.writerows(records)
    # The ZIP includes per-image saved JSON + overlay PNG, not original photos.
    try:
        endpoint = f"/api/batch/{batch}/export.zip"
        req = urllib.request.Request(args.url.rstrip("/") + endpoint)
        tmp = args.output / "wecfu_ultra_export.zip.part"
        with urllib.request.urlopen(req, timeout=900) as response, tmp.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        tmp.replace(args.output / "wecfu_ultra_export.zip")
        print("Export:", args.output / "wecfu_ultra_export.zip", flush=True)
    except Exception as exc:
        print("Export failed; saved results remain in WeCFU workspace:", exc, file=sys.stderr)
        failed += 1
    print("Report:", args.output / "ultra_run_report.csv", flush=True)
    print("All SAM markers are CANDIDATES, not verified colony boxes.", flush=True)
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())

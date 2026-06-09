"""Download top-N most-downloaded ClawHub skills as the noise pool.

For each skill:
  - GET /api/v1/download?slug=<slug>  -> zip (SKILL.md + attachments + _meta.json)
  - extract into noise_pool/<slug>/
  - parse `description` from SKILL.md frontmatter
  - write metadata.json: skill_name, source, url, description, download_time, hash, downloads, version
  - dedup by zip-content sha256 (skip near-identical re-uploads)
  - retry transient failures; log permanent failures
  - resume-aware: skip slugs that already have metadata.json
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path.home() / "Desktop" / "skill_retrieval"
POOL = ROOT / "noise_pool"
MANIFEST = ROOT / "clawhub_full_manifest.json"
ERR_LOG = ROOT / "download_errors.log"
DEDUP_LOG = ROOT / "download_dedup.log"
DL_URL = "https://clawhub.ai/api/v1/download?slug="

TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
WORKERS = 12
RETRIES = 3

_seen_hashes: dict[str, str] = {}   # zip_sha256 -> first slug that had it
_lock = threading.Lock()
_err_lock = threading.Lock()


def _frontmatter_description(skill_md: str) -> str:
    """Pull `description` from the YAML frontmatter (handles block scalars)."""
    lines = skill_md.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return ""
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(data, dict):
        return ""
    desc = data.get("description") or ""
    return str(desc).strip()


def _body_fallback(skill_md: str) -> str:
    """First meaningful paragraph of the body when frontmatter has no description."""
    text = skill_md
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                text = "\n".join(lines[i + 1:])
                break
    for para in text.split("\n\n"):
        t = para.strip().lstrip("#").strip()
        if len(t) >= 15:
            return t.replace("\n", " ").strip()[:500]
    return ""


def _description(skill_md: str) -> str:
    return _frontmatter_description(skill_md) or _body_fallback(skill_md)


def _extract_files(raw: bytes, depth: int = 0) -> dict[str, bytes]:
    """basename -> bytes; recurse one level into nested zips (publisher mistakes)."""
    out: dict[str, bytes] = {}
    zf = zipfile.ZipFile(io.BytesIO(raw))
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        data = zf.read(name)
        if data[:4] == b"PK\x03\x04" and depth < 2:
            try:
                out.update(_extract_files(data, depth + 1))
                continue
            except zipfile.BadZipFile:
                pass
        out[Path(name).name] = data
    return out


def _log(path: Path, msg: str):
    with _err_lock:
        with path.open("a") as f:
            f.write(msg + "\n")


def fetch_zip(slug: str) -> bytes:
    url = DL_URL + urllib.parse.quote(slug)
    req = urllib.request.Request(url, headers={"User-Agent": "noise-pool-collector/0.1"})
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.0 * (2 ** attempt))
    raise RuntimeError(f"fetch failed after {RETRIES}: {last}")


def process(rec: dict) -> str:
    slug = rec["slug"]
    out_dir = POOL / slug
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        return "skip"

    raw = fetch_zip(slug)
    digest = hashlib.sha256(raw).hexdigest()

    with _lock:
        dup_of = _seen_hashes.get(digest)
        if dup_of is None:
            _seen_hashes[digest] = slug
    if dup_of is not None:
        _log(DEDUP_LOG, f"{slug}\tDUPLICATE_OF\t{dup_of}\t{digest}")
        return "dup"

    try:
        files = _extract_files(raw)
    except zipfile.BadZipFile:
        _log(ERR_LOG, f"{slug}\tBAD_ZIP")
        return "err"

    out_dir.mkdir(parents=True, exist_ok=True)
    for base, data in files.items():
        (out_dir / base).write_bytes(data)

    skill_md_text = files.get("SKILL.md", b"").decode("utf-8", errors="replace")
    if not skill_md_text:
        _log(ERR_LOG, f"{slug}\tNO_SKILL_MD")

    meta = {
        "skill_name": slug,
        "source": "clawhub",
        "url": DL_URL + urllib.parse.quote(slug),
        "description": _description(skill_md_text),
        "download_time": int(time.time()),
        "hash": digest,
        "downloads": rec.get("downloads", 0),
        "version": rec.get("ver"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return "ok"


def main():
    recs = json.load(open(MANIFEST))
    recs.sort(key=lambda x: x.get("downloads", 0), reverse=True)
    targets = recs[:TOP_N]
    print(f"target top-{TOP_N} by downloads; range "
          f"[{targets[0]['downloads']} .. {targets[-1]['downloads']}]", flush=True)

    counts = {"ok": 0, "skip": 0, "dup": 0, "err": 0}
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process, r): r["slug"] for r in targets}
        for fut in as_completed(futs):
            slug = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                _log(ERR_LOG, f"{slug}\tFAIL\t{e}")
                res = "err"
            counts[res] = counts.get(res, 0) + 1
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                rate = done / max(el, 1)
                eta = (len(targets) - done) / max(rate, 0.001)
                print(f"[{done}/{len(targets)} {el:.0f}s eta={eta:.0f}s] {counts}", flush=True)

    print(f"\nDONE in {time.time()-t0:.0f}s: {counts}")
    print(f"noise pool dir: {POOL}")


if __name__ == "__main__":
    main()

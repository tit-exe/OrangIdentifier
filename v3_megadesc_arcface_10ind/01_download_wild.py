"""
V2_1_download_wild_orangs.py
=============================
CNRS IPHC Strasbourg — Orang-outan V2 pipeline
Author: Titouane

SOURCES
-------
1. iNaturalist API     — research grade observations (Pongo abelii, pygmaeus, tapanuliensis)
2. GBIF multimedia.txt — pre-downloaded GBIF media manifest (StillImage only)
3. Flickr group pool   — https://www.flickr.com/groups/orangutans/pool/
4. Internet Archive    — orangutanpics.zip (direct download + extract)

All images saved to: D:\OrangIdentifier\V2\WILD_ORANGS\
  raw\iNaturalist\     <- iNaturalist photos
  raw\GBIF\            <- GBIF photos
  raw\Flickr\          <- Flickr group pool
  raw\InternetArchive\ <- Archive.org zip

RUN
---
    conda activate orangs
    pip install requests tqdm
    python D:\OrangIdentifier\V2\scripts\V2_1_download_wild_orangs.py

Fully resumable — already downloaded files are skipped.
Flickr note: set FLICKR_API_KEY env var for full access (free key at flickr.com/services/api/keys/apply)
"""

import os
import sys
import time
import json
import shutil
import zipfile
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

# ==============================================================================
# FORCE CACHE TO D:
# ==============================================================================

os.environ["HF_HOME"]    = r"D:\HuggingFaceCache"
os.environ["TORCH_HOME"] = r"D:\TorchCache"

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR      = Path(r"D:\OrangIdentifier\V2\WILD_ORANGS")
RAW_DIR       = BASE_DIR / "raw"
LOG_FILE      = BASE_DIR / "download_log.json"
STATS_FILE    = BASE_DIR / "download_stats.txt"

MULTIMEDIA_TXT    = BASE_DIR / "multimedia.txt"
ARCHIVE_ZIP_URL   = "https://archive.org/compress/orangutanpics/formats=JPEG&file=/orangutanpics.zip"
ARCHIVE_ZIP_LOCAL = BASE_DIR / "orangutanpics.zip"

FLICKR_API_BASE = "https://api.flickr.com/services/rest/"
FLICKR_GROUP_ID = "909974@N25"

# Corrected iNaturalist taxon IDs (verified)
TAXON_IDS = {
    "Pongo_abelii":        67982,
    "Pongo_pygmaeus":      43582,
    "Pongo_tapanuliensis": 569674,
}

IMAGE_SIZE          = "large"
QUALITY_GRADE       = "research"
PER_PAGE            = 200
MAX_WORKERS         = 6
DELAY_PAGES         = 0.8
MIN_FREE_GB         = 5.0
MAX_INATURALIST     = 60_000
REQUEST_TIMEOUT     = 45
API_BASE            = "https://api.inaturalist.org/v1"

# ==============================================================================
# HELPERS
# ==============================================================================

def get_free_gb() -> float:
    return shutil.disk_usage(str(RAW_DIR.drive) + "\\").free / 1e9

def format_size(b: int) -> str:
    if b < 1e6:  return f"{b/1e3:.1f} KB"
    if b < 1e9:  return f"{b/1e6:.1f} MB"
    return f"{b/1e9:.2f} GB"

def load_log() -> dict:
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "total_bytes": 0}

def save_log(log: dict):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

def safe_filename(url: str, prefix: str = "") -> str:
    name = Path(urlparse(url).path).name
    if not name or "." not in name:
        name = str(abs(hash(url))) + ".jpg"
    return (prefix + "_" + name) if prefix else name

def download_one(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest.stat().st_size
    try:
        headers = {"User-Agent": "OrangutanResearch/1.0 (CNRS IPHC Strasbourg)"}
        r = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True, headers=headers)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "image" not in ct and "octet" not in ct:
            return 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        size = dest.stat().st_size
        if size < 1000:
            dest.unlink(); return 0
        return size
    except Exception:
        if dest.exists(): dest.unlink()
        return 0

def parallel_download(tasks, log, downloaded_set, label, workers=MAX_WORKERS):
    pending = [(u, d, k) for u, d, k in tasks if k not in downloaded_set]
    if not pending:
        return 0, 0
    n_ok = 0; n_bytes = 0
    bar = tqdm(total=len(pending), desc=f"  {label}", unit="img", ncols=90,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, u, d): (u, d, k) for u, d, k in pending}
        for fut in as_completed(futures):
            u, d, k = futures[fut]
            size = fut.result()
            if size > 0:
                downloaded_set.add(k)
                log["downloaded"].append(k)
                log["total_bytes"] = log.get("total_bytes", 0) + size
                n_ok += 1; n_bytes += size
                bar.set_postfix({"tot": format_size(log["total_bytes"]), "free": f"{get_free_gb():.1f}G"})
            else:
                log["failed"].append(k)
            bar.update(1)
            if get_free_gb() < MIN_FREE_GB:
                bar.close()
                print(f"\n  [STOP] Less than {MIN_FREE_GB} GB free on D:")
                save_log(log); sys.exit(0)
    bar.close()
    return n_ok, n_bytes

# ==============================================================================
# SOURCE 1 — iNaturalist
# ==============================================================================

def download_inaturalist(log, downloaded_set):
    print("\n" + "=" * 70)
    print("  SOURCE 1/4 — iNaturalist (research-grade)")
    print("=" * 70)

    out_dir = RAW_DIR / "iNaturalist"
    for sp in TAXON_IDS:
        (out_dir / sp).mkdir(parents=True, exist_ok=True)

    total_done = sum(1 for k in downloaded_set if k.startswith("inat/"))

    for species_name, taxon_id in TAXON_IDS.items():
        sp_dir = out_dir / species_name
        try:
            r = requests.get(f"{API_BASE}/observations", params={
                "taxon_id": taxon_id, "quality_grade": QUALITY_GRADE,
                "has[]": "photos", "per_page": 1, "page": 1
            }, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            total = r.json().get("total_results", 0)
        except Exception as e:
            print(f"  [ERROR] {species_name}: {e}"); continue

        total_pages = min((total // PER_PAGE) + 1, 500)
        print(f"\n  {species_name.replace('_',' ')} — {total:,} obs → {total_pages} pages")

        for page in tqdm(range(1, total_pages + 1),
                         desc=f"  {species_name[:18]}", unit="pg", ncols=90):
            if total_done >= MAX_INATURALIST:
                print(f"  Cap {MAX_INATURALIST:,} reached."); break
            try:
                r = requests.get(f"{API_BASE}/observations", params={
                    "taxon_id": taxon_id, "quality_grade": QUALITY_GRADE,
                    "has[]": "photos", "per_page": PER_PAGE, "page": page,
                    "order": "created_at", "order_by": "desc"
                }, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                obs_list = r.json().get("results", [])
            except Exception:
                time.sleep(5); continue
            if not obs_list: break

            tasks = []
            for obs in obs_list:
                obs_id = obs.get("id")
                for i, photo in enumerate(obs.get("photos", [])[:2]):
                    url = photo.get("url", "")
                    if not url: continue
                    for sz in ["square","small","medium","large","original"]:
                        url = url.replace(f"/{sz}.", f"/{IMAGE_SIZE}.")
                    fname = f"{obs_id}_{i:02d}.jpg"
                    tasks.append((url, sp_dir / fname, f"inat/{species_name}/{fname}"))

            n_ok, _ = parallel_download(tasks, log, downloaded_set,
                                        f"iNat {species_name[:12]} {page}/{total_pages}")
            total_done += n_ok
            if page % 5 == 0: save_log(log)
            time.sleep(DELAY_PAGES)

        save_log(log)

    print(f"\n  iNaturalist done: {sum(1 for k in downloaded_set if k.startswith('inat/')):,} images")

# ==============================================================================
# SOURCE 2 — GBIF multimedia.txt
# ==============================================================================

def download_gbif(log, downloaded_set):
    print("\n" + "=" * 70)
    print("  SOURCE 2/4 — GBIF multimedia.txt")
    print("=" * 70)

    if not MULTIMEDIA_TXT.exists():
        print(f"  [SKIP] Not found: {MULTIMEDIA_TXT}")
        return

    out_dir = RAW_DIR / "GBIF"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Parsing {MULTIMEDIA_TXT.name} ...")
    tasks = []; seen = set()

    with open(MULTIMEDIA_TXT, encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        # Find column indices
        try:
            type_col = header.index("type")
            url_col  = header.index("identifier")
        except ValueError:
            type_col, url_col = 1, 3   # fallback positions

        for line in f:
            cols = line.strip().split("\t")
            if len(cols) <= max(type_col, url_col): continue
            media_type = cols[type_col] if type_col < len(cols) else ""
            url        = cols[url_col]  if url_col  < len(cols) else ""
            if "StillImage" not in media_type: continue
            if not url.startswith("http"): continue
            if url in seen: continue
            seen.add(url)
            fname = safe_filename(url, "gbif")
            tasks.append((url, out_dir / fname, f"gbif/{fname}"))

    print(f"  {len(tasks):,} unique image URLs found")

    BATCH = 500
    for i in range(0, len(tasks), BATCH):
        parallel_download(tasks[i:i+BATCH], log, downloaded_set,
                          f"GBIF {i//BATCH+1}/{(len(tasks)-1)//BATCH+1}")
        save_log(log)

    print(f"\n  GBIF done: {sum(1 for k in downloaded_set if k.startswith('gbif/')):,} images")

# ==============================================================================
# SOURCE 3 — Flickr
# ==============================================================================

def download_flickr(log, downloaded_set):
    print("\n" + "=" * 70)
    print("  SOURCE 3/4 — Flickr: orangutans group")
    print("=" * 70)
    print(f"  Group URL: https://www.flickr.com/groups/orangutans/pool/")

    out_dir = RAW_DIR / "Flickr"
    out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("FLICKR_API_KEY", "")

    if api_key:
        print(f"  API key found — full access mode")
        _flickr_full(out_dir, log, downloaded_set, api_key)
    else:
        print("  No FLICKR_API_KEY in environment.")
        print("  Get a FREE key at: https://www.flickr.com/services/api/keys/apply/")
        print("  Then run: set FLICKR_API_KEY=your_key_here")
        print("  Trying public RSS feed (recent photos only)...")
        _flickr_rss(out_dir, log, downloaded_set)

def _build_flickr_url(photo, size="b"):
    farm   = photo.get("farm")
    server = photo.get("server")
    pid    = photo.get("id")
    secret = photo.get("secret")
    if not all([farm, server, pid, secret]): return ""
    return f"https://farm{farm}.staticflickr.com/{server}/{pid}_{secret}_{size}.jpg"

def _flickr_full(out_dir, log, downloaded_set, api_key):
    per_page = 500
    try:
        r = requests.get(FLICKR_API_BASE, params={
            "method": "flickr.groups.pools.getPhotos", "api_key": api_key,
            "group_id": FLICKR_GROUP_ID, "per_page": 1, "page": 1,
            "format": "json", "nojsoncallback": 1,
        }, timeout=30)
        r.raise_for_status()
        total      = int(r.json().get("photos", {}).get("total", 0))
        total_pages = min((total // per_page) + 1, 200)
        print(f"  {total:,} photos in group")
    except Exception as e:
        print(f"  [ERROR] {e}"); return

    for page in tqdm(range(1, total_pages + 1), desc="  Flickr pages", unit="pg", ncols=90):
        try:
            r = requests.get(FLICKR_API_BASE, params={
                "method": "flickr.groups.pools.getPhotos", "api_key": api_key,
                "group_id": FLICKR_GROUP_ID, "per_page": per_page, "page": page,
                "format": "json", "nojsoncallback": 1,
                "extras": "url_b,url_c,url_z",
            }, timeout=30)
            r.raise_for_status()
            photos = r.json().get("photos", {}).get("photo", [])
        except Exception:
            time.sleep(5); continue

        tasks = []
        for p in photos:
            url = p.get("url_b") or p.get("url_c") or p.get("url_z") or _build_flickr_url(p, "b")
            if not url: continue
            fname = f"flickr_{p['id']}.jpg"
            tasks.append((url, out_dir / fname, f"flickr/{fname}"))

        parallel_download(tasks, log, downloaded_set, f"Flickr {page}/{total_pages}")
        if page % 5 == 0: save_log(log)
        time.sleep(1)

    save_log(log)
    print(f"\n  Flickr done: {sum(1 for k in downloaded_set if k.startswith('flickr/')):,} images")

def _flickr_rss(out_dir, log, downloaded_set):
    tasks = []
    try:
        r = requests.get("https://api.flickr.com/services/feeds/groups_pool.gne",
                         params={"id": FLICKR_GROUP_ID, "format": "json", "nojsoncallback": 1},
                         timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        print(f"  Public feed: {len(items)} recent photos")
        for item in items:
            url = item.get("media", {}).get("m", "").replace("_m.jpg", "_b.jpg")
            if not url: continue
            pid   = url.split("/")[-1].split("_")[0]
            fname = f"flickr_{pid}.jpg"
            tasks.append((url, out_dir / fname, f"flickr/{fname}"))
    except Exception as e:
        print(f"  RSS feed error: {e}")

    if tasks:
        parallel_download(tasks, log, downloaded_set, "Flickr RSS")
        save_log(log)

    print(f"  Flickr done: {sum(1 for k in downloaded_set if k.startswith('flickr/')):,} images (set FLICKR_API_KEY for more)")

# ==============================================================================
# SOURCE 4 — Internet Archive
# ==============================================================================

def download_internet_archive(log, downloaded_set):
    print("\n" + "=" * 70)
    print("  SOURCE 4/4 — Internet Archive: orangutanpics.zip")
    print("=" * 70)

    out_dir = RAW_DIR / "InternetArchive"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download ZIP
    if not ARCHIVE_ZIP_LOCAL.exists() or ARCHIVE_ZIP_LOCAL.stat().st_size < 1_000_000:
        print(f"  Downloading: {ARCHIVE_ZIP_URL}")
        try:
            headers = {"User-Agent": "OrangutanResearch/1.0"}
            r = requests.get(ARCHIVE_ZIP_URL, stream=True, timeout=120, headers=headers)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            print(f"  Size: {format_size(total_size) if total_size else 'unknown'}")
            with open(ARCHIVE_ZIP_LOCAL, "wb") as f, tqdm(
                desc="  Downloading ZIP", total=total_size,
                unit="B", unit_scale=True, unit_divisor=1024, ncols=90
            ) as bar:
                for chunk in r.iter_content(65536):
                    f.write(chunk); bar.update(len(chunk))
            print(f"  ZIP saved: {format_size(ARCHIVE_ZIP_LOCAL.stat().st_size)}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            print(f"  Download manually from: {ARCHIVE_ZIP_URL}")
            print(f"  Save to: {ARCHIVE_ZIP_LOCAL}")
            return
    else:
        print(f"  ZIP already present: {format_size(ARCHIVE_ZIP_LOCAL.stat().st_size)}")

    # Extract
    print(f"  Extracting to {out_dir} ...")
    try:
        with zipfile.ZipFile(ARCHIVE_ZIP_LOCAL, "r") as zf:
            members = [m for m in zf.infolist()
                       if m.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                       and "__MACOSX" not in m.filename]
            print(f"  {len(members):,} images in ZIP")
            for m in tqdm(members, desc="  Extracting", unit="file", ncols=90):
                fname = Path(m.filename).name
                dest  = out_dir / fname
                key   = f"archive/{fname}"
                if key in downloaded_set or dest.exists():
                    continue
                try:
                    with zf.open(m) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    size = dest.stat().st_size
                    if size > 1000:
                        downloaded_set.add(key)
                        log["downloaded"].append(key)
                        log["total_bytes"] = log.get("total_bytes", 0) + size
                except Exception:
                    if dest.exists(): dest.unlink()
            save_log(log)
    except zipfile.BadZipFile:
        print("  [ERROR] Corrupted ZIP. Delete it and re-run.")
        return

    # Delete ZIP to free space
    try:
        ARCHIVE_ZIP_LOCAL.unlink()
        print("  ZIP deleted after extraction.")
    except Exception:
        pass

    print(f"\n  Archive done: {sum(1 for k in downloaded_set if k.startswith('archive/')):,} images")

# ==============================================================================
# FINAL REPORT
# ==============================================================================

def print_final_report(log, downloaded_set):
    sources = {
        "iNaturalist":      sum(1 for k in downloaded_set if k.startswith("inat/")),
        "GBIF":             sum(1 for k in downloaded_set if k.startswith("gbif/")),
        "Flickr":           sum(1 for k in downloaded_set if k.startswith("flickr/")),
        "InternetArchive":  sum(1 for k in downloaded_set if k.startswith("archive/")),
    }
    total_imgs  = sum(sources.values())
    total_bytes = log.get("total_bytes", 0)

    report = f"""
{'=' * 70}
  DOWNLOAD COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'=' * 70}

  Source breakdown:
    iNaturalist      : {sources['iNaturalist']:>7,} images
    GBIF             : {sources['GBIF']:>7,} images
    Flickr           : {sources['Flickr']:>7,} images
    Internet Archive : {sources['InternetArchive']:>7,} images
    ──────────────────────────────────────
    TOTAL            : {total_imgs:>7,} images
    Total size       : {format_size(total_bytes)}
    Free on D:       : {get_free_gb():.1f} GB remaining

  Output: {RAW_DIR}

  NEXT STEP:
    Run V2_2_extract_faces_wild.py
    YOLO will detect faces in all {total_imgs:,} images.
    Expected face crops: ~{total_imgs//5:,} to {total_imgs//3:,}
{'=' * 70}
"""
    print(report)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        f.write(report)

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("=" * 70)
    print("  ORANG-OUTAN UNIVERSAL IMAGE DOWNLOADER — V2")
    print("  CNRS IPHC Strasbourg")
    print("=" * 70)

    for p in [RAW_DIR, RAW_DIR/"iNaturalist", RAW_DIR/"GBIF",
              RAW_DIR/"Flickr", RAW_DIR/"InternetArchive"]:
        p.mkdir(parents=True, exist_ok=True)

    free = get_free_gb()
    print(f"\n  Free on D: {free:.1f} GB")
    print(f"  Output   : {RAW_DIR}")

    log            = load_log()
    downloaded_set = set(log["downloaded"])
    print(f"  Already downloaded: {len(downloaded_set):,} ({format_size(log.get('total_bytes',0))})")

    download_inaturalist(log, downloaded_set)
    download_gbif(log, downloaded_set)
    download_flickr(log, downloaded_set)
    download_internet_archive(log, downloaded_set)

    print_final_report(log, downloaded_set)

if __name__ == "__main__":
    main()
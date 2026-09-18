"""fetch every file an OpenNeuro manifest lists that is not already on disk, verified.

    python scripts/fetch_openneuro_manifest.py data/sources/ds008037/raw [--workers 6]

`raw/.openneuro-manifest-1.0.0.json` is the dataset's own file list: one entry per file,
with its size and a VERSIONED S3 url (`?versionId=...`), so a fetch reproduces the exact
snapshot the manifest describes rather than whatever the bucket holds today.  written for
ds008037, whose TMS-EEG recordings were staged and whose working-memory and rest EEG --
the delay-period data research step 2 needs (docs/LOG.md 2026-09-18) -- were not.

what it guarantees, because each of these has cost a run in this repo:
  * RESUMABLE.  a file is written to `<name>.part` and renamed only after its size matches
    the manifest.  an interrupted run leaves no file that looks complete and is not; a
    re-run skips every file already present at the right size.
  * VERIFIED.  size against the manifest for every file.  a present file at the WRONG size
    is reported and re-fetched, never trusted.
  * SUMMARISED AS IT GOES.  `<root>/.fetch_summary.json` is rewritten after every file, so
    a crash leaves a record of exactly what landed (CLAUDE.md: write the cheap summary
    before anything that can raise).
  * no credential is sent and no terms are accepted: OpenNeuro serves these anonymously,
    which is the acquisition record the card's DUA note relies on.
"""
from __future__ import annotations

import argparse, json, os, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MANIFEST = ".openneuro-manifest-1.0.0.json"
CHUNK = 1 << 20


def fetch_one(url: str, dest: str, size: int, retries: int = 4) -> tuple[bool, str]:
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url)
            if 0 < have < size:
                req.add_header("Range", f"bytes={have}-")
            else:
                have = 0
            with urllib.request.urlopen(req, timeout=120) as r:
                resumed = have and r.status == 206
                with open(part, "ab" if resumed else "wb") as fh:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        fh.write(b)
            got = os.path.getsize(part)
            if got != size:
                if got > size:
                    os.remove(part)
                raise IOError(f"size {got} != manifest {size}")
            os.replace(part, dest)
            return True, "ok"
        except Exception as e:  # noqa: BLE001 -- every failure is retried then reported
            err = f"{type(e).__name__}: {e}"
            time.sleep(min(30, 2 ** attempt))
    return False, err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.root, MANIFEST)))
    todo, present, wrong = [], 0, []
    for path, meta in man:
        if meta.get("directory"):
            continue
        dest = os.path.join(a.root, path)
        if os.path.isfile(dest):
            if os.path.getsize(dest) == meta["size"]:
                present += 1
                continue
            wrong.append(path)                       # re-fetch, never trust
        todo.append((path, meta["size"], meta["urls"][0]))
    total = sum(s for _, s, _ in todo)
    print(f"{present} present and verified, {len(wrong)} present at the WRONG size, "
          f"{len(todo)} to fetch ({total / 1e9:.1f} GB)", flush=True)
    if a.dry_run:
        return 0

    summ_path = os.path.join(a.root, ".fetch_summary.json")
    lock = threading.Lock()
    summ = {"manifest": MANIFEST, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "present_before": present, "wrong_size_before": wrong, "to_fetch": len(todo),
            "bytes_to_fetch": total, "fetched": 0, "bytes_fetched": 0, "failed": {}}

    def record():
        tmp = summ_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(summ, fh, indent=2)
        os.replace(tmp, summ_path)

    record()
    t0, done_b = time.time(), 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch_one, url, os.path.join(a.root, p), s): (p, s) for p, s, url in todo}
        for i, f in enumerate(as_completed(futs), 1):
            p, s = futs[f]
            ok, msg = f.result()
            with lock:
                if ok:
                    summ["fetched"] += 1
                    summ["bytes_fetched"] += s
                    done_b += s
                else:
                    summ["failed"][p] = msg
                record()
            if i % 50 == 0 or not ok:
                rate = done_b / max(1e-9, time.time() - t0) / 1e6
                print(f"  {i}/{len(todo)}  {done_b / 1e9:.1f}/{total / 1e9:.1f} GB  {rate:.1f} MB/s"
                      + ("" if ok else f"  FAILED {p}: {msg}"), flush=True)
    summ["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record()
    print(f"done: {summ['fetched']} fetched, {len(summ['failed'])} failed", flush=True)
    return 0 if not summ["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())

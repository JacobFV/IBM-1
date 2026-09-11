#!/usr/bin/env python3
"""Fetch the CMU Graphics Lab Motion Capture Database into data/sources/cmu-mocap/raw.

Why this corpus, from `docs/LOG.md` 2026-09-11: this body has two motions and neither is
admissible -- all 67 stored trajectories are outside the model's own declared ranges, and the depth
says the excursions are the operating regime rather than an overshoot (crawl-best's left knee is
past its limit for 95.6% of 1,600 frames). Recorded human motion is admissible BY CONSTRUCTION, so
it sidesteps the parameter search rather than solving it.

Licence: CMU states the database is free for all uses; the card records that the terms are not an
SPDX identifier and are not verified here. The bytes are gitignored and nothing is redistributed.

KNOWN ANSWER, checked against the card rather than against itself: `card.yaml` claims 144 subjects
and 2,600+ trials, written before any byte was fetched. The archive's own central directory must
corroborate both. If it does not, one of the two is wrong and the mismatch is reported rather than
absorbed -- a count that agrees with the count you already believed is the one nobody checks.

The download resumes. Re-running after a partial fetch continues it rather than starting over.
"""
import argparse, hashlib, json, re, sys, time, urllib.request, zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/sources/cmu-mocap"
RAW = SRC / "raw"
URL = "http://mocap.cs.cmu.edu/allasfamc.zip"
ARCHIVE = RAW / "allasfamc.zip"


def advertised_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["content-length"])


def fetch(url, dest, total):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        print(f"already complete: {have:,} bytes")
        return
    if have > total:
        sys.exit(f"local file is LARGER than advertised ({have:,} > {total:,}); not touching it")
    req = urllib.request.Request(url)
    if have:
        req.add_header("Range", f"bytes={have}-")
        print(f"resuming at {have:,} of {total:,}")
    started, last = time.time(), 0.0
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            have += len(chunk)
            now = time.time()
            if now - last > 10:
                last = now
                rate = have / max(now - started, 1e-9) / 1e6
                print(f"  {have:,} / {total:,}  ({have/total:.1%})  {rate:.1f} MB/s", flush=True)
    got = dest.stat().st_size
    if got != total:
        sys.exit(f"size mismatch after fetch: {got:,} against an advertised {total:,}")
    print(f"fetched {got:,} bytes")


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def card_claims():
    """The counts the card asserted BEFORE any byte was fetched."""
    text = (SRC / "card.yaml").read_text()
    n = re.search(r"^\s*n:\s*(\d+)", text, re.M)
    trials = re.search(r"([\d,]+)\+?\s*trials", text)
    return (int(n.group(1)) if n else None,
            int(trials.group(1).replace(",", "")) if trials else None)


def inventory(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    asf = [n for n in names if n.lower().endswith(".asf")]
    amc = [n for n in names if n.lower().endswith(".amc")]
    subj = Counter()
    for n in amc:
        m = re.search(r"(\d+)[_/]", Path(n).name)
        if m:
            subj[m.group(1)] += 1
    return names, asf, amc, subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-only", action="store_true")
    a = ap.parse_args()

    total = advertised_length(URL)
    print(f"{URL}\n  advertised {total:,} bytes")
    if not a.inventory_only:
        fetch(URL, ARCHIVE, total)

    if not ARCHIVE.exists():
        sys.exit("no archive to inventory")
    md5 = digest(ARCHIVE)
    names, asf, amc, subj = inventory(ARCHIVE)
    n_claim, t_claim = card_claims()

    print(f"\nmd5 {md5}   (recorded here; CMU publishes no checksum, so this is what we received,")
    print("            not a verification against the provider)")
    print(f"archive: {len(names):,} entries, {len(asf)} .asf skeletons, {len(amc):,} .amc trials,"
          f" {len(subj)} subjects by filename")

    print(f"\nKNOWN ANSWER against the card, written before the fetch:")
    ok = True
    for label, got, claim, rule in (("subjects", len(subj), n_claim, "equal"),
                                    ("trials", len(amc), t_claim, "at least")):
        if claim is None:
            print(f"  {label:9s} {got:6,}  card makes no claim")
            continue
        good = (got == claim) if rule == "equal" else (got >= claim)
        ok &= good
        print(f"  {label:9s} {got:6,} against the card's {claim:,} ({rule})"
              f"  {'PASS' if good else 'MISMATCH'}")
    if not ok:
        print("\n  MISMATCH is reported, not absorbed: one of the archive and the card is wrong,")
        print("  and a count that agrees with what you already believed is the one nobody checks.")

    rec = {"url": URL, "bytes": total, "md5": md5, "entries": len(names),
           "asf_skeletons": len(asf), "amc_trials": len(amc), "subjects": len(subj),
           "card_claims": {"subjects": n_claim, "trials": t_claim},
           "known_answer_pass": bool(ok),
           "licence_note": "CMU states free for all uses; terms are not an SPDX identifier and are "
                           "not verified here. Bytes are gitignored; nothing is redistributed.",
           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = SRC / "evidence/acquisition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

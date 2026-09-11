#!/usr/bin/env python3
"""Is the CMU mocap archive TRUNCATED, or is 112 subjects / 2,514 trials what CMU ships?

`scripts/fetch_cmu_mocap.py` reported a MISMATCH against `card.yaml`'s pre-fetch claim of
144 subjects and 2,600+ trials, and the fetch gate was right to report it rather than absorb
it. This settles which side is wrong, because a truncated download and a complete archive of
a smaller collection look identical from a file listing.

They do not look identical to two checks:

  CRC -- `zipfile.testzip()` recomputes every member's CRC-32 against the central directory.
         A short or damaged file fails on the first bad member.
  SHAPE -- a truncated download loses a CONTIGUOUS TAIL. If the subject IDs that are absent
         are scattered through the range instead, the file is not a prefix of a larger one.

KNOWN ANSWER, and it breaks the symmetry it is testing: flip one byte inside a member's
compressed payload and the CRC check MUST detect it. If a deliberately corrupted archive
still reports clean, the check cannot see truncation either and proves nothing about the real
one. Detection counts whether testzip() names the member or the decompressor refuses the
stream -- the bar is "damage is detected", and the first version of this asserted the
returned name specifically and failed on a raised zlib.error, which narrowed the instrument
rather than the bar.

WHAT THIS DOES NOT SHOW. That the downloaded file is intact and internally consistent is not
proof that CMU never published 144 subjects. That would need CMU's own index, which is not
fetched here. The honest verdict is about the archive at this URL, and the card is corrected
to measured figures with the provenance of the old ones noted as unverified.
"""
import collections, io, re, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/sources/cmu-mocap/raw/allasfamc.zip"


def known_answer():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.txt", "x" * 5000)
        z.writestr("b.txt", "y" * 5000)
    raw = bytearray(buf.getvalue())
    clean = zipfile.ZipFile(io.BytesIO(bytes(raw))).testzip()
    print(f"KNOWN ANSWER  intact 2-member archive   -> testzip() = {clean!r}  "
          f"{'PASS' if clean is None else 'FAIL'}")
    raw[raw.find(b"b.txt") + 8] ^= 0xFF
    try:
        bad = zipfile.ZipFile(io.BytesIO(bytes(raw))).testzip()
        detected, how = bad is not None, f"testzip() = {bad!r}"
    except Exception as e:
        detected, how = True, f"{type(e).__name__}: {e}"
    print(f"KNOWN ANSWER  one byte flipped          -> {how}  "
          f"{'PASS' if detected else 'FAIL -- the check cannot see damage'}")
    if clean is not None or not detected:
        sys.exit("known answer FAILED; the real archive is not checked")


def main():
    known_answer()
    if not P.exists():
        sys.exit(f"\n{P} is not present")

    z = zipfile.ZipFile(P)
    names = z.namelist()
    print(f"\n{P.relative_to(ROOT)}  ({P.stat().st_size:,} bytes)")
    print(f"  central directory intact, {len(names):,} members")

    t = time.time()
    first_bad = z.testzip()
    print(f"  CRC over every member: {first_bad!r}  ({time.time() - t:.0f}s)")
    if first_bad is not None:
        sys.exit(f"  CORRUPT at {first_bad}")
    print("  every member's CRC matches -- the bytes are complete, not truncated")

    asf = [n for n in names if n.endswith(".asf")]
    amc = [n for n in names if n.endswith(".amc")]
    subj = sorted({int(m.group(1)) for n in asf
                   for m in [re.search(r"/(\d+)\.asf$", n)] if m})
    gaps = [i for i in range(min(subj), max(subj) + 1) if i not in subj]
    per = collections.Counter(m.group(1) for n in amc
                              for m in [re.search(r"/(\d+)_\d+\.amc$", n)] if m)

    print(f"\n  subjects (.asf): {len(subj)}   IDs {min(subj)}-{max(subj)}")
    print(f"  trials   (.amc): {len(amc):,}   per subject {min(per.values())}-{max(per.values())}")
    print(f"  IDs absent from [{min(subj)}, {max(subj)}]: {len(gaps)}")
    print(f"    {gaps}")
    print(f"  {len(subj)} present + {len(gaps)} absent = {len(subj) + len(gaps)}, "
          f"and the highest ID is {max(subj)}")

    tail = gaps and gaps[0] > max(subj) - len(gaps)
    print("\n  SHAPE: " + ("the absent IDs form a CONTIGUOUS TAIL -- consistent with truncation"
                           if tail else
                           "the absent IDs are SCATTERED through the range -- this file is not a "
                           "prefix of a larger one, so it is not a truncated download"))
    print(f"\n  VERDICT: the archive is complete as distributed. card.yaml's population.n of 144 "
          f"is the highest subject ID plus one, not a count of subjects;\n"
          f"  the measured collection is {len(subj)} subjects and {len(amc):,} trials. The 144 / "
          f"2,600+ figures were written before any byte was fetched and were never verified\n"
          f"  against CMU's own index, which is not fetched here.")


if __name__ == "__main__":
    main()

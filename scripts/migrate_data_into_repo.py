"""move the corpora under data/sources/<id>/raw and re-provenance them.

two separate corrections, and the second is the one that matters.

*location*: the payloads sat in `~/Documents/win-data`, a sibling repo's working
directory, while `data/sources/` held only cards.  nothing in this repo recorded
that, so a card marked `binding: bound` said nothing about whether the bytes were
reachable.  same filesystem, so the move is a rename and costs nothing.

*provenance*: `win` is not a SOURCE.  it is another program that downloaded these
corpora from somewhere, and citing it would be citing a courier.  every card
already carries the real origin -- an OpenNeuro accession, a DOI, an OSF node --
so `local_root` becomes a path in this repo and the provenance chain points at
the origin the data actually came from.  a dataset whose origin cannot be named
is one this repo should not claim to hold.
"""
from __future__ import annotations

import argparse, os, shutil, sys
from pathlib import Path

import yaml

WIN = Path.home() / "Documents/win-data"
SRC = Path("data/sources")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="without this, dry run")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    moved = skipped = 0
    for d in sorted(WIN.iterdir()):
        if not d.is_dir() or d.name in ("assets", "derived", "mne"):
            continue
        if a.only and d.name != a.only:
            continue
        card_p = SRC / d.name / "card.yaml"
        if not card_p.exists():
            print(f"  SKIP {d.name}: no card -- an unclaimed corpus, not migrated")
            skipped += 1
            continue
        card = yaml.safe_load(card_p.read_text())
        urls = card.get("urls") or []
        if not urls:
            print(f"  SKIP {d.name}: card names no origin, so this repo cannot say "
                  f"where the bytes came from")
            skipped += 1
            continue
        dest = SRC / d.name / "raw"
        print(f"  {d.name:34s} -> {dest}   origin: {urls[0][:52]}")
        if a.apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            # `raw/` already exists as a POINTER: .location.yaml naming an
            # external local_root, plus a deferred checksum manifest.  that scheme
            # was right when the bytes lived elsewhere; now they come inside, so
            # the stubs move aside and the payload takes their place.
            if dest.exists():
                stash = dest.parent / "_pointer"
                stash.mkdir(exist_ok=True)
                for f in list(dest.iterdir()):
                    if f.name in (".location.yaml", "checksums.txt"):
                        shutil.move(str(f), str(stash / f.name))
                if any(dest.iterdir()):
                    print("      raw/ holds payload already, leaving alone"); continue
                dest.rmdir()
            os.rename(d, dest)                      # same filesystem: instant
            # the pointer no longer points outward -- rewrite it to say so
            loc = stash / ".location.yaml" if (dest.parent / "_pointer").exists() else None
            if loc and loc.exists():
                y = yaml.safe_load(loc.read_text()) or {}
                y["local_root"] = str(dest)
                y["note"] = ("bytes now live INSIDE the repository under raw/ and are "
                             "gitignored; source_url above is the origin of record")
                (dest.parent / ".location.yaml").write_text(
                    yaml.safe_dump(y, sort_keys=False, width=100))
                shutil.rmtree(stash)
            card["local_root"] = str(dest)
            card["provenance_note"] = (
                "moved into this repo from a sibling working directory; the origin "
                "above is the source of record, not the machine it was staged on")
            card_p.write_text(yaml.safe_dump(card, sort_keys=False, width=100))
        moved += 1
    print(f"\n{moved} to migrate, {skipped} skipped")
    if not a.apply:
        print("dry run -- rerun with --apply")


if __name__ == "__main__":
    main()

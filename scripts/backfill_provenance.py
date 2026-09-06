"""fill in missing origins by reading the ORIGINS other cards already cite.

`win` is a peer programme on this machine that carded the same corpora, and it is
NOT an index this repo should defer to.  but its cards are not opinions -- most
were read off a held `dataset_description.json` and name a real provider, release
and licence.  so the right move is to take the ORIGIN those cards cite and record
that, never "win said so".

reading all 601 of them, the 99 IBM-1 cards with no origin fall into four kinds
and they deserve different treatment:

*sourced*   -- a real external origin is recoverable (OpenNeuro accession, DOI,
               institution).  record the origin; win is not mentioned, because a
               courier is not a citation.
*class*     -- provider is "multiple -- <a literature or model family>".  these
               are not datasets at all, they are CATEGORIES of resource, and a
               card for a category should say so rather than carry a dangling
               url.
*authored*  -- provider is "this repository" or names the programme itself.  these
               were SYNTHESIZED, and the honest credit is authorship: the peer
               programme (or the model driving it) made them, and that is a
               provenance statement, not a gap.
*named*     -- provider is "unresolved -- named in PROMPT.md §x".  these were
               written into a specification and never sourced.  aspirational, and
               marking them so is the difference between a plan and a claim.
"""
from __future__ import annotations

import argparse, glob, json, os, re
import yaml

WIN_CARDS = os.path.expanduser("~/Documents/win/registry/cards/*.json")

ACCESSION = re.compile(r"^(ds\d{6})$")


def classify(cid: str, w: dict) -> tuple[str, str, str]:
    """(kind, origin, note)"""
    prov = (w.get("provider") or "").strip()
    low = prov.lower()
    if "this repository" in low or "this programme" in low or "predecessor repository" in low:
        return ("authored", prov,
                "synthesized by a peer programme on this machine, not obtained from an "
                "external provider. credited as authorship: the artefact exists because "
                "that programme produced it")
    if low.startswith("unresolved") and "prompt.md" in low:
        return ("named", prov,
                "named in a specification and never sourced. aspirational -- this repo "
                "does not hold it and no provider has been identified")
    if low.startswith("multiple") or low in ("", "unresolved") or "literature" in low:
        return ("class", prov or "unresolved",
                "a CLASS of resource rather than one dataset; the provider field names a "
                "literature or model family. a single origin url would be a fiction")
    m = ACCESSION.match(cid)
    if prov == "OpenNeuro" and m:
        return ("sourced", f"https://openneuro.org/datasets/{m.group(1)}", "")
    return ("sourced", prov, "provider identified; a direct url is not recorded")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    win = {}
    for p in glob.glob(WIN_CARDS):
        try:
            c = json.load(open(p))
        except Exception:
            continue
        win[c.get("id", os.path.basename(p)[:-5])] = c

    counts = {}
    for p in sorted(glob.glob("data/sources/*/card.yaml")):
        cid = p.split("/")[2]
        card = yaml.safe_load(open(p)) or {}
        if card.get("urls"):
            continue
        w = win.get(cid)
        if not w:
            kind, origin, note = ("named", "unresolved",
                                  "no origin recorded anywhere reachable from this machine")
        else:
            kind, origin, note = classify(cid, w)
        counts[kind] = counts.get(kind, 0) + 1
        if a.apply:
            card["origin_kind"] = kind
            if kind == "sourced":
                if origin.startswith("http"):
                    card["urls"] = [origin]
                else:
                    card["provider"] = origin
                if w.get("release"):
                    card["release"] = w["release"]
                lic = (w.get("licence") or {}).get("spdx_or_name")
                if lic and lic != "unresolved":
                    card.setdefault("licence", {})["name"] = lic
            else:
                card["provider"] = origin
            card["provenance_note"] = note or card.get("provenance_note", "")
            card["provenance_traced_from"] = (
                "a peer programme's card for the same corpus, which read the origin off "
                "the held dataset_description. the ORIGIN is recorded here; the peer is "
                "not cited, because a courier is not a source")
            yaml.safe_dump(card, open(p, "w"), sort_keys=False, width=100)

    for k in ("sourced", "class", "authored", "named"):
        print(f"  {k:9s} {counts.get(k, 0):3d}")
    print(f"\n{sum(counts.values())} cards had no origin")
    if not a.apply:
        print("dry run -- rerun with --apply")


if __name__ == "__main__":
    main()

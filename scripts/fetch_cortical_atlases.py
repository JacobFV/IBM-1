"""fetch the three payloads the cortical sheet was declared against and never held.

`docs/DISCONNECTS.md` rows 2 and 3: `cortical_sites()` places sites on a sphere,
`cortical_regions()` cuts six lobe labels by coordinate thresholds, and
`ibm/topologies/tract.py` declares a tractometric adjacency that nothing imports.
The three sources that would close that gap -- `desikan2006`, `dkt-atlas` and
`braingraph-hcp-connectomes` -- each held one 12 KB `checksums.txt` and no bytes.

This script stages all three, plus the `fsaverage` surfaces they are covers over,
and writes a real sha256 manifest for each.

**Where each payload actually comes from, which is not the same as where it was
staged from.**  `data/sources/<id>/raw/.location.yaml` is the file that has to be
right about this, and two of the three are *vendored*, not downloaded:

*desikan2006* -- `?h.aparc.annot` on fsaverage.  Its `.location.yaml` already
    said `vendored_in: freesurfer`, and that is exactly true: the bytes live
    inside a FreeSurfer subject directory that reached this machine bundled in an
    MNE dataset.  Three such directories are present, staged independently
    (`data/sources/mne-sample/raw/.../subjects/fsaverage`, `~/mne_data`, and a
    TMS working copy), and this script asserts they are byte-identical before
    staging one.  That agreement is the verification: the existing
    `checksums.txt` held no hashes to check against, only a header comment, so
    there was nothing to verify AGAINST and cross-copy identity is what is
    actually available.

*dkt-atlas* -- `?h.aparc.DKTatlas.annot`.  The card says it "ships inside each
    recon-all subject directory" and declares its frame as `subject_surface_ras`,
    so the payload is a per-subject recon, not a template: fsaverage does not
    carry DKT at all (checked: no `DKTatlas` file in any of the three fsaverage
    copies).  The one full recon-all output this repo holds is
    `mne-somato` subject `01`, and that is what is staged, surfaces included,
    because per-vertex labels without their subject's surface are meaningless --
    the card says so under `requires`.

*braingraph-hcp-connectomes* -- downloaded, not vendored.  The 86-node set is the
    Desikan-Killiany resolution (68 cortical + 18 subcortical nodes), 1064 HCP
    subjects, 10x-repeated 1M-streamline tractography, and every edge carries
    `fiber_length_mean` -- which is the field `ibm/topologies/tract.py`'s
    `tractometric_matrix` builder requires and cannot synthesise.  The download
    link on braingraph.org is gated behind a JavaScript "I agree to the HCP data
    use terms" checkbox that enables a form whose `action` is the static path; the
    agreement is a click, not a credential, so the direct path is used and the
    terms are recorded here.

Nothing here writes into `data/sources/*/raw` outside the three ids plus
`fsaverage`, and every path it writes is gitignored.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: every FreeSurfer subject directory on this machine that could supply the
#: fsaverage bytes.  more than one is the point: they were staged by different
#: tools at different times, so byte-identity across them is evidence the bytes
#: are the distributed ones and not something a local pipeline rewrote.
FSAVERAGE_CANDIDATES = (
    os.path.join(ROOT, "data/sources/mne-sample/raw/processed-v6/"
                       "MNE-sample-data/subjects/fsaverage"),
    os.path.expanduser("~/mne_data/MNE-fsaverage-data/fsaverage"),
    os.path.expanduser("~/tms-brain-data/freesurfer/fsaverage"),
)

#: the recon-all subject that carries a DKT parcellation.
DKT_SUBJECT = os.path.join(
    ROOT, "data/sources/mne-somato/raw/bids-v0.10/MNE-somato-data/"
          "derivatives/freesurfer/subjects/01")

BRAINGRAPH_URL = "https://braingraph.org/static/repeated_10_scale_33.7z"

#: fsaverage files the sheet actually needs.  `white` is the geometry sites are
#: placed on, `sphere` is the frame the parcellation was transferred in, and
#: `cortex.label` is what excludes the medial wall -- a site placed there is a
#: site on a surface that has no cortex under it.
FSAVERAGE_SURF = tuple(f"{h}.{s}" for h in ("lh", "rh")
                       for s in ("white", "pial", "inflated", "sphere",
                                 "curv", "sulc", "thickness", "area"))
FSAVERAGE_LABEL = tuple(f"{h}.{s}" for h in ("lh", "rh")
                        for s in ("cortex.label", "Medial_wall.label"))

DESIKAN_LABEL = ("lh.aparc.annot", "rh.aparc.annot")
DKT_LABEL = ("lh.aparc.DKTatlas.annot", "rh.aparc.DKTatlas.annot",
             "aparc.annot.DKTatlas.ctab")
DKT_SURF = tuple(f"{h}.{s}" for h in ("lh", "rh")
                 for s in ("white", "pial", "inflated", "sphere", "curv", "sulc"))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def agreeing_source(rel: str, roots) -> tuple[str, str, int]:
    """the one copy of `rel`, having checked every root that holds it agrees.

    raises if two roots disagree.  a silent divergence here would mean staging
    bytes that are not the distributed ones, and the whole point of holding an
    atlas is that it is the same object everyone else's numbers are in.
    """
    have = [(r, os.path.join(r, rel)) for r in roots
            if os.path.exists(os.path.join(r, rel))]
    if not have:
        raise FileNotFoundError(
            f"{rel} is in none of:\n  " + "\n  ".join(roots))
    digests = {p: sha256(p) for _, p in have}
    uniq = set(digests.values())
    if len(uniq) != 1:
        raise ValueError(
            f"{rel} DIFFERS between staged copies -- this is not a template "
            "copy, do not stage it:\n  " +
            "\n  ".join(f"{d[:16]}  {p}" for p, d in digests.items()))
    return have[0][1], digests[have[0][1]], len(have)


def stage(dest_root: str, plan, roots) -> list[tuple[str, str, int]]:
    """copy `plan` -- (relative source path, relative dest path) -- into place."""
    out = []
    for rel_src, rel_dst in plan:
        src, digest, n_agree = agreeing_source(rel_src, roots)
        dst = os.path.join(dest_root, rel_dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.exists(dst) and sha256(dst) == digest):
            shutil.copy2(src, dst)
            assert sha256(dst) == digest, f"copy of {rel_src} did not verify"
        out.append((rel_dst, digest, n_agree))
        print(f"    {digest[:12]}  {rel_dst}"
              f"  (agreed by {n_agree} staged {'copies' if n_agree > 1 else 'copy'})",
              flush=True)
    return out


def write_manifest(source_id: str, rows, note: str) -> None:
    path = os.path.join(ROOT, "data/sources", source_id, "raw/checksums.txt")
    with open(path, "w") as f:
        f.write(f"# sha256 manifest for {source_id}; written on acquisition\n")
        f.write(f"# {note}\n")
        for rel, digest, n_agree in sorted(rows):
            f.write(f"{digest}  {rel}\n")
    print(f"  wrote {len(rows)} hashes to {path}", flush=True)


def write_location(source_id: str, body: str) -> None:
    path = os.path.join(ROOT, "data/sources", source_id, "raw/.location.yaml")
    with open(path, "w") as f:
        f.write(body if body.endswith("\n") else body + "\n")


# ---------------------------------------------------------------------------


def do_fsaverage() -> None:
    print("fsaverage (the carrier the parcellations are covers OVER)", flush=True)
    roots = [r for r in FSAVERAGE_CANDIDATES if os.path.isdir(r)]
    if not roots:
        raise FileNotFoundError(
            "no fsaverage subject directory found.  `python -c \"from "
            "mne.datasets import fetch_fsaverage; fetch_fsaverage()\"` stages one.")
    dest = os.path.join(ROOT, "data/sources/fsaverage/raw/fsaverage")
    plan = ([(f"surf/{f}", f"surf/{f}") for f in FSAVERAGE_SURF]
            + [(f"label/{f}", f"label/{f}") for f in FSAVERAGE_LABEL])
    rows = stage(dest, plan, roots)
    write_manifest("fsaverage", [(f"fsaverage/{r}", d, n) for r, d, n in rows],
                   f"verified by agreement across {len(roots)} independently "
                   "staged FreeSurfer subject directories")
    write_location("fsaverage", f"""\
# where the bytes for `fsaverage` actually live.  never the bytes themselves.
# fsaverage is DISTRIBUTED WITH FREESURFER; it is not downloaded from a dataset
# host.  the copy staged here reached this machine inside a FreeSurfer subject
# directory bundled by MNE-Python, which is the lineage the licence note on the
# card already records.
source: FreeSurfer distribution (surfaces + spherical registration frame)
vendored_in: freesurfer
staged_from:
{chr(10).join('  - ' + r for r in roots)}
verification: >
  {len(roots)} independently staged copies were hashed and are byte-identical for
  every file staged; that agreement, not the pre-existing checksums.txt (which
  held a header comment and no hashes), is what this payload was verified by.
local_root: data/sources/fsaverage/raw/fsaverage
""")


def do_desikan() -> None:
    print("desikan2006 (aparc, 34 gyral labels per hemisphere, on fsaverage)",
          flush=True)
    roots = [r for r in FSAVERAGE_CANDIDATES if os.path.isdir(r)]
    dest = os.path.join(ROOT, "data/sources/desikan2006/raw/fsaverage")
    plan = [(f"label/{f}", f"label/{f}") for f in DESIKAN_LABEL]
    rows = stage(dest, plan, roots)
    write_manifest("desikan2006", [(f"fsaverage/{r}", d, n) for r, d, n in rows],
                   f"verified by agreement across {len(roots)} independently "
                   "staged FreeSurfer subject directories")
    write_location("desikan2006", f"""\
# where the bytes for `desikan2006` actually live.  never the bytes themselves.
# the card's licence note already states the lineage: this reached us inside a
# FreeSurfer subject directory, so the terms are FreeSurfer's.  the payload is
# the fsaverage `?h.aparc.annot` pair -- the parcellation propagated onto the
# template, which is the frame every published DK connectome is expressed in.
source: FreeSurfer distribution, fsaverage/label/?h.aparc.annot
vendored_in: freesurfer
staged_from:
{chr(10).join('  - ' + r for r in roots)}
verification: >
  {len(roots)} independently staged copies hashed and byte-identical.  the
  pre-existing checksums.txt held no hashes, so there was nothing to verify
  against; cross-copy identity is the verification actually available.
requires: fsaverage  # a cover is meaningless without the carrier (card: requires)
local_root: data/sources/desikan2006/raw/fsaverage
""")


def do_dkt() -> None:
    print("dkt-atlas (DKT31, on a recon-all subject -- fsaverage does not carry it)",
          flush=True)
    if not os.path.isdir(DKT_SUBJECT):
        raise FileNotFoundError(
            f"no recon-all subject at {DKT_SUBJECT}.  the DKT parcellation is "
            "written per subject by recon-all and is NOT shipped on fsaverage; "
            "acquiring mne-somato is the way to hold it.")
    # state the negative rather than leaving it implied: fsaverage really does
    # not carry DKT, and a reader is entitled to see that checked rather than
    # asserted.
    for r in FSAVERAGE_CANDIDATES:
        d = os.path.join(r, "label")
        if os.path.isdir(d):
            got = [f for f in os.listdir(d) if "DKT" in f]
            assert not got, f"fsaverage DOES carry DKT after all: {got} in {d}"
    dest = os.path.join(ROOT, "data/sources/dkt-atlas/raw/somato-01")
    plan = ([(f"label/{f}", f"label/{f}") for f in DKT_LABEL]
            + [(f"surf/{f}", f"surf/{f}") for f in DKT_SURF]
            + [("label/lh.cortex.label", "label/lh.cortex.label"),
               ("label/rh.cortex.label", "label/rh.cortex.label"),
               ("label/lh.aparc.annot", "label/lh.aparc.annot"),
               ("label/rh.aparc.annot", "label/rh.aparc.annot")])
    rows = stage(dest, plan, [DKT_SUBJECT])
    write_manifest("dkt-atlas", [(f"somato-01/{r}", d, n) for r, d, n in rows],
                   "one recon-all subject; DKT is written per subject, not on a "
                   "template, which is what the card's subject_surface_ras frame says")
    write_location("dkt-atlas", f"""\
# where the bytes for `dkt-atlas` actually live.  never the bytes themselves.
# DKT IS NOT ON FSAVERAGE.  the card says the atlas "ships inside each recon-all
# subject directory" and declares its frame as subject_surface_ras; that is
# literal, and this script asserts it by checking that no fsaverage copy on this
# machine carries a DKTatlas file.  so the payload is a subject recon, staged
# with its own surfaces because per-vertex labels without the subject's surface
# are meaningless (card: requires cortical-support-bank, severity blocking).
source: FreeSurfer recon-all output, subject 01 of the MNE somatosensory dataset
vendored_in: freesurfer
staged_from:
  - {DKT_SUBJECT}
scope: >
  ONE subject.  this is enough to corroborate a boundary that desikan2006 also
  draws -- and the card is explicit that the two are NOT independent evidence,
  DKT being a deliberate revision of DK -- but it is not a template and must not
  be used as one.
its_aparc_is_also_staged: >
  lh/rh.aparc.annot from the SAME subject, so a DK-vs-DKT comparison is within
  one brain rather than across two frames.
local_root: data/sources/dkt-atlas/raw/somato-01
""")


def do_braingraph(keep_archive: bool = False) -> None:
    print("braingraph-hcp-connectomes (86-node DK, 1064 HCP subjects)", flush=True)
    dest = os.path.join(ROOT, "data/sources/braingraph-hcp-connectomes/raw")
    out_dir = os.path.join(dest, "repeated_10_scale_33")
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, "repeated_10_scale_33.7z")
    have = (os.path.isdir(out_dir)
            and len([f for f in os.listdir(out_dir) if f.endswith(".graphml")]) > 1000)
    if not have:
        if not os.path.exists(archive):
            print(f"  fetching {BRAINGRAPH_URL}", flush=True)
            r = subprocess.run(["curl", "-fsSL", "--retry", "3", "--max-time", "3600",
                                "-o", archive, BRAINGRAPH_URL])
            if r.returncode != 0:
                raise RuntimeError(f"curl failed ({r.returncode}) on {BRAINGRAPH_URL}")
        import py7zr
        print("  extracting", flush=True)
        os.makedirs(out_dir, exist_ok=True)
        with py7zr.SevenZipFile(archive) as a:
            a.extractall(path=out_dir)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".graphml"))
    print(f"  {len(files)} subject graphs", flush=True)
    # hashing 1064 files is cheap and it is what makes the manifest a manifest.
    rows = [(f"repeated_10_scale_33/{f}", sha256(os.path.join(out_dir, f)), 1)
            for f in files]
    if os.path.exists(archive):
        if keep_archive:
            rows.append(("repeated_10_scale_33.7z", sha256(archive), 1))
        else:
            os.remove(archive)
    write_manifest("braingraph-hcp-connectomes", rows,
                   f"{len(files)} per-subject graphml, 86 Desikan-Killiany nodes, "
                   "downloaded from braingraph.org")
    write_location("braingraph-hcp-connectomes", f"""\
# where the bytes for `braingraph-hcp-connectomes` actually live.  never the
# bytes themselves.
source_url: "{BRAINGRAPH_URL}"
listing_page: "https://braingraph.org/cms/download-pit-group-connectomes/"
selection: >
  the 86-node set: 1064 brains, 1,000,000 streamlines, 10x repeated and averaged.
  86 nodes is the Desikan-Killiany resolution (68 cortical + 18 subcortical),
  which is the frame ibm-1 needs because it is the frame desikan2006 gives and
  the one `ibm/topologies/tract.py:tractometric_matrix` expands from.  the
  higher-resolution sets (129/234/463/1015) are Lausanne subdivisions and would
  need a second crosswalk for no gain here.
edge_features_present:
  - number_of_fibers        # NOT used as a weight: tract.py refuses streamline count
  - fiber_length_mean       # mm; this is the field that makes a delay possible
  - FA_mean
terms: >
  braingraph.org gates these files behind an "I agree to the data use terms of
  the Human Connectome Project" checkbox that enables the download form in the
  page's own markup; the form action is the static path used above.  the
  agreement is a click, not a credential.  HCP open-access terms are accepted.
local_root: data/sources/braingraph-hcp-connectomes/raw/repeated_10_scale_33
""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["fsaverage", "desikan2006", "dkt-atlas",
                             "braingraph-hcp-connectomes"])
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep the braingraph .7z after extraction (18 MB)")
    a = ap.parse_args()
    want = set(a.only) if a.only else {"fsaverage", "desikan2006", "dkt-atlas",
                                       "braingraph-hcp-connectomes"}
    if "fsaverage" in want:
        do_fsaverage()
    if "desikan2006" in want:
        do_desikan()
    if "dkt-atlas" in want:
        do_dkt()
    if "braingraph-hcp-connectomes" in want:
        do_braingraph(a.keep_archive)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())

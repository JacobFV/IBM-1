"""export the released checkpoints to site/data/releases.js.

the site lists models by a legible name; the weights behind one are a run
of checkpoints uploaded every `--upload-every` steps by the trainer itself
(ibm/release.py).  this walks the release repo, reads each sidecar for the one
thing the filename does not carry -- the parameter count -- and writes the list
per model, newest step first, so the page can show the progress rather than a
single frozen link.

    python scripts/export_releases.py            # reads HF_TOKEN, or `hf auth token`

the mapping from a checkpoint family to the model it is a checkpoint *of* is
declared in FAMILIES below, because nothing in the filename says it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "jacob-valdez/ibm-1"
OUT = Path(__file__).resolve().parents[1] / "site" / "data" / "releases.js"

# family prefix (everything before .step) -> the model in §4 it belongs to
FAMILIES = {
    "ibm1.m-multi.s300k.e128.k48.dyn-r4.obj-av+meg.vw1": "av_meg_predictor",
    "ibm1.m-multi.s250k.e128.k48.dyn-r4.obj-av+meg.vw1": "av_meg_predictor",
    "ibm1.m-av.s250k.e128.k48.dyn-r4.obj-av.vw1": "av_predictor",
    "ibm1.m-p.s150k.e128.k48.dyn-r4.obj-meg.vw1": "meg_encoder",
    "ibm1.m-ve.s30k.e128.k48.dyn-r4.obj-evoked.vw1": "visual_evoked",
    "ibm1.m-vc.s30k.e128.k48.dyn-r4.obj-contrastive.vw1": "eeg_to_image",
}

STEP_RE = re.compile(r"^checkpoints/(?P<family>.+)\.step(?P<step>\d+)\.git-(?P<sha>\w+)\.pt$")

# the whole-substrate kernels, which live beside the per-task checkpoints under
# implicit/.  ONE of these carries the parameters for ANY materialization; the
# per-task checkpoints under checkpoints/ are a convenience for compute and
# memory, not different models.  this directory went unexported for a long time,
# so the site had no link to the artifact its central claim is about.
IMPLICIT_RE = re.compile(
    r"^implicit/(?P<name>ibm1\.implicit\.s(?P<sites>[\w]+)\.e\d+\.(?P<variant>[a-z0-9]+)"
    r"(?:\.step(?P<step>\d+))?)\.pt$"
)


def token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:  # the cli keeps one, and this script is usually run by whoever trained
        out = subprocess.run(["hf", "auth", "token"], capture_output=True, text=True, timeout=20)
        for line in out.stdout.splitlines():
            line = line.strip()
            if line and not line.lower().startswith("hint"):
                return line
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def get(url: str, tok: str | None) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=REPO, help="release repo holding the checkpoints")
    ap.add_argument("--link-repo", default=None, help="repo the site should link to (default: --repo)")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    link_repo = a.link_repo or a.repo
    tok = token()

    listing = json.loads(get(f"https://huggingface.co/api/models/{a.repo}?full=true", tok))
    if "siblings" not in listing:
        print(f"{a.repo}: {listing.get('error', 'no file listing')}", file=sys.stderr)
        return 1

    found: dict[str, list[dict]] = {}
    unclaimed: set[str] = set()
    for sib in listing["siblings"]:
        m = STEP_RE.match(sib["rfilename"])
        if not m:
            continue
        model = FAMILIES.get(m["family"])
        if model is None:
            unclaimed.add(m["family"])
            continue
        found.setdefault(model, []).append(
            {"name": Path(sib["rfilename"]).stem, "path": sib["rfilename"], "step": int(m["step"]), "sha": m["sha"]}
        )
    for fam in sorted(unclaimed):
        print(f"note: no model declared for {fam}", file=sys.stderr)

    kernels: list[dict] = []
    for sib in listing["siblings"]:
        m = IMPLICIT_RE.match(sib["rfilename"])
        if not m:
            continue
        kernels.append({
            "name": m["name"], "path": sib["rfilename"], "variant": m["variant"],
            "sites": m["sites"], "step": int(m["step"]) if m["step"] else None,
        })

    # the sidecar carries what the name cannot: how big the thing actually is
    def params(ck: dict) -> None:
        url = f"https://huggingface.co/{a.repo}/resolve/main/" + urllib.parse.quote(ck["path"][:-3] + ".json")
        try:
            side = json.loads(get(url, tok))
        except Exception as e:  # a checkpoint without its sidecar still lists
            print(f"note: no sidecar for {ck['name']}: {e}", file=sys.stderr)
            return
        ck["params"] = side.get("n_params")
        ck["association"] = side.get("n_association_params")

    # an implicit kernel's sidecar is a DIFFERENT schema (ibm1/implicit-v1): it has no
    # n_params, and carries sites/embed instead, plus the transfer numbers it was
    # measured at.  reading them here is what stops the page hand-typing 88.7%.
    def kparams(k: dict) -> None:
        url = f"https://huggingface.co/{a.repo}/resolve/main/" + urllib.parse.quote(k["path"][:-3] + ".json")
        try:
            side = json.loads(get(url, tok))
        except Exception as e:
            print(f"note: no sidecar for {k['name']}: {e}", file=sys.stderr)
            return
        sites, embed = side.get("sites"), side.get("embed")
        if sites and embed:
            k["params"] = sites * embed
        k["sites_n"] = sites
        k["embed"] = embed
        k["n_sources"] = side.get("n_sources")
        k["what"] = side.get("what_it_is")
        k["measured"] = side.get("measured") or {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(params, [c for cks in found.values() for c in cks]))
        list(pool.map(kparams, kernels))

    models = {}
    for model, cks in found.items():
        cks.sort(key=lambda c: c["step"], reverse=True)
        models[model] = [
            {
                "name": c["name"],
                "step": c["step"],
                "params": c.get("params"),
                "url": f"https://huggingface.co/{link_repo}/blob/main/" + urllib.parse.quote(c["path"]),
            }
            for c in cks
        ]

    # fused first -- it is the one the page points at -- then by step, newest first
    kernels.sort(key=lambda k: (0 if k["variant"].startswith("fused") else 1, -(k["step"] or 0)))
    kernel_out = [
        {
            "name": k["name"], "path": k["path"], "variant": k["variant"],
            "sites": k["sites"], "step": k["step"],
            "params": k.get("params"), "sites_n": k.get("sites_n"), "embed": k.get("embed"),
            "n_sources": k.get("n_sources"), "what": k.get("what"), "measured": k.get("measured"),
            "url": f"https://huggingface.co/{link_repo}/blob/main/" + urllib.parse.quote(k["path"]),
        }
        for k in kernels
    ]

    a.out.write_text(
        "/* generated by scripts/export_releases.py -- the checkpoints released per model,\n"
        f"   newest first, from {a.repo}.  do not edit by hand. */\n"
        "window.IBM_RELEASES = "
        + json.dumps({"repo": link_repo, "kernels": kernel_out, "models": models},
                      indent=1, sort_keys=True)
        + ";\n"
    )
    total = sum(len(v) for v in models.values())
    print(f"{a.out}: {total} checkpoints across {len(models)} models, "
          f"{len(kernel_out)} implicit kernels")
    for k in kernel_out:
        print(f"  kernel {k['variant']:14} sites={k['sites']:5} step={k['step']} "
              f"({k['params']} params) {k.get('measured') or ''}")
    for model, cks in sorted(models.items()):
        print(f"  {model}: {len(cks)}, latest step {cks[0]['step']} ({cks[0]['params']} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

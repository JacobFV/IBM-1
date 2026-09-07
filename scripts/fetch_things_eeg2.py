"""fetch THINGS-EEG2: 10 subjects, 63-channel EEG paired with 16,540 images.

the visual-pathway counterpart to LibriBrain.  the model already holds ~4 hours of
speech->MEG; it holds no image->EEG at all, so the occipital port of every
materialisation is currently trained only against a self-supervised video loop
with no measured brain on the other side of it.

40 GB for all ten subjects at 63 channels.  the 17-channel variants are skipped
deliberately: spatial degrees of freedom are the scarce quantity for a forward
model to fit, and saving 3 GB per subject by discarding two thirds of the
channels is not a trade worth making at this bandwidth budget.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.request

DEST = "data/sources/things-eeg2/raw"
NODE = "anp5v"          # "Preprocessed EEG data" under the THINGS-EEG2 project


def listing() -> list[dict]:
    """every file in the storage node, following pagination.

    the default page size is 10, which is why a first pass saw only three of the
    ten subjects and reported the corpus as 12 GB rather than 51 GB.
    """
    url = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/?page[size]=100"
    out: list[dict] = []
    while url:
        d = json.load(urllib.request.urlopen(url, timeout=60))
        out += [i for i in d["data"] if i["attributes"]["kind"] == "file"]
        url = d["links"].get("next")
    return out


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    sel = sorted((i for i in listing() if "63_channels" in i["attributes"]["name"]),
                 key=lambda i: i["attributes"]["name"])
    total = sum(i["attributes"]["size"] for i in sel) / 1e9
    print(f"{len(sel)} subjects at 63 channels, {total:.1f} GB", flush=True)

    got = 0.0
    for i in sel:
        a = i["attributes"]
        out = os.path.join(DEST, a["name"])
        if os.path.exists(out) and os.path.getsize(out) > a["size"] * 0.98:
            print(f"  have {a['name']}", flush=True)
            continue
        print(f"  fetching {a['name']} ({a['size']/1e9:.2f} GB)", flush=True)
        r = subprocess.run(["curl", "-sL", "--retry", "3", "--max-time", "7200",
                            "-o", out, i["links"]["download"]])
        if r.returncode == 0 and os.path.exists(out):
            got += os.path.getsize(out) / 1e9
            print(f"    ok  ({got:.1f} GB this run)", flush=True)
        else:
            print(f"    failed (curl {r.returncode})", flush=True)
    print(f"done: {got:.1f} GB downloaded into {DEST}", flush=True)


if __name__ == "__main__":
    main()

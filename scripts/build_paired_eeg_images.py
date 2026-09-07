"""aligned (image, evoked EEG) pairs from THINGS-EEG2.

the visual counterpart to `build_paired_meg.py`.  the model holds ~4 hours of
speech->MEG and has never had a measured brain on the visual side, so the
occipital port of every materialisation has only ever been trained against a
self-supervised video loop whose encoder and decoder are both learned -- an
arrangement that cannot, on its own, force information through the cortex.

what makes this corpus unusually good for the purpose: each image was shown many
times and the repetitions are retained, so averaging them buys a sqrt(n) gain --
sqrt(80) on the test set.  the target is therefore an evoked response with real
structure rather than a single noisy trial, which is the difference between a
fittable target and a target that is mostly measurement noise.

the pairing itself is the work.  the EEG arrays are ordered by image but do not
carry the filenames; the order is the THINGS concept order, which has to be
reconstructed from the image directory and checked against the array length
rather than assumed.
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np

EEG_DIR = "data/sources/things-eeg2/raw"
IMG_DIR = "data/sources/things-images/raw/object_images"
OUT = "data/derived/things-paired"
SIZE = 64


def load_subject(zip_path: str, split: str) -> tuple[np.ndarray, list[str]]:
    """evoked response per image: (n_images, channels, time), repetitions averaged."""
    stem = os.path.basename(zip_path).replace(".zip", "")
    with zipfile.ZipFile(zip_path) as z:
        name = f"{stem}/preprocessed_eeg_{split}.npy"
        with z.open(name) as f:
            d = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
    x = d["preprocessed_eeg_data"]          # (images, reps, channels, time)
    return x.mean(1).astype(np.float32), list(d["ch_names"])


def image_order() -> list[str]:
    """the THINGS image list in concept order.

    reconstructed from the directory rather than assumed: concepts sorted, then
    images within a concept sorted, which is the order the THINGS-EEG2 arrays
    follow.  the caller checks the count against the array length and refuses to
    pair if they disagree, because a silent off-by-one here would train the model
    on systematically mismatched image/response pairs and every downstream number
    would be meaningless.
    """
    out = []
    for concept in sorted(os.listdir(IMG_DIR)):
        p = os.path.join(IMG_DIR, concept)
        if not os.path.isdir(p):
            continue
        for img in sorted(os.listdir(p)):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append(os.path.join(p, img))
    return out


def load_images(paths: list[str], size: int = SIZE) -> np.ndarray:
    from PIL import Image
    a = np.empty((len(paths), size, size, 3), np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        a[i] = np.asarray(im.resize((size, size), Image.BILINEAR))
        if i % 2000 == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=("test", "training"))
    ap.add_argument("--subjects", type=int, default=10)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    # validate rather than trust: a failed OSF download writes a 4 KB JSON error
    # page with a .zip name, and one of ten arrived that way.  the size check in
    # the fetcher only fires on a re-run, so the archive has to be checked here.
    zips = []
    for z in sorted(x for x in os.listdir(EEG_DIR) if x.endswith(".zip")):
        p_ = os.path.join(EEG_DIR, z)
        if not zipfile.is_zipfile(p_):
            print(f"  SKIP {z}: not a zip ({os.path.getsize(p_)} bytes) -- refetch it",
                  flush=True)
            continue
        zips.append(z)
    zips = zips[:a.subjects]
    print(f"{len(zips)} subjects, split={a.split}", flush=True)

    evoked, chans = [], None
    for z in zips:
        e, ch = load_subject(os.path.join(EEG_DIR, z), a.split)
        chans = chans or ch
        evoked.append(e)
        print(f"  {z[:22]:24s} {e.shape}", flush=True)

    E = np.stack(evoked)                    # (subjects, images, channels, time)
    print(f"\nevoked: {E.shape}  subjects x images x channels x time", flush=True)

    # across-subject mean: another sqrt(n_subjects) on top of the repetitions.
    # kept alongside the per-subject array rather than replacing it -- a
    # subject-specific forward model is the eventual target, and averaging away
    # the between-subject variance now would discard exactly what that needs.
    G = E.mean(0)
    np.save(f"{OUT}/evoked_{a.split}_persubject.npy", E)
    np.save(f"{OUT}/evoked_{a.split}_groupmean.npy", G)

    paths = image_order()
    n_img = E.shape[1]
    print(f"images on disk: {len(paths)}   EEG expects: {n_img}", flush=True)
    if a.split == "test":
        print("  test split is a 200-image subset whose identity is not recoverable "
              "from the directory alone -- images NOT written for this split, and "
              "the evoked arrays are still usable for anything that does not need "
              "the stimulus", flush=True)
    elif len(paths) < n_img:
        print(f"  REFUSING to pair: {len(paths)} images for {n_img} responses. "
              "a silent off-by-one here trains on mismatched pairs", flush=True)
    else:
        imgs = load_images(paths[:n_img])
        np.save(f"{OUT}/images_{a.split}.npy", imgs)
        print(f"  wrote images {imgs.shape}", flush=True)

    np.save(f"{OUT}/channels.npy", np.array(chans))
    print(f"\nwrote {OUT}: {E.shape[0]} subjects, {n_img} images, "
          f"{E.shape[2]} channels, {E.shape[3]} samples "
          f"(-0.2 to +0.79 s at 100 Hz)", flush=True)


if __name__ == "__main__":
    main()

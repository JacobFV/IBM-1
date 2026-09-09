"""recover, and VERIFY, the row -> (run, chapter time, word) index for
`data/derived/libribrain-paired`.

`build_paired_meg.py` concatenates twelve runs into one array and saves no index,
so nothing in the derived corpus says which chapter, or which spoken word, a row
belongs to.  the prompt bar needs that for its audio->text and text->audio paths.

THIS IS A RECONSTRUCTION, and a reconstruction that merely has the right LENGTH is
exactly the failure this repository has already paid for once (ledger row 8: the
THINGS image order was rebuilt by walking a directory, matched on 10 of 16,540
pairs, and passed a count check because the counts were equal).  so the row map is
not trusted on its length.  it is verified by REPRODUCING THE STORED MEG BYTES:
for every run the recovered sample indices are applied to the raw h5, normalised
the way the builder normalises, and compared elementwise against the corresponding
slice of `meg_250hz.npy`.  a run is accepted only if that comparison passes.  the
chapter assignment for the ambiguous runs (two chapters 1.5 s apart) is likewise
DECIDED by which candidate reproduces the bytes, not by which one is closer in
duration.

the words themselves come from `derivatives/events/*.tsv`, which is the file the
dataset defines the alignment in.  they are a FORCED ALIGNER'S output -- the
source card marks the `alignments` stream `context` for exactly this reason -- so
a word onset here is where an aligner put it, not when a phoneme occurred.  that
caveat travels with the index and is written into the output.
"""
from __future__ import annotations

import csv, glob, json, os, sys
import av, h5py, numpy as np

ROOT = "/home/brandonin/Documents/IBM-1/data/sources/libribrain/raw/Sherlock1"
OUT = "/home/brandonin/Documents/IBM-1/data/derived/libribrain-paired-v3"
FS, RATE, NFFT, HOP = 16000, 250, 1024, 16000 // 250


def n_frames(path):
    """len(cochleagram) and duration for a chapter, without the FFT.

    `build_paired_meg.cochleagram` decodes to mono 16 kHz and then takes
    n = (len(x) - nfft) // hop frames.  only those two numbers are needed here, so
    the transform itself is skipped -- but the decode path is byte-identical.
    """
    c = av.open(path)
    rs = av.AudioResampler(format="s16", layout="mono", rate=FS)
    total = 0
    for fr in c.decode(c.streams.audio[0]):
        for r in rs.resample(fr):
            total += r.to_ndarray().reshape(-1).shape[0]
    return (total - NFFT) // HOP, total / FS


def knots(rows):
    tmeg = np.array([float(r["timemeg"]) for r in rows])
    tchap = np.array([float(r["timechapter"]) for r in rows])
    nb = max(4, int((tchap.max() - tchap.min()) // 30))
    edges = np.quantile(tchap, np.linspace(0, 1, nb + 1))
    kc, km = [], []
    for i in range(nb):
        m = (tchap >= edges[i]) & (tchap <= edges[i + 1])
        if m.sum() < 5:
            continue
        q, w = np.polyfit(tchap[m], tmeg[m], 1)
        for c in (edges[i], edges[i + 1]):
            kc.append(float(c)); km.append(float(q * c + w))
    kc, km = np.array(kc), np.array(km)
    o = np.argsort(kc); kc, km = kc[o], km[o]
    return kc, np.maximum.accumulate(km), tchap.max()


def main():
    stored = np.load(f"{OUT}/meg_250hz.npy", mmap_mode="r")
    print(f"stored MEG {stored.shape}", flush=True)

    wavs = sorted(glob.glob(f"{ROOT}/stimuli/audio/*.wav"))
    lens, durs = {}, {}
    for w in wavs:
        lens[w], durs[w] = n_frames(w)
        print(f"  {os.path.basename(w)} {lens[w]} frames {durs[w]:.1f}s", flush=True)

    off = 0
    runs, row_run, row_tchap = [], [], []
    for ev_path in sorted(glob.glob(f"{ROOT}/derivatives/events/*.tsv")):
        all_rows = list(csv.DictReader(open(ev_path), delimiter="\t"))
        rows = [r for r in all_rows if r["kind"] == "word" and r["timechapter"]]
        if len(rows) < 50:
            continue
        kc, km, span = knots(rows)
        stem = os.path.basename(ev_path).replace("_events.tsv", "")
        h5 = f"{ROOT}/derivatives/serialised/{stem}_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
        if not os.path.exists(h5):
            continue
        with h5py.File(h5, "r") as h:
            meg = h["data"][:]

        cand = sorted((abs(durs[k] - span), k) for k in durs if durs[k] >= span - 1.0)
        margin = (cand[1][0] - cand[0][0]) if len(cand) > 1 else float("inf")
        picks = [cand[0][1]] if margin >= 5.0 else [c[1] for c in cand[:2]]

        chosen = None
        for key in picks:
            f = lambda i: (np.interp(i / RATE, km, kc) - NFFT / (2 * FS)) * RATE
            i0 = max(0, int(np.ceil(km[0] * RATE)))
            i1 = min(meg.shape[1] - 1, int(np.floor(km[-1] * RATE)))
            if i1 - i0 < RATE * 60:
                continue
            idx = np.arange(i0, i1 + 1)
            pos = f(idx)
            ok = (pos >= 0) & (pos <= lens[key] - 1)
            idx = idx[ok]
            if off + len(idx) > len(stored):
                continue
            m = meg[:, idx].T.astype(np.float32)
            m = (m - m.mean(0)) / (m.std(0) + 1e-9)
            # THE VERIFICATION.  not a length check -- the actual samples.
            ref = np.asarray(stored[off:off + len(idx)])
            err = float(np.abs(m - ref).max())
            print(f"    {stem} vs {os.path.basename(key)}: "
                  f"{len(idx)} rows, max|dm| {err:.2e}", flush=True)
            if err < 1e-3:
                chosen = (key, idx, len(idx))
                break
        if chosen is None:
            print(f"  !! {stem} REPRODUCED NO CANDIDATE -- index abandoned", flush=True)
            sys.exit(1)
        key, idx, n = chosen
        # meg sample -> chapter time, the knot map read backwards.
        tchap = np.interp(idx / RATE, km, kc).astype(np.float32)
        row_run.append(np.full(n, len(runs), np.int16))
        row_tchap.append(tchap)
        words = [{"word": r["segment"], "t": float(r["timechapter"]),
                  "dur": float(r["duration"] or 0.0)} for r in rows]
        runs.append({"stem": stem, "chapter": os.path.basename(key),
                     "row0": off, "n_rows": n, "n_words": len(words),
                     "words": words})
        off += n
        print(f"  OK {stem} -> {os.path.basename(key)}  rows {n}  total {off}",
              flush=True)
        del meg

    if off != len(stored):
        print(f"  !! recovered {off} rows, stored has {len(stored)} -- abandoned")
        sys.exit(1)
    np.savez_compressed(
        f"{OUT}/row_index.npz",
        run=np.concatenate(row_run), chapter_time=np.concatenate(row_tchap))
    meta = {
        "verified": "every run's recovered sample indices reproduce the stored "
                    "meg_250hz.npy elementwise to max|dm| < 1e-3; the chapter "
                    "assignment was decided by that reproduction, not by duration",
        "n_rows": off, "rate_hz": RATE,
        "word_source": "data/sources/libribrain/raw/Sherlock1/derivatives/events/*.tsv",
        "word_caveat": "word onsets are a FORCED ALIGNER's output, not measured "
                       "event times; the source card grades the alignments stream "
                       "as context for this reason",
        "runs": [{k: v for k, v in r.items() if k != "words"} for r in runs],
    }
    json.dump(meta, open(f"{OUT}/row_index.json", "w"), indent=2)
    json.dump({r["stem"]: r["words"] for r in runs},
              open(f"{OUT}/row_words.json", "w"))
    print(f"\nsaved row_index.npz ({off} rows, {len(runs)} runs) and words",
          flush=True)


if __name__ == "__main__":
    main()

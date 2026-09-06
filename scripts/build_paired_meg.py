"""aligned (speech cochleagram, MEG) pairs from LibriBrain.

the paired materialization: a stimulus the model transduces, and the MEG a real
head produced while hearing it.  this is what stops the cortex being decorative.
in the self-supervised video loop both the encoder and the decoder are learned, so
nothing prevents the model routing information around the dynamics and using them
as a delay line.  here the target is measured neural activity, so the cortical
state cannot be arbitrary -- it has to be the state that produces this MEG.

alignment: the events file gives `timemeg` and `timechapter` per word, and their
difference is constant within a run (the chapter starts at a fixed offset into the
recording).  which chapter a run is, is not written down anywhere, so it is
recovered by matching the run's chapter-time span against the wav durations --
14 chapters with distinct lengths, so the match is unambiguous.
"""
from __future__ import annotations

import csv, glob, os, re
import av, h5py, numpy as np

ROOT = "/home/brandonin/Documents/win-data/libribrain/Sherlock1"
OUT = "/home/brandonin/Documents/win-data/derived/libribrain-paired"
FS, N_BANDS, RATE = 16000, 64, 250          # MEG is 250 Hz; cochleagram matches it


def erb(n, lo=50.0, hi=7000.0):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    inv = lambda E: (10 ** (E / 21.4) - 1) / 0.00437
    return inv(np.linspace(e(lo), e(hi), n))


def cochleagram(path, rate=RATE):
    c = av.open(path)
    rs = av.AudioResampler(format="s16", layout="mono", rate=FS)
    ch = []
    for fr in c.decode(c.streams.audio[0]):
        for r in rs.resample(fr):
            ch.append(r.to_ndarray().reshape(-1))
    x = np.concatenate(ch).astype(np.float32) / 32768.0
    nfft, hop = 1024, FS // rate
    n = (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(nfft)[None, :], axis=1))
    f = np.fft.rfftfreq(nfft, 1 / FS); cf = erb(N_BANDS)
    W = np.exp(-0.5 * ((f[None, :] - cf[:, None]) / (0.12 * cf[:, None])) ** 2)
    W /= W.sum(1, keepdims=True)
    a = np.log1p(S @ W.T * 20.0).astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6), len(x) / FS


def main():
    os.makedirs(OUT, exist_ok=True)
    wavs = sorted(glob.glob(f"{ROOT}/stimuli/audio/*.wav"))
    print(f"cochleagramming {len(wavs)} chapters at {RATE} Hz", flush=True)
    cochs, durs = {}, {}
    for w in wavs:
        key = os.path.basename(w)
        cochs[key], durs[key] = cochleagram(w)
        print(f"  {key} {cochs[key].shape} {durs[key]:.0f}s", flush=True)

    X, Y = [], []
    for ev_path in sorted(glob.glob(f"{ROOT}/derivatives/events/*.tsv")):
        rows = [r for r in csv.DictReader(open(ev_path), delimiter="\t")
                if r["kind"] == "word" and r["timechapter"]]
        if len(rows) < 50:
            continue
        tmeg = np.array([float(r["timemeg"]) for r in rows])
        tchap = np.array([float(r["timechapter"]) for r in rows])
        offset = float(np.median(tmeg - tchap))
        span = tchap.max()
        # identify the chapter by duration: the last word must fall inside it, and
        # the closest-length chapter that still contains it is the match.
        cand = [(abs(durs[k] - span), k) for k in durs if durs[k] >= span - 1.0]
        if not cand:
            continue
        _, key = min(cand)

        stem = os.path.basename(ev_path).replace("_events.tsv", "")
        h5 = f"{ROOT}/derivatives/serialised/{stem}_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
        if not os.path.exists(h5):
            continue
        with h5py.File(h5, "r") as h:
            meg = h["data"][:]                       # (306, T)
        coch = cochs[key]
        # MEG sample i is chapter time (i/RATE - offset)
        t0 = int(round(offset * RATE))
        n = min(meg.shape[1] - t0, coch.shape[0])
        if n < RATE * 60:
            continue
        m = meg[:, t0:t0 + n].T.astype(np.float32)   # (n, 306)
        m = (m - m.mean(0)) / (m.std(0) + 1e-9)
        X.append(coch[:n]); Y.append(m)
        print(f"  {stem} -> {key}  offset {offset:.1f}s  {n/RATE:.0f}s paired", flush=True)

    if not X:
        print("NO PAIRS"); return
    Xc, Yc = np.concatenate(X), np.concatenate(Y)
    np.save(f"{OUT}/cochleagram_250hz.npy", Xc)
    np.save(f"{OUT}/meg_250hz.npy", Yc)
    print(f"\nsaved {Xc.shape} cochleagram and {Yc.shape} MEG "
          f"= {len(Xc)/RATE/60:.1f} minutes paired", flush=True)


if __name__ == "__main__":
    main()

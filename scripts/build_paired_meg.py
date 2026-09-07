"""aligned (speech cochleagram, MEG) pairs from LibriBrain.

the paired materialization: a stimulus the model transduces, and the MEG a real
head produced while hearing it.  this is what stops the cortex being decorative.
in the self-supervised video loop both the encoder and the decoder are learned, so
nothing prevents the model routing information around the dynamics and using them
as a delay line.  here the target is measured neural activity, so the cortical
state cannot be arbitrary -- it has to be the state that produces this MEG.

alignment: the events file gives `timemeg` and `timechapter` per word.  an earlier
version of this file asserted their difference "is constant within a run" and
sliced the MEG at its median.  **it is not constant.**  measured, that difference
has a standard deviation of 1.0-2.1 SECONDS within every run, because the two
clocks run at different RATES: fitting tmeg = a*tchap + b gives a - 1 of
4,300-5,300 ppm consistently across all 12 runs, and drops the residual to
0.04-0.16 s -- a factor of 15-20.

that is not a detail.  0.48% over a 1,400 s chapter accumulates ~6.7 s of drift,
so a median offset is correct in the middle of a run and off by +/-3.4 s at its
ends.  cortical speech tracking is a 1-8 Hz effect, so that smears it across 1-12
cycles and destroys it: the arrays built the old way carried no detectable
envelope tracking at all (scripts/check_pairing.py, and a forward TRF).  the
cochleagram is therefore RESAMPLED onto the MEG clock through the fitted line,
not sliced at an offset.

which chapter a run is, is not written down anywhere, so it is recovered by
matching the run's chapter-time span against the wav durations.  the same earlier
version claimed "14 chapters with distinct lengths, so the match is unambiguous";
that is also false -- two chapters differ by 1.5 s and two runs sit between them.
those are disambiguated by which candidate actually produces envelope-MEG
coupling, which is only a well-posed question once the drift is corrected.
"""
from __future__ import annotations

import csv, glob, os, re
import av, h5py, numpy as np

ROOT = "/home/brandonin/Documents/IBM-1/data/sources/libribrain/raw/Sherlock1"
OUT = "/home/brandonin/Documents/IBM-1/data/derived/libribrain-paired-v3"
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
    # frame j spans samples [j*hop, j*hop+nfft), so it is CENTRED at chapter time
    # j/rate + nfft/(2*FS) -- 32 ms, carried so the resampling is exact.
    return (a - a.mean()) / (a.std() + 1e-6), len(x) / FS, nfft / (2 * FS)


def main():
    os.makedirs(OUT, exist_ok=True)
    wavs = sorted(glob.glob(f"{ROOT}/stimuli/audio/*.wav"))
    print(f"cochleagramming {len(wavs)} chapters at {RATE} Hz", flush=True)
    cochs, durs, centres = {}, {}, {}
    for w in wavs:
        key = w                                  # full path: basenames collide
        cochs[key], durs[key], centres[key] = cochleagram(w)
        print(f"  {os.path.basename(w)} {cochs[key].shape} "
              f"{durs[key]:.0f}s", flush=True)

    X, Y = [], []
    for ev_path in sorted(glob.glob(f"{ROOT}/derivatives/events/*.tsv")):
        rows = [r for r in csv.DictReader(open(ev_path), delimiter="\t")
                if r["kind"] == "word" and r["timechapter"]]
        if len(rows) < 50:
            continue
        tmeg = np.array([float(r["timemeg"]) for r in rows])
        tchap = np.array([float(r["timechapter"]) for r in rows])
        # tmeg = a*tchap + b is the first-order story; `a` is NOT 1, the clocks
        # differ by ~4,800 ppm.  but the residual around that line is structured,
        # so the map is built piecewise through knots instead.
        a_rate, b_off = (float(v) for v in np.polyfit(tchap, tmeg, 1))
        lin_resid = float((tmeg - (a_rate * tchap + b_off)).std())
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
        km = np.maximum.accumulate(km)                  # the map must be monotone
        resid = float((tmeg - np.interp(tchap, kc, km)).std())
        span = tchap.max()
        cand = sorted((abs(durs[k] - span), k) for k in durs if durs[k] >= span - 1.0)
        if not cand:
            continue

        stem = os.path.basename(ev_path).replace("_events.tsv", "")
        h5 = f"{ROOT}/derivatives/serialised/{stem}_proc-bads+headpos+sss+notch+bp+ds_meg.h5"
        if not os.path.exists(h5):
            continue
        with h5py.File(h5, "r") as h:
            meg = h["data"][:]                       # (306, T)

        def take(key):
            """resample the chapter's cochleagram onto this run's MEG clock.

            MEG sample i sits at meg time i/RATE, which is chapter time
            (i/RATE - b)/a, which is cochleagram frame (that - centre)*RATE.
            """
            coch = cochs[key]
            c = centres[key]
            # meg time -> chapter time is the knot map read backwards
            f = lambda i: (np.interp(i / RATE, km, kc) - c) * RATE
            i0 = max(0, int(np.ceil(km[0] * RATE)))
            i1 = min(meg.shape[1] - 1, int(np.floor(km[-1] * RATE)))
            if i1 - i0 < RATE * 60:
                return None
            idx = np.arange(i0, i1 + 1)
            pos = f(idx)
            ok = (pos >= 0) & (pos <= len(coch) - 1)
            idx, pos = idx[ok], pos[ok]
            xr = np.stack([np.interp(pos, np.arange(len(coch)), coch[:, k])
                           for k in range(coch.shape[1])], 1).astype(np.float32)
            m = meg[:, idx].T.astype(np.float32)
            m = (m - m.mean(0)) / (m.std(0) + 1e-9)
            return xr, m

        # the two 1,247/1,248 s chapters are 1.5 s apart and two runs sit between
        # them, so duration alone cannot choose.  ask which candidate actually
        # couples -- a question that only became well-posed once the drift was out.
        margin = (cand[1][0] - cand[0][0]) if len(cand) > 1 else float("inf")
        picks = [cand[0][1]] if margin >= 5.0 else [c[1] for c in cand[:2]]
        best = None
        for key in picks:
            got = take(key)
            if got is None:
                continue
            xr, m = got
            e = xr.mean(1); e = (e - e.mean()) / (e.std() + 1e-9)
            g = m.std(1); g = (g - g.mean()) / (g.std() + 1e-9)
            lag = int(0.1 * RATE)
            r = float(abs((e[:len(e) - lag] * g[lag:]).mean()))
            if best is None or r > best[0]:
                best = (r, key, xr, m)
        if best is None:
            continue
        r, key, xr, m = best
        X.append(xr); Y.append(m)
        chosen = "" if len(picks) == 1 else f"  [chose by coupling r={r:.4f}]"
        print(f"  {stem} -> {os.path.basename(key)}  "
              f"rate {1e6*(a_rate-1):+.0f} ppm  resid {lin_resid:.3f}->{resid:.3f}s  "
              f"{len(xr)/RATE:.0f}s paired{chosen}", flush=True)

    if not X:
        print("NO PAIRS"); return
    Xc, Yc = np.concatenate(X), np.concatenate(Y)
    np.save(f"{OUT}/cochleagram_250hz.npy", Xc)
    np.save(f"{OUT}/meg_250hz.npy", Yc)
    print(f"\nsaved {Xc.shape} cochleagram and {Yc.shape} MEG "
          f"= {len(Xc)/RATE/60:.1f} minutes paired", flush=True)


if __name__ == "__main__":
    main()

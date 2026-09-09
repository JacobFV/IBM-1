"""recover the per-chapter log-magnitude standardization the cochleagram build threw away.

`build_paired_meg.cochleagram` returns `(a - a.mean()) / (a.std() + 1e-6)` where
`a = log1p(20 * |STFT| @ W.T)`, and saves neither scalar.  without them the stored
array cannot be taken back to band magnitudes, so a resynthesis from it would be
guessing its own gain and spectral tilt.  this recomputes mu and sigma per chapter
-- the FFT, but accumulating only two moments -- so the inversion in
`ihm/app/brain_prompt.py` is exact up to the two losses that are genuinely
irreversible: the 513 -> 64 band projection and the discarded phase.

VERIFIED, not assumed: for each chapter the recomputed normalized cochleagram is
compared against the run rows the index says came from it.  the stored rows are a
RESAMPLING of this array onto the MEG clock, so they are compared through the same
interpolation rather than elementwise at the same index.
"""
from __future__ import annotations

import glob, json, os
import av, numpy as np

ROOT = "/home/brandonin/Documents/IBM-1/data/sources/libribrain/raw/Sherlock1"
OUT = "/home/brandonin/Documents/IBM-1/data/derived/libribrain-paired-v3"
FS, N_BANDS, RATE, NFFT = 16000, 64, 250, 1024
HOP = FS // RATE
BLOCK = 32768


def erb(n, lo=50.0, hi=7000.0):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    inv = lambda E: (10 ** (E / 21.4) - 1) / 0.00437
    return inv(np.linspace(e(lo), e(hi), n))


def bank():
    f = np.fft.rfftfreq(NFFT, 1 / FS); cf = erb(N_BANDS)
    W = np.exp(-0.5 * ((f[None, :] - cf[:, None]) / (0.12 * cf[:, None])) ** 2)
    return (W / W.sum(1, keepdims=True)).astype(np.float32), cf


def decode(path):
    c = av.open(path)
    rs = av.AudioResampler(format="s16", layout="mono", rate=FS)
    ch = []
    for fr in c.decode(c.streams.audio[0]):
        for r in rs.resample(fr):
            ch.append(r.to_ndarray().reshape(-1))
    return np.concatenate(ch).astype(np.float32) / 32768.0


def main():
    W, cf = bank()
    win = np.hanning(NFFT).astype(np.float32)
    out = {}
    for path in sorted(glob.glob(f"{ROOT}/stimuli/audio/*.wav")):
        x = decode(path)
        n = (len(x) - NFFT) // HOP
        s1 = s2 = 0.0
        cnt = 0
        for i in range(0, n, BLOCK):
            j = min(i + BLOCK, n)
            idx = np.arange(NFFT)[None, :] + HOP * np.arange(i, j)[:, None]
            S = np.abs(np.fft.rfft(x[idx] * win[None, :], axis=1))
            a = np.log1p((S @ W.T) * 20.0).astype(np.float64)
            s1 += a.sum(); s2 += (a * a).sum(); cnt += a.size
        mu = s1 / cnt
        sd = float(np.sqrt(max(s2 / cnt - mu * mu, 0.0)))
        out[os.path.basename(path)] = {"mean": float(mu), "std": sd,
                                       "n_frames": int(n),
                                       "duration_s": len(x) / FS}
        print(f"  {os.path.basename(path)}  mu {mu:.6f}  sd {sd:.6f}  "
              f"{n} frames", flush=True)

    meta = {
        "what": "per-chapter mean and sd of a = log1p(20 * |STFT| @ W.T), the "
                "scalars build_paired_meg.cochleagram divided out and did not save",
        "inverse": "band_magnitude = expm1(std * stored + mean) / 20",
        "stft": {"n_fft": NFFT, "hop": HOP, "window": "hanning",
                 "sample_rate_hz": FS, "frame_rate_hz": RATE},
        "erb_centres_hz": [float(v) for v in cf],
        "chapters": out,
    }
    json.dump(meta, open(f"{OUT}/cochleagram_scale.json", "w"), indent=2)
    print(f"wrote {OUT}/cochleagram_scale.json", flush=True)


if __name__ == "__main__":
    main()

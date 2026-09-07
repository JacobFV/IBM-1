"""does a paired corpus carry the effect it is supposed to carry?

**run this before building a training term on any paired corpus.**  three of this
programme's withdrawn claims came from training against arrays that could not
have supported the result: an image-EEG pairing that was 99.94% wrong (ledger
row 8) produced three hours of confident negative results about a corpus that was
fine, and the check that would have caught it costs under a minute.

the principle: every paired corpus has some coupling that is so well established
that its ABSENCE indicts the arrays rather than the model.  for speech and MEG
that is envelope tracking in 1-8 Hz at a 50-200 ms lag -- one of the most
reproduced effects in the field.  for evoked designs it is a stimulus-locked
deflection.  if the canonical effect is not there, no objective will rescue the
term, and a negative result from it says nothing about the modality.

**this test has a floor, and the floor is measured.**  injecting a known coupling
into one channel of real MEG at a known SNR, the gate returns PASS at r ~ 0.058
(finding the right channel and the right lag) and FAIL at r ~ 0.023.  so a FAIL
means "nothing above about r = 0.05 at a single channel and lag" -- it does NOT
mean "nothing".  a real effect can live under that floor, because this statistic
takes the single best sample of a response rather than pooling its shape.  when a
FAIL matters, follow it with a forward TRF, which regresses on all lags jointly
and evaluates held out; that is both the sensitive instrument and the one the
literature uses.

three things this gets right that a naive version does not:

*the band*.  broadband MEG carries far more power above 8 Hz than in the band the
effect is defined in, so a broadband correlation is the right quantity in the
wrong units -- ledger row 7's shape.  the test is run in-band.

*the null*.  both signals are heavily autocorrelated, so the i.i.d. standard
error 1/sqrt(N) understates the null width by more than an order of magnitude
here (0.0018 against a measured 0.0145).  the null comes from CIRCULAR SHIFTS,
which destroy the correspondence while preserving every spectral property.

*the statistic*.  the observed value is a maximum over channels x lags, so the
null must be the same maximum over the same grid.  comparing a grid maximum
against a single-draw null is the multiple-comparisons form of comparing against
the wrong population.
"""
from __future__ import annotations

import argparse

import numpy as np


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """zero-phase brick-wall bandpass along axis 0."""
    n = x.shape[0]
    f = np.fft.rfftfreq(n, 1 / fs)
    X = np.fft.rfft(x, axis=0)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=n, axis=0)


def grid_max(sig: np.ndarray, Y: np.ndarray, lags: list[int]) -> tuple[float, int, int]:
    """largest |r| over channels x lags, with where it sat."""
    n = len(sig)
    best = (0.0, -1, -1)
    for lag in lags:
        r = (sig[:n - lag, None] * Y[lag:]).mean(0)
        j = int(np.abs(r).argmax())
        if abs(r[j]) > abs(best[0]):
            best = (float(r[j]), lag, j)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stim", required=True, help="(T, F) stimulus array, .npy")
    ap.add_argument("--neural", required=True, help="(T, C) neural array, .npy")
    ap.add_argument("--scale", help="(2,) centre and scale for the neural array")
    ap.add_argument("--fs", type=float, default=250.0)
    ap.add_argument("--band", type=float, nargs=2, default=(1.0, 8.0),
                    help="the band the effect is DEFINED in, not the recorded band")
    ap.add_argument("--lag-ms", type=float, nargs=2, default=(20.0, 200.0))
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--skip-minutes", type=float, default=0.0,
                    help="start the window here, so disjoint windows can replicate "
                         "the test instead of resting on one draw")
    ap.add_argument("--shifts", type=int, default=50)
    ap.add_argument("--min-shift-s", type=float, default=30.0)
    a = ap.parse_args()

    stim = np.load(a.stim, mmap_mode="r")
    neur = np.load(a.neural, mmap_mode="r")
    n = min(len(stim), len(neur))
    off = int(a.skip_minutes * 60 * a.fs)
    N = min(n - off, int(a.minutes * 60 * a.fs))
    step = max(1, int(round(a.fs * 0.02)))
    lags = list(range(int(a.lag_ms[0] * a.fs / 1000),
                      int(a.lag_ms[1] * a.fs / 1000) + 1, step))

    env = np.asarray(stim[off:off + N]).astype(np.float64)
    env = env.mean(1) if env.ndim > 1 else env
    Y = np.asarray(neur[off:off + N]).astype(np.float64)
    if a.scale:
        sc = np.load(a.scale)
        Y = np.clip((Y - sc[0]) / sc[1], -6, 6)

    e = bandpass(env, a.fs, *a.band)
    Yb = bandpass(Y, a.fs, *a.band)
    e = (e - e.mean()) / (e.std() + 1e-9)
    Yb = (Yb - Yb.mean(0)) / (Yb.std(0) + 1e-9)

    print(f"{N:,} samples ({N/a.fs/60:.1f} min at {a.fs:g} Hz) | "
          f"{Yb.shape[1]} channels x {len(lags)} lags | "
          f"band {a.band[0]:g}-{a.band[1]:g} Hz")

    obs, lag, ch = grid_max(e, Yb, lags)
    print(f"observed  max |r| = {abs(obs):.5f}  (r = {obs:+.5f}) at "
          f"{lag*1000/a.fs:.0f} ms, channel {ch}")

    rng = np.random.default_rng(0)
    lo = int(a.min_shift_s * a.fs)
    null = np.array([abs(grid_max(np.roll(e, int(rng.integers(lo, N - lo))), Yb, lags)[0])
                     for _ in range(a.shifts)])
    hits = int((null >= abs(obs)).sum())
    p = (hits + 1) / (a.shifts + 1)
    print(f"null      mean {null.mean():.5f}  sd {null.std():.5f}  max {null.max():.5f}")
    print(f"          {hits}/{a.shifts} shifts reach the observed value   p = {p:.3f}")

    if p < 0.05:
        print("\nPASS -- the canonical coupling is present.  a negative result from "
              "this corpus is about the model.")
    else:
        print("\nFAIL -- no coupling ABOVE THIS TEST'S FLOOR (measured at r ~ 0.05 "
              "for a single channel and lag; it misses r ~ 0.02).  that is a reason "
              "to suspect the arrays and to check the pairing order and the "
              "derivation -- it is NOT proof the effect is absent.  before "
              "concluding anything about the corpus, run a forward TRF, which pools "
              "the whole response shape instead of its best single sample.")


if __name__ == "__main__":
    main()

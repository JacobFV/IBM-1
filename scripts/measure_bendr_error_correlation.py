"""how correlated is a real teacher's output, measured instead of assumed.

the source cards have to declare a `correlated_fraction` -- the fraction rho of a
teacher's residual variance that is shared across the values it writes -- and no
paper reports one.  the usual response is a prior with a rationale, which is
honest but weak.  for BENDR we hold the actual pretrained weights, so the weak
option is unnecessary: run the encoder over real EEG and look at what comes out.

the schema asks for the correlation of the teacher's *errors*, and errors need
ground truth, which for a self-supervised representation does not exist.  so two
different things are measured and they are not interchangeable.

**output redundancy** is the correlation of the representation itself across the
values it writes, over a population of real recordings.  it says how much of the
representation is spent saying the same thing twice.  it is *not* an error
correlation and does not bound one in either direction: a shared additive bias
gives errors that correlate at 1 while leaving the outputs as decorrelated as
they were.  it is reported because it is the thing everyone reaches for, and
because knowing it is small stops a reader assuming the answer was obvious.

**induced-error correlation** is the real measurement.  the residual covariance
the schema is asking about is the covariance of `x_true - x_teacher` over
whatever the model gets applied to, and the reason it is not diagonal is that
deployment differs from the benchmark in ways that act on the *input* and
therefore hit every output at once.  those shifts are nameable and reproducible:
a different amplifier gain, a swapped or missing electrode, mains interference.
perturb the input, encode again, and `z_perturbed - z_clean` is a real sample of
the teacher's error under a real distribution shift.  the correlation of that
across the values it writes is exactly `correlated_fraction`, and its
eigenspectrum is exactly `error_rank`.  no ground truth is needed, because the
error is induced rather than estimated.

the perturbations are chosen from BENDR's own card: it fixes a 20-channel subset
by construction and carries no electrode geometry, so a swapped label and a
missing electrode are the failure modes it is least equipped to notice.

BENDR is also the honest case to measure, for a reason its own card records: it
takes a FIXED 20-channel subset, so there is no channel axis in its output at all
and the "values it writes" are (feature, time-position) pairs of a 512-dimensional
representation.  the correlation reported here is over that product.

data: PhysioNet EEG Motor Movement/Imagery (eegmmidb), the corpus BENDR itself
evaluates on as MMI, at 160 Hz, resampled to the 256 Hz the encoder expects.

run:  ./.venv/bin/python scripts/measure_bendr_error_correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import torch
import torch.nn as nn

WEIGHTS = Path("/home/brandonin/Documents/win-data/bendr/encoder.pt")
EEGMMIDB = Path("/home/brandonin/Documents/win-data/eegmmidb/1.0.0")

#: the 19 electrodes of the 10-20 system, in the eegmmidb naming.  BENDR's
#: pretrained encoder takes 20 input channels: these plus one relative-amplitude
#: channel, which is the DEEP1010 mapping's way of telling a min-max scaled
#: signal how large it originally was.
TEN_TWENTY = ["Fp1.", "Fp2.", "F7..", "F3..", "Fz..", "F4..", "F8..",
              "T7..", "C3..", "Cz..", "C4..", "T8..",
              "P7..", "P3..", "Pz..", "P4..", "P8..", "O1..", "O2.."]

SFREQ = 256.0
WINDOW_S = 6.0                 # BENDR's own MMI downstream trial length, table 2
STRIDE_S = 6.0


def build_encoder(path: Path) -> nn.Module:
    """`ConvEncoderBENDR` rebuilt from the checkpoint's own shapes.

    the architecture is not guessed: six blocks, widths and strides, the channel
    count and the group count are all read off the state dict, so a mismatch
    between this and the published model shows up as a load error rather than as
    a plausible number.  the one thing the shapes cannot tell us is that index 1
    is a dropout and index 3 a GELU -- those come from the BENDR source, and both
    are inert at eval time (dropout) or shape-preserving (GELU), so a wrong guess
    there would change the values and not the structure.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    n_blocks = len({k.split(".")[1] for k in sd})
    strides = (3, 2, 2, 2, 2, 2)                       # BENDR enc_downsample default
    blocks = []
    for i in range(n_blocks):
        w = sd[f"encoder.Encoder_{i}.0.weight"]
        out_c, in_c, width = w.shape
        blocks.append((f"Encoder_{i}", nn.Sequential(
            nn.Conv1d(in_c, out_c, width, stride=strides[i], padding=width // 2),
            nn.Dropout2d(0.0),
            nn.GroupNorm(out_c // 2, out_c),
            nn.GELU(),
        )))
    enc = nn.Sequential()
    for name, blk in blocks:
        enc.add_module(name, blk)
    model = nn.Module()
    model.encoder = enc
    model.load_state_dict(sd)                          # strict: any mismatch raises
    model.eval()
    return model


def load_windows(subjects: list[str], runs: list[int], variant: str) -> np.ndarray:
    """(n_windows, 20, n_samples) of real EEG, preprocessed two different ways.

    two variants because the exact DEEP1010 scaling BENDR trained under is not
    reproduced here -- dn3 is not a dependency and guessing its constants would
    be inventing a number in a script written to stop people inventing numbers.
    so the preprocessing is stated rather than claimed: `minmax` is a per-window
    per-channel scaling to [-1, 1], `robust` divides by a per-channel 99th
    percentile taken over the whole recording.  if the correlation structure is a
    property of the encoder it will survive both; if it is a property of my
    scaling choice, the two will disagree and the measurement is worthless.  the
    script prints both and lets the reader see which.
    """
    out = []
    n = int(WINDOW_S * SFREQ)
    step = int(STRIDE_S * SFREQ)
    for s in subjects:
        for r in runs:
            f = EEGMMIDB / s / f"{s}R{r:02d}.edf"
            if not f.exists():
                continue
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            missing = [c for c in TEN_TWENTY if c not in raw.ch_names]
            if missing:
                continue
            raw.pick(TEN_TWENTY)
            raw.resample(SFREQ, verbose="ERROR")
            x = raw.get_data()                          # (19, T), volts
            if variant == "robust":
                scale = np.percentile(np.abs(x), 99, axis=1, keepdims=True)
                x = np.clip(x / np.maximum(scale, 1e-12), -1.0, 1.0)
            for a in range(0, x.shape[1] - n + 1, step):
                w = x[:, a:a + n]
                if variant == "minmax":
                    lo = w.min(1, keepdims=True)
                    hi = w.max(1, keepdims=True)
                    w = 2.0 * (w - lo) / np.maximum(hi - lo, 1e-12) - 1.0
                amp = np.full((1, n), np.clip(np.log10(np.abs(w).mean() + 1e-12), -1, 1))
                out.append(np.concatenate([w, amp], 0))
    return np.asarray(out, dtype=np.float32)


@torch.no_grad()
def encode(model: nn.Module, windows: np.ndarray, batch: int = 32) -> np.ndarray:
    """(n_windows, 512, n_positions) BENDR features from real recordings."""
    reps = []
    for a in range(0, len(windows), batch):
        reps.append(model.encoder(torch.from_numpy(windows[a:a + batch])).numpy())
    return np.concatenate(reps, 0)


def _corr_stats(e: np.ndarray, rng: np.random.Generator, center: bool, n_sub: int = 128):
    """the shared fraction of an error population, two estimators and a held-out count.

    `e` is (n_samples, n_values).  `center` picks which matrix is being described,
    and the choice is not cosmetic.

    the residual covariance the precision formula inverts is `E[e e^T]` about
    *zero*, because a teacher's bias is part of its error -- a model that reads
    every alpha amplitude 10% high is wrong by 10%, not right on average.  a
    systematic bias is also precisely the rank-1 structure the low-rank
    correction exists to catch.  subtracting the mean across the population, the
    reflex of every correlation routine, deletes exactly that term and reports
    only the part of the error that varies from window to window.  so the
    uncentered second moment is the number the card wants and the centered one is
    reported beside it to show how much of the answer the reflex would have
    thrown away.

    two estimators of rho, because the obvious one has a signed failure mode.
    the mean off-diagonal correlation is rho exactly under a one-factor model
    with equal positive loadings.  but a shared factor may load with mixed signs
    -- half the features go up when the gain rises and half go down -- and then
    the pairwise correlations are +rho and -rho in roughly equal numbers and
    average to nothing, reporting independence for a perfectly rank-1 error.  the
    leading eigenvalue's share of the correlation matrix's trace does not care
    about the signs: for a one-factor model with |loading| = sqrt(rho) it is
    exactly rho.  where the two disagree, the eigenvalue is right.

    the effective-constraint count is scored on a *held-out half*.  a sample
    covariance inverted on the data that produced it reports the precision of an
    estimator fitted to noise, and at these sizes that inflation is a factor of
    several -- large enough to invent an answer.
    """
    w, n_all = e.shape
    idx = rng.choice(n_all, size=min(n_sub, n_all), replace=False)
    x = e[:, idx]
    n = x.shape[1]
    xc = x - x.mean(0, keepdims=True) if center else x
    m = xc.T @ xc / (w - 1 if center else w)          # covariance, or second moment
    sd = np.sqrt(np.maximum(np.diag(m), 1e-30))
    r = m / np.outer(sd, sd)
    off = r[~np.eye(n, dtype=bool)]
    rho_bar = float(off.mean())
    ev = np.linalg.eigvalsh(r)[::-1]
    frac = np.cumsum(ev) / ev.sum()

    half = w // 2
    ma = xc[:half].T @ xc[:half] / half
    mb = xc[half:].T @ xc[half:] / (w - half)
    one = np.ones(n)
    a = np.linalg.solve(ma + 1e-6 * np.trace(ma) / n * np.eye(n), one)   # GLS weights, fitted
    eff = float((one @ a) ** 2 / (a @ mb @ a))                           # scored held out
    eff /= 1.0 / float(np.mean(np.diag(mb)))
    return n, rho_bar, float(frac[0]), frac, eff


def report(e: np.ndarray, rng: np.random.Generator, label: str) -> tuple[float, float]:
    print(f"  {label}")
    out = {}
    for center, tag in ((False, "uncentered E[e e^T]  <- the card's number"),
                        (True, "centered  Cov[e]     (bias removed)")):
        n, rho_bar, lam1, frac, eff = _corr_stats(e, rng, center)
        model_eff = n / ((1 - lam1) + n * lam1) if lam1 > 0 else float(n)
        print(f"    {tag}")
        print(f"      rho, mean off-diagonal    : {rho_bar: .4f}")
        print(f"      rho, leading eigenvalue   : {lam1: .4f}   "
              f"(top 4 {frac[3]:.3f}, top 16 {frac[15]:.3f})")
        print(f"      effective constraints     : {eff:8.2f} measured (held out), "
              f"{model_eff:8.2f} rank-1 at that rho, out of {n}")
        out[center] = lam1
    return out[False], out[True]


def perturb(windows: np.ndarray, kind: str) -> np.ndarray:
    """a named, reproducible distribution shift applied to the encoder's input.

    each one is a thing that actually happens between a benchmark corpus and a
    deployment, and each is chosen because BENDR's own card says it has no
    defence against it: the model fixes a 20-channel subset, identifies channels
    by position in that subset, and carries no electrode geometry and no forward
    model, so nothing in it can notice that two labels were exchanged or that an
    electrode is missing.
    """
    x = windows.copy()
    if kind == "gain x1.5":
        # a different amplifier or a different unit convention.  BENDR's input is
        # min-max scaled, so this is not a no-op: it changes the clipping and the
        # relative-amplitude channel.
        x[:, :19] = np.clip(x[:, :19] * 1.5, -1.0, 1.0)
    elif kind == "swap C3/C4":
        i, j = TEN_TWENTY.index("C3.."), TEN_TWENTY.index("C4..")
        x[:, [i, j]] = x[:, [j, i]]
    elif kind == "drop Cz":
        x[:, TEN_TWENTY.index("Cz..")] = 0.0
    elif kind == "50 Hz mains":
        t = np.arange(x.shape[2], dtype=np.float32) / SFREQ
        x[:, :19] += 0.15 * np.sin(2 * np.pi * 50.0 * t)[None, None, :]
        x[:, :19] = np.clip(x[:, :19], -1.0, 1.0)
    else:
        raise ValueError(kind)
    return x


def main() -> None:
    subjects = [f"S{i:03d}" for i in range(1, 9)]
    runs = list(range(3, 15))                          # the motor imagery/movement runs
    model = build_encoder(WEIGHTS)
    print("=" * 78)
    print("BENDR encoder over eegmmidb: measured correlation structure")
    print("=" * 78)
    print(f"weights: {WEIGHTS}")
    print(f"subjects {subjects[0]}..{subjects[-1]}, runs R03..R14, "
          f"{WINDOW_S:g}s windows at {SFREQ:g} Hz")

    rng = np.random.default_rng(20260904)
    wins = load_windows(subjects, runs, "robust")
    z = encode(model, wins)
    w, c, t = z.shape
    print(f"\n{w} windows -> representation ({c} features x {t} positions) "
          f"= {c * t} values per window\n")

    print("-" * 78)
    print("1. output redundancy -- NOT an error correlation, reported for context")
    print("-" * 78)
    report(z.reshape(w, c * t), rng, "over (feature, position) pairs")

    print("\n" + "-" * 78)
    print("2. induced-error correlation -- this is what correlated_fraction wants")
    print("-" * 78)
    rhos = {}
    for kind in ("gain x1.5", "swap C3/C4", "drop Cz", "50 Hz mains"):
        zp = encode(model, perturb(wins, kind))
        err = (zp - z).reshape(w, c * t)
        rel = float(np.linalg.norm(zp - z) / np.linalg.norm(z))
        print(f"\n  perturbation: {kind}   (relative representation change "
              f"{rel:.3f})")
        rhos[kind] = report(err, rng, "error over (feature, position) pairs")

    print("\n" + "=" * 78)
    print("summary")
    print("=" * 78)
    print(f"  {'perturbation':>14}  {'rho (uncentered)':>17}  {'rho (centered)':>15}")
    for k, (u, c_) in rhos.items():
        print(f"  {k:>14}  {u:>17.4f}  {c_:>15.4f}")
    vals = [u for u, _ in rhos.values()]
    print(f"\n  uncentered rho across perturbations: {min(vals):.4f} to {max(vals):.4f}, "
          f"median {float(np.median(vals)):.4f}")
    print("  each perturbation is one deployment shift, not a sample from the")
    print("  distribution of them, so the spread is a floor on the uncertainty in")
    print("  rho and not a confidence interval.  the card declares a prior.")


if __name__ == "__main__":
    main()

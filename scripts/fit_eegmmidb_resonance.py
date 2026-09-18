#!/usr/bin/env python
"""occipital alpha and sensorimotor mu: two more partitions of the declared resonator.

`fit_sleep_resonance.py` put `thalamocortical_coupling:alpha_resonator` where it could
fail on sleep spectra, split by subject, and it held (STATE.md §3 row 1).  this script
applies the same design to two further circuits the site draws, "Retino-Geniculo-
Striate Alpha" and "Sensorimotor Mu", using PhysioNet eegmmidb, and adds three things
the sleep run did not have:

    a baseline    an aperiodic-only spectrum g f^-e with NO resonance.  beating the
                  untouched prior is not enough; the resonance has to beat a power law
    a known answer the Berger effect (eyes-closed posterior alpha > eyes-open) and mu
                  desynchronisation (executed movement < rest at C3/C4).  if the
                  pipeline cannot recover those, nothing it fits is reported
    a control     the EC/EO (or rest/movement) labels relabelled within the training
                  subjects.  any state contrast the model claims must vanish under it

THE PRE-REGISTRATION IS docs/LOG.md "PRE-REGISTRATION: occipital alpha and sensorimotor
mu, on eegmmidb".  every threshold, the band, the grid, the split and the seeds below
are the ones fixed there; change none of them after a run.  a gate that fails is FAILED.

the declared prior is the registry's `alpha_resonator` (f0 ~ N(10, 1.2) Hz, q ~
LogNormal(4, x/ 1.8)), per_partition.  there is no mu- or visual-specific resonator in
the registry, so the two circuits are two PARTITIONS of that one declaration, each
collapsed to a single (f0, q) within its montage, as the sleep script collapsed its.
nothing here writes a prior.

two departures from the sleep template, both pre-registered:

- the drive-exponent grid runs from -6, not -1.  `alpha_resonance_transfer` is a
  LOW-pass resonance with unit DC gain, so |H|^2 falls as (f0/f)^4 above the peak;
  with e >= -1 the resonant model must fall at least as f^-3 above its peak, and the
  aperiodic comparison would be decided by where the grid stopped.
- the profile likelihood is vectorised over recordings and exponents.  it is checked
  against `SpectralEvidence.log_likelihood` at identical inputs before anything is
  fitted (gate K0); if they disagree the run stops.

what this CANNOT test, said before it is asked: the card forbids reading imagery trials
as verified covert motor state (no behavioural check), so the imagery contrast is
reported and never gated; every electrode is a template position with an unrecorded
reference, so "occipital" and "sensorimotor" are montage statements, not cortical
ones (`eeg_predict`'s own prior_dominated list says as much); and mu's ~20 Hz harmonic
sits at the band's edge and is modelled by nobody.

run:  CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/fit_eegmmidb_resonance.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

SOURCE = "eegmmidb"
IMPL = "thalamocortical_coupling:alpha_resonator"

#: pre-registered.  see the LOG entry for why each is what it is.
BAND_LO, BAND_HI = 4.0, 20.0
E_GRID = np.linspace(-6.0, 5.0, 111)
ALPHA_LO, ALPHA_HI = 8.0, 13.0          # the known-answer band, K1 and K2
N_GATES = 6
ALPHA_GATE = 0.01 / N_GATES             # 0.00167, Bonferroni over A1-A3, M1-M3
SPLIT_SEED, TRAIN_FRAC = 0, 0.6
BOOT_SEED, N_BOOT = 2, 10_000
K_RELABEL = 20
MIN_WINDOWS = 5
FS = 160.0

POSTERIOR_CH = ("p3", "p1", "pz", "p2", "p4", "po3", "poz", "po4", "o1", "oz", "o2")
LAPLACIAN = {"c3": ("fc3", "c5", "c1", "cp3"), "c4": ("fc4", "c6", "c2", "cp4")}
EXEC_RUNS = (3, 5, 7, 9, 11, 13)
IMAG_RUNS = (4, 6, 8, 10, 12, 14)
#: window codes for the mu per-window store
REST_EXEC, EXEC, REST_IMAG, IMAG = 0, 1, 2, 3
MU_WIN_S, MU_SKIP_S, MU_MIN_SEG_S = 3.0, 0.5, 3.5
ALPHA_WIN_S = 4.0


# ---------------------------------------------------------------------------
# bookkeeping: the summary is written before anything that can raise
# ---------------------------------------------------------------------------


def _coerce(o):
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 10_000 else f"<ndarray {o.shape}>"
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    return repr(o)


class Summary:
    def __init__(self, path: Path):
        self.path, self.d = path, {}
        path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, key, value):
        self.d[key] = value
        self.path.write_text(json.dumps(self.d, indent=1, default=_coerce))


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


# ---------------------------------------------------------------------------
# where the bytes are
# ---------------------------------------------------------------------------


def data_root() -> tuple[Path, str]:
    """`local_root` per `.location.yaml` if there is one, else the card's own field.

    eegmmidb has no `.location.yaml` on this machine; its card carries
    `local_root: data/sources/eegmmidb/raw`.  reading that is reading the card, not
    inventing a path, and the route taken is printed.
    """
    from ibm.forge.spectra import local_root
    try:
        return local_root(SOURCE) / "1.0.0", ".location.yaml"
    except FileNotFoundError:
        import yaml
        card = yaml.safe_load((REPO / "data" / "sources" / SOURCE / "card.yaml").read_text())
        p = Path(card["local_root"])
        p = (p if p.is_absolute() else REPO / p) / "1.0.0"
        if not p.is_dir():
            raise FileNotFoundError(f"card local_root {p} does not exist")
        return p, "card.yaml local_root"


def split_subjects() -> tuple[list[str], list[str]]:
    """the pre-registered draw: all 109 ids, sorted, shuffled by default_rng(0)."""
    subs = [f"S{i:03d}" for i in range(1, 110)]
    order = list(subs)
    np.random.default_rng(SPLIT_SEED).shuffle(order)
    n = int(round(TRAIN_FRAC * len(order)))
    return sorted(order[:n]), sorted(order[n:])


# ---------------------------------------------------------------------------
# reading ONE subject (runs in a worker; returns small arrays only)
# ---------------------------------------------------------------------------


def _norm(names):
    return [n.strip().strip(".").lower() for n in names]


def read_subject(args) -> dict:
    """one subject's alpha evidence and mu per-window psds, or the reason it is excluded.

    streamed: one EDF in memory at a time, and only derived arrays leave the worker.
    """
    root, s = args
    import mne
    from ibm.fields.uncertainty.spectral import TemporalBasis
    from ibm.forge.spectra import clean_windows, evidence_from_signal, periodogram, window
    from ibm.vocabulary import Band
    mne.set_log_level("ERROR")
    band = Band(BAND_LO, BAND_HI)
    out: dict = {"subject": s, "alpha": None, "mu": None, "excluded": {}}

    # -- alpha: R01 eyes open, R02 eyes closed, 11 posterior channels -----------
    try:
        got = {}
        for run, lab in ((1, "EO"), (2, "EC")):
            raw = mne.io.read_raw_edf(Path(root) / s / f"{s}R{run:02d}.edf", preload=True)
            if abs(raw.info["sfreq"] - FS) > 1e-6:
                raise ValueError(f"R{run:02d} at {raw.info['sfreq']:g} Hz")
            nm = _norm(raw.ch_names)
            ch = [nm.index(c) for c in POSTERIOR_CH if c in nm]
            if len(ch) != len(POSTERIOR_CH):
                raise ValueError(f"R{run:02d} missing posterior channels")
            ev = evidence_from_signal(raw.get_data(picks=ch), FS, window_s=ALPHA_WIN_S,
                                      band=band, source=f"{SOURCE}/{s}/R{run:02d}", note=lab)
            got[lab] = {"psd": ev.psd.astype(float), "n_windows": int(ev.n_windows)}
            del raw
        if min(v["n_windows"] for v in got.values()) < MIN_WINDOWS:
            raise ValueError(f"fewer than {MIN_WINDOWS} windows survive")
        out["alpha"] = got
    except Exception as exc:                                  # noqa: BLE001
        out["excluded"]["alpha"] = f"{type(exc).__name__}: {exc}"

    # -- mu: R03-R14, small laplacian at C3 and C4, one 3 s window per segment --
    try:
        n = int(round(MU_WIN_S * FS))
        basis = TemporalBasis(n, 1.0 / FS)
        segs, labs = [], []
        for run in EXEC_RUNS + IMAG_RUNS:
            raw = mne.io.read_raw_edf(Path(root) / s / f"{s}R{run:02d}.edf", preload=True)
            if abs(raw.info["sfreq"] - FS) > 1e-6:
                raise ValueError(f"R{run:02d} at {raw.info['sfreq']:g} Hz")
            nm = _norm(raw.ch_names)
            need = [c for k, v in LAPLACIAN.items() for c in (k,) + v]
            if any(c not in nm for c in need):
                raise ValueError(f"R{run:02d} missing laplacian channels")
            x = raw.get_data()
            lap = np.stack([x[nm.index(k)] - x[[nm.index(c) for c in v]].mean(0)
                            for k, v in LAPLACIAN.items()])          # (2, T)
            ex = run in EXEC_RUNS
            for on, du, de in zip(raw.annotations.onset, raw.annotations.duration,
                                  raw.annotations.description):
                if du < MU_MIN_SEG_S:
                    continue
                i0 = int(round((on + MU_SKIP_S) * FS))
                if i0 + n > lap.shape[1]:
                    continue
                if de == "T0":
                    code = REST_EXEC if ex else REST_IMAG
                elif de in ("T1", "T2"):
                    code = EXEC if ex else IMAG
                else:
                    continue
                segs.append(lap[:, i0:i0 + n])
                labs.append(code)
            del raw, x
        w = window(np.stack(segs, 1), basis)[..., 0, :]  # (2, W, 1, n) -> (2, W, n), detrended
        keep = clean_windows(w)                          # pooled across labels, ONCE
        p = periodogram(w[:, keep], basis).mean(0)       # (W', k), psd averaged over C3', C4'
        labs = np.asarray(labs)[keep]
        counts = {c: int((labs == c).sum()) for c in (REST_EXEC, EXEC, REST_IMAG, IMAG)}
        if min(counts.values()) < MIN_WINDOWS:
            raise ValueError(f"window counts {counts} below {MIN_WINDOWS}")
        out["mu"] = {"psd": p.astype(float), "labels": labs.astype(np.int8),
                     "n_segments": len(segs), "n_rejected": int((~keep).sum())}
    except Exception as exc:                                  # noqa: BLE001
        out["excluded"]["mu"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# the model and a vectorised profile likelihood
# ---------------------------------------------------------------------------


class Cohort:
    """a stack of recordings on one basis, prepared for the profile likelihood.

    `S` (R, B) in-band psds, `A` (R, B) gamma shapes dof/2, and the parts of the gamma
    log density that do not depend on the model.  `loglik` returns (R, E) over the
    exponent grid; the caller maximises over E.  this is the same density as
    `SpectralEvidence.log_likelihood` with the gain profiled -- K0 checks that.
    """

    def __init__(self, evs, names):
        from scipy.special import gammaln
        self.evs, self.names = list(evs), list(names)
        b = self.evs[0].basis
        self.idx = b.band_indices(_band())
        self.f = b.freqs_hz[self.idx]
        self.S = np.stack([np.asarray(e.psd, float)[self.idx] for e in self.evs])
        self.A = np.stack([0.5 * e.dof[self.idx] for e in self.evs])
        s = np.maximum(self.S, 1e-300)
        self.const = (self.A * np.log(self.A) - gammaln(self.A)
                      + (self.A - 1.0) * np.log(s)).sum(1)
        self.sumA = self.A.sum(1)
        self.logf = np.log(np.maximum(self.f, 1e-6))

    def __len__(self):
        return len(self.evs)

    def loglik(self, shape: np.ndarray) -> np.ndarray:
        """(R, E): log p(psd_r | g_r, e) at the ML g_r for every e, for one shape(f)."""
        logM = np.log(np.maximum(shape, 1e-300))[None, :] - E_GRID[:, None] * self.logf[None, :]
        invM = np.exp(-logM)                                        # (E, B)
        c = (self.A * self.S) @ invM.T / self.sumA[:, None]         # (R, E) ML gains
        c = np.maximum(c, 1e-300)
        return (self.const[:, None] - self.sumA[:, None] * np.log(c)
                - self.A @ logM.T - self.sumA[:, None])

    def best(self, shape: np.ndarray):
        ll = self.loglik(shape)
        j = ll.argmax(1)
        return ll[np.arange(len(j)), j], E_GRID[j]


def _band():
    from ibm.vocabulary import Band
    return Band(BAND_LO, BAND_HI)


def resonance_shape(f: np.ndarray, f0: float, q: float) -> np.ndarray:
    """|H(f; f0, q)|^2 from the registry's own `alpha_resonance_transfer`."""
    from ibm.processes.neural import alpha_resonance_transfer

    class _B:                                    # the transfer reads only `omega`
        omega = 2.0 * np.pi * f
    return np.abs(alpha_resonance_transfer(_B, f0_hz=float(f0), q=float(q), gain=1.0)) ** 2


def aperiodic_shape(f: np.ndarray) -> np.ndarray:
    return np.ones_like(f)


def mean_shape(cohort: Cohort) -> np.ndarray:
    """the train cohort's mean normalised log-psd: a non-parametric reference."""
    y = np.log(np.maximum(cohort.S, 1e-300))
    return np.exp((y - y.mean(1, keepdims=True)).mean(0))


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def fit(cohort: Cohort, space, name: str):
    """one (f0, q) for a cohort, MAP under the declared prior, through ibm.forge.fit."""
    from ibm.forge.fit import Task, fit_map
    bf, bq = space["f0_hz"], space["q"]

    def logp(theta):
        f0, q = float(theta[bf.slice][0]), float(theta[bq.slice][0])
        if not (1.0 < f0 < 45.0 and 0.2 < q < 200.0):
            return -1e12
        v = float(cohort.best(resonance_shape(cohort.f, f0, q))[0].sum())
        return v if np.isfinite(v) else -1e12

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        th, rep = fit_map(space, [Task(name=f"{SOURCE}.{name}", logp=logp,
                                       moves=(bf.key, bq.key), source=SOURCE, kind="fit",
                                       note=f"{len(cohort)} recordings")], max_iter=200)
    return (float(th[bf.slice][0]), float(th[bq.slice][0])), rep


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def gate_stats(d: np.ndarray, rng: np.random.Generator) -> dict:
    """paired difference over test SUBJECTS: t, bootstrap over subjects, win rate."""
    from scipy import stats
    d = np.asarray(d, float)
    n = d.size
    t, p = stats.ttest_1samp(d, 0.0)
    bs = d[rng.integers(0, n, size=(N_BOOT, n))].mean(1)
    lo, hi = np.percentile(bs, [100 * ALPHA_GATE / 2, 100 * (1 - ALPHA_GATE / 2)])
    lo95, hi95 = np.percentile(bs, [2.5, 97.5])
    sd = float(d.std(ddof=1))
    out = {"n": n, "mean": float(d.mean()), "sd": sd, "sem": sd / math.sqrt(n),
           "t": float(t), "p": float(p), "dz": float(d.mean() / sd) if sd else float("nan"),
           "win_rate": float((d > 0).mean()), "ci95": [float(lo95), float(hi95)],
           "ci_gate": [float(lo), float(hi)], "ci_gate_level": 1 - ALPHA_GATE}
    out["passed"] = bool(out["mean"] > 0 and out["p"] < ALPHA_GATE and lo > 0
                         and out["win_rate"] > 0.5)
    return out


def show(tag: str, st: dict, unit: str = "nats/subject") -> None:
    print(f"  {tag:46s} {st['mean']:+10.2f} {unit}  95% CI [{st['ci95'][0]:+.2f}, "
          f"{st['ci95'][1]:+.2f}]  t {st['t']:+6.2f}  p {st['p']:.3g}  "
          f"wins {st['win_rate']:.0%} of {st['n']}"
          + (f"  -> {'PASS' if st['passed'] else 'FAIL'}" if "passed" in st else ""),
          flush=True)


def sign_test(x: np.ndarray) -> dict:
    from scipy.stats import binomtest
    k, n = int((x > 0).sum()), int(x.size)
    return {"k": k, "n": n, "fraction": k / n, "p": float(binomtest(k, n, 0.5).pvalue)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=None,
                    help="npz of per-subject derived arrays (never commit it)")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "sources" / SOURCE / "evidence"
                    / "fit_eegmmidb_resonance@v1" / "result.json")
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N subjects")
    args = ap.parse_args()
    t0 = time.time()
    summ = Summary(args.out)
    summ.put("preregistration", "docs/LOG.md 2026-09-18 PRE-REGISTRATION: occipital alpha "
                                "and sensorimotor mu, on eegmmidb, held out by subject")
    summ.put("constants", {"band_hz": [BAND_LO, BAND_HI], "e_grid": [float(E_GRID[0]),
             float(E_GRID[-1]), len(E_GRID)], "alpha_gate": ALPHA_GATE, "n_boot": N_BOOT,
             "k_relabel": K_RELABEL, "split_seed": SPLIT_SEED, "boot_seed": BOOT_SEED})

    import ibm
    ibm.load_all(seal=True, strict=True)
    from fit_sleep_resonance import parameter_space
    from ibm.forge.spectra import SpectralEvidence
    from ibm.fields.uncertainty.spectral import TemporalBasis
    space = parameter_space()
    rule("p(theta): the declared prior, from REGISTRY, unedited")
    print(space.describe())
    prior = {b.name: {"dist": b.prior.dist, "loc": b.prior.loc,
                      "scale": getattr(b.prior, "scale", None),
                      "source": b.prior.source, "note": b.prior.note} for b in space.blocks}
    th0 = space.median()
    F0P, QP = float(th0[space["f0_hz"].slice][0]), float(th0[space["q"].slice][0])
    print(f"prior median  f0 {F0P:.3f} Hz  q {QP:.3f}")
    summ.put("prior", {"impl": IMPL, "blocks": prior, "median": {"f0_hz": F0P, "q": QP},
                       "written_by_this_script": False})

    # -- split, then data ------------------------------------------------------
    tr_all, te_all = split_subjects()
    rule("split (pre-registered, by SUBJECT)")
    print(f"train {len(tr_all)}: {' '.join(tr_all)}\ntest  {len(te_all)}: {' '.join(te_all)}")
    summ.put("split", {"train": tr_all, "test": te_all})

    root, how = data_root()
    rule("data")
    print(f"{SOURCE} root {root}  (via {how})")
    subjects = tr_all + te_all
    if args.limit:
        subjects = sorted(subjects)[: args.limit]
    recs: dict[str, dict] = {}
    if args.cache and args.cache.is_file():
        z = np.load(args.cache, allow_pickle=True)
        recs = z["recs"].item()
        print(f"loaded {len(recs)} subjects from cache {args.cache}")
    todo = [s for s in subjects if s not in recs]
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(read_subject, [(str(root), s) for s in todo])):
                recs[r["subject"]] = r
                if (i + 1) % 10 == 0:
                    print(f"  read {i + 1}/{len(todo)}  ({time.time() - t0:.0f} s)", flush=True)
        if args.cache:
            np.savez(args.cache, recs=np.array(recs, dtype=object))
    excl = {s: r["excluded"] for s, r in recs.items() if r["excluded"]}
    for s, e in sorted(excl.items()):
        print(f"  excluded {s}: {e}")
    summ.put("exclusions", excl)

    basis_a = TemporalBasis(int(round(ALPHA_WIN_S * FS)), 1.0 / FS)
    basis_m = TemporalBasis(int(round(MU_WIN_S * FS)), 1.0 / FS)
    band = _band()

    def ev_alpha(s, lab):
        a = recs[s]["alpha"][lab]
        return SpectralEvidence(basis_a, a["psd"], a["n_windows"], band=band,
                                source=f"{SOURCE}/{s}/{lab}", note=lab)

    def ev_mu(s, codes, labels=None):
        m = recs[s]["mu"]
        lab = m["labels"] if labels is None else labels
        sel = np.isin(lab, codes)
        return SpectralEvidence(basis_m, m["psd"][sel].mean(0), int(sel.sum()), band=band,
                                source=f"{SOURCE}/{s}/mu{tuple(codes)}")

    A_ok = [s for s in subjects if recs[s]["alpha"] is not None]
    M_ok = [s for s in subjects if recs[s]["mu"] is not None]
    a_tr, a_te = [s for s in A_ok if s in tr_all], [s for s in A_ok if s in te_all]
    m_tr, m_te = [s for s in M_ok if s in tr_all], [s for s in M_ok if s in te_all]
    print(f"alpha: {len(a_tr)} train / {len(a_te)} test subjects;  "
          f"mu: {len(m_tr)} train / {len(m_te)} test")
    ex0 = ev_alpha(A_ok[0], "EC")
    print(f"  example alpha: {ex0}\n  fitted over {BAND_LO:g}-{BAND_HI:g} Hz: "
          f"{len(ex0.basis.band_indices(band))} bins")
    exm = ev_mu(M_ok[0], (REST_EXEC, REST_IMAG))
    print(f"  example mu:    {exm}\n  fitted over {BAND_LO:g}-{BAND_HI:g} Hz: "
          f"{len(exm.basis.band_indices(band))} bins")
    wc = np.array([[int((recs[s]["mu"]["labels"] == c).sum()) for c in range(4)] for s in M_ok])
    print(f"  mu windows per subject (median): rest_exec {np.median(wc[:, 0]):.0f}  exec "
          f"{np.median(wc[:, 1]):.0f}  rest_imag {np.median(wc[:, 2]):.0f}  imag "
          f"{np.median(wc[:, 3]):.0f}")
    summ.put("n", {"alpha_train": len(a_tr), "alpha_test": len(a_te), "mu_train": len(m_tr),
                   "mu_test": len(m_te), "mu_windows_median": np.median(wc, 0)})

    # =========================================================================
    rule("K0: the instrument, against answers known in advance")
    # (a) the vectorised likelihood equals SpectralEvidence.log_likelihood
    coh = Cohort([ev_alpha(s, "EC") for s in A_ok[:5]], A_ok[:5])
    worst = 0.0
    for f0, q in ((10.0, 4.0), (12.3, 1.7), (8.1, 9.0)):
        ll = coh.loglik(resonance_shape(coh.f, f0, q))
        for j in (0, 37, 80, 110):
            m = np.zeros(coh.evs[0].basis.k)
            m[coh.idx] = resonance_shape(coh.f, f0, q) * np.maximum(coh.f, 1e-6) ** (-E_GRID[j])
            for r, e in enumerate(coh.evs):
                ref = e.log_likelihood(m, band=band)
                worst = max(worst, abs(ll[r, j] - ref) / abs(ref))
    lik_ok = worst < 1e-9
    print(f"  vectorised vs SpectralEvidence.log_likelihood: worst relative error {worst:.2e}"
          f"  -> {'PASS' if lik_ok else 'FAIL'}")
    # (b) synthetic recovery at f0 11.5, q 3, e 1.5, one synthetic spectrum per train
    #     subject at that subject's own dof -- the regime the fit actually runs in
    rng_syn = np.random.default_rng(7)
    true_f0, true_q, true_e = 11.5, 3.0, 1.5
    syn = []
    for s in a_tr:
        e = ev_alpha(s, "EC")
        f = e.basis.freqs_hz
        mm = resonance_shape(f, true_f0, true_q) * np.maximum(f, 1e-6) ** (-true_e) * 1e-11
        a = np.maximum(0.5 * e.dof, 0.5)
        syn.append(SpectralEvidence(e.basis, mm * rng_syn.gamma(a, 1.0 / a), e.n_windows,
                                    band=band, source=f"synthetic/{s}"))
    (sf0, sq), _ = fit(Cohort(syn, a_tr), space, "k0.synthetic")
    rec_ok = abs(sf0 - true_f0) <= 0.2 and abs(sq / true_q - 1) <= 0.2
    print(f"  synthetic recovery: f0 {sf0:.3f} (true {true_f0})  q {sq:.3f} (true {true_q})"
          f"  -> {'PASS' if rec_ok else 'FAIL'}")
    summ.put("K0", {"lik_rel_err": worst, "lik_ok": lik_ok, "synthetic": {
        "true": [true_f0, true_q, true_e], "fitted": [sf0, sq], "n": len(syn)},
        "recovery_ok": rec_ok, "passed": bool(lik_ok and rec_ok)})
    if not (lik_ok and rec_ok):
        print("\nK0 FAILED: the instrument is broken; nothing is fitted.")
        return 3

    # =========================================================================
    rule("K1 Berger / K2 mu ERD: known answers, all included subjects, no model")

    from ibm.vocabulary import Band

    def bandpow(e):
        return float(np.mean(e.psd[e.basis.band_indices(Band(ALPHA_LO, ALPHA_HI))]))
    ec = np.array([bandpow(ev_alpha(s, "EC")) for s in A_ok])
    eo = np.array([bandpow(ev_alpha(s, "EO")) for s in A_ok])
    k1 = sign_test(np.log(ec / eo))
    k1["median_ratio_ec_over_eo"] = float(np.median(ec / eo))
    k1["passed"] = bool(k1["fraction"] >= 0.75 and k1["p"] < 1e-3)
    print(f"  K1 Berger: EC > EO in {k1['k']}/{k1['n']} ({k1['fraction']:.0%}), sign p "
          f"{k1['p']:.2g}, median EC/EO 8-13 Hz power {k1['median_ratio_ec_over_eo']:.2f}x"
          f"  -> {'PASS' if k1['passed'] else 'FAIL'}")
    rx = np.array([bandpow(ev_mu(s, (REST_EXEC,))) for s in M_ok])
    mx = np.array([bandpow(ev_mu(s, (EXEC,))) for s in M_ok])
    ri = np.array([bandpow(ev_mu(s, (REST_IMAG,))) for s in M_ok])
    mi = np.array([bandpow(ev_mu(s, (IMAG,))) for s in M_ok])
    k2 = sign_test(np.log(rx / mx))
    k2["median_ratio_exec_over_rest"] = float(np.median(mx / rx))
    k2["passed"] = bool(k2["fraction"] >= 0.70 and k2["p"] < 1e-3)
    k2i = sign_test(np.log(ri / mi))
    k2i["median_ratio_imag_over_rest"] = float(np.median(mi / ri))
    print(f"  K2 mu ERD (executed): EXEC < REST in {k2['k']}/{k2['n']} ({k2['fraction']:.0%}),"
          f" sign p {k2['p']:.2g}, median EXEC/REST {k2['median_ratio_exec_over_rest']:.2f}x"
          f"  -> {'PASS' if k2['passed'] else 'FAIL'}")
    print(f"     (imagery, reported not gated: IMAG < REST in {k2i['k']}/{k2i['n']} "
          f"({k2i['fraction']:.0%}), p {k2i['p']:.2g}, median {k2i['median_ratio_imag_over_rest']:.2f}x)")
    summ.put("K1_berger", k1)
    summ.put("K2_mu_erd", {"executed": k2, "imagery_not_gated": k2i})
    if not k1["passed"]:
        print("\nK1 FAILED: the shared pipeline cannot recover the Berger effect. No fit "
              "result from either circuit is reported.")
        return 4

    rng_boot = np.random.default_rng(BOOT_SEED)
    results: dict = {}

    def score_models(cohort, shapes: dict) -> dict:
        """per-recording best ll, chosen e, and r2 for each named shape."""
        out = {}
        for k, shp in shapes.items():
            ll, e = cohort.best(shp)
            r2 = []
            for ev, ee in zip(cohort.evs, e):
                m = np.zeros(ev.basis.k)
                m[cohort.idx] = shp * np.maximum(cohort.f, 1e-6) ** (-ee)
                r2.append(ev.r2_log(m, band=band))
            out[k] = {"ll": ll, "e": e, "r2": np.array(r2),
                      "edge": float(np.mean((e <= E_GRID[0]) | (e >= E_GRID[-1])))}
        return out

    def contrast_D(te_a: Cohort, te_b: Cohort, pa, pb) -> np.ndarray:
        """D = [ll(a|post_a) - ll(a|post_b)] + [ll(b|post_b) - ll(b|post_a)] per subject."""
        la_a = te_a.best(resonance_shape(te_a.f, *pa))[0]
        la_b = te_a.best(resonance_shape(te_a.f, *pb))[0]
        lb_b = te_b.best(resonance_shape(te_b.f, *pb))[0]
        lb_a = te_b.best(resonance_shape(te_b.f, *pa))[0]
        return (la_a - la_b) + (lb_b - lb_a)

    def relabel_verdict(true_mean: float, perm_means: list[float]) -> dict:
        pm = np.asarray(perm_means)
        ok = bool(true_mean > np.max(np.abs(pm)) and
                  abs(np.median(pm)) <= 0.1 * true_mean)
        return {"true_mean": true_mean, "perm_means": pm, "max_abs": float(np.max(np.abs(pm))),
                "median": float(np.median(pm)), "passed": ok}

    # =========================================================================
    rule("ALPHA: fit on train subjects")
    tr_ec = Cohort([ev_alpha(s, "EC") for s in a_tr], a_tr)
    tr_eo = Cohort([ev_alpha(s, "EO") for s in a_tr], a_tr)
    te_ec = Cohort([ev_alpha(s, "EC") for s in a_te], a_te)
    te_eo = Cohort([ev_alpha(s, "EO") for s in a_te], a_te)
    post_ec, rep_ec = fit(tr_ec, space, "alpha.ec.train")
    post_eo, rep_eo = fit(tr_eo, space, "alpha.eo.train")
    print(f"  prior    f0 {F0P:7.3f} Hz  q {QP:7.3f}")
    print(f"  post_EC  f0 {post_ec[0]:7.3f} Hz  q {post_ec[1]:7.3f}   converged {rep_ec.converged}")
    print(f"  post_EO  f0 {post_eo[0]:7.3f} Hz  q {post_eo[1]:7.3f}   converged {rep_eo.converged}")

    # bookkeeping: twice at the same input, and with the train order shuffled
    again, _ = fit(tr_ec, space, "alpha.ec.train.again")
    perm = np.random.default_rng(99).permutation(len(a_tr))
    shuf, _ = fit(Cohort([tr_ec.evs[i] for i in perm], [a_tr[i] for i in perm]), space,
                  "alpha.ec.train.shuffled")
    idem = bool(again == post_ec)
    order_rel = max(abs(a - b) / abs(b) for a, b in zip(shuf, post_ec))
    book_ok = bool(idem and order_rel < 1e-6)
    print(f"  bookkeeping: identical on refit {idem}; train-order shuffle moves theta by "
          f"{order_rel:.1e} relative  -> {'PASS' if book_ok else 'FAIL (run void)'}")
    results["bookkeeping"] = {"idempotent": idem, "order_rel": order_rel, "passed": book_ok}
    results["alpha_fit"] = {"post_EC": post_ec, "post_EO": post_eo,
                            "converged": [rep_ec.converged, rep_eo.converged]}
    summ.put("results", results)
    if not book_ok:
        print("\nbookkeeping FAILED: the fit is not a function of its input. Run void.")
        return 5

    rule("ALPHA: held out, on unseen participants")
    ms_ec, ms_eo = mean_shape(tr_ec), mean_shape(tr_eo)
    sc_ec = score_models(te_ec, {"post": resonance_shape(te_ec.f, *post_ec),
                                 "prior": resonance_shape(te_ec.f, F0P, QP),
                                 "aperiodic": aperiodic_shape(te_ec.f),
                                 "mean_shape": ms_ec})
    sc_eo = score_models(te_eo, {"post": resonance_shape(te_eo.f, *post_eo),
                                 "prior": resonance_shape(te_eo.f, F0P, QP),
                                 "aperiodic": aperiodic_shape(te_eo.f),
                                 "mean_shape": ms_eo})
    nb = len(te_ec.idx)
    for tag, sc in (("EC", sc_ec), ("EO", sc_eo)):
        print(f"\n  test {tag}: {len(te_ec)} subjects x {nb} bins. mean ll/subject, r2(log psd), "
              f"fraction of fits on a grid edge:")
        for k, v in sc.items():
            print(f"    {k:11s} ll {v['ll'].mean():10.2f}   r2 {v['r2'].mean():6.3f}   "
                  f"e median {np.median(v['e']):+5.2f}   edge {v['edge']:.0%}")
    A1 = gate_stats(sc_ec["post"]["ll"] - sc_ec["prior"]["ll"], rng_boot)
    A2 = gate_stats(sc_ec["post"]["ll"] - sc_ec["aperiodic"]["ll"], rng_boot)
    DA = contrast_D(te_ec, te_eo, post_ec, post_eo)
    A3 = gate_stats(DA, rng_boot)
    print()
    show("A1  EC: post_EC - prior", A1)
    show("A2  EC: post_EC - aperiodic", A2)
    show("A3  EC/EO contrast D (before relabelling)", A3)
    sec_a = {"EO_post_minus_prior": gate_stats(sc_eo["post"]["ll"] - sc_eo["prior"]["ll"], rng_boot),
             "EO_post_minus_aperiodic": gate_stats(sc_eo["post"]["ll"] - sc_eo["aperiodic"]["ll"], rng_boot),
             "EC_prior_minus_aperiodic": gate_stats(sc_ec["prior"]["ll"] - sc_ec["aperiodic"]["ll"], rng_boot),
             "EO_prior_minus_aperiodic": gate_stats(sc_eo["prior"]["ll"] - sc_eo["aperiodic"]["ll"], rng_boot),
             "EC_post_minus_meanshape": gate_stats(sc_ec["post"]["ll"] - sc_ec["mean_shape"]["ll"], rng_boot),
             "EO_post_minus_meanshape": gate_stats(sc_eo["post"]["ll"] - sc_eo["mean_shape"]["ll"], rng_boot)}
    for k, v in sec_a.items():
        v.pop("passed", None)
        show(f"    (not gated) {k}", v)
    results["alpha"] = {"A1": A1, "A2": A2, "A3_raw": A3, "secondary": sec_a,
                        "scores": {"EC": {k: {"ll_mean": v["ll"].mean(), "r2_mean": v["r2"].mean(),
                                              "e_median": np.median(v["e"]), "edge": v["edge"]}
                                          for k, v in sc_ec.items()},
                                   "EO": {k: {"ll_mean": v["ll"].mean(), "r2_mean": v["r2"].mean(),
                                              "e_median": np.median(v["e"]), "edge": v["edge"]}
                                          for k, v in sc_eo.items()}},
                        "per_subject": {"subjects": a_te, "post_minus_prior": sc_ec["post"]["ll"] - sc_ec["prior"]["ll"],
                                        "post_minus_aperiodic": sc_ec["post"]["ll"] - sc_ec["aperiodic"]["ll"],
                                        "D": DA}}
    summ.put("results", results)

    rule(f"ALPHA relabelling control: {K_RELABEL} balanced EC/EO swaps within train subjects")
    perm_means = []
    for k in range(K_RELABEL):
        swap = np.zeros(len(a_tr), bool)
        swap[np.random.default_rng(1000 + k).choice(len(a_tr), len(a_tr) // 2, replace=False)] = True
        c_ec = Cohort([ev_alpha(s, "EO" if w else "EC") for s, w in zip(a_tr, swap)], a_tr)
        c_eo = Cohort([ev_alpha(s, "EC" if w else "EO") for s, w in zip(a_tr, swap)], a_tr)
        pe, _ = fit(c_ec, space, f"alpha.relabel{k}.ec")
        po, _ = fit(c_eo, space, f"alpha.relabel{k}.eo")
        dm = float(contrast_D(te_ec, te_eo, pe, po).mean())
        perm_means.append(dm)
        print(f"  draw {k:2d}: 'EC' f0 {pe[0]:6.3f} q {pe[1]:6.3f} | 'EO' f0 {po[0]:6.3f} "
              f"q {po[1]:6.3f} | held-out mean D {dm:+9.2f}", flush=True)
    rc_a = relabel_verdict(A3["mean"], perm_means)
    A3_final = dict(A3, passed=bool(A3["passed"] and rc_a["passed"]))
    print(f"\n  true mean D {A3['mean']:+.2f}; relabelled: max |mean D| {rc_a['max_abs']:.2f}, "
          f"median {rc_a['median']:+.2f}  -> control {'PASS' if rc_a['passed'] else 'FAIL'}")
    show("A3  EC/EO contrast D (with its control)", A3_final)
    results["alpha"]["relabel"] = rc_a
    results["alpha"]["A3"] = A3_final
    summ.put("results", results)

    # =========================================================================
    if not k2["passed"]:
        print("\nK2 FAILED: executed-movement mu ERD is not recovered. No mu result reported.")
        results["mu"] = {"skipped": "K2 failed"}
        summ.put("results", results)
    else:
        rule("MU: fit on train subjects (laplacian C3/C4)")
        REST = (REST_EXEC, REST_IMAG)
        tr_rest = Cohort([ev_mu(s, REST) for s in m_tr], m_tr)
        te_rest = Cohort([ev_mu(s, REST) for s in m_te], m_te)
        tr_rx = Cohort([ev_mu(s, (REST_EXEC,)) for s in m_tr], m_tr)
        tr_mx = Cohort([ev_mu(s, (EXEC,)) for s in m_tr], m_tr)
        te_rx = Cohort([ev_mu(s, (REST_EXEC,)) for s in m_te], m_te)
        te_mx = Cohort([ev_mu(s, (EXEC,)) for s in m_te], m_te)
        tr_ri = Cohort([ev_mu(s, (REST_IMAG,)) for s in m_tr], m_tr)
        tr_mi = Cohort([ev_mu(s, (IMAG,)) for s in m_tr], m_tr)
        te_ri = Cohort([ev_mu(s, (REST_IMAG,)) for s in m_te], m_te)
        te_mi = Cohort([ev_mu(s, (IMAG,)) for s in m_te], m_te)
        fits = {}
        for name, c in (("REST", tr_rest), ("REST_exec", tr_rx), ("EXEC", tr_mx),
                        ("REST_imag", tr_ri), ("IMAG", tr_mi)):
            p, rp = fit(c, space, f"mu.{name}.train")
            fits[name] = p
            print(f"  post_{name:10s} f0 {p[0]:7.3f} Hz  q {p[1]:7.3f}   converged {rp.converged}")
        results["mu_fit"] = fits
        summ.put("results", results)

        rule("MU: held out, on unseen participants")
        sc_r = score_models(te_rest, {"post": resonance_shape(te_rest.f, *fits["REST"]),
                                      "prior": resonance_shape(te_rest.f, F0P, QP),
                                      "aperiodic": aperiodic_shape(te_rest.f),
                                      "mean_shape": mean_shape(tr_rest)})
        print(f"  test REST: {len(te_rest)} subjects x {len(te_rest.idx)} bins")
        for k, v in sc_r.items():
            print(f"    {k:11s} ll {v['ll'].mean():10.2f}   r2 {v['r2'].mean():6.3f}   "
                  f"e median {np.median(v['e']):+5.2f}   edge {v['edge']:.0%}")
        M1 = gate_stats(sc_r["post"]["ll"] - sc_r["prior"]["ll"], rng_boot)
        M2 = gate_stats(sc_r["post"]["ll"] - sc_r["aperiodic"]["ll"], rng_boot)
        DM = contrast_D(te_rx, te_mx, fits["REST_exec"], fits["EXEC"])
        M3 = gate_stats(DM, rng_boot)
        DI = contrast_D(te_ri, te_mi, fits["REST_imag"], fits["IMAG"])
        print()
        show("M1  REST: post_REST - prior", M1)
        show("M2  REST: post_REST - aperiodic", M2)
        show("M3  REST/EXEC contrast D (before relabelling)", M3)
        sec_m = {"REST_prior_minus_aperiodic": gate_stats(sc_r["prior"]["ll"] - sc_r["aperiodic"]["ll"], rng_boot),
                 "REST_post_minus_meanshape": gate_stats(sc_r["post"]["ll"] - sc_r["mean_shape"]["ll"], rng_boot),
                 "imagery_contrast_D": gate_stats(DI, rng_boot)}
        for k, v in sec_m.items():
            v.pop("passed", None)
            show(f"    (not gated) {k}", v)
        results["mu"] = {"M1": M1, "M2": M2, "M3_raw": M3, "secondary": sec_m,
                         "scores": {k: {"ll_mean": v["ll"].mean(), "r2_mean": v["r2"].mean(),
                                        "e_median": np.median(v["e"]), "edge": v["edge"]}
                                    for k, v in sc_r.items()},
                         "per_subject": {"subjects": m_te,
                                         "post_minus_prior": sc_r["post"]["ll"] - sc_r["prior"]["ll"],
                                         "post_minus_aperiodic": sc_r["post"]["ll"] - sc_r["aperiodic"]["ll"],
                                         "D": DM}}
        summ.put("results", results)

        rule(f"MU relabelling control: {K_RELABEL} within-subject permutations of REST/EXEC")
        perm_means = []
        for k in range(K_RELABEL):
            g = np.random.default_rng(1000 + k)
            new_labels = {}
            for s in m_tr:                                  # sorted order, one draw each
                lab = recs[s]["mu"]["labels"].copy()
                sel = np.isin(lab, (REST_EXEC, EXEC))
                lab[sel] = g.permutation(lab[sel])
                new_labels[s] = lab
            c_r = Cohort([ev_mu(s, (REST_EXEC,), new_labels[s]) for s in m_tr], m_tr)
            c_m = Cohort([ev_mu(s, (EXEC,), new_labels[s]) for s in m_tr], m_tr)
            pr, _ = fit(c_r, space, f"mu.relabel{k}.rest")
            pm, _ = fit(c_m, space, f"mu.relabel{k}.exec")
            dm = float(contrast_D(te_rx, te_mx, pr, pm).mean())
            perm_means.append(dm)
            print(f"  draw {k:2d}: 'REST' f0 {pr[0]:6.3f} q {pr[1]:6.3f} | 'EXEC' f0 {pm[0]:6.3f} "
                  f"q {pm[1]:6.3f} | held-out mean D {dm:+9.2f}", flush=True)
        rc_m = relabel_verdict(M3["mean"], perm_means)
        M3_final = dict(M3, passed=bool(M3["passed"] and rc_m["passed"]))
        print(f"\n  true mean D {M3['mean']:+.2f}; relabelled: max |mean D| {rc_m['max_abs']:.2f}, "
              f"median {rc_m['median']:+.2f}  -> control {'PASS' if rc_m['passed'] else 'FAIL'}")
        show("M3  REST/EXEC contrast D (with its control)", M3_final)
        results["mu"]["relabel"] = rc_m
        results["mu"]["M3"] = M3_final
        summ.put("results", results)

    # =========================================================================
    rule("summary")
    allq = [results["alpha_fit"]["post_EC"][1], results["alpha_fit"]["post_EO"][1]]
    allq += [v[1] for v in results.get("mu_fit", {}).values()]
    print(f"fitted q above the declared pathological 10: {sum(q > 10 for q in allq)} of {len(allq)}")
    gates = {"K0": True, "K1": k1["passed"], "K2": k2["passed"],
             "A1": results["alpha"]["A1"]["passed"], "A2": results["alpha"]["A2"]["passed"],
             "A3": results["alpha"]["A3"]["passed"]}
    if "M1" in results.get("mu", {}):
        gates.update(M1=results["mu"]["M1"]["passed"], M2=results["mu"]["M2"]["passed"],
                     M3=results["mu"]["M3"]["passed"])
    for k, v in gates.items():
        print(f"  {k:3s} {'PASS' if v else 'FAIL'}")
    summ.put("gates", gates)
    summ.put("q_above_10", {"q": allq, "count": sum(q > 10 for q in allq)})
    summ.put("runtime_s", time.time() - t0)
    print(f"\nwritten to {args.out}\n{time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

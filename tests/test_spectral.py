"""known-answer tests for ibm/spectral.py.

run as a script (pytest is not installed in this repo's venv):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python tests/test_spectral.py

every check below is a case whose answer is known before the instrument is run --
Parseval on a Gaussian, A**2/2 for a sine, an exponent of 2 for a synthesised 1/f^2,
a coherence floor of 1/M for two unrelated signals.  CLAUDE.md's ledger says the
recurring error is a quantity computed correctly and compared against the wrong thing,
and a spectral instrument has four independent ways to be wrong by a constant (the
window power, the one-sided doubling, the `fs` in the density, the df in the integral)
that a plot cannot show.  so each check prints its MEASURED number beside the EXPECTED
one, and a check that passes still tells you by how much.

the last group is not a known-answer check at all: it calls every public function twice
on identical input and demands bit-identical output.  a known answer tests the value;
idempotence tests whether the thing is a function.

all signals are drawn from explicit `torch.Generator`s.  nothing here, and nothing in
`ibm/spectral.py`, touches the global RNG.
"""
from __future__ import annotations

import math
import sys
import traceback

import torch

from ibm.spectral import (
    aperiodic_fit,
    band_coherence,
    band_power,
    bandpass_analytic,
    pac_mi,
    peak_frequency,
    peak_prominence,
    phase_locking_value,
    relative_band_power,
    spectral_loss,
    welch_psd,
)

torch.manual_seed(0)  # belt and braces: nothing below should depend on it

_RESULTS: list = []


def check(name: str, ok: bool, measured, expected: str) -> bool:
    """print one line and record it.  never raises, so every number in a test prints
    even when an earlier one has already failed -- a run that stops at the first
    failure hides the pattern in the rest."""
    _RESULTS.append((name, bool(ok)))
    if isinstance(measured, torch.Tensor):
        measured = measured.detach().reshape(-1).tolist()
        measured = measured[0] if len(measured) == 1 else measured
    def _fmt(v):
        return f"{v:.6g}" if isinstance(v, float) else str(v)

    if isinstance(measured, (list, tuple)):
        m = "[" + ", ".join(_fmt(v) for v in measured) + "]"
    else:
        m = _fmt(measured)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: measured {m}, expected {expected}")
    return bool(ok)


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _t(n: int, fs: float, dtype=torch.float64) -> torch.Tensor:
    return torch.arange(n, dtype=dtype) / float(fs)


def _pink(n: int, fs: float, exponent: float, generator: torch.Generator, dtype=torch.float64) -> torch.Tensor:
    """a series whose psd falls as f**-exponent, shaped in the frequency domain.

    white noise -> rFFT -> divide the AMPLITUDE by f**(exponent/2) -> irFFT.  DC is set
    to zero rather than scaled, because f**-a at f = 0 is infinite and a mean offset is
    not part of what "1/f^a" means.
    """
    w = torch.randn(n, generator=generator, dtype=dtype)
    W = torch.fft.rfft(w)
    f = torch.fft.rfftfreq(n, d=1.0 / fs, dtype=dtype)
    h = torch.zeros_like(f)
    h[1:] = f[1:] ** (-exponent / 2.0)
    x = torch.fft.irfft(W * h, n=n)
    return x / x.std()


# --------------------------------------------------------------------------------
# 1. Parseval
# --------------------------------------------------------------------------------

def test_01_parseval():
    """sum(psd) * df must be the variance of x.  this is the check that pins the window
    power, the one-sided doubling and the 1/fs in the density all at once: get any one
    of them wrong and the ratio below is off by 2, by 8/3, or by fs."""
    fs, T = 256.0, 32768
    x = torch.randn(T, generator=_gen(11), dtype=torch.float64)
    freqs, psd = welch_psd(x, fs, nperseg=1024)
    df = float(freqs[1] - freqs[0])
    got = float(psd.sum()) * df
    want = float(x.var())
    rel = abs(got - want) / want
    check("parseval_gaussian", rel < 0.02, got, f"{want:.6g} (var of x), rel err {rel:.4f} < 0.02")

    # the same identity on a two-sided spectrum, where there is no doubling to get wrong
    freqs2, psd2 = welch_psd(x, fs, nperseg=1024, onesided=False)
    got2 = float(psd2.sum()) * float(fs / 1024)
    rel2 = abs(got2 - want) / want
    check("parseval_twosided", rel2 < 0.02, got2, f"{want:.6g}, rel err {rel2:.4f} < 0.02")


# --------------------------------------------------------------------------------
# 2. a sine
# --------------------------------------------------------------------------------

def test_02_single_sine():
    """a sine of amplitude A carries A**2/2 of power, all of it at its own frequency.

    fs = 250 and nperseg = 500 puts 10 Hz exactly on bin 20, so the hann taper spreads
    it over bins 19-21 and nothing else -- the whole of it is inside 8-13 Hz and the
    expected integrated power is exact, not approximate."""
    fs, T, A, f0 = 250.0, 10000, 1.7, 10.0
    x = A * torch.sin(2 * math.pi * f0 * _t(T, fs))
    freqs, psd = welch_psd(x, fs, nperseg=500)

    p = float(band_power(psd, freqs, 8.0, 13.0))
    want = A * A / 2
    check("sine_band_power", abs(p - want) / want < 0.02, p, f"A^2/2 = {want:.6g} (2%)")

    pf = float(peak_frequency(psd, freqs, 8.0, 13.0))
    check("sine_peak_frequency", abs(pf - f0) < 0.1, pf, "10.0 +/- 0.1 Hz")

    rel = float(relative_band_power(psd, freqs, 8.0, 13.0))
    check("sine_relative_alpha", rel > 0.95, rel, "> 0.95")

    # the documented failure mode, checked rather than only written down: a flat
    # spectrum makes `peak_frequency` return the band CENTRE, so a peak frequency read
    # off a band with no peak in it is the band you chose and not a measurement.
    flat = torch.ones_like(psd)
    pf_flat = float(peak_frequency(flat, freqs, 8.0, 13.0))
    check("peak_frequency_flat_is_band_centre", abs(pf_flat - 10.5) < 1e-9, pf_flat,
          "10.5 Hz = (8+13)/2, the documented bias")
    # and a band with no power at all lands on the centre too, rather than near 0 Hz
    pf_empty = float(peak_frequency(torch.zeros_like(psd), freqs, 8.0, 13.0))
    check("peak_frequency_empty_is_band_centre", abs(pf_empty - 10.5) < 1e-9, pf_empty, "10.5 Hz")


# --------------------------------------------------------------------------------
# 3. two sines
# --------------------------------------------------------------------------------

def test_03_two_sines():
    """6 Hz at amplitude 1.0 and 20 Hz at amplitude 0.4: the band powers must come back
    in the ratio (1.0/0.4)**2 = 6.25, and each must equal its own A**2/2."""
    fs, T = 250.0, 10000
    a6, a20 = 1.0, 0.4
    t = _t(T, fs)
    x = a6 * torch.sin(2 * math.pi * 6.0 * t) + a20 * torch.cos(2 * math.pi * 20.0 * t)
    freqs, psd = welch_psd(x, fs, nperseg=500)

    p6 = float(band_power(psd, freqs, 5.0, 7.0))
    p20 = float(band_power(psd, freqs, 19.0, 21.0))
    check("two_sines_theta_power", abs(p6 - a6 ** 2 / 2) / (a6 ** 2 / 2) < 0.02, p6, f"{a6**2/2:.6g} (2%)")
    check("two_sines_beta_power", abs(p20 - a20 ** 2 / 2) / (a20 ** 2 / 2) < 0.02, p20, f"{a20**2/2:.6g} (2%)")
    ratio = p6 / p20
    want = (a6 / a20) ** 2
    check("two_sines_ratio", abs(ratio - want) / want < 0.02, ratio, f"{want:.6g} (2%)")


# --------------------------------------------------------------------------------
# 4. scipy cross-check
# --------------------------------------------------------------------------------

def test_04_scipy_cross_check():
    """the same segmenting, taper, detrend and scaling as scipy.signal.welch.

    this is the only check here that compares against another implementation rather
    than against a known answer, and it is worth exactly what scipy's own conventions
    are worth -- which is why tests 1-3 exist as well.  it catches the conventions the
    known answers cannot see: the PERIODIC vs symmetric hann, and mean vs median
    averaging over segments."""
    try:
        import numpy as np
        from scipy import signal as sp_signal
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"  SKIP  scipy_cross_check: scipy/numpy not importable ({exc})")
        return

    fs, T = 200.0, 4096
    x = torch.randn(T, generator=_gen(5), dtype=torch.float64)
    freqs, psd = welch_psd(x, fs, nperseg=256, noverlap=128, window="hann", detrend="constant")
    sf, sp = sp_signal.welch(
        x.numpy(), fs=fs, window="hann", nperseg=256, noverlap=128,
        detrend="constant", return_onesided=True, scaling="density", average="mean",
    )
    df_err = float(np.max(np.abs(freqs.numpy() - sf)))
    rel = float(np.max(np.abs(psd.numpy() - sp) / np.maximum(np.abs(sp), 1e-300)))
    check("scipy_freqs", df_err < 1e-9, df_err, "max |df| < 1e-9 Hz")
    check("scipy_psd_rel", rel < 1e-5, rel, "max rel err < 1e-5")

    # and once more with no overlap and a hamming taper, so the window power and the
    # segment count are both exercised on a different setting
    _, psd2 = welch_psd(x, fs, nperseg=512, noverlap=0, window="hamming", detrend="constant")
    _, sp2 = sp_signal.welch(
        x.numpy(), fs=fs, window="hamming", nperseg=512, noverlap=0,
        detrend="constant", return_onesided=True, scaling="density", average="mean",
    )
    rel2 = float(np.max(np.abs(psd2.numpy() - sp2) / np.maximum(np.abs(sp2), 1e-300)))
    check("scipy_psd_rel_hamming_nooverlap", rel2 < 1e-5, rel2, "max rel err < 1e-5")


# --------------------------------------------------------------------------------
# 5. the aperiodic background
# --------------------------------------------------------------------------------

def test_05_aperiodic():
    """a synthesised 1/f^2 process must read exponent 2, and white noise must read 0.

    white noise is the check that matters most: an instrument that reported a spurious
    positive exponent on white noise would report "the model has a 1/f background"
    about a model that has nothing at all."""
    fs, T = 256.0, 256 * 120
    x2 = _pink(T, fs, 2.0, _gen(3))
    freqs, psd = welch_psd(x2, fs, nperseg=2048)
    off2, exp2 = aperiodic_fit(psd, freqs, lo=2.0, hi=45.0)
    check("aperiodic_f2_exponent", abs(float(exp2) - 2.0) < 0.2, float(exp2), "2.0 +/- 0.2")

    xw = torch.randn(T, generator=_gen(4), dtype=torch.float64)
    _, psdw = welch_psd(xw, fs, nperseg=2048)
    offw, expw = aperiodic_fit(psdw, freqs, lo=2.0, hi=45.0)
    check("aperiodic_white_exponent", abs(float(expw)) < 0.15, float(expw), "0.0 +/- 0.15")

    # the offset of white noise is log10 of its (flat) density: var * 2 / fs
    want_off = math.log10(float(xw.var()) * 2.0 / fs)
    check("aperiodic_white_offset", abs(float(offw) - want_off) < 0.05, float(offw),
          f"log10(2*var/fs) = {want_off:.6g} (+/- 0.05)")

    # an exponent read on a batch must equal the exponent read on each row alone
    batch = torch.stack([x2, xw])
    _, pb = welch_psd(batch, fs, nperseg=2048)
    _, eb = aperiodic_fit(pb, freqs, lo=2.0, hi=45.0)
    same = abs(float(eb[0]) - float(exp2)) < 1e-9 and abs(float(eb[1]) - float(expw)) < 1e-9
    check("aperiodic_batched_matches_single", same, eb, f"[{float(exp2):.6g}, {float(expw):.6g}]")


# --------------------------------------------------------------------------------
# 6. prominence
# --------------------------------------------------------------------------------

def test_06_prominence():
    """1/f noise plus a 10 Hz sine has a positive alpha prominence; the SAME 1/f noise
    without the sine has one near zero.

    the pairing is the point.  prominence is `log10(measured / background)` and the
    background is fitted from the same spectrum, so the only difference between the two
    numbers below is the sine -- which is the thing the number claims to be about."""
    fs, T = 256.0, 256 * 120
    noise = _pink(T, fs, 1.0, _gen(7))
    t = _t(T, fs)
    A = 0.35
    x = noise + A * torch.sin(2 * math.pi * 10.0 * t)

    freqs, psd_n = welch_psd(noise, fs, nperseg=1024)
    _, psd_x = welch_psd(x, fs, nperseg=1024)
    prom_n = float(peak_prominence(psd_n, freqs, 8.0, 13.0))
    prom_x = float(peak_prominence(psd_x, freqs, 8.0, 13.0))
    check("prominence_noise_only", abs(prom_n) < 0.1, prom_n, "|.| < 0.1 decades (no peak)")
    check("prominence_with_alpha", prom_x > 0.3, prom_x, "> 0.3 decades (a peak)")
    check("prominence_separates", prom_x - prom_n > 0.3, prom_x - prom_n, "> 0.3 decades of separation")

    # the control that says prominence is not just "more low-frequency power": scale the
    # whole 1/f trace up by 3x and the prominence must not move, because the background
    # moved with it.
    _, psd_big = welch_psd(3.0 * noise, fs, nperseg=1024)
    prom_big = float(peak_prominence(psd_big, freqs, 8.0, 13.0))
    check("prominence_scale_invariant", abs(prom_big - prom_n) < 1e-6, prom_big,
          f"{prom_n:.6g} (unchanged by a 3x gain)")


# --------------------------------------------------------------------------------
# 7. phase-amplitude coupling
# --------------------------------------------------------------------------------

def test_07_pac():
    """a 40 Hz carrier whose amplitude follows the phase of a 6 Hz wave gives a high MI;
    the identical signal with the modulation removed gives an MI near zero."""
    fs, T = 500.0, 500 * 30
    t = _t(T, fs)
    theta = torch.sin(2 * math.pi * 6.0 * t)
    carrier = torch.sin(2 * math.pi * 40.0 * t)
    env = 0.5 * (1.0 + 0.9 * torch.cos(2 * math.pi * 6.0 * t))   # 0.05 .. 0.95
    n = 0.05 * torch.randn(T, generator=_gen(21), dtype=torch.float64)

    coupled = theta + 0.6 * env * carrier + n
    flat = theta + 0.6 * float(env.mean()) * carrier + n        # same power, no modulation

    mi_c = float(pac_mi(coupled, fs, (4.0, 8.0), (30.0, 50.0)))
    mi_f = float(pac_mi(flat, fs, (4.0, 8.0), (30.0, 50.0)))
    check("pac_coupled", mi_c > 0.05, mi_c, "> 0.05 (theta-gamma nesting present)")
    check("pac_uncoupled", mi_f < 0.005, mi_f, "< 0.005 (no nesting)")
    check("pac_separates", mi_c > 10 * max(mi_f, 1e-6), (mi_c, mi_f), "coupled >> uncoupled")

    # a uniform amplitude-by-phase distribution must read EXACTLY 0, not 'small'.  the
    # soft binning is a circular smoother and smoothing a uniform leaves it uniform, so
    # the zero point of the index is not moved by the differentiability.
    # `abs`, not `<`: rounding in (log n - H)/log n puts an exactly-uniform distribution
    # a few times 1e-16 BELOW zero, and a one-sided check would pass on any negative
    # number however large -- a control that can pass for a reason unrelated to what it
    # tests is worse than one that cannot fail.
    mi_pure = float(pac_mi(theta + 0.6 * carrier, fs, (4.0, 8.0), (30.0, 50.0)))
    check("pac_zero_point", abs(mi_pure) < 1e-6, mi_pure, "|.| < 1e-6 (unmodulated carrier)")


# --------------------------------------------------------------------------------
# 8. coherence and PLV
# --------------------------------------------------------------------------------

def test_08_coherence():
    """a delayed, scaled copy is coherent; two independent noises sit on the 1/M floor.

    `noverlap=0` on purpose: overlapping segments are not independent and the floor is
    then a number nobody can state.  with M = 20 independent segments the expected
    coherence of two unrelated signals is 1/20 = 0.05, and reporting that beside the
    measured value is the whole difference between 'low coherence' and 'no coherence'."""
    fs, T = 250.0, 10000
    g = _gen(31)
    x = torch.randn(T, generator=g, dtype=torch.float64)
    y = 0.7 * torch.roll(x, 3) + 0.05 * torch.randn(T, generator=g, dtype=torch.float64)
    c_hi = float(band_coherence(x, y, fs, 8.0, 13.0, nperseg=500, noverlap=0))
    check("coherence_delayed_copy", c_hi > 0.95, c_hi, "> 0.95")

    a = torch.randn(T, generator=_gen(32), dtype=torch.float64)
    b = torch.randn(T, generator=_gen(33), dtype=torch.float64)
    M = T // 500
    # averaged over 1-45 Hz (176 bins) rather than over the 11 bins of the alpha band:
    # per bin the estimate has sd ~ 1/M, so the narrow band would sit ~1 sd from the
    # floor and the check would be testing luck.  the wide band pins 1/M to ~0.4%.
    c_lo = float(band_coherence(a, b, fs, 1.0, 45.0, nperseg=500, noverlap=0))
    check("coherence_independent", abs(c_lo - 1.0 / M) < 0.015, c_lo, f"1/M = {1.0/M:.4g} (M = {M} segments)")
    c_lo_alpha = float(band_coherence(a, b, fs, 8.0, 13.0, nperseg=500, noverlap=0))
    check("coherence_independent_alpha_band", abs(c_lo_alpha - 1.0 / M) < 0.04, c_lo_alpha,
          f"1/M = {1.0/M:.4g}, 11 bins so +/- 0.04")

    plv_hi = float(phase_locking_value(x, y, fs, 8.0, 13.0))
    plv_lo = float(phase_locking_value(a, b, fs, 8.0, 13.0))
    check("plv_delayed_copy", plv_hi > 0.95, plv_hi, "> 0.95")
    check("plv_independent", plv_lo < 0.2, plv_lo, "< 0.2")

    # a signal is perfectly coherent and perfectly phase-locked with itself: 1.0 exactly
    c_self = float(band_coherence(x, x, fs, 8.0, 13.0, nperseg=500, noverlap=0))
    p_self = float(phase_locking_value(x, x, fs, 8.0, 13.0))
    check("coherence_self", abs(c_self - 1.0) < 1e-9, c_self, "1.0")
    check("plv_self", abs(p_self - 1.0) < 1e-9, p_self, "1.0")


# --------------------------------------------------------------------------------
# 9. gradients
# --------------------------------------------------------------------------------

def _loss_signal(B: int, C: int, T: int, fs: float, generator: torch.Generator) -> torch.Tensor:
    t = _t(T, fs)
    base = torch.stack([torch.sin(2 * math.pi * (8.0 + 2.0 * i) * t) for i in range(C)])
    base = base.unsqueeze(0).repeat(B, 1, 1)
    return base + 0.5 * torch.randn(B, C, T, generator=generator, dtype=torch.float64)


def test_09_gradients():
    """the loss must pass a finite, non-zero gradient back to the trace, and the psd
    path must survive gradcheck in float64."""
    fs, T = 128.0, 128 * 8
    x = _loss_signal(2, 2, T, fs, _gen(41)).requires_grad_(True)
    targets = [
        {"kind": "relative_power", "channels": [0, 1], "band": (8.0, 13.0), "target": 0.6, "name": "alpha"},
        {"kind": "peak_frequency", "channels": [0], "band": (6.0, 14.0), "target": 10.0, "scale": 1.0, "name": "peak"},
        {"kind": "aperiodic_exponent", "channels": [0, 1], "band": (2.0, 45.0), "target": 1.0, "name": "slope"},
        {"kind": "peak_prominence", "channels": [0], "band": (8.0, 13.0), "target": 0.5, "name": "prom"},
        {"kind": "pac", "channels": [0, 1], "phase_band": (4.0, 8.0), "amp_band": (30.0, 55.0),
         "target": 0.02, "name": "pac"},
        {"kind": "coherence", "channels": [[0, 1]], "band": (8.0, 13.0), "target": 0.5, "name": "coh"},
    ]
    total, parts = spectral_loss(x, fs, targets)
    total.backward()
    g = x.grad
    check("grad_finite", bool(torch.isfinite(g).all()), float(g.abs().max()), "all finite")
    check("grad_nonzero", float(g.abs().max()) > 0, float(g.abs().max()), "> 0")
    check("grad_all_terms_present", set(parts) == {"alpha", "peak", "slope", "prom", "pac", "coh"},
          sorted(parts), "every named term reported")
    for k, v in parts.items():
        print(f"        {k:6s} value {float(v['value']):+.6g}   penalty {float(v['penalty']):.6g}")

    # gradcheck on the welch -> band_power path, float64, a deliberately small case
    xs = torch.randn(2, 128, generator=_gen(42), dtype=torch.float64, requires_grad=True)

    def f(z):
        fr, ps = welch_psd(z, 64.0, nperseg=32, noverlap=16)
        return band_power(ps, fr, 5.0, 15.0)

    ok = torch.autograd.gradcheck(f, (xs,), eps=1e-6, atol=1e-8, rtol=1e-4)
    check("gradcheck_band_power", bool(ok), ok, "True")


# --------------------------------------------------------------------------------
# 10. the loss knows its own answer
# --------------------------------------------------------------------------------

def _known_spectrum(peak_hz: float, fs: float, T: int) -> torch.Tensor:
    """a trace whose relative power in 8-13 Hz is EXACTLY 0.6 when `peak_hz` is 10.

    every component sits on an exact bin of a 2 s segment (a multiple of 0.5 Hz) and no
    two are closer than 1.5 Hz = 3 bins.  that second condition is what makes the answer
    exact rather than approximate: the windowed mean square of a sum of sinusoids carries
    a cross term at each frequency DIFFERENCE, and a hann-squared taper has spectral
    content only at bins 0, +/-1 and +/-2, so a spacing of 3 bins makes every cross term
    identically zero and the total power is the plain sum of A**2/2.

        peak power   = 1.0**2 / 2               = 0.5
        other power  = 8 * b**2 / 2 = 4 b**2    = 0.333333   (b = sqrt(1/12))
        relative     = 0.5 / 0.833333           = 0.6
    """
    others = [1.5, 3.0, 5.0, 25.0, 27.0, 30.0, 35.0, 40.0]
    b = math.sqrt(1.0 / 12.0)
    t = _t(T, fs)
    x = torch.cos(2 * math.pi * peak_hz * t)
    for k, f0 in enumerate(others):
        x = x + b * torch.cos(2 * math.pi * f0 * t + 0.37 * k)
    return x


def test_10_loss_knows_its_answer():
    fs, T, nperseg = 256.0, 256 * 20, 512
    good = _known_spectrum(10.0, fs, T).reshape(1, 1, T)
    moved = _known_spectrum(20.0, fs, T).reshape(1, 1, T)

    freqs, psd = welch_psd(good[0], fs, nperseg=nperseg)
    rel = float(relative_band_power(psd, freqs, 8.0, 13.0))
    check("known_relative_alpha_is_06", abs(rel - 0.6) < 0.002, rel, "0.6 (analytic, +/- 0.002)")

    targets = [
        {"kind": "relative_power", "channels": [0], "band": (8.0, 13.0), "target": 0.6,
         "nperseg": nperseg, "name": "alpha"},
        {"kind": "peak_frequency", "channels": [0], "band": (8.0, 13.0), "target": 10.0,
         "nperseg": nperseg, "name": "peak"},
    ]
    t_good, p_good = spectral_loss(good, fs, targets)
    t_bad, p_bad = spectral_loss(moved, fs, targets)
    print(f"        satisfied: alpha {float(p_good['alpha']['value']):.6g}, "
          f"peak {float(p_good['peak']['value']):.6g} Hz")
    print(f"        moved:     alpha {float(p_bad['alpha']['value']):.6g}, "
          f"peak {float(p_bad['peak']['value']):.6g} Hz")
    check("loss_zero_when_satisfied", float(t_good) < 1e-4, float(t_good), "< 1e-4")
    check("loss_rises_when_peak_moves", float(t_bad) > 100 * max(float(t_good), 1e-12),
          float(t_bad), f"much larger than {float(t_good):.6g}")

    # `parts` reports the SPECTRUM, not only the cost: the measured alpha must be the
    # same number `relative_band_power` gives on its own.
    check("parts_report_the_measurement", abs(float(p_good["alpha"]["value"]) - rel) < 1e-12,
          float(p_good["alpha"]["value"]), f"{rel:.6g} (identical to the direct call)")

    # an interval target costs EXACTLY zero inside the interval -- 0.0, not 1e-9.
    interval = [{"kind": "relative_power", "channels": [0], "band": (8.0, 13.0),
                 "target": (0.5, 0.7), "nperseg": nperseg, "name": "alpha_band"}]
    t_int, p_int = spectral_loss(good, fs, interval)
    check("interval_target_costs_exactly_zero",
          torch.equal(t_int.detach(), torch.zeros_like(t_int)), float(t_int), "exactly 0.0")
    # and it must still cost something outside
    t_out, _ = spectral_loss(moved, fs, interval)
    check("interval_target_costs_outside", float(t_out) > 0.0, float(t_out), "> 0")

    # the interval must be flat inside, not merely zero at one point: two different
    # signals whose alpha is inside [0.5, 0.7] both cost exactly the same (nothing).
    mild = (_known_spectrum(10.0, fs, T) + 0.05 * torch.cos(2 * math.pi * 17.0 * _t(T, fs))).reshape(1, 1, T)
    t_mild, p_mild = spectral_loss(mild, fs, interval)
    inside = 0.5 <= float(p_mild["alpha_band"]["value"]) <= 0.7
    check("interval_flat_inside", inside and torch.equal(t_mild.detach(), torch.zeros_like(t_mild)),
          float(p_mild["alpha_band"]["value"]), "in [0.5, 0.7] and costing 0.0")


# --------------------------------------------------------------------------------
# 11. idempotence
# --------------------------------------------------------------------------------

def test_11_idempotence():
    """call it twice at the same input.

    CLAUDE.md: this is not a known-answer check.  a known answer tests the value;
    this tests whether the thing is a FUNCTION at all.  the failure mode it catches --
    an instrument that quietly redraws, or caches on something mutable -- produces
    confident wrong conclusions from reasoning that is itself sound, and the symptom
    looks like a clean result."""
    fs, T = 256.0, 256 * 10
    g = _gen(51)
    x = _pink(T, fs, 1.0, g) + 0.4 * torch.sin(2 * math.pi * 10.0 * _t(T, fs))
    y = 0.6 * torch.roll(x, 5) + 0.2 * torch.randn(T, generator=g, dtype=torch.float64)
    traces = torch.stack([x, y]).reshape(1, 2, T)

    f1, p1 = welch_psd(x, fs, nperseg=512)
    f2, p2 = welch_psd(x, fs, nperseg=512)
    check("idem_welch_psd", torch.equal(f1, f2) and torch.equal(p1, p2), "bitwise", "bit-identical")

    pairs = [
        ("band_power", lambda: band_power(p1, f1, 8.0, 13.0)),
        ("relative_band_power", lambda: relative_band_power(p1, f1, 8.0, 13.0)),
        ("aperiodic_offset", lambda: aperiodic_fit(p1, f1, 1.0, 45.0)[0]),
        ("aperiodic_exponent", lambda: aperiodic_fit(p1, f1, 1.0, 45.0)[1]),
        ("peak_frequency", lambda: peak_frequency(p1, f1, 8.0, 13.0)),
        ("peak_prominence", lambda: peak_prominence(p1, f1, 8.0, 13.0)),
        ("bandpass_analytic", lambda: bandpass_analytic(x, fs, 8.0, 13.0)),
        ("pac_mi", lambda: pac_mi(x, fs, (4.0, 8.0), (30.0, 60.0))),
        ("band_coherence", lambda: band_coherence(x, y, fs, 8.0, 13.0, nperseg=512)),
        ("phase_locking_value", lambda: phase_locking_value(x, y, fs, 8.0, 13.0)),
    ]
    for name, fn in pairs:
        a, b = fn(), fn()
        check(f"idem_{name}", torch.equal(a, b), "bitwise", "bit-identical")

    targets = [
        {"kind": "relative_power", "channels": [0, 1], "band": (8.0, 13.0), "target": 0.5, "name": "a"},
        {"kind": "peak_prominence", "channels": [0], "band": (8.0, 13.0), "target": 0.4, "name": "p"},
        {"kind": "pac", "channels": [0], "phase_band": (4.0, 8.0), "amp_band": (30.0, 60.0),
         "target": 0.01, "name": "m"},
        {"kind": "coherence", "channels": [[0, 1]], "band": (8.0, 13.0), "target": 0.5, "name": "c"},
        {"kind": "aperiodic_exponent", "channels": [0, 1], "band": (2.0, 45.0), "target": 1.0, "name": "e"},
    ]
    ta, pa = spectral_loss(traces, fs, targets)
    tb, pb = spectral_loss(traces, fs, targets)
    same = torch.equal(ta, tb) and all(
        torch.equal(pa[k][f], pb[k][f]) for k in pa for f in ("value", "penalty", "weighted")
    )
    check("idem_spectral_loss", same, float(ta), f"bit-identical (total {float(ta):.6g})")

    # and float32 must agree with float64 to float32's own precision -- an instrument
    # that only works in one dtype is one silent cast away from being wrong
    x32 = x.to(torch.float32)
    f32, p32 = welch_psd(x32, fs, nperseg=512)
    rel = float((p32.double() - p1).abs().max() / p1.max())
    check("float32_matches_float64", rel < 1e-5, rel, "max rel err < 1e-5")
    check("float32_dtype_preserved", p32.dtype == torch.float32, str(p32.dtype), "torch.float32")


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    crashed = 0
    for fn in fns:
        print(fn.__name__)
        try:
            fn()
        except Exception:
            crashed += 1
            print(f"  FAIL  {fn.__name__} raised")
            traceback.print_exc()
    bad = [n for n, ok in _RESULTS if not ok]
    print(f"\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} checks passed, {crashed} test(s) raised")
    for n in bad:
        print(f"  failing: {n}")
    sys.exit(1 if (bad or crashed) else 0)

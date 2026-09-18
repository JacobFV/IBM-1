"""regression tests for ibm/substrate.py (the v2 cortical field).

cheap, CPU-only, and each pins a property the v2 design depends on and that a later
edit could silently remove.  see docs/LOG.md 2026-09-18 for the network-level gates.
"""
import torch

from ibm.substrate import CorticalField, Priors, HIERARCHY, build_sheet, hierarchy_of


def _columns(hs, priors=None):
    names = [min(HIERARCHY, key=lambda g: abs(HIERARCHY[g] - h)) for h in hs]
    n = len(hs)
    pr = priors or Priors()
    pr.G_L = 0.0; pr.G_F = 0.0
    return CorticalField(torch.zeros(n, 3), torch.arange(n)[:, None], torch.ones(n, 1), names, priors=pr)


def test_hierarchy_unknowns_are_reported_not_hidden():
    h, unknown = hierarchy_of(["lh.insula", "rh.pericalcarine", "lh.not_a_gyrus"])
    assert torch.allclose(h, torch.tensor([0.6, 0.0, 0.5]))
    assert unknown == ["lh.not_a_gyrus"]


@torch.no_grad()
def test_bounded_under_absurd_drive_and_dt():
    """exponential Euler: the box cannot be left, whatever the input or step size."""
    f = _columns([0.0, 0.5, 0.9])
    g = torch.Generator().manual_seed(0)
    for dt in (1e-4, 1e-2, 0.5):
        st = f.init_state(2)
        for _ in range(200):
            d = (torch.rand(2, 3, generator=g) - 0.5) * 1e4
            st = f.step(st, d, dt, noise=torch.randn(2, 3, generator=g))
            for k in ("E", "I", "x"):
                assert float(st[k].min()) >= 0.0 and float(st[k].max()) <= 1.0, (k, dt)
            assert float(st["a"].min()) >= 0.0
            assert all(bool(torch.isfinite(st[k]).all()) for k in ("E", "I", "a", "x", "eta"))


@torch.no_grad()
def test_idempotent_step():
    f = build_sheet(64, 8, seed=0)
    g = torch.Generator().manual_seed(1)
    s = f.init_state(2, random=True, generator=g)
    d = torch.rand(2, 64, generator=g); z = torch.randn(2, 64, generator=g)
    a, b = f.step(s, d, 1e-3, noise=z), f.step(s, d, 1e-3, noise=z)
    assert all(torch.equal(a[k], b[k]) for k in ("E", "I", "a", "x", "eta"))


def test_random_start_refuses_to_draw_its_own():
    f = build_sheet(32, 8, seed=0)
    try:
        f.init_state(1, random=True)
    except AssertionError:
        return
    raise AssertionError("a random start without a generator must refuse")


@torch.no_grad()
def test_declared_prior_gives_the_hierarchy_of_stability():
    """primary cortex monostable, association cortex bistable -- the design's premise."""
    bist = _columns([0.0, 0.25, 0.9]).column_bistable()
    assert bist.tolist() == [False, False, True]


@torch.no_grad()
def test_up_state_dwell_rises_along_the_hierarchy_and_is_released():
    f = _columns([0.6, 0.9])
    st = f.init_state(1); dt = 1e-3; E = []
    for t in range(int(6.0 / dt)):
        st = f.step(st, torch.full((1, 2), 1.0 if t * dt < 0.2 else 0.0), dt); E.append(st["E"][0].clone())
    E = torch.stack(E)[int(0.2 / dt):]
    dwell = [(E[:, i] > 0.5).float().argmin().item() * dt for i in range(2)]
    assert 0.0 < dwell[0] < dwell[1] < 6.0 - 0.3      # ordered, and h=0.9 is RELEASED in time


def test_residual_cannot_leave_its_band():
    f = _columns([0.9])
    with torch.no_grad():
        f.res_w_EE.fill_(1e6)
    frac = Priors().residual_frac
    assert float(f.site("w_EE")) <= float(f.prior_w_EE) * (1 + frac) + 1e-5


if __name__ == "__main__":
    # pytest is not installed in this repo's venv; the other tests run as scripts too
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for fn in fns:
        try:
            fn(); print("PASS", fn.__name__)
        except Exception:
            bad += 1; print("FAIL", fn.__name__); traceback.print_exc()
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)

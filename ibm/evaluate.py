"""evaluation over a PROFILE of materializations, against baselines that bite.

three things this exists to prevent, all of which already happened:

*a loss without a baseline is not a result.*  the paired MEG run was reported at
91-97% of variance explained on the strength of an MSE of 0.03-0.09 against an
ASSUMED variance of 1.0.  the array's true variance is 0.0065, because MEG
artifacts dominate a per-run standard deviation and 99.9% of the standardized
values land under 0.1 -- so that MSE was WORSE THAN PREDICTING ZERO.  every metric
here is reported as skill against an explicit baseline, and the baselines include
the trivial ones precisely because a trivial baseline is what caught this.

*a held-out batch is not a held-out split.*  sampling random indices from an
11-minute movie the model has trained on end to end measures memorization.  splits
here are contiguous in time and disjoint, so a test frame's neighbours were never
seen.

*one materialization is not the model.*  ibm-1's claim is that many explicit
models are projections of one implicit one, so evaluating a single projection
answers almost nothing.  a `Profile` is a set of materializations evaluated
together against one shared theta, which is the only evaluation whose result is
about the implicit model rather than about a head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Split:
    """contiguous, disjoint index ranges.  never random over a correlated series."""
    name: str
    lo: int
    hi: int

    def __len__(self) -> int:
        return self.hi - self.lo


def time_splits(n: int, *, train=0.8, val=0.1, gap: int = 0) -> dict[str, Split]:
    """train / val / test as contiguous blocks, with an optional guard band.

    `gap` discards samples at each boundary.  at 250 Hz with an autocorrelated
    signal, a test sample one step after a training sample is not independent of
    it, and a guard band is the cheapest way to stop that leaking.
    """
    a = int(n * train); b = int(n * (train + val))
    return {"train": Split("train", 0, a - gap),
            "val": Split("val", a + gap, b - gap),
            "test": Split("test", b + gap, n)}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def baseline_scores(y: np.ndarray, x_prev: np.ndarray | None = None) -> dict:
    """what a model must beat before any of its numbers mean anything.

    *zero*        -- predict the array's own zero.  for standardized data this is
                     the variance, and it is the baseline that caught the MEG
                     normalization error.
    *mean*        -- predict the training mean per channel.
    *persistence* -- predict the previous frame.  for video at 25 fps this is a
                     very strong baseline and a next-frame model that does not
                     clearly beat it has learned nothing about dynamics.
    """
    out = {"zero": float((y ** 2).mean()),
           "mean": float(((y - y.mean(0, keepdims=True)) ** 2).mean())}
    if x_prev is not None:
        out["persistence"] = float(((y - x_prev) ** 2).mean())
    return out


def skill(mse: float, baselines: dict) -> dict:
    """fraction of a baseline's error removed.  negative means worse than it."""
    return {f"skill_vs_{k}": float(1.0 - mse / v) if v > 0 else float("nan")
            for k, v in baselines.items()}


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

@dataclass
class Materialization:
    """one explicit model drawn from the shared implicit one."""
    name: str
    kind: str                       # video | audio | av | paired | rollout
    target: str
    horizon: int = 1
    rollout_steps: int = 1
    note: str = ""


@dataclass
class Profile:
    """a set of materializations evaluated together against ONE theta.

    the point of the type: a result about ibm-1 is a result about the shared
    substrate, and a shared substrate is only demonstrated by holding several
    projections to account at once.  a profile that improves on one member while
    degrading another has not improved the implicit model, it has reallocated it.
    """
    name: str
    members: tuple[Materialization, ...]
    results: dict = field(default_factory=dict)

    def summarize(self) -> str:
        w = max(len(m.name) for m in self.members) + 2
        lines = [f"profile: {self.name}", ""]
        lines.append(f"  {'materialization':{w}} {'mse':>10} {'vs zero':>9} "
                     f"{'vs persist':>11}")
        for m in self.members:
            r = self.results.get(m.name)
            if not r:
                lines.append(f"  {m.name:{w}} {'-- not run --':>10}")
                continue
            lines.append(f"  {m.name:{w}} {r['mse']:10.5f} "
                         f"{r.get('skill_vs_zero', float('nan')):9.3f} "
                         f"{r.get('skill_vs_persistence', float('nan')):11.3f}")
        worst = [m.name for m in self.members
                 if (self.results.get(m.name, {}).get("skill_vs_zero", 1) or 0) <= 0]
        if worst:
            lines += ["", f"  FAILS the zero baseline: {', '.join(worst)}",
                      "  a model that cannot beat predicting zero has no result to report."]
        return "\n".join(lines)


#: the standing profile.  rollout members are the same weights read at longer
#: horizons -- a rollout predictor is not a different model, it is the same
#: materialization asked a harder question, which is why they belong in one
#: profile rather than in a separate benchmark.
STANDARD = Profile("standard", (
    Materialization("video.h8", "video", "next frame at t+8", horizon=8),
    Materialization("audio.h8", "audio", "next cochleagram at t+8", horizon=8),
    Materialization("av.h8", "av", "both, jointly", horizon=8),
    Materialization("paired.meg", "paired", "measured MEG", horizon=0),
    Materialization("video.roll8", "rollout", "8 steps fed its own output",
                    horizon=8, rollout_steps=8,
                    note="prediction is not generation: a model can be excellent "
                         "one step ahead and drift immediately when closed on "
                         "itself"),
    Materialization("video.roll32", "rollout", "32 steps fed its own output",
                    horizon=8, rollout_steps=32),
))

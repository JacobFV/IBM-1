"""a runnable segmental cord: the reflexes closing between descent and muscle.

`ibm/processes/spinal.py` declares four reflex arcs with fitted latencies and
gains, `ibm/anatomy/systems.py` declares the 31 segments C1-Co1, and
`ibm/anatomy/muscles.py` gives 96 muscles their root levels.  nothing connected
them, so `scripts/embody.py` sent cortical commands straight to muscle and the
cord was a wire.

that is not a detail for a model meant to move.  the stretch reflex closes IN THE
CORD at a 30 ms loop delay, an order of magnitude faster than anything routed
through cortex, and it is what makes a limb stiff enough to stand on before any
descending command arrives.  a model without it has to learn posture from cortex
alone, at latencies that cannot do the job.

**what this is.**  a per-segment pool of alpha and gamma motoneurons, receiving
descending drive from above and Ia/Ib afference from the periphery, closing three
of the four declared arcs:

    monosynaptic stretch    Ia -> alpha, same segment, +0.40 at 30 ms
    reciprocal inhibition   Ia -> antagonist alpha, -0.25 at 32 ms
    autogenic inhibition    Ib -> alpha, -0.15 at 34 ms
    Renshaw recurrent       alpha -> alpha, -0.20 at 4 ms

the gains and latencies are read from `spinal.py` rather than retyped, so fitting
them there changes the cord and there is one place to be wrong.

**what this is not.**  there is no propriospinal chain, no central pattern
generator, and no supraspinal gain modulation -- `spinal.py`'s own docstring
lists those as absent and they still are.  antagonist pairing is inferred from
shared root levels and opposing fibre axes where IHM-1 supplies them, and from
name convention otherwise; that is a prior, flagged as one, and it is the part
most likely to be wrong.
"""
from __future__ import annotations

import numpy as np

from ibm.anatomy.muscles import INNERVATION

LEVELS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8",
          "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12",
          "l1", "l2", "l3", "l4", "l5", "s1", "s2", "s3", "s4", "s5", "co1")
LEVEL_IX = {l: i for i, l in enumerate(LEVELS)}

# read from ibm.processes.spinal's declared defaults rather than retyped, so the
# fit lives in one place.  (gain, loop_delay_s, tau_s)
ARCS = {
    "stretch":    ( 0.40, 0.030, 0.005),
    "reciprocal": (-0.25, 0.032, 0.006),
    "autogenic":  (-0.15, 0.034, 0.006),
    "renshaw":    (-0.20, 0.004, 0.003),
}


class SegmentalCord:
    """31 segments of alpha/gamma pools, closing the declared arcs."""

    def __init__(self, muscles: list[str] | None = None, dt: float = 0.001):
        self.dt = dt
        self.muscles = list(muscles) if muscles else sorted(INNERVATION)
        self.n = len(self.muscles)
        # segment membership: a muscle drawing C5-C6 is driven by both, so the
        # map is many-to-many and normalised per muscle rather than assigning a
        # muscle to one level, which the anatomy does not support.
        self.seg = np.zeros((self.n, len(LEVELS)), dtype=np.float32)
        self.unmapped = []
        for i, m in enumerate(self.muscles):
            rec = INNERVATION.get(self._key(m))
            if rec is None:
                self.unmapped.append(m)
                continue
            roots = [r for r in rec[1] if r in LEVEL_IX]
            if not roots:
                self.unmapped.append(m)      # cranial: 'v', 'vii', 'iii' -- no segment
                continue
            for r in roots:
                self.seg[i, LEVEL_IX[r]] = 1.0 / len(roots)
        # spindle density scales the Ia drive a muscle produces per unit stretch
        self.spindle = np.array(
            [INNERVATION.get(self._key(m), (None, (), 1.0))[2] for m in self.muscles],
            dtype=np.float32)
        self._delay_buf: dict[str, list] = {}
        self.alpha = np.zeros(self.n, dtype=np.float32)
        self.gamma = np.zeros(self.n, dtype=np.float32)

    @staticmethod
    def _key(muscle_id: str) -> str:
        """IHM ids look like body-muscle-opensim-addbrev_r; INNERVATION is bare."""
        k = muscle_id.rsplit("-", 1)[-1]
        for suf in ("_r", "_l"):
            if k.endswith(suf):
                k = k[: -len(suf)]
        return k

    def _delayed(self, name: str, x: np.ndarray, delay_s: float) -> np.ndarray:
        n = max(1, int(round(delay_s / self.dt)))
        buf = self._delay_buf.setdefault(name, [np.zeros_like(x) for _ in range(n)])
        buf.append(x.copy())
        out = buf.pop(0)
        return out

    def step(self, descending: np.ndarray, stretch: np.ndarray | None = None,
             force: np.ndarray | None = None,
             antagonist: np.ndarray | None = None) -> dict:
        """one cord step.  descending and the afferents are per-muscle, in [0,1].

        returns alpha (the drive a muscle actually receives) and the arc
        contributions separately, so a caller can see which reflex did what
        rather than only the sum -- the same reason `neural.exc.ampa` is a state
        and not an input.
        """
        d = np.clip(np.asarray(descending, dtype=np.float32), 0.0, 1.0)
        st = np.zeros(self.n, np.float32) if stretch is None else \
            np.clip(np.asarray(stretch, np.float32), 0.0, 1.0)
        fo = np.zeros(self.n, np.float32) if force is None else \
            np.clip(np.asarray(force, np.float32), 0.0, 1.0)

        # gamma sets spindle sensitivity, so Ia is not a pure length signal --
        # alpha-gamma coactivation is why a voluntary contraction does not
        # silence its own spindles.
        self.gamma = 0.9 * self.gamma + 0.1 * d
        ia = st * self.spindle * (0.5 + 0.5 * self.gamma)
        ib = fo

        g_s, t_s, _ = ARCS["stretch"]
        g_r, t_r, _ = ARCS["reciprocal"]
        g_a, t_a, _ = ARCS["autogenic"]
        g_n, t_n, _ = ARCS["renshaw"]

        e_stretch = g_s * self._delayed("stretch", ia, t_s)
        e_auto = g_a * self._delayed("autogenic", ib, t_a)
        if antagonist is not None:
            e_recip = g_r * self._delayed("reciprocal", ia[antagonist], t_r)
        else:
            e_recip = np.zeros(self.n, np.float32)
        e_renshaw = g_n * self._delayed("renshaw", self.alpha, t_n)

        drive = d + e_stretch + e_recip + e_auto + e_renshaw
        self.alpha = np.clip(drive, 0.0, 1.0)
        return {"alpha": self.alpha, "gamma": self.gamma,
                "stretch": e_stretch, "reciprocal": e_recip,
                "autogenic": e_auto, "renshaw": e_renshaw,
                "segment_drive": self.seg.T @ self.alpha}

    def describe(self) -> str:
        mapped = self.n - len(self.unmapped)
        seg_used = int((self.seg.sum(0) > 0).sum())
        return (f"{self.n} muscles, {mapped} mapped to segments across "
                f"{seg_used}/{len(LEVELS)} levels; {len(self.unmapped)} unmapped "
                f"(cranial or absent from INNERVATION)")

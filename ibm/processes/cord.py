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

import re

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


# OpenSim Rajagopal component names. These are name equivalences, not inferred
# nerve-root assignments. Unlisted muscles remain explicitly unmapped.
OPENSIM_ALIASES = {
    "addlong": "adductor_longus", "addmagdist": "adductor_magnus",
    "addmagmid": "adductor_magnus", "addmagprox": "adductor_magnus",
    "edl": "extensor_digitorum_longus", "ehl": "extensor_hallucis_longus",
    "fdl": "flexor_digitorum_longus", "fhl": "flexor_hallucis_longus",
    "gaslat": "gastrocnemius_lateral", "gasmed": "gastrocnemius_medial",
    "grac": "gracilis", "perbrev": "fibularis_brevis", "perlong": "fibularis_longus",
    "recfem": "rectus_femoris", "sart": "sartorius",
    "semimem": "semimembranosus", "semiten": "semitendinosus",
    "tfl": "tensor_fasciae_latae", "tibant": "tibialis_anterior",
    "tibpost": "tibialis_posterior", "vasint": "vastus_intermedius",
    "vaslat": "vastus_lateralis", "vasmed": "vastus_medialis",
    "addbrev": "adductor_brevis", "iliacus": "iliacus",
    # the ischiocondylar head is a hamstring on the tibial division, not an
    # adductor on the obturator, and now has its own entry to say so.
    "addmagisch": "adductor_magnus_ischiocondylar",
    "perter": "fibularis_tertius", "popli": "popliteus",
    **{f"{short}{i}": full for short, full in
       (("glmax", "gluteus_maximus"), ("glmed", "gluteus_medius"),
        ("glmin", "gluteus_minimus")) for i in (1, 2, 3)},
}

#: catalog names that are anatomically one muscle under several BodyParts3D
#: labels.  each is a NAME equivalence, never an inferred innervation: the
#: interossei are numbered by ray and share one nerve, the two heads of adductor
#: hallucis share one nerve, and quadratus plantae is catalogued under its older
#: name.  a name this table does not know stays unmapped and is reported.
CATALOG_ALIASES = {
    **{f"{o}_plantar_interosseous_of_foot": "plantar_interossei"
       for o in ("first", "second", "third")},
    **{f"{o}_dorsal_interosseous_of_foot": "dorsal_interossei_foot"
       for o in ("first", "second", "third", "fourth")},
    "quadratus_plantae": "flexor_accessorius",
    "levator_palpebrae": "levator_palpebrae_superioris",
}

#: catalog entries in the muscle channel list that are not contractile at all.
#: IHM's `muscle_bindings` carries tendon sheaths, tendons and check ligaments
#: alongside real muscles; they have no motor pool, and counting them as
#: "unmapped muscles" inflates the gap with things that were never muscles.
NON_CONTRACTILE = re.compile(
    r"\b(tendon|sheath|ligament|aponeurosis|fascia|raphe|retinaculum|bursa)\b",
    re.I)


def _catalog_key(name: str) -> str:
    """Normalize explicit catalog names; never guess from opaque BP3D IDs."""
    name = re.sub(r"\b(left|right)\b", "", name.lower())
    name = " ".join(name.split())
    name = re.sub(r"^(acromial|clavicular|spinal|ascending|descending|transverse) part of ", "", name)
    name = re.sub(r"^(long|short|lateral|medial|humeral|ulnar|oblique|transverse) head of ", "", name)
    # "first plantar interosseous of left foot": the laterality is stripped above
    # and leaves a doubled space plus a dangling "of foot", which no table entry
    # can match.  Normalising it is a name equivalence, not an anatomy guess.
    name = re.sub(r"\s+of\s+(foot|hand)$", r"_of_\g<1>", name)
    key = name.removesuffix(" muscle").replace(" ", "_")
    return CATALOG_ALIASES.get(key, key)


class SegmentalCord:
    """31 segments of alpha/gamma pools, closing the declared arcs.

    Opaque IHM IDs require ``muscle_bindings`` catalog records with a muscle_id
    (or native id) and anatomical name. Mapping uses exact IBM anatomy entries
    and explicit component aliases only. Unmapped channels pass descending drive
    through, but receive no spinal arcs; they are not evidence of spinal control.
    """

    def __init__(self, muscles: list[str] | None = None, dt: float = 0.001,
                 muscle_bindings: list[dict] | None = None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.dt = float(dt)
        self.muscles = list(muscles) if muscles is not None else sorted(INNERVATION)
        if any(not isinstance(m, str) or not m or m.startswith("proprio:") for m in self.muscles):
            raise ValueError("muscles must contain bare, nonempty muscle IDs")
        if len(set(self.muscles)) != len(self.muscles):
            raise ValueError("muscle IDs must be unique")
        bindings = {}
        for binding in muscle_bindings or []:
            mid = binding.get("muscle_id", binding.get("id"))
            if not isinstance(mid, str) or not mid:
                raise ValueError("muscle binding requires a nonempty muscle_id or id")
            if mid in bindings:
                raise ValueError(f"duplicate muscle binding: {mid}")
            bindings[mid] = binding
        self.mapping_keys = []
        self.channel_names = []
        for mid in self.muscles:
            name = ""
            if mid in bindings:
                name = bindings[mid].get("name", bindings[mid].get("source_name", "")) or ""
            self.channel_names.append(name)
            key = self._key(mid)
            if mid.startswith("body-connective-") or NON_CONTRACTILE.search(name):
                # a tendon sheath is not a weak motor pool, it is not a motor
                # pool.  counting it as an unmapped muscle would inflate the gap
                # with things that were never muscles.
                key = ""
            elif key not in INNERVATION and name:
                key = self._key(name)
                if key not in INNERVATION:
                    key = _catalog_key(name)
            self.mapping_keys.append(key)
        self.n = len(self.muscles)
        # segment membership: a muscle drawing C5-C6 is driven by both, so the
        # map is many-to-many and normalised per muscle rather than assigning a
        # muscle to one level, which the anatomy does not support.
        self.seg = np.zeros((self.n, len(LEVELS)), dtype=np.float32)
        # THREE REASONS A CHANNEL GETS NO ARC, AND THEY ARE NOT THE SAME FACT.
        # lumping them into one `unmapped` count made a tendon sheath, an
        # extraocular muscle and a genuinely missing innervation entry read as
        # the same failure.  only `no_innervation_entry` is a gap to close;
        # `cranial_no_segment` is innervated by a cranial nerve and correctly has
        # no spinal segment, and `non_contractile` was never a muscle.
        self.non_contractile = []
        self.cranial_no_segment = []
        self.no_innervation_entry = []
        for i, m in enumerate(self.muscles):
            key = self.mapping_keys[i]
            rec = INNERVATION.get(key)
            if not key:
                self.non_contractile.append(m)
                continue
            if rec is None:
                self.no_innervation_entry.append(m)
                continue
            roots = [r for r in rec[1] if r in LEVEL_IX]
            if not roots:
                self.cranial_no_segment.append(m)   # 'v', 'vii', 'iii': no segment
                continue
            for r in roots:
                self.seg[i, LEVEL_IX[r]] = 1.0 / len(roots)
        # kept as the union for callers that only want "received no arc"
        self.unmapped = [m for m in self.muscles
                         if m in set(self.non_contractile)
                         | set(self.cranial_no_segment)
                         | set(self.no_innervation_entry)]
        # spindle density scales the Ia drive a muscle produces per unit stretch
        self.spindle = np.array(
            [INNERVATION.get(k, (None, (), 0.0))[2] for k in self.mapping_keys],
            dtype=np.float32)
        self.spinal_mask = (self.seg.sum(axis=1) > 0).astype(np.float32)
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
        return OPENSIM_ALIASES.get(k.lower(), k)

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
        def vector(value, name):
            if value is None:
                return np.zeros(self.n, np.float32)
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (self.n,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must have shape ({self.n},) and finite values")
            return np.clip(array, 0.0, 1.0)

        if descending is None:
            raise ValueError("descending is required")
        d = vector(descending, "descending")
        st = vector(stretch, "stretch")
        fo = vector(force, "force")
        if antagonist is not None:
            antagonist = np.asarray(antagonist)
            if (antagonist.shape != (self.n,) or antagonist.dtype.kind not in "iu"
                    or np.any(antagonist < -1) or np.any(antagonist >= self.n)):
                raise ValueError("antagonist must be one valid integer muscle index per muscle or -1 (unpaired)")

        # gamma sets spindle sensitivity, so Ia is not a pure length signal --
        # alpha-gamma coactivation is why a voluntary contraction does not
        # silence its own spindles.
        self.gamma = 0.9 * self.gamma + 0.1 * d
        ia = st * self.spindle * (0.5 + 0.5 * self.gamma) * self.spinal_mask
        ib = fo * self.spinal_mask

        g_s, t_s, _ = ARCS["stretch"]
        g_r, t_r, _ = ARCS["reciprocal"]
        g_a, t_a, _ = ARCS["autogenic"]
        g_n, t_n, _ = ARCS["renshaw"]

        e_stretch = g_s * self._delayed("stretch", ia, t_s)
        e_auto = g_a * self._delayed("autogenic", ib, t_a)
        if antagonist is not None:
            e_recip = g_r * self._delayed("reciprocal", ia[np.maximum(antagonist, 0)] * (antagonist >= 0), t_r) * self.spinal_mask
        else:
            e_recip = np.zeros(self.n, np.float32)
        e_renshaw = g_n * self._delayed("renshaw", self.alpha * self.spinal_mask, t_n)

        drive = d + e_stretch + e_recip + e_auto + e_renshaw
        self.alpha = np.clip(drive, 0.0, 1.0)
        return {"alpha": self.alpha, "gamma": self.gamma,
                "stretch": e_stretch, "reciprocal": e_recip,
                "autogenic": e_auto, "renshaw": e_renshaw,
                "segment_drive": self.seg.T @ self.alpha}

    def describe(self) -> str:
        """coverage with its denominators spelled out.

        the denominator that matters is contractile channels, not raw channels:
        a tendon sheath in the channel list is not a muscle that failed to be
        innervated.  the three residual categories are reported apart because
        only one of them is a gap anyone can close.
        """
        mapped = self.n - len(self.unmapped)
        contractile = self.n - len(self.non_contractile)
        seg_used = int((self.seg.sum(0) > 0).sum())
        return (f"{self.n} channels, {contractile} contractile; "
                f"{mapped} of {contractile} under spinal arcs across "
                f"{seg_used}/{len(LEVELS)} levels; "
                f"{len(self.cranial_no_segment)} cranial (innervated, no segment); "
                f"{len(self.no_innervation_entry)} with no INNERVATION entry; "
                f"{len(self.non_contractile)} non-contractile")

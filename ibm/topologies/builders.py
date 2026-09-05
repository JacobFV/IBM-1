"""the builder protocol: how a declared topology becomes an actual edge set.

a topology declaration says *which class of adjacency exists*.  a builder says
how to compute it for one materialization.  they are separate because the
declaration is ontology and must be printable, comparable and sealable without
touching any data, while the builder is arithmetic over whatever sites a request
happened to instantiate.

    builder(sites) -> EdgeSet(src, dst, features)

`sites` is the materialized site table -- positions, spacings and support-specific
columns -- keyed by support, because most topologies span more than one.  indices
in an EdgeSet are *global* site indices, not per-table ones, which is why every
SiteTable carries the `offset` of its first row.  a topology that couples tissue
to the vascular tree would otherwise silently produce edges between two different
things numbered the same.

three deliberate choices.

*numpy and scipy are imported inside the functions, not at module scope.*  the
ontology has to be importable and printable on a machine with no numeric stack;
nothing here computes until a materialization asks it to.  scipy is preferred
where it exists and there is a numpy fallback everywhere it does not, because a
spatial index is a convenience and not a dependency of the idea.

*a builder that needs data ibm-1 does not synthesize takes it as an argument and
says so.*  there is no way to invent a tractogram or a vascular segmentation, and
a bare NotImplementedError tells the caller nothing about what would fix it.
`MissingInput` names the argument, what it must contain, and where such a thing
comes from.

*edge features are named columns, not a weight.*  a topology is a support for
interaction, not a weighting of it (ARCHITECTURE.md §3): distance, delay, contact
area and orientation are facts about the pair, and how strongly they actually
couple is the process's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field, replace
from typing import Any, Callable, Iterable, Mapping


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class MissingInput(RuntimeError):
    """a builder was asked to run without data that cannot be invented.

    the message is the whole point of this class.  "not implemented" is false --
    the algorithm is implemented -- and it hides the only useful fact, which is
    which file the caller has to go and find.
    """

    def __init__(self, builder: str, needed: str, what: str, where: str = "") -> None:
        msg = (f"topology builder {builder!r} cannot run: it needs {needed!r}.\n"
               f"  what it must contain: {what}")
        if where:
            msg += f"\n  where such a thing comes from: {where}"
        super().__init__(msg)
        self.builder, self.needed = builder, needed


def _numpy(builder: str):
    try:
        import numpy as np
    except ImportError as exc:                                    # pragma: no cover
        raise MissingInput(builder, "numpy", "the numeric stack; declarations import "
                           "without it but builders do not run without it",
                           "pip install numpy (and scipy, for the spatial index)") from exc
    return np


def _kdtree():
    """scipy's spatial index if present.  its absence is a slowdown, not a failure."""
    try:
        from scipy.spatial import cKDTree
        return cKDTree
    except ImportError:                                           # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# the materialized site table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteTable:
    """the positions of one support in one materialization.

    `columns` is where everything support-specific lives -- cortical depth,
    vascular parent index and radius, surface triangulation, relay stage, contact
    impedance.  it is a bag rather than a schema because the supports genuinely do
    not have a common one: a tree has a parent array and a surface has faces, and
    forcing either into the other's shape is how the universal-graph mistake gets
    made.

    `offset` places this table inside the global site index.  builders that span
    two supports must add it; there is a helper (`gidx`) so they cannot forget.
    """

    support: str
    frame: str
    xyz: Any                                   # (n, 3) positions in mm, in `frame`
    offset: int = 0                            # index of row 0 in the global table
    spacing_mm: Any = 1.0                      # scalar or (n,) -- r(q) at each site
    columns: Mapping[str, Any] = _field(default_factory=dict)
    partitions: Mapping[str, Any] = _field(default_factory=dict)   # system -> (n, k)

    @property
    def n(self) -> int:
        return int(len(self.xyz))

    def __len__(self) -> int:
        return self.n

    def col(self, name: str, builder: str, what: str, where: str = "") -> Any:
        """a required support-specific column, or an error that says what to supply."""
        if name not in self.columns or self.columns[name] is None:
            raise MissingInput(builder, f"{self.support}.columns[{name!r}]", what, where)
        return self.columns[name]

    def opt(self, name: str, default: Any = None) -> Any:
        v = self.columns.get(name, default)
        return default if v is None else v

    def gidx(self, np) -> Any:
        """global indices of this table's rows."""
        return np.arange(self.n) + self.offset

    def spacing(self, np) -> Any:
        s = np.asarray(self.spacing_mm, dtype=float)
        return np.broadcast_to(s, (self.n,)) if s.ndim == 0 else s

    def with_columns(self, **cols: Any) -> "SiteTable":
        merged = dict(self.columns)
        merged.update(cols)
        return replace(self, columns=merged)


@dataclass(frozen=True)
class Sites:
    """every site table a materialization instantiated, keyed by support.

    a builder asks for the supports it needs by name.  asking for one that was not
    materialized is a request error, not an empty graph -- a topology silently
    producing no edges because a support was left out of the request is exactly
    the kind of hole the registry exists to refuse.
    """

    tables: Mapping[str, SiteTable] = _field(default_factory=dict)

    @classmethod
    def of(cls, table: SiteTable) -> "Sites":
        return cls({table.support: table})

    def __getitem__(self, support: str) -> SiteTable:
        return self.tables[support]

    def __contains__(self, support: str) -> bool:
        return support in self.tables

    def get(self, support: str) -> SiteTable | None:
        return self.tables.get(support)

    def require(self, support: str, builder: str, why: str = "") -> SiteTable:
        t = self.tables.get(support)
        if t is None:
            raise MissingInput(
                builder, f"sites[{support!r}]",
                why or f"materialized positions on the {support!r} support",
                "add that support to the materialization request; a topology defined "
                "over it cannot be built from the supports that were materialized "
                f"instead ({', '.join(sorted(self.tables)) or 'none'})")
        return t

    @property
    def n_total(self) -> int:
        return sum(t.n for t in self.tables.values())

    def frames(self) -> set[str]:
        return {t.frame for t in self.tables.values()}


def as_sites(x: Sites | SiteTable | Mapping[str, SiteTable]) -> Sites:
    if isinstance(x, Sites):
        return x
    if isinstance(x, SiteTable):
        return Sites.of(x)
    return Sites(dict(x))


# ---------------------------------------------------------------------------
# the edge set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSet:
    """the result of applying one topology to one materialized state graph.

    `src` and `dst` are global site indices.  `features` are named per-edge
    columns, each either (m,) or (m, k) -- orientation is the reason for the
    second case, and squeezing it into three scalar columns would lose the fact
    that they are one vector.

    an undirected topology stores each pair once; `symmetrized()` produces both
    directions when a process wants to scatter over them.  storing one direction
    is not an optimization, it is a statement: a symmetric adjacency has one fact
    per pair, and duplicating it invites the two copies to drift apart.
    """

    topology: str
    src: Any
    dst: Any
    n_sites: int
    features: Mapping[str, Any] = _field(default_factory=dict)
    directed: bool = False
    note: str = ""

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    def __len__(self) -> int:
        return self.n_edges

    def feature(self, name: str) -> Any:
        if name not in self.features:
            raise KeyError(f"edge set {self.topology!r} carries "
                           f"{sorted(self.features) or 'no features'}, not {name!r}")
        return self.features[name]

    def with_features(self, **cols: Any) -> "EdgeSet":
        merged = dict(self.features)
        merged.update(cols)
        return replace(self, features=merged)

    def filtered(self, mask: Any) -> "EdgeSet":
        return replace(self, src=self.src[mask], dst=self.dst[mask],
                       features={k: v[mask] for k, v in self.features.items()})

    def symmetrized(self) -> "EdgeSet":
        """both directions of an undirected set; a no-op on a directed one."""
        if self.directed:
            return self
        np = _numpy(self.topology)
        cat = lambda a, b: np.concatenate([a, b], axis=0)
        return replace(self, src=cat(self.src, self.dst), dst=cat(self.dst, self.src),
                       features={k: cat(v, v) for k, v in self.features.items()},
                       directed=True, note=(self.note + " [symmetrized]").strip())

    def describe(self) -> str:
        f = ", ".join(f"{k}{tuple(getattr(v, 'shape', ('?',)))[1:] or ''}"
                      for k, v in sorted(self.features.items()))
        return (f"{self.topology}: {self.n_edges} edges over {self.n_sites} sites, "
                f"{'directed' if self.directed else 'undirected'}"
                + (f", features [{f}]" if f else ""))

    def __repr__(self) -> str:
        return f"<EdgeSet {self.describe()}>"


def empty(topology: str, n_sites: int, features: Iterable[str] = (),
          directed: bool = False, note: str = "") -> EdgeSet:
    """a well-formed edge set with no edges.

    a materialization can legitimately instantiate a region in which a topology
    has nothing to connect; the columns must still exist so that downstream code
    reads a zero-length array rather than a KeyError.
    """
    np = _numpy(topology)
    z = np.zeros(0, dtype=np.int64)
    return EdgeSet(topology, z, z, n_sites,
                   {k: np.zeros(0, dtype=float) for k in features}, directed, note)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuilderSpec:
    """a named way of computing one topology's edges.

    `requires` is the honest part: the external data the algorithm cannot proceed
    without.  it is declared rather than discovered at runtime so a materialization
    request can be checked before anything is instantiated, and so the inventory of
    what ibm-1 is actually missing is readable off the registry instead of off a
    pile of exceptions.
    """

    name: str
    doc: str
    fn: Callable[..., EdgeSet]
    produces: tuple[str, ...] = ()          # edge feature names
    supports: tuple[str, ...] = ()          # supports it reads
    requires: tuple[str, ...] = ()          # external inputs with no default
    directed: bool = False
    metric: str = ""                        # the notion of distance it encodes

    def __call__(self, sites, **kw) -> EdgeSet:
        return self.fn(as_sites(sites), **kw)


BUILDERS: dict[str, BuilderSpec] = {}


def builder(name: str, *, produces: Iterable[str] = (), supports: Iterable[str] = (),
            requires: Iterable[str] = (), directed: bool = False, metric: str = "",
            doc: str = ""):
    """declare a builder under a name a Topology can reference.

    the name, not the function object, is what a Topology carries: declarations
    have to survive being printed, diffed and sealed without importing anything
    that computes.
    """

    def deco(fn: Callable[..., EdgeSet]) -> Callable[..., EdgeSet]:
        if name in BUILDERS:
            raise ValueError(
                f"builder {name!r} already registered by {BUILDERS[name].fn.__module__}; "
                "two builders with one name is the same failure the component registry "
                "refuses -- rename one or reference the existing one")
        spec = BuilderSpec(name, doc or (fn.__doc__ or "").strip(), fn,
                           tuple(produces), tuple(supports), tuple(requires),
                           directed, metric)
        BUILDERS[name] = spec
        fn.spec = spec                                            # type: ignore[attr-defined]
        return fn

    return deco


def get(name: str) -> BuilderSpec:
    if name not in BUILDERS:
        raise KeyError(f"no builder {name!r}; registered: {', '.join(sorted(BUILDERS))}")
    return BUILDERS[name]


def build(name: str, sites, **kw) -> EdgeSet:
    """run a named builder over a site table."""
    return get(name)(sites, **kw)


def table() -> str:
    """the builders as one printable table, in the spirit of REGISTRY.table()."""
    head = ("builder", "metric", "supports", "requires", "features")
    rows = [(b.name, b.metric or "-", ",".join(b.supports) or "-",
             ",".join(b.requires) or "-", ",".join(b.produces) or "-")
            for b in sorted(BUILDERS.values(), key=lambda x: x.name)]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(head)]
    line = "  ".join(h.ljust(x) for h, x in zip(head, w))
    rule = "  ".join("-" * x for x in w)
    body = "\n".join("  ".join(c.ljust(x) for c, x in zip(r, w)) for r in rows)
    return f"{line}\n{rule}\n{body}"


# ---------------------------------------------------------------------------
# shared geometry
# ---------------------------------------------------------------------------


def pairs_within(xyz, radius_mm: float, builder_name: str):
    """all unordered pairs closer than a radius, as (i, j, d).

    scipy's KD-tree where available; a chunked brute-force sweep otherwise, which
    is O(n^2) in time but not in memory and is perfectly adequate for the site
    counts a local topology is ever built at.
    """
    np = _numpy(builder_name)
    xyz = np.asarray(xyz, dtype=float)
    n = len(xyz)
    if n < 2:
        return (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros(0, dtype=float),)
    KD = _kdtree()
    if KD is not None:
        pairs = KD(xyz).query_pairs(radius_mm, output_type="ndarray")
        if len(pairs) == 0:
            return (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros(0, dtype=float),)
        i, j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
        return i, j, np.linalg.norm(xyz[i] - xyz[j], axis=1)
    ii, jj, dd = [], [], []
    chunk = max(1, int(4e6 // max(n, 1)))
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d = np.linalg.norm(xyz[a:b, None, :] - xyz[None, :, :], axis=-1)
        loc_i, loc_j = np.nonzero(d <= radius_mm)
        keep = (loc_i + a) < loc_j
        ii.append(loc_i[keep] + a); jj.append(loc_j[keep]); dd.append(d[loc_i[keep], loc_j[keep]])
    i = np.concatenate(ii) if ii else np.zeros(0, dtype=np.int64)
    j = np.concatenate(jj) if jj else np.zeros(0, dtype=np.int64)
    d = np.concatenate(dd) if dd else np.zeros(0, dtype=float)
    return i.astype(np.int64), j.astype(np.int64), d


def nearest(query_xyz, target_xyz, k: int, builder_name: str):
    """k nearest targets for each query, as (idx, dist) of shape (nq, k)."""
    np = _numpy(builder_name)
    q = np.asarray(query_xyz, dtype=float)
    t = np.asarray(target_xyz, dtype=float)
    k = int(min(k, len(t)))
    if k == 0 or len(q) == 0:
        return np.zeros((len(q), 0), dtype=np.int64), np.zeros((len(q), 0), dtype=float)
    KD = _kdtree()
    if KD is not None:
        d, idx = KD(t).query(q, k=k)
        return np.atleast_2d(idx.reshape(len(q), k)).astype(np.int64), \
            np.atleast_2d(np.asarray(d, dtype=float).reshape(len(q), k))
    d = np.linalg.norm(q[:, None, :] - t[None, :, :], axis=-1)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx.astype(np.int64), np.take_along_axis(d, idx, axis=1)


def unit(vec, np):
    """unit vectors, with a zero row left at zero rather than made into a nan."""
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    return np.divide(vec, n, out=np.zeros_like(vec), where=n > 0)


def conduction_delay_s(length_mm, velocity_m_s, np):
    """length over velocity, in seconds.

    delay is a feature of the edge and not of the process because it is a fact
    about the anatomy.  in the spectral form it is a phase ramp, so carrying it
    per edge costs nothing at run time -- which is the reason it is worth carrying
    exactly rather than lumping into a single mean latency per pathway.
    """
    v = np.maximum(np.asarray(velocity_m_s, dtype=float), 1e-6)
    return np.asarray(length_mm, dtype=float) * 1e-3 / v


def velocity_from_myelination(myelination, np, unmyelinated_m_s: float = 0.5,
                              myelinated_m_s: float = 20.0):
    """crude interpolation between unmyelinated and myelinated conduction speed.

    real velocity follows axon diameter (roughly 5.5 m/s per micron of outer
    diameter in myelinated fibre), and the myelin maps ibm-1 can actually obtain
    -- T1w/T2w ratio, MT saturation, g-ratio estimates -- are ordinal proxies
    rather than diameters.  a monotone interpolation is therefore the honest
    amount of structure to claim; the endpoints are literature and the middle is
    not, and a process that needs better should fit theta rather than trust this.
    """
    m = np.clip(np.asarray(myelination, dtype=float), 0.0, 1.0)
    return unmyelinated_m_s + m * (myelinated_m_s - unmyelinated_m_s)

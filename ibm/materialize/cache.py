"""content-addressed cache for the artefacts a build cannot afford to repeat.

the separation between a request and its materialization exists partly for this.
a `MaterializationRequest` is symbolic and costs microseconds; the things it
implies -- an octree over a segmented head, a geodesic graph over a 160k-vertex
surface, an atlas resampled into a subject's frame, a laplacian eigendecomposition,
a boundary-element lead field -- cost minutes to hours each and are *completely
determined* by a spec that is known before any of them start.  so the key is a
hash of that spec, and each is computed once per spec rather than once per run.

the design follows from what goes wrong with caches, not from what they are for.

*the key is derived, never passed.*  a caller who names its own cache key has
taken responsibility for noticing that the thing it names has changed, and
eventually will not.  `spec_hash` walks the spec structurally -- dataclasses,
enums, arrays, nested containers -- so that changing a resolution rule, a subject
id or a random seed changes the key automatically, and changing a docstring does
not.

*a hit is verified, not trusted.*  every entry stores its spec next to its payload
and the loader compares.  a hash collision is astronomically unlikely and a stale
directory left over from a format change is not, and both present identically:
plausible numbers, quietly wrong.

*the directory is readable.*  each entry also writes a small json sidecar naming
what it is, because a directory of 64-character hex filenames is a thing nobody
can reason about at 2am, and the request's own `spec()` docstring makes the same
argument about keeping `name` and `notes` in the hash.

nothing here is required.  `cache=None` everywhere in `build.py` means the build
recomputes, which is slower and identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass, field as _field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

#: bumped when the on-disk payload format changes in a way that makes old entries
#: unreadable or wrong.  it is part of every key, so a bump invalidates the whole
#: store without anyone having to remember to delete it.
FORMAT = 3

_HASH_BYTES = 16                 # 32 hex characters; ample for a per-user cache


# ---------------------------------------------------------------------------
# hashing a spec
# ---------------------------------------------------------------------------


def canonical(obj: Any) -> Any:
    """a spec reduced to json-shaped primitives, deterministically.

    the walk is structural rather than by `repr`, because reprs carry memory
    addresses for anything without a `__repr__` and two identical requests would
    then miss each other's cache entries.  it is also *typed*: a dataclass records
    its class name alongside its fields, so a `Ball` and a `Near` with the same
    numbers do not collide, which they would under a bare tuple of values.

    a callable is reduced to its qualified name and not its bytecode.  that is the
    one deliberate hole: changing the body of a builder without renaming it will
    not invalidate entries that used it.  hashing bytecode was the alternative and
    it invalidates the entire store on every unrelated edit, which in practice
    means the cache is never warm and the whole thing was pointless.  bump
    `FORMAT` when a builder's behaviour changes.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return "nan" if math.isnan(obj) else ("inf" if math.isinf(obj)
                                              else round(obj, 12) + 0.0)
    if isinstance(obj, Enum):
        return {"enum": type(obj).__name__, "value": obj.value}
    if isinstance(obj, bytes):
        return {"bytes": hashlib.blake2b(obj, digest_size=16).hexdigest()}
    if isinstance(obj, Path):
        return {"path": str(obj)}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {"@": type(obj).__name__,
                **{f: canonical(getattr(obj, f)) for f in sorted(_fields_of(obj))}}
    if isinstance(obj, Mapping):
        return {"{}": [[canonical(k), canonical(v)] for k, v in
                       sorted(obj.items(), key=lambda kv: repr(kv[0]))]}
    if isinstance(obj, (set, frozenset)):
        return {"set": sorted(repr(canonical(x)) for x in obj)}
    if isinstance(obj, (list, tuple)):
        return [canonical(x) for x in obj]
    arr = _as_array(obj)
    if arr is not None:
        return arr
    if callable(obj):
        return {"fn": f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', repr(obj))}"}
    return {"repr": repr(obj)}


def _fields_of(obj: Any) -> list[str]:
    from dataclasses import fields as _dc_fields
    return [f.name for f in _dc_fields(obj)]


def _as_array(obj: Any) -> Any:
    """an ndarray reduced to shape, dtype and a digest of its bytes.

    a digest rather than the values because a site table's positions are megabytes
    and the key has to be cheap; and shape and dtype alongside it because two
    arrays with the same bytes and different shapes are different arrays.
    """
    try:
        import numpy as np
    except ImportError:                                            # pragma: no cover
        return None
    if not isinstance(obj, np.ndarray):
        return None
    a = np.ascontiguousarray(obj)
    return {"ndarray": [list(a.shape), a.dtype.str],
            "digest": hashlib.blake2b(a.view(np.uint8).reshape(-1).tobytes(),
                                      digest_size=16).hexdigest()}


def spec_hash(*parts: Any) -> str:
    """the content address of a spec.  stable across processes and machines."""
    blob = json.dumps([FORMAT, *[canonical(p) for p in parts]],
                      sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.blake2b(blob.encode(), digest_size=_HASH_BYTES).hexdigest()


def readable(spec: Any, limit: int = 400) -> str:
    """a one-line human summary for the sidecar.  what the hash is *of*."""
    try:
        s = json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):                                # pragma: no cover
        s = repr(spec)
    return s if len(s) <= limit else s[:limit] + f"...(+{len(s) - limit} chars)"


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def default_root() -> Path:
    """`$IBM_CACHE_DIR`, else the XDG cache, else `~/.cache/ibm-1`.

    deliberately outside the repository.  these artefacts are derived, large, and
    subject-specific, and a cache that lands inside a working tree ends up in
    somebody's commit.
    """
    env = os.environ.get("IBM_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg).expanduser() / "ibm-1" if xdg else Path.home() / ".cache" / "ibm-1"


@dataclass
class Cache:
    """a content-addressed store on disk, with an in-process layer in front.

    the in-process layer is not an optimization afterthought.  one build asks for
    the same site table once per topology that spans it, and unpickling a
    hundred-megabyte octree four times to satisfy four builders is slower than
    recomputing it once would have been.

    `enabled=False` turns the whole thing into a pass-through, which is what the
    tests of a build step want and what a caller debugging a stale artefact wants.
    """

    root: Path = _field(default_factory=default_root)
    enabled: bool = True
    verify: bool = True
    memory: bool = True
    _mem: dict[str, Any] = _field(default_factory=dict, repr=False)
    hits: int = 0
    misses: int = 0
    writes: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    # -- paths -----------------------------------------------------------

    def key(self, kind: str, spec: Any) -> str:
        return spec_hash(kind, spec)

    def _dir(self, kind: str, key: str) -> Path:
        return self.root / kind / key[:2]

    def path(self, kind: str, key: str) -> Path:
        return self._dir(kind, key) / f"{key}.pkl"

    def sidecar(self, kind: str, key: str) -> Path:
        return self._dir(kind, key) / f"{key}.json"

    # -- get / put -------------------------------------------------------

    def get(self, kind: str, spec: Any) -> Any:
        """the cached artefact, or `None`.

        `None` is never a cached value -- a build step that legitimately produces
        nothing should cache a sentinel -- so a `None` return unambiguously means
        "not present", which keeps the call sites from needing a second lookup.
        """
        if not self.enabled:
            return None
        key = self.key(kind, spec)
        if self.memory and key in self._mem:
            self.hits += 1
            return self._mem[key]
        p = self.path(kind, key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with p.open("rb") as fh:
                blob = pickle.load(fh)
        except Exception as exc:                                   # pragma: no cover
            self.misses += 1
            self._quarantine(p, f"unreadable: {exc}")
            return None
        if self.verify and blob.get("spec") != canonical(spec):
            self.misses += 1
            self._quarantine(p, "the stored spec does not match the requested one")
            return None
        if blob.get("format") != FORMAT:
            self.misses += 1
            return None
        self.hits += 1
        value = blob["value"]
        if self.memory:
            self._mem[key] = value
        return value

    def put(self, kind: str, spec: Any, value: Any) -> str:
        """store an artefact under its content address, atomically.

        written to a temporary file in the same directory and then renamed, so a
        run interrupted mid-write leaves no half-file behind.  a cache that can
        hand back a truncated eigendecomposition is worse than no cache: the
        failure surfaces as an assertion three modules away, or not at all.
        """
        if not self.enabled:
            return ""
        key = self.key(kind, spec)
        if self.memory:
            self._mem[key] = value
        d = self._dir(kind, key)
        d.mkdir(parents=True, exist_ok=True)
        blob = {"format": FORMAT, "kind": kind, "spec": canonical(spec),
                "value": value, "written": time.time()}
        fd, tmp = tempfile.mkstemp(dir=str(d), suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self.path(kind, key))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.sidecar(kind, key).write_text(json.dumps(
            {"kind": kind, "key": key, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
             "spec": readable(spec)}, indent=1))
        self.writes += 1
        return key

    def memo(self, kind: str, spec: Any, fn: Callable[[], Any]) -> Any:
        """`fn()`, computed once per spec.  the only call this module really needs."""
        hit = self.get(kind, spec)
        if hit is not None:
            return hit
        value = fn()
        if value is not None:
            self.put(kind, spec, value)
        return value

    def __call__(self, kind: str, spec: Any, fn: Callable[[], Any]) -> Any:
        return self.memo(kind, spec, fn)

    # -- housekeeping ----------------------------------------------------

    def _quarantine(self, p: Path, why: str) -> None:
        """move a suspect entry aside rather than deleting it.

        an entry that fails verification is evidence of something -- a format
        change, an interrupted write, a genuine collision -- and deleting it
        destroys the only copy of that evidence.  the build carries on and
        recomputes either way.
        """
        try:
            bad = p.with_suffix(p.suffix + ".rejected")
            os.replace(p, bad)
            bad.with_suffix(".why").write_text(why)
        except OSError:                                            # pragma: no cover
            pass

    def clear(self, kind: str | None = None) -> int:
        """drop everything, or one kind.  returns the bytes reclaimed."""
        target = self.root / kind if kind else self.root
        n = self.size(kind)
        shutil.rmtree(target, ignore_errors=True)
        self._mem.clear()
        return n

    def size(self, kind: str | None = None) -> int:
        target = self.root / kind if kind else self.root
        if not target.exists():
            return 0
        return sum(f.stat().st_size for f in target.rglob("*") if f.is_file())

    def kinds(self) -> dict[str, int]:
        if not self.root.exists():
            return {}
        return {d.name: sum(1 for _ in d.rglob("*.pkl"))
                for d in sorted(self.root.iterdir()) if d.is_dir()}

    def entries(self, kind: str) -> list[dict[str, Any]]:
        """the sidecars of one kind: what is actually in there, in words."""
        d = self.root / kind
        if not d.exists():
            return []
        out = []
        for f in sorted(d.rglob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except (OSError, ValueError):                          # pragma: no cover
                continue
        return out

    def describe(self) -> str:
        by = self.kinds()
        lines = [f"cache at {self.root}"
                 + ("" if self.enabled else "  [DISABLED: every lookup recomputes]"),
                 f"  {self.hits} hits, {self.misses} misses, {self.writes} writes this session",
                 f"  {sum(by.values())} entries, {self.size() / 2 ** 20:.1f} MiB on disk"]
        for k, n in sorted(by.items()):
            lines.append(f"    {k:22s} {n:5d} entries  {self.size(k) / 2 ** 20:8.1f} MiB")
        return "\n".join(lines)


#: the kinds a build actually stores, named so that `cache.clear(kind)` is usable
#: and so that two call sites cannot disagree about a spelling.  each is expensive
#: for a different reason: the first three are geometry, the last two are physics.
SITES = "sites"                  # octree / poisson-disk / tree traversal
GEODESIC = "geodesic"            # mesh graphs and distance matrices
ATLAS = "atlas"                  # a partitioning system resampled into a frame
EIGEN = "eigen"                  # laplacian eigendecompositions
LEADFIELD = "leadfield"          # BEM/FEM forward solutions
EDGES = "edges"                  # built topology edge sets

KINDS = (SITES, GEODESIC, ATLAS, EIGEN, LEADFIELD, EDGES)


#: the process-wide default.  a module-level singleton because the alternative is
#: threading a cache through every call in `build.py`, and a build step that
#: quietly created its own would silently never hit.
DEFAULT = Cache()


def disabled() -> Cache:
    """a pass-through cache, for a caller who wants the timing without the store."""
    return Cache(enabled=False)


__all__ = ["Cache", "DEFAULT", "FORMAT", "KINDS", "SITES", "GEODESIC", "ATLAS", "EIGEN",
           "LEADFIELD", "EDGES", "spec_hash", "canonical", "readable", "default_root",
           "disabled"]

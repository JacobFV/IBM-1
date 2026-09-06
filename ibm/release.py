"""checkpoint identity: a name that says what the weights actually are.

a checkpoint from this project is not identified by a step number.  the substrate
it was trained on changes, the ontology it was declared against changes, the
objective changes, and a tensor of 32 million association embeddings is
meaningless without knowing how many sites they index and what the dynamics
between them were.  so the name carries all of it.

the schema, dot-separated so it sorts and greps:

    ibm1.m-{modality}.s{sites}.e{embed}.k{degree}.dyn-{rev}.obj-{objective}
        .vw{viability}.step{n}.git-{sha7}

field by field, and each one is here because getting it wrong silently produces a
checkpoint that loads and is wrong:

*m*  -- `v` video, `a` audio, `av` joint.  a joint checkpoint's cortical weights
        are shared across ports and a single-modality one's are not, so they are
        not interchangeable even at identical shapes.
*s*  -- site count, the leading dimension of the embedding table.  a shape
        mismatch is caught on load; a site count that matches while the GEOMETRY
        differs is not, which is why `geom` is in the sidecar.
*e*  -- embedding dimension, the learned rank of the association kernel.
*k*  -- association degree.  it sets the fan-in normalization, so two checkpoints
        with different k have differently scaled weights even at equal shape.
*dyn* -- dynamics revision.  bumped by hand whenever the E/I equations, the
        shunting form or the fan-in convention change.  this is the field that
        catches "the weights load and the model is wrong".
*obj* -- what was minimized: `nf1` next frame, `nfh{n}` an n-step horizon,
        `av` cross-modal.
*vw* -- negative log10 of the viability weight.  it belongs in the name because
        run v1 learned a 430 mV cortex under vw=inf and looked fine by loss.
*step*, *git* -- when, and against which declaration.

the git sha is the one that matters most in this project, because the ONTOLOGY is
versioned in the same repo as the weights: a checkpoint trained before
`ASSOCIATION_KERNEL` became PER_SITE indexes a different object by the same name.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: bump by hand when the dynamics change shape or convention.
#: r1  initial E/I + learned association
#: r2  conductance-based shunting (STATE.md 4.11), fan-in normalization (4.10)
#: r3  viability penalty on membrane potential (ONTOLOGY.md 7)
#: r4  multi-step prediction horizon; cross-modal ports
DYNAMICS_REV = "r4"

REPO_ID = "brandonin/ibm-1"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


@dataclass
class CheckpointName:
    modality: str
    sites: int
    embed: int
    degree: int
    objective: str
    viability_weight: float
    step: int
    dynamics_rev: str = DYNAMICS_REV
    sha: str = field(default_factory=git_sha)

    def __str__(self) -> str:
        s = f"{self.sites // 1000}k" if self.sites >= 1000 else str(self.sites)
        vw = ("inf" if self.viability_weight <= 0 else
              f"{round(-__import__('math').log10(self.viability_weight))}")
        return (f"ibm1.m-{self.modality}.s{s}.e{self.embed}.k{self.degree}"
                f".dyn-{self.dynamics_rev}.obj-{self.objective}.vw{vw}"
                f".step{self.step:06d}.git-{self.sha}")


def sidecar(name: CheckpointName, *, geometry: str, n_params: int,
            n_assoc: int, metrics: dict, config: dict) -> dict:
    """what the filename cannot carry but a loader still needs."""
    return {
        "name": str(name),
        "schema": "ibm1/v1",
        "identity": asdict(name),
        "geometry": geometry,
        "n_params": n_params,
        "n_association_params": n_assoc,
        "metrics": metrics,
        "config": config,
        "notes": {
            "association_factorization":
                "w_ij = M[parcel] x exp(-d/l) x sigma(<e_i,e_j>); only the third "
                "factor is trained here",
            "held_out_diagnostics":
                "effective rank is reported and deliberately NOT in the loss "
                "(STATE.md 7d): a predictive loss alone is a capture curriculum, "
                "and holding the expansion measure out is what keeps it diagnostic",
        },
    }


def upload(ckpt_path: str | Path, meta: dict, *, repo_id: str = REPO_ID,
           private: bool = True) -> str | None:
    """push one checkpoint and its sidecar.  returns the remote path, or None."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    name = meta["name"]
    p = Path(ckpt_path)
    side = p.with_suffix(".json")
    side.write_text(json.dumps(meta, indent=2))
    api.upload_file(path_or_fileobj=str(p), path_in_repo=f"checkpoints/{name}.pt",
                    repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(side), path_in_repo=f"checkpoints/{name}.json",
                    repo_id=repo_id, repo_type="model")
    return f"{repo_id}/checkpoints/{name}.pt"

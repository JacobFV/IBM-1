# data

source cards, and the indices we draw them from.

## what a source card is

a card is not a description of a dataset. it is a statement of **which ibm state
variables a dataset realizes, and how**.

that is the whole reason this schema differs from the one in `~/Documents/win/registry`.
win's cards carry an eight-way epistemic role taxonomy — observation, boundary, prior,
calibration, oracle, teacher, evaluation, negative_control — which is a second ontology
living alongside the four primitives, with its own vocabulary to keep consistent. here
the same information falls out of the architecture:

| win concept | here |
|---|---|
| `role: observation` | a stream with `kind: measured` |
| `role: boundary` | a stream with `kind: imposed` |
| `role: calibration` | a stream that constrains device or geometry state, never tissue |
| `role: prior` | `use: prior` — shapes $p(\theta)$, contributes no likelihood |
| `role: teacher` | `use: distil` — a target for $p(\theta)$, never a likelihood |
| `role: evaluation` | `use: evaluate` — scores a materialization, never updates $\theta$ |
| `role: negative_control` | `use: control` — must produce a null |
| `role: oracle` | `use: surrogate` — synthetic truth for a surrogate fit |
| `observation_process` (free text) | `via:` a registered process id |

and three things win has no place for, which this system needs:

- **`band`** per stream. ARCHITECTURE.md §1 makes bandwidth a laziness axis co-equal
  with resolution, and a source's sampling rate determines which temporal laplacian
  components it can constrain at all. an fMRI run adds precision below
  $0.25\ \mathrm{Hz}$ and none above it. without this, heterogeneous sources appear to
  conflict where they simply do not overlap.
- **`constrains`** as registered component ids rather than prose. a card cannot claim to
  constrain state that does not exist in the registry.
- **`via`** as a registered process id. §6 says an observation is evidence about state
  that an ordinary process produces; naming that process is what makes the claim
  checkable.

## what a dataset is

not $p(y\mid x)$ — that presumes a direction most of these do not have. a simultaneous
EEG-fMRI recording constrains electromagnetic and haemodynamic state through two
different processes and neither conditions the other.

a dataset is a finite set of **jointly realized values of ibm state variables**, split
by how they came to be:

$$
p(\theta\mid D)\ \propto\ p(\theta)\ \prod_d\ p\!\left(x_{\text{measured}}^{(d)}\ \middle|\ x_{\text{imposed}}^{(d)},\ \theta\right)
$$

both are ordinary state. the difference is what they are permitted to do: an imposed
variable is conditioned on, a measured one contributes a likelihood term. a stimulus was
chosen by an experimenter rather than drawn from the brain's joint distribution, and
treating it as jointly sampled lets a model learn the experimental design as biology.

## layout

```
data/
  schema/source.schema.json      the card schema
  index/                         catalogues we draw from; rows, not commitments
    neuro2/datasets.jsonl          26 399 datasets, 2.55 PB, from datasets.neuro2.ai
    neuro2/summary.json            breakdown by source, modality, species, licence
    win/cards.jsonl                601 cards migrated from ~/Documents/win/registry
  sources/<id>/                  sources we have committed to using
    card.yaml                      the card. the only file you must read
    raw/                           gitignored mount point for the bytes
      .location.yaml               committed: where the bytes actually live
      checksums.txt                committed: what they hashed to
    derive/                        committed: recipes, not their output
      <recipe>.yaml                inputs, tool and version, parameters, outputs
    evidence/                      the bound result
      <recipe>@<version>/
        manifest.yaml              committed: what was produced, from which raw hash
        *.zarr                     gitignored
```

three deliberate departures from the obvious `meta / raw / processing / processed`
shape:

**`raw/` holds a pointer, not bytes.** the catalogue is 2.55 PB and most of these
licences prohibit redistribution. the repository commits where the bytes are and what
they hashed to; acquiring them is a separate, local act.

**`processing/` is `derive/`, and it holds recipes rather than runs.** a recipe is
declared — tool, version, parameters — and committed. its *output* goes under
`evidence/`, keyed by recipe and version, because more than one recipe may legitimately
be applied to one raw source and they will not agree. fixation-related potentials need
an overlap-correction model and which one is used changes the result; a single
`processed/` directory quietly hides that behind whichever ran last.

**`processed/` is `evidence/`.** what comes out is not cleaned data, it is values bound
to registered components over a declared band at a declared precision — a likelihood
term. calling it processed data invites using it as though it were the state itself.

## promotion

a row in `index/` is a candidate. a directory in `sources/` is a commitment. promoting
one means writing its card, resolving its licence, and deciding which components it
binds to. only sources we intend to use get directories; the other 26 000 stay rows.

## binding status

`constrains` and `via` reference registry ids that do not exist yet — there is no code.
migrated cards carry `binding: unbound` and record what the source's own documentation
said, so rebinding is a lookup rather than a re-reading. an unbound card is usable for
planning and not for fitting.

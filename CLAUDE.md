# working notes for agents on IBM-1

Conventions and traps specific to this repository. Read before touching
publishing, metrics, or long-running jobs.

## Accounts and publishing

**The HuggingFace account is `jacob-valdez` (with the hyphen).**

- `jacobvaldez`, `jacobfv123`, `jacobfv`, `JacobFV` — **none of these exist on
  HuggingFace.** Links written to them 404. The GitHub account IS `JacobFV`; the
  two are different and it is an easy substitution to make.
- `brandonin` is the machine account. It is **not** the programme's account.
  Anything published there has to be migrated, and migration across accounts is
  not a `move_repo` — HF refuses that without one token holding write on both
  namespaces, so it is copy → **verify** → delete, never delete-first. 52
  checkpoints once existed only on HF with 3 on local disk; a delete-first
  migration would have destroyed 49.
- `HF_TOKEN` in the environment is **`brandonin`'s** and silently shadows the CLI's
  active token. Every `HfApi()` built without an explicit token picks it up. To act
  as `jacob-valdez`, run with `env -u HF_TOKEN`. Assert `whoami()` before any
  destructive step — that guard once caught a script about to "migrate"
  `brandonin/ibm-1` onto itself and then delete it.

**Changing `REPO_ID` does not redirect a running job.** A training process loads
`ibm/release.py` at launch and holds the old value for its lifetime, and
`create_repo(exist_ok=True)` will happily **recreate a repo you just deleted**.
A deleted namespace stays deleted only after the last process holding the stale
constant has exited. Sync and delete *after* stopping those jobs, not before.

**Changing a constant is not the same as updating the prose.** `REPO_ID` was
changed in code while README, RELEASE.md, TRAINING.md and PROGRAMME.md kept
pointing at the old repo, and the public README kept asserting results the log had
already withdrawn. When a fact changes, grep the docs for it.

## Metrics

**A raw loss is not a result. Report skill against an explicit baseline**, via
`ibm/evaluate.py`. Include the trivial baselines — predicting zero, predicting the
mean, persistence — because a trivial baseline is what caught the worst error here
(a loss computed in the wrong units that read as "91–97% of variance explained"
while being *worse than predicting nothing*).

`docs/LOG.md` keeps a ledger of every withdrawn claim. Ten so far, and they share
one shape: **a quantity computed correctly, then compared against the wrong
thing** — wrong population, wrong units, wrong split, wrong baseline, or none.
Specifically:

- **Magnitude is not contribution.** A weight can be 4.4× baseline and cost
  −0.06% to sever. Gate on an **ablation**, never on a magnitude.
- **Check a metric against a case whose answer you know.** Chance must print
  1.0×. Effective rank over a batch of 4 cannot exceed 3.
- **A training loss is not skill** even when a baseline is at hand.
- **Single-pool retrieval has sd ≈ 2.8%.** Average pools before comparing, and
  never checkpoint on a single-pool best — that selects for lucky draws.

## Randomness

**A shared generator makes two things vary that should have varied
independently.** This bit twice in one day, in the same file, with the same
signature both times.

- `CorticalDynamics.__init__` drew its long-range partners with `torch.randint`
  from the **global** RNG. Seed 0 reproduces them on the *same device*, but cpu
  and cuda draw unrelated graphs from the same seed (measured coincidence
  0.00062 against a chance of 0.00050). `embody.py --seed` therefore redrew the
  **topology** while claiming to vary only the initialisation.
- An ablation drew `torch.randperm` *inside* the feature extractor, from a
  generator shared with the caller, and the extractor is called twice per arm —
  once for training features, once for held-out. The head was fitted on one
  permutation and evaluated on another. The arm read exactly chance while
  preserving 99.7% of the across-image variance, which looks exactly like a
  clean ablation result and is not one.

Draw it **once**, outside the loop, pass it in, and have the callee assert it
received one rather than drawing its own. When two passes must see the same
draw, `assert torch.equal(...)` between them — printing both for a human to
compare is only as good as the reader, and this one got past two of us for an
hour.

**The general form of row 23, and it is one line: call it twice at the same
input.** A function that is meant to depend only on its arguments must return
the same answer when called twice with the same arguments. Row 23's extractor
did not — it drew a fresh permutation each call — and the symptom was an
arm reading *exactly chance while preserving 99.7% of the variance*, which took
hours to unpick. The sister repo hit the identical shape on a geometry query
this week: called twice at identical positions it moved 141 of 3,123
associations by up to 6.5 mm, and the resulting artefact was read as a property
of the mesh, the solver and the physics for a full round of work before anyone
called the function twice.

This is **not** a known-answer check. A known answer tests the *value*;
idempotence tests whether the thing is a *function* at all, and a
non-idempotent instrument produces confident wrong conclusions from reasoning
that is itself sound. Run it on anything stateful, cached, seeded from a
previous result, or drawing from a generator.

**A gate that compares like with like is blind to a difference between the two
likes.** Testing whether recorded human motion respects IHM-1's declared joint
ranges, the instrument gate (G1) checked each motion against *the model it was
generated from* — a genuine known answer, since OpenSim produced that motion with
that model. It passed 42 of 48. Then the real comparison failed at 4.8%, almost
entirely on `knee_angle`, because gait2392 declares the knee `[−120°, +10°]` and
the target declares `[0°, +140°]`: **mirror images, opposite sign conventions.**
G1 could not see it — the source model shares the motion's convention, so the flip
cancels on both sides. The pre-registered fork had two branches, "the reference is
wrong" and "my mapping is wrong", and the truth was a third that **presents as the
first**. When two artefacts must be compared, check what they each declare about
the shared quantity before trusting any gate that only ever looks at one of them.

**Exactly chance, with the signal still present, means look at the bookkeeping
before you believe the ablation.** A destroyed-information result and a
train/test mismatch are indistinguishable from the accuracy alone. The
separating control is a pure-relabelling arm — permute something that *cannot*
change the answer, using the same draw in both passes, and confirm it scores
what the intact arm scores. If that arm moves, every arm in the run is void.

## Data

**Take an ordering from the dataset that defines it, never from a reconstruction
that happens to be the right length.** The THINGS-EEG2 image order was rebuilt by
walking a directory; it matched on **10 of 16,540 pairs** and the count check
passed because counts were equal. Three hours of confident negative results
followed. `image_metadata.npy` had the answer.

Corpora live in `data/sources/<id>/raw` (gitignored); `card.yaml` names the origin
the bytes actually came from — never the machine that staged them. A card marked
`binding: bound` does not guarantee the bytes are present.

## Jobs

- **`PYTHONPATH` is not optional for `scripts/*.py`.** Python puts the SCRIPT's
  directory on `sys.path`, never the working directory, so `cd ibm-1 && venv/bin/python
  scripts/train_proprioceptive_motor.py` dies on `ModuleNotFoundError: No module named
  'ibm'` however right the cwd looks. Launch with `PYTHONPATH=<repo root>`. It cost two
  failed launches on two machines in one day, and the failure is silent in a nohup log
  until you read it — the process exits in seconds and the GPU sits idle looking busy-free.
- **Never commit weights or caches.** History was rewritten once to purge 2.9 GB.
- **Save the checkpoint before attempting to upload it.** A failed upload once
  destroyed 2,000 steps because `torch.save` sat after the import that threw.
- `pkill -f <pattern>` **matches your own shell's command line** if the pattern
  appears in it, and returns 1 when nothing matches, aborting a `&&` chain. This
  has silently killed commands mid-sequence several times. Match on
  `venv/bin/python -u scripts/<name>` and tolerate a non-zero exit.
- **The bracket trick is not enough either.** `grep "[v]env/bin/python ..."` stops
  grep from matching *itself*, but the harness runs every command inside a
  wrapper -- `bash -c "... eval '<your whole command>'"` -- and that wrapper's argv
  contains your pattern as literal text, so the bracketed grep matches the
  wrapper. It happened twice in a row on a "is a fetch already running?" guard,
  which refused to relaunch because it found itself. Match on the EXECUTABLE, not
  the arguments: `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /scripts\/<name>/'`.
  The wrapper's `comm` is `bash`, whatever its arguments say.
- **`ast.parse` proves syntax, not scope.** `global X` declared after the same function has
  already used `X` (for example as an argparse default), or an inner `from m import n` that
  makes `n` local to the whole function, is a SyntaxError or NameError that `ast.parse`
  does NOT raise: it surfaces only when Python compiles the function's scopes, or when the
  code runs. It shipped twice in one day behind an "ast.parse: OK" check. Verify an edited
  script with `python -m py_compile <file>`, and treat a patch that adds `global` inside a
  function as suspect by default -- a local variable is almost always the right fix.
- Publish checkpoints for runs that **failed** too. A negative result without a
  checkpoint is an anecdote; the sidecar should say plainly what was falsified.
- **A save gated on a period the run is too short to reach never fires, and nothing
  says so.** `train_multi_materialization.py` saved inside `if a.upload_every and
  step % a.upload_every == 0` with `upload_every` defaulting to **2000**; the whole
  lead-rank sweep ran `--steps 1500`. Three arms trained to completion, printed
  1,499 healthy-looking lines each, and wrote **no weights at all**. There was no
  terminal save to catch it. Always save unconditionally on the last step, keep
  saving separate from uploading, and when you set `--steps`, check it against every
  `% N == 0` in the loop. Same shape as `| tail -1` swallowing an exit status: a
  guard that cannot fire looks exactly like a guard that passed.
- **Check that the trainer has a split before reading any number it prints.** This
  one drew `j = np.random.randint(pctx, lim)` over the entire array for every term
  and had no train/test split anywhere — so a day of `skill/0` readings, and a
  published sweep table, were all in-sample. `grep` the sampling line, not the log.
  A checkpoint should **store the split it was trained under**, and an evaluator
  should refuse any checkpoint whose stored split does not match the set it is
  about to be scored on.
- The remote has no git repo, so `git rev-parse` there yields `git-unknown` and the
  checkpoint becomes unciteable. Pass `IBM_GIT_SHA` from the launcher.

## Write it down where it will be found

**Every agent on this programme records what it did, in the repo, as it goes.** Not at the end, not
only when it works.

- **A finding goes in the doc that owns the subject** — `docs/LOG.md` for the brain's sequence,
  `docs/BODY_PARAMETERS.md`, `docs/SEGMENT_CONTACT_SURFACES.md`, `docs/TISSUE_MECHANICS.md` for the
  body's — with the evidence, and the commit carries the same reasoning rather than a one-line
  summary. Someone reading the log in a month gets the argument, not the headline.
- **Negatives and withdrawals are the point.** The corrections ledger above is the most reused thing
  in this repository. A claim that turned out wrong, with the check that caught it, is worth more
  than a claim that held. Record it at the moment it turns, not after the next result buries it.
- **A gate that fails is recorded as FAILED.** Never rescored, never quietly re-run with a different
  bar. Instruments may change after a failure; thresholds may not.
- **CLAUDE.md is for traps, not results.** If something cost real time and would cost it again, it
  belongs here in a few lines: what happened, what the shape of it is, what to do instead. If it is
  a finding about the body or the brain, it belongs in a doc and not here. Keep this file current —
  prune a trap that no longer exists, and add one the moment it bites.
- **A caution that lives only in a conversation is one revision from being lost.** Put it in the
  script's header, the artefact, or the doc.

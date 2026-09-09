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

- **Never commit weights or caches.** History was rewritten once to purge 2.9 GB.
- **Save the checkpoint before attempting to upload it.** A failed upload once
  destroyed 2,000 steps because `torch.save` sat after the import that threw.
- `pkill -f <pattern>` **matches your own shell's command line** if the pattern
  appears in it, and returns 1 when nothing matches, aborting a `&&` chain. This
  has silently killed commands mid-sequence several times. Match on
  `venv/bin/python -u scripts/<name>` and tolerate a non-zero exit.
- Publish checkpoints for runs that **failed** too. A negative result without a
  checkpoint is an anecdote; the sidecar should say plainly what was falsified.
- The remote has no git repo, so `git rev-parse` there yields `git-unknown` and the
  checkpoint becomes unciteable. Pass `IBM_GIT_SHA` from the launcher.

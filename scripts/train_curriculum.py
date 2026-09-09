"""one kernel, many objectives, a phase-varying schedule, periodic consolidation.

this is the training loop the architecture has been claiming and not doing.  every
run so far trained ONE materialisation from scratch; what is published is a pile
of independent models that share an architecture.  here the association kernel is
the model, several materialisations are trained against it simultaneously, and
their kernels are periodically averaged back into one.

**the schedule.**  `--phases` is a comma-separated list of
`until_frac:obj=weight;obj=weight`.  the default walks from video-heavy to
paired-heavy, because the volume term is the cheapest structure available (40 h
of unpaired film needs no recordings) and the paired terms are the scarce ones
worth spending later capacity on.  a phase is a DISTRIBUTION, not a stage: every
objective keeps a nonzero weight throughout, because a term dropped to zero is a
term whose kernel drifts away from the consensus and has to be re-learned.

**the consolidation.**  each objective holds its own replica of the kernel and
trains it independently for `--consolidate-every` steps; then the replicas are
orthogonal-Procrustes-aligned onto the highest-weighted one and averaged, and
every replica is reset to the average.  alignment is not optional: the kernel
enters as sigma(<e_i, e_j>), so e and Re are the same function and replicas that
have drifted into different frames would average toward zero.  measured on two
independently trained kernels, aligning before averaging recovered 97.2% against
95.8% naive -- small there because both started from the same seed, and expected
to matter more as replicas diverge.

**what is shared and what is not.**  the kernel is shared; the heads are not, and
cannot be -- a 64-channel EEG head and a 3-channel frame decoder have nothing to
average.  that split is the architecture's own claim, and at 250k sites it was
measured as 32.0M shared against heads of 9.9M and 3.0M.

**evaluation.**  every objective reports against its own explicit baseline every
`--eval-every` steps, and the consolidated kernel is scored by TRANSFER onto the
held-out EEG task -- the one measurement that asks whether the shared thing is
actually shared.  a raw loss is not a result.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sp = importlib.util.spec_from_file_location("ptrain", os.path.join(HERE, "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
fu = importlib.util.spec_from_file_location("fuse", os.path.join(HERE, "fuse_implicit.py"))
FU = importlib.util.module_from_spec(fu); fu.loader.exec_module(FU)

THINGS = "data/derived/things-paired"
LIBRI = "data/derived/libribrain-paired"
FILM = "data/derived/pd-film"
ONSET, KEEP = 20, 50


# --------------------------------------------------------------------------
# the objectives.  each owns its data, its head, its loss and its baseline.
# --------------------------------------------------------------------------

class VisualEEG:
    """image -> cortex -> embedding, contrastive against measured 64-ch EEG."""
    name = "visual_eeg"

    def __init__(self, dyn, dev, a):
        self.dev, self.a = dev, a
        self.imgs = np.load(f"{THINGS}/images_training.npy", mmap_mode="r")
        ev = np.load(f"{THINGS}/evoked_training_groupmean.npy", mmap_mode="r")
        self.ev = ev
        self.n = min(len(self.imgs), len(ev))
        self.ntr = int(self.n * 0.8)
        s = np.asarray(ev[:self.ntr:7]).astype(np.float32)
        self.med = np.median(s, 0)
        self.iqr = ((np.percentile(s, 75, 0) - np.percentile(s, 25, 0)) / 1.349).clip(1e-9)
        T = self._eeg(np.arange(4)).shape[-1]
        self.model = P.VisualContrastiveLoop(dyn, n_sensors=ev.shape[1], n_times=T).to(dev)
        self.temp = nn.Parameter(torch.tensor(0.07, device=dev))
        self.chance = 1.0 / a.pool

    def _eeg(self, i):
        y = (np.asarray(self.ev[i]).astype(np.float32) - self.med) / self.iqr
        return np.clip(y, -6, 6)[..., ONSET:ONSET + KEEP:2]

    def params(self):
        return [p for n, p in self.model.named_parameters()
                if not n.startswith("dyn.")] + [self.temp]

    def _batch(self, lo, hi, m, rng):
        i = rng.integers(lo, hi, m)
        x = torch.from_numpy(np.ascontiguousarray(self.imgs[i])).to(self.dev)
        return (x.permute(0, 3, 1, 2).float() / 127.5) - 1, \
               torch.from_numpy(self._eeg(i)).to(self.dev)

    def loss(self, rng):
        x, y = self._batch(0, self.ntr, self.a.batch, rng)
        z, s = self.model.embed_image(x, substeps=4, dt=2e-2, n_steps=4)
        lg = z @ self.model.embed_eeg(y).T / self.temp.clamp(0.01, 1.0)
        lb = torch.arange(len(x), device=self.dev)
        return 0.5 * (F.cross_entropy(lg, lb) + F.cross_entropy(lg.T, lb)) \
            + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        accs = []
        rng = np.random.default_rng(90_000 + step)
        for _ in range(self.a.eval_pools):
            x, y = self._batch(self.ntr, self.n - 1, self.a.pool, rng)
            z, _ = self.model.embed_image(x, substeps=4, dt=2e-2, n_steps=4)
            sim = z @ self.model.embed_eeg(y).T
            accs.append(float((sim.argmax(1) ==
                        torch.arange(len(x), device=self.dev)).float().mean()))
        m, sd = float(np.mean(accs)), float(np.std(accs))
        return {"top1": m, "sd": sd, "x_chance": m / self.chance,
                "report": f"top-1 {100*m:5.2f}%+/-{100*sd:4.2f} ({m/self.chance:5.1f}x)"}


class AudioMEG:
    """cochleagram -> cortex -> embedding, contrastive against 306-ch MEG."""
    name = "audio_meg"
    CEILING = 0.0712                       # dynamics-free control, 8 pools

    def __init__(self, dyn, dev, a):
        self.dev, self.a = dev, a
        self.stim = np.load(f"{LIBRI}/cochleagram_250hz.npy", mmap_mode="r")
        self.neur = np.load(f"{LIBRI}/meg_250hz.npy", mmap_mode="r")
        self.n = min(len(self.stim), len(self.neur))
        self.ntr = int(self.n * 0.8)
        self.gap, self.win, self.ctx = 2500, 250, 250
        self.model = P.AudioContrastiveLoop(
            dyn, n_sensors=self.neur.shape[1], n_bands=self.stim.shape[1],
            stim_len=self.ctx, n_times=self.win).to(dev)
        self.temp = nn.Parameter(torch.tensor(0.07, device=dev))
        self.chance = 1.0 / a.pool

    def params(self):
        return [p for n, p in self.model.named_parameters()
                if not n.startswith("dyn.")] + [self.temp]

    def _batch(self, lo, hi, m, rng):
        i = rng.integers(lo, hi, m)
        c = np.empty((m, self.ctx, self.stim.shape[1]), np.float32)
        g = np.empty((m, self.neur.shape[1], self.win), np.float32)
        for b, q in enumerate(i):
            c[b] = np.asarray(self.stim[q - self.ctx:q])
            g[b] = np.clip(np.asarray(self.neur[q:q + self.win]).astype(np.float32), -6, 6).T
        return (torch.from_numpy(c).to(self.dev).permute(0, 2, 1),
                torch.from_numpy(g).to(self.dev))

    def loss(self, rng):
        # FULL batch, not half.  contrastive collapse is driven by too few
        # negatives, and halving this to save memory put audio at 32 while the
        # other terms ran at 64 -- combined with the lowest phase weight, it
        # collapsed completely by step 1,000: every audio embedding identical
        # (pairwise cos +0.9999), every MEG embedding identical (+1.0000), one
        # distinct prediction out of 24, reported as exactly 0.50% +/- 0.00.
        # the standalone audio run reached 6.75% at batch 64.
        x, y = self._batch(self.ctx, self.ntr - self.win, self.a.batch, rng)
        z, s = self.model.embed_audio(x, substeps=4, dt=2e-2, n_steps=4)
        lg = z @ self.model.embed_meg(y).T / self.temp.clamp(0.01, 1.0)
        lb = torch.arange(len(x), device=self.dev)
        return 0.5 * (F.cross_entropy(lg, lb) + F.cross_entropy(lg.T, lb)) \
            + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        accs = []
        rng = np.random.default_rng(90_000 + step)
        for _ in range(self.a.eval_pools):
            x, y = self._batch(self.ntr + self.gap, self.n - self.win - 1,
                               self.a.pool, rng)
            z, _ = self.model.embed_audio(x, substeps=4, dt=2e-2, n_steps=4)
            sim = z @ self.model.embed_meg(y).T
            accs.append(float((sim.argmax(1) ==
                        torch.arange(len(x), device=self.dev)).float().mean()))
        m, sd = float(np.mean(accs)), float(np.std(accs))
        # exactly 1/pool with zero variance is the signature of collapse, not of
        # chance: genuine chance on a pool of 200 varies ~0.5 points between
        # draws.  say so rather than printing a number that reads as "training".
        flag = "  COLLAPSED" if (sd == 0.0 and abs(m - self.chance) < 1e-9) else ""
        return {"top1": m, "sd": sd, "ceiling": self.CEILING, "collapsed": bool(flag),
                "report": f"top-1 {100*m:5.2f}%+/-{100*sd:4.2f} "
                          f"({m/self.chance:5.1f}x, ceiling {100*self.CEILING:.2f}%)"
                          f"{flag}"}


class VideoNext:
    """frame t -> cortex -> frame t+H, scored against persistence."""
    name = "video"

    def __init__(self, dyn, dev, a):
        self.dev, self.a = dev, a
        films = sorted(glob.glob(f"{FILM}/*_frames.npy"))
        self.TR = [np.load(f, mmap_mode="r") for f in films[:-2]]
        self.TE = [np.load(f, mmap_mode="r") for f in films[-2:]]
        self.H = 8
        self.model = P.VideoLoop(dyn).to(dev)

    def params(self):
        return [p for n, p in self.model.named_parameters() if not n.startswith("dyn.")]

    def _batch(self, pool, m, rng):
        xs, ys = [], []
        for _ in range(m):
            v = pool[rng.integers(len(pool))]
            i = int(rng.integers(0, len(v) - self.H - 1))
            xs.append(np.asarray(v[i])); ys.append(np.asarray(v[i + self.H]))
        f = lambda t: (torch.from_numpy(np.stack(t)).to(self.dev)
                       .permute(0, 3, 1, 2).float() / 127.5) - 1.0
        return f(xs), f(ys)

    def loss(self, rng):
        x, y = self._batch(self.TR, max(4, self.a.batch // 8), rng)
        pred, s = self.model(x, 8, 5e-3)
        return F.mse_loss(pred, y) + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        rng = np.random.default_rng(90_000 + step)
        x, y = self._batch(self.TE, 64, rng)
        pred, _ = self.model(x, 8, 5e-3)
        held = float(F.mse_loss(pred, y))
        per = float(F.mse_loss(x, y))
        sk = 1 - held / per
        return {"held": held, "persistence": per, "skill_vs_persistence": sk,
                "report": f"skill vs persistence {sk:+.4f}"}


class VisualEEGSubject(VisualEEG):
    """image -> cortex -> embedding, against ONE subject's evoked response.

    THINGS-EEG2 ships 10 subjects, and `build_paired_eeg_images.py` keeps the
    per-subject array alongside the group mean for exactly this reason: a
    subject-specific forward model is the eventual target, and averaging away the
    between-subject variance now would discard what that needs.

    ten subjects sharing one kernel is the architecture's claim in its cleanest
    form.  the stimulus is identical across them, so anything the kernel learns is
    common structure and anything the head learns is that subject's head, skull
    and cortical folding.  if the shared substrate is real, ten subjects should
    cost far less than ten times one subject.

    the group-mean term is kept separately: averaging 10 subjects buys sqrt(10) of
    SNR, so it is an easier task and a different one, not a substitute.
    """

    def __init__(self, dyn, dev, a, subject: int):
        self.subject = subject
        self.dev, self.a = dev, a
        self.imgs = np.load(f"{THINGS}/images_training.npy", mmap_mode="r")
        ev = np.load(f"{THINGS}/evoked_training_persubject.npy", mmap_mode="r")[subject]
        self.ev = ev
        self.n = min(len(self.imgs), len(ev))
        self.ntr = int(self.n * 0.8)
        s = np.asarray(ev[:self.ntr:7]).astype(np.float32)
        self.med = np.median(s, 0)
        self.iqr = ((np.percentile(s, 75, 0) - np.percentile(s, 25, 0)) / 1.349).clip(1e-9)
        T = self._eeg(np.arange(4)).shape[-1]
        self.model = P.VisualContrastiveLoop(dyn, n_sensors=ev.shape[1], n_times=T).to(dev)
        self.temp = nn.Parameter(torch.tensor(0.07, device=dev))
        self.chance = 1.0 / a.pool
        self.name = f"visual_eeg_s{subject:02d}"


class AudioVisual:
    """film frame + its cochleagram -> cortex -> next frame.

    the only term here that drives the sheet from TWO ports at once, so it is the
    only one that can test cross-modal binding.  the last attempt at that measured
    cross-modal edge magnitude at 0.885 while severing those exact edges cost
    -0.06% -- magnitude without contribution -- and dropping the audio drive
    entirely cost only +3.94%, so the objective was video-dominated.  it is
    included at a low weight to keep that question alive rather than to carry the
    schedule.
    """
    name = "audio_visual"

    def __init__(self, dyn, dev, a):
        self.dev, self.a = dev, a
        fr = sorted(glob.glob(f"{FILM}/*_frames.npy"))
        co = [f.replace("_frames.npy", "_coch.npy") for f in fr]
        keep = [(f, c) for f, c in zip(fr, co) if os.path.exists(c)]
        self.TRf = [np.load(f, mmap_mode="r") for f, _ in keep[:-2]]
        self.TRc = [np.load(c, mmap_mode="r") for _, c in keep[:-2]]
        self.TEf = [np.load(f, mmap_mode="r") for f, _ in keep[-2:]]
        self.TEc = [np.load(c, mmap_mode="r") for _, c in keep[-2:]]
        # AudioVisualLoop defaults to ctx=8 cochleagram frames; pass ours through
        # rather than assuming, since a mismatch shows up as an opaque shape error
        # inside the encoder rather than at construction.
        self.H, self.ctx = 8, 8
        self.model = P.AudioVisualLoop(dyn, n_bands=self.TRc[0].shape[1],
                                       ctx=self.ctx).to(dev)

    def params(self):
        return [p for n, p in self.model.named_parameters() if not n.startswith("dyn.")]

    def _batch(self, F_, C_, m, rng):
        xs, cs, ys = [], [], []
        for _ in range(m):
            j = rng.integers(len(F_))
            v, c = F_[j], C_[j]
            lim = min(len(v), len(c)) - self.H - 1
            i = int(rng.integers(self.ctx, lim))
            xs.append(np.asarray(v[i])); ys.append(np.asarray(v[i + self.H]))
            cs.append(np.asarray(c[i - self.ctx:i]))
        f = lambda t: (torch.from_numpy(np.stack(t)).to(self.dev)
                       .permute(0, 3, 1, 2).float() / 127.5) - 1.0
        return f(xs), torch.from_numpy(np.stack(cs)).float().to(self.dev), f(ys)

    def loss(self, rng):
        x, c, y = self._batch(self.TRf, self.TRc, max(4, self.a.batch // 8), rng)
        pred, _, s = self.model(x, c, 8, 5e-3)
        return F.mse_loss(pred, y) + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        rng = np.random.default_rng(90_000 + step)
        x, c, y = self._batch(self.TEf, self.TEc, 32, rng)
        pred, _, _ = self.model(x, c, 8, 5e-3)
        held = float(F.mse_loss(pred, y)); per = float(F.mse_loss(x, y))
        return {"held": held, "persistence": per,
                "skill_vs_persistence": 1 - held / per,
                "report": f"skill vs persistence {1-held/per:+.4f}"}


class OpticNerve(VisualEEG):
    """image -> OPTIC NERVE -> occipital cortex -> measured EEG.

    the same corpus and target as `visual_eeg`, and a different question.  that
    term drives `drive[:, :dyn.n // 8]` -- an arbitrary eighth of a seeded random
    point cloud, called occipital in a comment.  this one drives the sites
    `cortical_regions` labels occipital, through a nerve declared in
    `ibm/topologies/nerve.py` with a length and three retinal ganglion
    populations whose conduction velocities differ by a factor of three.

    so it is the ablation for the anatomy itself.  run beside `visual_eeg` on
    identical data, the difference between them is what the declared pathway is
    worth -- and if it is worth nothing, that is the finding, because the
    embodiment story rests on these routes meaning something.
    """
    name = "optic_nerve"

    def __init__(self, dyn, dev, a):
        VisualEEG.__init__(self, dyn, dev, a)
        T = self._eeg(np.arange(4)).shape[-1]
        self.model = P.CranialNerveLoop(dyn, nerve="optic", lobe="occipital",
                                        n_sensors=self.ev.shape[1], n_times=T).to(dev)
        self.temp = nn.Parameter(torch.tensor(0.07, device=dev))

    def loss(self, rng):
        x, y = self._batch(0, self.ntr, self.a.batch, rng)
        z, s = self.model.embed_stimulus(x)
        lg = z @ self.model.embed_eeg(y).T / self.temp.clamp(0.01, 1.0)
        lb = torch.arange(len(x), device=self.dev)
        return 0.5 * (F.cross_entropy(lg, lb) + F.cross_entropy(lg.T, lb)) \
            + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        accs = []
        rng = np.random.default_rng(90_000 + step)
        for _ in range(self.a.eval_pools):
            x, y = self._batch(self.ntr, self.n - 1, self.a.pool, rng)
            z, _ = self.model.embed_stimulus(x)
            sim = z @ self.model.embed_eeg(y).T
            accs.append(float((sim.argmax(1) ==
                        torch.arange(len(x), device=self.dev)).float().mean()))
        m, sd = float(np.mean(accs)), float(np.std(accs))
        flag = "  COLLAPSED" if (sd == 0.0 and abs(m - self.chance) < 1e-9) else ""
        return {"top1": m, "sd": sd, "collapsed": bool(flag),
                "report": f"top-1 {100*m:5.2f}%+/-{100*sd:4.2f} "
                          f"({m/self.chance:5.1f}x){flag}"}


class CochlearNerve(AudioMEG):
    """cochleagram -> COCHLEAR NERVE -> temporal cortex -> measured MEG.

    the auditory twin, and the same ablation.  type I fibres are 95% of the nerve
    and myelinated at 25 m/s; type II are unmyelinated at 3 m/s, so over 25 mm
    they arrive 1.0 ms and 8.3 ms after the same transient.  `audio_meg` drives
    `dyn.n // 4` and calls it temporal; this drives the sites the region
    assignment actually labels temporal.
    """
    name = "cochlear_nerve"

    def __init__(self, dyn, dev, a):
        AudioMEG.__init__(self, dyn, dev, a)
        self.model = P.CranialNerveLoop(dyn, nerve="cochlear", lobe="temporal",
                                        n_sensors=self.neur.shape[1],
                                        n_times=self.win).to(dev)
        self.temp = nn.Parameter(torch.tensor(0.07, device=dev))

    def loss(self, rng):
        x, y = self._batch(self.ctx, self.ntr - self.win, self.a.batch, rng)
        z, s = self.model.embed_stimulus(x)
        lg = z @ self.model.embed_eeg(y).T / self.temp.clamp(0.01, 1.0)
        lb = torch.arange(len(x), device=self.dev)
        return 0.5 * (F.cross_entropy(lg, lb) + F.cross_entropy(lg.T, lb)) \
            + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        accs = []
        rng = np.random.default_rng(90_000 + step)
        for _ in range(self.a.eval_pools):
            x, y = self._batch(self.ntr + self.gap, self.n - self.win - 1,
                               self.a.pool, rng)
            z, _ = self.model.embed_stimulus(x)
            sim = z @ self.model.embed_eeg(y).T
            accs.append(float((sim.argmax(1) ==
                        torch.arange(len(x), device=self.dev)).float().mean()))
        m, sd = float(np.mean(accs)), float(np.std(accs))
        flag = "  COLLAPSED" if (sd == 0.0 and abs(m - self.chance) < 1e-9) else ""
        return {"top1": m, "sd": sd, "ceiling": self.CEILING, "collapsed": bool(flag),
                "report": f"top-1 {100*m:5.2f}%+/-{100*sd:4.2f} "
                          f"({m/self.chance:5.1f}x, ceiling {100*self.CEILING:.2f}%)"
                          f"{flag}"}


class BodyStance:
    """muscle state -> cortex -> motor command.  the body, as one more corpus.

    the same shape as every other materialization here: a fixed paired corpus, a
    head, and a loss against an explicit baseline.  vision is (image, evoked
    EEG); hearing is (cochleagram, MEG); this is (muscle state, motor command),
    recorded once from IHM-1's engineered LQR holding a 70 kg body upright for
    12 s under perturbation.  the physics ran during collection, not here -- a
    curriculum step stays 1.45 s instead of becoming 300.

    **the teacher is the LQR and that is the design constraint, not a detail.**
    measured on IHM-1's body: the bare postural servo falls at 1.19 s, the servo
    with equilibrium excitations falls at 1.97 s, the LQR holds.  the first two
    produce a corpus of a body falling over, and an earlier attempt that cloned
    one had every ablation arm lose to predicting the mean -- there is nothing to
    learn from a controller that is failing.  the LQR is also the only teacher
    INDEPENDENT of this kernel: IHM's cortical stance controller also holds, but
    it is built from this kernel by an offline decoder fit, so cloning it would
    be circular.

    **the ceiling is known and it is high.**  the LQR is u = -Kx, so the map is
    linear and a ridge regression on the same split reaches skill +0.9801 against
    predicting the training mean.  that is what this term is being asked to do,
    and reporting anything below it as success would be reporting a failure.
    """
    name = "body_stance"
    CEILING = 0.9801          # ridge on the same split, measured

    def __init__(self, dyn, dev, a):
        self.dev, self.a = dev, a
        d = "data/derived/body-corpus"
        X = np.load(f"{d}/state.npy"); Y = np.load(f"{d}/command.npy")
        meta = json.load(open(f"{d}/meta.json"))
        self.muscles = meta["muscles"]
        n = len(X); self.ntr = int(n * 0.8)
        # contiguous split with a guard band: consecutive body states at 100 Hz
        # are near-duplicates and a random split leaks the test set.
        self.gap = max(1, n // 50)
        self.X = torch.from_numpy(X).to(dev); self.Y = torch.from_numpy(Y).to(dev)
        self.n = n
        self.mean_mse = float(((self.Y[self.ntr + self.gap:] -
                                self.Y[:self.ntr].mean(0)) ** 2).mean())
        self.model = P.SensorimotorLoop(dyn, muscles=self.muscles,
                                        afferent_channels=X.shape[1]).to(dev)

    def params(self):
        return [p for n, p in self.model.named_parameters() if not n.startswith("dyn.")]

    def loss(self, rng):
        i = rng.integers(0, self.ntr, min(self.a.batch, self.ntr))
        idx = torch.from_numpy(i).to(self.dev)
        pred, s = self.model(self.X[idx], n_steps=8, substeps=2)
        return F.mse_loss(pred, self.Y[idx]) + 1e-1 * P.viability_penalty(s[0])

    @torch.no_grad()
    def evaluate(self, step):
        te = slice(self.ntr + self.gap, self.n)
        pred, _ = self.model(self.X[te], n_steps=8, substeps=2)
        mse = float(F.mse_loss(pred, self.Y[te]))
        skill = 1 - mse / self.mean_mse
        return {"held_mse": mse, "skill_vs_mean": skill, "ceiling": self.CEILING,
                "report": f"skill vs mean {skill:+.4f} (ridge ceiling "
                          f"{self.CEILING:+.4f})"}


OBJECTIVES = {"visual_eeg": VisualEEG, "audio_meg": AudioMEG, "video": VideoNext,
              "audio_visual": AudioVisual,
              # the nerve-routed pair: same data and target as visual_eeg and
              # audio_meg, but entering through a DECLARED pathway into the lobe
              # the region assignment names, rather than an index slice.  run
              # beside them, the difference is what the anatomy is worth.
              "optic_nerve": OpticNerve, "cochlear_nerve": CochlearNerve,
              # the body: one more corpus in the soup, sharing the kernel
              "body_stance": BodyStance}
# ten per-subject visual terms, built from the array the pairing builder keeps
# for exactly this purpose.  they share the stimulus, so what the kernel learns
# across them is common structure and what each head learns is that subject.
for _s in range(10):
    OBJECTIVES[f"visual_eeg_s{_s:02d}"] = (
        lambda dyn, dev, a, _i=_s: VisualEEGSubject(dyn, dev, a, _i))


def parse_phases(spec: str):
    out = []
    for part in spec.split(","):
        frac, rest = part.split(":")
        w = {}
        for kv in rest.split(";"):
            k, v = kv.split("=")
            w[k] = float(v)
        out.append((float(frac), w))
    return out


def phase_weights(phases, frac):
    for until, w in phases:
        if frac <= until:
            return w
    return phases[-1][1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", default="ckpt/ibm1_implicit.pt",
                    help="the implicit model to start from; '' for a fresh kernel")
    ap.add_argument("--sites", type=int, default=30_000)
    ap.add_argument("--embed", type=int, default=128)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--long-range", type=float, default=0.25)
    ap.add_argument("--objectives",
                    default="visual_eeg,audio_meg,video,audio_visual," +
                            "optic_nerve,cochlear_nerve," +
                            ",".join(f"visual_eeg_s{i:02d}" for i in range(10)),
                    help="14 materializations by default: the group-mean visual "
                         "term, speech->MEG, video continuation, the audio-visual "
                         "loop, and TEN per-subject visual terms sharing one "
                         "kernel -- which is the architecture's claim in its "
                         "cleanest form, since the stimulus is identical across "
                         "subjects and only head, skull and folding differ")
    ap.add_argument("--phases", default="",
                    help="explicit schedule; empty means build one from GROUPS, "
                         "which is the only readable way to schedule 14 terms")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--eval-pools", type=int, default=8)
    ap.add_argument("--warm-start", default="ckpt/visual_contrastive_v2.pt,"
                                            "ckpt/audio_contrastive_v1.pt,"
                                            "ckpt/video_fixedread.pt",
                    help="checkpoints whose HEADS initialise matching materialisations.  "
                         "splitting a step budget 14 ways gives each contrastive term "
                         "a few dozen steps per thousand, and a contrastive head needs "
                         "hundreds to escape the trivial all-embeddings-equal solution "
                         "-- measured: 11 of 14 terms sat at exactly 0.50%% +/- 0.00 at "
                         "step 1,000, which is collapse and not chance.  starting from "
                         "a head already trained on the same architecture skips that "
                         "phase.  it is not a shortcut around the result: the KERNEL "
                         "still comes from the fused implicit model and is what the run "
                         "is measuring")
    ap.add_argument("--consolidate-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--ckpt", default="ckpt/ibm1_curriculum.pt")
    ap.add_argument("--out", default="out/curriculum.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    names = [n for n in a.objectives.split(",") if n]
    if a.phases:
        phases = parse_phases(a.phases)
    else:
        # schedule by GROUP, then split each group's mass evenly across its
        # members.  writing 14 weights by hand three times over is unreadable and
        # is how a term silently ends up at zero -- which is what collapsed the
        # audio head when it sat at 0.2 with half the batch.
        def grp(n):
            if n.startswith("visual_eeg_s"): return "subjects"
            if n in ("video", "audio_visual"): return "selfsup"
            if n.endswith("_nerve"): return "nerve"
            if n == "body_stance": return "body"
            return "paired"
        GROUPS = [(0.33, {"selfsup": .25, "paired": .22, "subjects": .21, "nerve": .17, "body": .15}),
                  (0.66, {"selfsup": .17, "paired": .22, "subjects": .29, "nerve": .17, "body": .15}),
                  (1.00, {"selfsup": .13, "paired": .22, "subjects": .33, "nerve": .17, "body": .15})]
        phases = []
        for until, gw in GROUPS:
            members = {}
            for g, w in gw.items():
                ms = [n for n in names if grp(n) == g]
                for m in ms:
                    members[m] = w / len(ms)
            phases.append((until, members))

    # one kernel per objective: a replica that trains independently between
    # consolidations.  they start identical, so the first consolidation is a no-op
    # and any later divergence is what the schedule actually produced.
    dyns, objs, opts = {}, {}, {}
    init = None
    if a.init and os.path.exists(a.init):
        d = torch.load(a.init, map_location="cpu")
        init = d["dyn.embed"]
        if init.shape[0] != a.sites:
            print(f"resampling implicit kernel {init.shape[0]:,} -> {a.sites:,} "
                  f"(k=1; a round trip costs ~25%, so this is not free)", flush=True)
            init = FU.resample(init, init.shape[0], a.sites)
        print(f"initialised from {a.init}: {len(d.get('sources', []))} fused sources",
              flush=True)
    for n in names:
        torch.manual_seed(0)                       # identical replicas at step 0
        dy = P.CorticalDynamics(a.sites, a.embed, a.k, dev, long_range=a.long_range).to(dev)
        if init is not None:
            dy.embed.data.copy_(init.to(dev))
        dyns[n] = dy
        objs[n] = OBJECTIVES[n](dy, dev, a)
        objs[n].name = n
        # warm-start the head from a trained one of the same architecture, when
        # there is one.  every visual_eeg* term is a VisualContrastiveLoop and
        # every audio term an AudioContrastiveLoop, so the head transfers exactly.
        # match the source to the objective TYPE, not to whatever happens to share
        # tensor names.  the first version accepted any checkpoint matching a
        # quarter of the head, which warm-started audio_meg and video from the
        # VISUAL checkpoint on 8 and 10 incidental tensors -- different modalities
        # wearing the same parameter names.
        want = ("visual" if n.startswith("visual_eeg") else
                "audio" if n.startswith("audio_meg") else
                "video" if n in ("video", "audio_visual") else "")
        cands = [x for x in a.warm_start.split(",")
                 if x and os.path.exists(x) and want and want in os.path.basename(x)]
        for w in cands:
            try:
                src = torch.load(w, map_location="cpu", weights_only=False)["model"]
            except Exception:
                continue
            tgt = objs[n].model.state_dict()
            share = {k: v for k, v in src.items()
                     if k in tgt and tgt[k].shape == v.shape and not k.startswith("dyn.")}
            if len(share) >= max(4, len(tgt) // 2):
                tgt.update(share)
                objs[n].model.load_state_dict(tgt)
                print(f"    {n}: head warm-started from {os.path.basename(w)} "
                      f"({len(share)} tensors)", flush=True)
                break
        opts[n] = torch.optim.AdamW(list(dy.parameters()) + objs[n].params(),
                                    lr=a.lr, weight_decay=1e-4)
        tot = sum(p.numel() for p in dy.parameters()) + \
              sum(p.numel() for p in objs[n].params())
        print(f"  {n:12s} {tot:>12,} params "
              f"({dy.embed.numel():,} kernel)", flush=True)

    log = {"config": vars(a), "steps": [], "consolidations": []}
    rng = np.random.default_rng(0)
    t0 = time.time()

    for step in range(a.steps + 1):
        w = phase_weights(phases, step / max(a.steps, 1))
        pick = [n for n in names if w.get(n, 0) > 0]
        p = np.array([w[n] for n in pick], dtype=float); p /= p.sum()
        n = pick[int(rng.choice(len(pick), p=p))]
        loss = objs[n].loss(rng)
        opts[n].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(dyns[n].parameters()) + objs[n].params(), 1.0)
        opts[n].step()

        if a.consolidate_every and step % a.consolidate_every == 0 and step:
            ks = [dyns[m].embed.data for m in names]
            ref = ks[0]
            aligned = [ks[0]] + [FU.procrustes(k, ref) for k in ks[1:]]
            drift = [float(F.cosine_similarity(k.flatten(), ref.flatten(), dim=0))
                     for k in ks]
            avg = torch.stack(aligned).mean(0)
            for m in names:
                dyns[m].embed.data.copy_(avg)
            log["consolidations"].append({"step": step, "drift_cos": drift})
            print(f"{step:6d}  consolidated {len(names)} replicas   "
                  f"drift cos {['%.3f'%d for d in drift]}", flush=True)

        if step % a.eval_every == 0:
            rec = {"step": step, "phase": w, "sec": round(time.time() - t0, 1)}
            parts = []
            for m in names:
                objs[m].model.eval()
                r = objs[m].evaluate(step)
                objs[m].model.train()
                rec[m] = r
                parts.append(f"{m} {r['report']}")
            log["steps"].append(rec)
            print(f"{step:6d}  " + " | ".join(parts) + f"   {time.time()-t0:5.0f}s",
                  flush=True)
            os.makedirs(os.path.dirname(a.ckpt) or ".", exist_ok=True)
            # ATOMIC: write to a temp file and rename.  torch.save streams a zip,
            # so a reader that opens the path mid-write gets "failed finding
            # central directory" -- a corrupt checkpoint, not a partial one.  that
            # bit me reading this very file, and IHM-1 is pulling checkpoints from
            # here now, so a torn read reaches another project.
            _tmp = a.ckpt + ".tmp"
            torch.save({"dyn.embed": dyns[names[0]].embed.data.cpu(),
                        "sites": a.sites, "embed": a.embed,
                        "schema": "ibm1/implicit-v1", "step": step,
                        "sources": [f"curriculum:{m}" for m in names],
                        "weights": [1.0 / len(names)] * len(names), "aligned": True,
                        "heads": {m: objs[m].model.state_dict() for m in names}},
                       _tmp)
            os.replace(_tmp, a.ckpt)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(log, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()

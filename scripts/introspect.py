"""decode the running cortical state into a retrieved image and a text label.

WHAT THIS IS.  `ckpt/visual_contrastive_v2.pt` maps image -> occipital port ->
E/I dynamics -> a 128-d embedding, trained contrastively against the embedding of
the *measured* THINGS-EEG2 evoked response.  the two embeddings live in one space.
so a cortical state can be matched against a bank of measured brain responses, and
whatever it matches carries an image and a concept name.  that is the decode.

WHAT IT IS NOT.  the retrieval bank is **measured EEG**, never the images.  if the
bank were the image embeddings the query would be a member of its own corpus and
the "decode" would be an identity lookup returning the input.  matching a cortical
state against evoked responses is the same cross-modal task `eval_things_test.py`
scores, so the accuracy printed here is the accuracy of that task and nothing
better.

THE THING TO DISTRUST is that the dynamics are decorative -- that `cortex_head`
is really reading `to_cortex(enc(img))` with a differential equation wrapped
around it, in which case this decodes the input image and means nothing.  four
controls are measured, all on the designated test set at the trained readout:

    shuffle_sites   permute the read sites within each sample.  the same values,
                    the wrong topography.  must fall to chance.
    gauss_state     replace the state with noise of matched mean and sd.  must
                    fall to chance.
    swap_state      decode image i's *label* from image i+1's cortical state.
                    this is the direct test that the input is not leaking: the
                    panel still shows image i.  must fall to chance.
    bypass          no dynamics at all -- the encoder's drive read straight off
                    the readout sites.  this one is NOT expected to be chance; it
                    is expected to be much worse than `full`.

THE BYPASS GAP IS NOT A CLAIM THAT THE CORTEX HELPS.  it says the trained readout
depends on the dynamics -- the cortex is load-bearing *within this model*.  it
does not say a cortex beats an encoder without one, and on this same designated
test set a separately-trained dynamics-free control reaches 66.50% against this
model's 63.50% (ledger row 12; the 3-point gap is 0.89 sd, so the two are
indistinguishable and the directional claim in either direction is dead).  what
follows is an introspection tool for a model that HAS a cortex, not evidence for
having one.

THE TEXT LABELS ARE NOT AN INDEPENDENT DECODE.  they are the concept names of the
retrieved bank entries, parsed from `image_paths_test.npy`.  on the designated
test set the 200 images are 200 distinct concepts, one image each, so label
accuracy and image-retrieval accuracy are the SAME NUMBER by construction.  the
label ranking is a relabelling of the retrieval ranking, not a second result.

THE SETTLING CURVE is off-distribution past the trained horizon.  the checkpoint
was trained to read the state after n_steps*substeps = 16 integrator steps; this
runs longer so the trajectory is visible, and every step past 16 is the readout
applied to a state it never saw in training.  the reported accuracy is the one at
step 16.  the rest of the curve is a picture, not a claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, gridspec
from PIL import Image

sp = importlib.util.spec_from_file_location(
    "ptrain", os.path.join(os.path.dirname(__file__), "pretrain_video_loop.py"))
P = importlib.util.module_from_spec(sp)
sp.loader.exec_module(P)

D = "data/derived/things-paired"
ONSET, KEEP = 20, 50
# the contrastive temperature was a free parameter of training (init 0.07) and was
# NOT saved with the checkpoint, so the softmax probabilities below are computed at
# the initialization value.  they are a readable rescaling of the cosine
# similarities, not a calibrated posterior; the ranking and the margin are the
# quantities that do not depend on this choice.
TEMP = 0.07
# the resting firing rate the dynamics relax to: rate(e_rest) with the declared
# sigmoid, r_max/(1+exp((v_half-e_rest)/slope)) = 100/(1+exp(10/4)).
R_REST = 100.0 / (1.0 + float(np.exp((-55.0 + 65.0) / 4.0)))


def concepts_from_paths(paths) -> list[str]:
    """the free text vocabulary: THINGS-EEG2 names the concept in the directory.

    '.../00001_aircraft_carrier/aircraft_carrier_06s.jpg' -> 'aircraft carrier'.
    the leading index is stripped; everything after the first underscore is the
    name the corpus itself declares, so this is read from the dataset rather than
    reconstructed (ledger row 8).
    """
    out = []
    for p in paths:
        d = os.path.basename(os.path.dirname(str(p)))
        out.append(d.split("_", 1)[1].replace("_", " ") if "_" in d else d)
    return out


# --------------------------------------------------------------------------- load

def load(ckpt: str, device: str):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, sd = d["config"], d["model"]
    n_sites, embed = sd["dyn.embed"].shape
    k = sd["dyn.idx"].shape[1]

    # normalisation fitted on TRAINING only.  fitting it on the test set leaks it.
    tr = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
    ntr = int(min(len(tr), len(np.load(f"{D}/images_training.npy", mmap_mode="r"))) * 0.8)
    samp = np.asarray(tr[:ntr:7]).astype(np.float32)
    med = np.median(samp, 0)
    iqr = ((np.percentile(samp, 75, 0) - np.percentile(samp, 25, 0)) / 1.349).clip(1e-9)

    imgs = np.load(f"{D}/images_test.npy")
    paths = np.load(f"{D}/image_paths_test.npy")
    ev = np.load(f"{D}/evoked_test_groupmean.npy")
    dec = cfg["decimate"]
    y = np.clip((ev.astype(np.float32) - med) / iqr, -6, 6)[..., ONSET:ONSET + KEEP:dec]
    T, C = y.shape[-1], ev.shape[1]

    dyn = P.CorticalDynamics(n_sites, embed, k, device,
                             long_range=cfg["long_range"]).to(device)
    model = P.VisualContrastiveLoop(dyn, n_sensors=C, n_times=T).to(device)
    model.load_state_dict(sd)
    model.eval()
    model.read_idx = model.read_idx.to(device)
    return model, dyn, cfg, d["step"], imgs, paths, y


# ----------------------------------------------------------------- cortical trace

@torch.no_grad()
def drive_of(model, img):
    b = img.shape[0]
    drive = torch.zeros(b, model.dyn.n, device=img.device)
    drive[:, :model.port] = model.to_cortex(model.enc(img))
    return drive


@torch.no_grad()
def trace_read_sites(model, dyn, img, total: int, h: float, batch: int = 50):
    """run the dynamics and return the rate at the READ sites at every substep.

    (total, B, read_sites).  this is the cortical state as the readout sees it --
    the same tensor `embed_image` slices at its final step, captured all the way
    along instead of once at the end.

    chunked over the batch because `association` materialises (B, 30000, 48) --
    1.15 GB at B=200 -- and it is built once per substep.
    """
    w = dyn.edge_weights()
    rs, ds = [], []
    for i in range(0, img.shape[0], batch):
        xb = img[i:i + batch]
        drive = drive_of(model, xb)
        s = dyn.init_state(xb.shape[0], xb.device)
        out = []
        for _ in range(total):
            s = dyn.step(s, drive, h, w)
            out.append(s[1][:, model.read_idx].clone())
        rs.append(torch.stack(out)); ds.append(drive[:, model.read_idx].clone())
        del s, drive, out
    return torch.cat(rs, 1), torch.cat(ds, 0)


@torch.no_grad()
def head(model, r_read):
    return torch.nn.functional.normalize(model.cortex_head(r_read), dim=-1)


# ------------------------------------------------------------------------ decode

@torch.no_grad()
def decode(z, bank, concepts, k: int = 5):
    """match a cortical embedding against the bank of MEASURED evoked responses.

    returns per-query: the ranked bank indices, their cosine similarities, the
    concept ranking (max-pooled over the bank entries sharing a concept),
    and a confidence = (top-1 similarity, margin over rank 2).
    """
    sim = z @ bank.T                                     # (B, n_bank)
    order = sim.argsort(dim=1, descending=True)
    top = sim.gather(1, order)
    # pool similarity per concept.  on the designated test set every concept has
    # exactly one bank entry, so this is the identity -- it is written pooled so
    # the same code is correct on a bank with repeats, and so the degeneracy is
    # explicit rather than assumed.
    vocab = sorted(set(concepts))
    cidx = torch.tensor([vocab.index(c) for c in concepts], device=z.device)
    logits = sim / TEMP
    pooled = torch.full((z.shape[0], len(vocab)), -1e30, device=z.device)
    pooled = pooled.index_reduce(1, cidx, logits, "amax", include_self=True)
    cprob = torch.softmax(pooled, dim=1)
    corder = cprob.argsort(dim=1, descending=True)
    return {
        "order": order[:, :k].cpu().numpy(),
        "sim": top[:, :k].cpu().numpy(),
        "top1_sim": top[:, 0].cpu().numpy(),
        "margin": (top[:, 0] - top[:, 1]).cpu().numpy(),
        "concept_rank": corder[:, :k].cpu().numpy(),
        "concept_prob": cprob.gather(1, corder[:, :k]).cpu().numpy(),
        "vocab": vocab,
        "full_order": order.cpu().numpy(),
    }


# --------------------------------------------------------------------- ablations

@torch.no_grad()
def measure(model, dyn, x, bank, total: int, h: float, readout: int,
            seed: int = 0, batch: int = 50):
    """the settling curve and the four controls, on the whole designated test set."""
    n = x.shape[0]
    lbl = torch.arange(n, device=x.device)

    def top1(z):
        sim = z @ bank.T
        return float(((sim > sim.gather(1, lbl[:, None])).sum(1) == 0).float().mean())

    def topk(z, k):
        sim = z @ bank.T
        return float(((sim > sim.gather(1, lbl[:, None])).sum(1) < k).float().mean())

    rr, drive_read = trace_read_sites(model, dyn, x, total, h, batch)
    curve = [top1(head(model, rr[t])) for t in range(total)]

    r = rr[readout - 1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.stack([torch.randperm(r.shape[1], generator=g) for _ in range(n)])
    arms = {
        "full": r,
        "shuffle_sites": torch.gather(r, 1, perm.to(r.device)),
        "gauss_state": (torch.randn(r.shape, generator=g).to(r.device)
                        * r.std(1, keepdim=True) + r.mean(1, keepdim=True)),
        "swap_state": torch.roll(r, 1, dims=0),
        "bypass": drive_read,
    }
    res = {}
    for name, rv in arms.items():
        z = head(model, rv)
        sim = z @ bank.T
        srt = sim.sort(dim=1, descending=True).values
        # the CONFIDENCE is reported per arm too, because a confidence that does
        # not fall when the state is destroyed is not a confidence.  it is
        # measured here rather than asserted.
        res[name] = {"top1": top1(z), "top5": topk(z, 5),
                     "median_top1_sim": float(srt[:, 0].median()),
                     "median_margin": float((srt[:, 0] - srt[:, 1]).median())}

    # the two substrate ablations eval_things_test.py reports, recomputed here so
    # the introspection report stands alone.
    geo0, emb0 = dyn.geo.clone(), dyn.embed.data.clone()
    for name in ("frozen_embed", "no_assoc"):
        dyn.geo.copy_(geo0); dyn.embed.data.copy_(emb0)
        if name == "frozen_embed":
            dyn.embed.data.copy_(torch.randn_like(dyn.embed) * 0.02)
        else:
            dyn.geo.zero_()
        rr2, _ = trace_read_sites(model, dyn, x, readout, h, batch)
        z = head(model, rr2[-1])
        sim = z @ bank.T
        srt = sim.sort(dim=1, descending=True).values
        res[name] = {"top1": top1(z), "top5": topk(z, 5),
                     "median_top1_sim": float(srt[:, 0].median()),
                     "median_margin": float((srt[:, 0] - srt[:, 1]).median())}
    dyn.geo.copy_(geo0); dyn.embed.data.copy_(emb0)
    return curve, res, rr


# ---------------------------------------------------------------------- rendering

def _panel(fig, img_in, pos_lonlat, r_state, rvmax, retr_imgs, retr_ok,
           labels, probs, sims, title, sub, true_label, chance):
    gs = gridspec.GridSpec(3, 6, figure=fig, height_ratios=[1.30, 0.85, 1.05],
                           hspace=0.42, wspace=0.30,
                           left=0.045, right=0.985, top=0.86, bottom=0.07)

    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(img_in); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"input drive\n“{true_label}”", fontsize=10)

    ax = fig.add_subplot(gs[0, 2:6], projection="mollweide")
    # DEVIATION FROM REST on a symmetric log scale, not the raw rate.  the 512
    # readout sites that fall inside the occipital port swing 0-98 Hz while the
    # other 3584 sit in a 4-7 Hz band, so a linear 0-98 map paints seven eighths
    # of the cortex a single flat colour and the associative spread -- the part
    # the dynamics actually produce -- is invisible.  centring on the resting
    # rate and compressing logarithmically shows both.
    sc = ax.scatter(pos_lonlat[:, 0], pos_lonlat[:, 1], c=r_state - R_REST, s=5.0,
                    cmap="RdBu_r", norm=colors.SymLogNorm(
                        linthresh=0.25, linscale=0.5, vmin=-rvmax, vmax=rvmax),
                    linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)
    ax.set_title("cortical state — 4096 readout sites on the cortical sphere",
                 fontsize=10)
    cb = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.02,
                      ticks=[-rvmax, -1, 0, 1, rvmax], format="%.0f")
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Δ rate from rest (Hz, symlog)", fontsize=7.5)

    for j in range(5):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(retr_imgs[j]); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_edgecolor("#1a9850" if retr_ok[j] else "#b0b0b0")
            sp_.set_linewidth(2.6 if retr_ok[j] else 0.8)
        ax.set_title(f"#{j+1}  cos {sims[j]:+.3f}", fontsize=8)

    ax = fig.add_subplot(gs[1, 5]); ax.axis("off")
    ax.text(0.0, 0.95, sub, va="top", ha="left", fontsize=8.5, family="monospace",
            transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 0:6])
    y = np.arange(5)[::-1]
    cols = ["#1a9850" if l == true_label else "#4575b4" for l in labels]
    ax.barh(y, probs, color=cols, height=0.62)
    for i, (l, p) in enumerate(zip(labels, probs)):
        ax.text(max(p, 0) + 0.012, y[i], f"{l}   {100*p:.1f}%", va="center",
                fontsize=9.5)
    ax.axvline(chance, color="#d73027", lw=1.0, ls="--")
    ax.text(chance, 4.75, f" chance {100*chance:.2f}%", color="#d73027", fontsize=8,
            va="top")
    ax.set_yticks([]); ax.set_xlim(0, 1.0); ax.set_ylim(-0.7, 4.9)
    ax.set_xlabel("decoded label probability  (softmax over the 200-concept "
                  "vocabulary, τ=0.07)", fontsize=9)
    ax.set_title("top-5 text labels — concept names of the matched bank "
                 "entries, not a separate decode", fontsize=10, loc="left")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    fig.suptitle(title, fontsize=12.5, y=0.965)


def frame(fig, *a, **kw) -> Image.Image:
    fig.clf()
    _panel(fig, *a, **kw)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=76, facecolor="white")
    buf.seek(0)
    # quantised to a 128-colour adaptive palette: 128 full-resolution
    # frames as truecolour PNGs is a ~60 MB GIF, and the panel is flat
    # colour plus two small photographs.
    return Image.open(buf).convert("RGB").quantize(
        colors=128, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="ckpt/visual_contrastive_v2.pt")
    ap.add_argument("--outdir", default="out/introspect")
    ap.add_argument("--queries", default="", help="comma-separated test indices; "
                    "default picks a spread of correctly and incorrectly decoded ones")
    ap.add_argument("--total", type=int, default=32,
                    help="integrator substeps to simulate (trained readout is 16)")
    ap.add_argument("--gif", default="thought_sequence.gif")
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.outdir, exist_ok=True)
    model, dyn, cfg, step, imgs, paths, y = load(a.ckpt, dev)
    concepts = concepts_from_paths(paths)
    n = len(imgs)
    readout = cfg.get("n_steps", 4) * cfg["substeps"]
    h = cfg["dt"] / cfg["substeps"]
    chance = 1.0 / n
    print(f"{a.ckpt}: step {step}\ndesignated THINGS-EEG2 test set: {n} images, "
          f"{len(set(concepts))} distinct concepts, chance {100*chance:.2f}%\n"
          f"trained readout at substep {readout}; simulating {a.total} "
          f"(h = {1000*h:.1f} ms)\n", flush=True)

    x = torch.from_numpy(np.ascontiguousarray(imgs)).to(dev)
    x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1
    bank = model.embed_eeg(torch.from_numpy(y).to(dev))     # MEASURED EEG, (200,128)

    # ---- measurement, in batches so the (B, 30000, 48) association fits
    curve, arms, rr = measure(model, dyn, x, bank, a.total, h, readout,
                              batch=a.batch)
    print(f"  settling: top-1 by substep")
    for t in range(a.total):
        mark = "  <- trained readout" if t + 1 == readout else ""
        print(f"    {t+1:3d}  ({1000*h*(t+1):5.1f} ms)  {100*curve[t]:6.2f}%{mark}",
              flush=True)
    print()
    full = arms["full"]["top1"]
    print(f"  {'arm':14s} {'top-1':>8s} {'top-5':>8s} {'x chance':>10s} "
          f"{'retained':>9s} {'med cos':>9s} {'med margin':>11s}")
    for k, v in arms.items():
        print(f"  {k:14s} {100*v['top1']:7.2f}% {100*v['top5']:7.2f}% "
              f"{v['top1']/chance:9.1f}x {100*v['top1']/full:8.1f}% "
              f"{v['median_top1_sim']:+9.4f} {v['median_margin']:+11.4f}", flush=True)
    print("\n  note: the cosine similarity barely falls when the state is "
          "destroyed.\n  the MARGIN and the accuracy do.  a high top-1 cosine is "
          "therefore not\n  evidence that the decode is real; only the margin "
          "carries that.", flush=True)

    # ---- the decode at the trained readout, for every test image
    dec_ro = decode(head(model, rr[readout - 1]), bank, concepts)
    correct = dec_ro["order"][:, 0] == np.arange(n)

    if a.queries:
        qs = [int(v) for v in a.queries.split(",")]
    else:
        ok = np.flatnonzero(correct)
        bad = np.flatnonzero(~correct)
        qs = [int(v) for v in list(ok[:: max(1, len(ok) // 3)][:2]) + list(bad[:1])]
    print(f"\n  rendering queries {qs} "
          f"({sum(correct[q] for q in qs)}/{len(qs)} decoded correctly at readout)",
          flush=True)

    lon = np.arctan2(dyn.pos[model.read_idx, 1].cpu().numpy(),
                     dyn.pos[model.read_idx, 0].cpu().numpy())
    lat = np.arcsin(np.clip(dyn.pos[model.read_idx, 2].cpu().numpy()
                            / float(dyn.pos.norm(dim=1).median()), -1, 1))
    ll = np.stack([lon, lat], 1)

    fig = plt.figure(figsize=(13.2, 7.4))
    frames, durations = [], []
    g = torch.Generator(device="cpu").manual_seed(1)

    def render_clip(q: int, scramble: bool):
        rq = rr[:, q:q + 1]                                  # (total, 1, sites)
        if scramble:
            perm = torch.stack([torch.randperm(rq.shape[2], generator=g)
                                for _ in range(rq.shape[0])]).to(rq.device)
            rq = torch.gather(rq[:, 0], 1, perm)[:, None]
        z = head(model, rq.reshape(-1, rq.shape[2]))
        d = decode(z, bank, concepts)
        hi = float(np.abs(rq.cpu().numpy() - R_REST).max())
        for t in range(a.total):
            oidx = d["order"][t]
            labs = [d["vocab"][c] for c in d["concept_rank"][t]]
            tag = ("SCRAMBLED CORTICAL STATE (ablation)  —  "
                   if scramble else "")
            phase = ("settling" if t + 1 < readout else
                     "TRAINED READOUT" if t + 1 == readout else
                     "past the trained horizon (off-distribution)")
            title = (f"{tag}introspecting the cortical state  —  substep "
                     f"{t+1}/{a.total}   t = {1000*h*(t+1):.0f} ms   [{phase}]")
            sub = (f"top-1 cos {d['top1_sim'][t]:+.4f}\n"
                   f"margin   {d['margin'][t]:+.4f}\n"
                   f"chance   {100*chance:.2f}%\n"
                   f"decoded  {'HIT' if oidx[0] == q else 'miss'}\n"
                   f"bank     200 measured\n"
                   f"         evoked responses")
            frames.append(frame(
                fig, imgs[q], ll, rq[t, 0].cpu().numpy(), hi,
                [imgs[o] for o in oidx], [o == q for o in oidx],
                labs, d["concept_prob"][t], d["sim"][t], title, sub,
                concepts[q], chance))
            durations.append(1400 if t + 1 == readout else 180)
        durations[-1] = 900

    for q in qs:
        render_clip(q, False)
    render_clip(qs[0], True)          # the ablation, same input, scrambled state

    gif = os.path.join(a.outdir, a.gif)
    frames[0].save(gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=durations, optimize=False)
    print(f"  wrote {gif}  ({len(frames)} frames)", flush=True)

    # a still of the trained readout for the first query
    fig.clf()
    q = qs[0]
    d = decode(head(model, rr[readout - 1, q:q + 1]), bank, concepts)
    oidx = d["order"][0]
    _panel(fig, imgs[q], ll, rr[readout - 1, q].cpu().numpy(),
           float(np.abs(rr[:, q].cpu().numpy() - R_REST).max()),
           [imgs[o] for o in oidx], [o == q for o in oidx],
           [d["vocab"][c] for c in d["concept_rank"][0]], d["concept_prob"][0],
           d["sim"][0],
           f"introspection at the trained readout (substep {readout})",
           f"top-1 cos {d['top1_sim'][0]:+.4f}\nmargin   {d['margin'][0]:+.4f}\n"
           f"chance   {100*chance:.2f}%", concepts[q], chance)
    still = os.path.join(a.outdir, "readout_still.png")
    fig.savefig(still, dpi=110, facecolor="white")

    # the settling curve as its own figure
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(1000 * h * np.arange(1, a.total + 1), 100 * np.array(curve), "-o", ms=3)
    ax.axhline(100 * chance, color="#d73027", ls="--", lw=1,
               label=f"chance {100*chance:.2f}%")
    ax.axvline(1000 * h * readout, color="#1a9850", ls=":", lw=1.4,
               label=f"trained readout ({readout} substeps)")
    ax.set_xlabel("integration time (ms)")
    ax.set_ylabel("top-1 retrieval on the 200-image designated test set (%)")
    ax.set_title("the thought settling: decode accuracy at every substep\n"
                 "(past the green line the readout is off-distribution)",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    curvep = os.path.join(a.outdir, "settling_curve.png")
    fig.savefig(curvep, dpi=110, facecolor="white", bbox_inches="tight")

    report = {
        "ckpt": a.ckpt, "train_step": step,
        "set": "THINGS-EEG2 designated test set (200 images, 80 repetitions each)",
        "retrieval_bank": "the 200 MEASURED group-mean evoked EEG responses of the "
                          "test images, embedded by eeg_head.  the query is the "
                          "cortical state; the bank is never the images.",
        "n": n, "n_concepts": len(set(concepts)), "chance": chance,
        "trained_readout_substep": readout, "substeps_simulated": a.total,
        "substep_ms": 1000 * h,
        "top1": arms["full"]["top1"], "top5": arms["full"]["top5"],
        "x_chance": arms["full"]["top1"] / chance,
        "ablations": {k: {**v, "delta_top1": v["top1"] - full,
                          "retained_frac": v["top1"] / full} for k, v in arms.items()},
        "settling_top1_by_substep": curve,
        "median_top1_similarity": float(np.median(dec_ro["top1_sim"])),
        "median_margin_over_rank2": float(np.median(dec_ro["margin"])),
        "median_margin_when_correct": float(np.median(dec_ro["margin"][correct])),
        "median_margin_when_wrong": float(np.median(dec_ro["margin"][~correct])),
        "queries_rendered": qs,
        "dynamics_free_control_top1_same_set": 0.665,
        "artifacts": {"gif": gif, "still": still, "settling_curve": curvep},
        "caveats": [
            "label accuracy == image-retrieval accuracy by construction: the "
            "designated test set holds one image per concept, so the concept "
            "ranking is a relabelling of the retrieval ranking and is not an "
            "independent result.",
            "the softmax temperature 0.07 is the training INITIALIZATION; the "
            "learned value was not saved with the checkpoint, so the printed "
            "probabilities are a rescaling of cosine similarity, not calibrated.",
            "every substep past the trained readout is off-distribution; only the "
            "accuracy at substep %d is a claim." % readout,
            "the top-1 cosine similarity is NOT a usable confidence: it stays "
            "high under the scramble ablations while accuracy is at chance.  the "
            "margin over rank 2 is the quantity that separates them.",
            "bypass is not expected to be chance -- the encoder alone is "
            "informative.  the full-minus-bypass gap says the dynamics are "
            "load-bearing INSIDE this model; it does not say a cortex beats an "
            "encoder without one.  a separately-trained dynamics-free control "
            "reaches 66.50% on this same set against 63.50% here (ledger row 12), "
            "a 0.89 sd difference -- the two are indistinguishable.",
        ],
    }
    rp = os.path.join(a.outdir, "report.json")
    json.dump(report, open(rp, "w"), indent=2)
    print(f"  wrote {rp}, {still}, {curvep}", flush=True)


if __name__ == "__main__":
    main()

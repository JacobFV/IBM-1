"""build a naturalistic video+audio corpus from public-domain film.

data, not compute, is the binding constraint on every downstream milestone: the
self-supervised materialisation has been training on eleven minutes of one film,
which is memorisation territory.  the paired corpora cannot fix this -- the
naturalistic-video-with-fMRI datasets carry annotations and not the films,
because the films are commercial.  but the SELF-SUPERVISED term is the volume
term and it needs no neural pairing at all, so public-domain feature film is
exactly the right supply: hours of real scenes, real motion, real speech and
music, under a licence that permits it.

the pipeline downloads, extracts, and DELETES the source.  a 2 GB film becomes
~1.6 GB of frames at 64x64 and a few MB of cochleagram, so keeping the mp4 would
waste both disk and the 200 GB/day the connection allows.  what is retained is
the derived array plus a manifest recording the archive.org identifier, so any
frame is traceable to the film it came from.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, urllib.parse, urllib.request
import numpy as np

OUT = "data/derived/pd-film"
FS, N_BANDS, H, W = 16000, 64, 64, 64


def search(rows: int, min_gb: float, max_gb: float) -> list[dict]:
    q = 'collection:(feature_films) AND mediatype:(movies) AND format:(MPEG4)'
    u = ("https://archive.org/advancedsearch.php?q=" + urllib.parse.quote(q) +
         "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=item_size"
         f"&rows={rows}&page=1&output=json")
    docs = json.load(urllib.request.urlopen(u, timeout=60))["response"]["docs"]
    return [d for d in docs
            if min_gb * 1e9 <= (d.get("item_size") or 0) <= max_gb * 1e9]


def mp4_url(ident: str) -> str | None:
    meta = json.load(urllib.request.urlopen(
        f"https://archive.org/metadata/{ident}", timeout=60))
    best = None
    for f in meta.get("files", []):
        if f.get("name", "").lower().endswith(".mp4"):
            sz = int(f.get("size", 0) or 0)
            if best is None or sz > best[0]:
                best = (sz, f["name"])
    return f"https://archive.org/download/{ident}/{urllib.parse.quote(best[1])}" if best else None


def erb(n, lo=50.0, hi=7000.0):
    e = lambda f: 21.4 * np.log10(1 + 0.00437 * f)
    inv = lambda E: (10 ** (E / 21.4) - 1) / 0.00437
    return inv(np.linspace(e(lo), e(hi), n))


def extract(path: str, fps: float = 25.0):
    """frames at 64x64 and a cochleagram on the SAME clock, one row per frame."""
    import av
    c = av.open(path)
    vs = c.streams.video[0]; vs.thread_type = "AUTO"
    src_fps = float(vs.average_rate or 25)
    keep = max(int(round(src_fps / fps)), 1)
    frames = []
    for i, f in enumerate(c.decode(vs)):
        if i % keep: continue
        img = f.to_ndarray(format="rgb24")
        h, w, _ = img.shape
        s = min(h, w); y0, x0 = (h - s) // 2, (w - s) // 2
        sq = img[y0:y0 + s, x0:x0 + s]
        k = max(s // H, 1)
        sq = sq[:k * H, :k * W].reshape(H, k, W, k, 3).mean(axis=(1, 3))
        frames.append(sq.astype(np.uint8))
    if not frames: return None, None
    V = np.stack(frames)

    c2 = av.open(path)
    if not c2.streams.audio:
        return V, np.zeros((len(V), N_BANDS), np.float32)
    rs = av.AudioResampler(format="s16", layout="mono", rate=FS)
    ch = []
    for fr in c2.decode(c2.streams.audio[0]):
        for r in rs.resample(fr): ch.append(r.to_ndarray().reshape(-1))
    x = np.concatenate(ch).astype(np.float32) / 32768.0 if ch else np.zeros(FS)
    nfft, hop = 1024, int(FS / fps)
    n = max((len(x) - nfft) // hop, 1)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    idx = idx[idx[:, -1] < len(x)]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(nfft)[None, :], axis=1))
    f_ = np.fft.rfftfreq(nfft, 1 / FS); cf = erb(N_BANDS)
    Wm = np.exp(-0.5 * ((f_[None, :] - cf[:, None]) / (0.12 * cf[:, None])) ** 2)
    Wm /= Wm.sum(1, keepdims=True)
    A = np.log1p(S @ Wm.T * 20.0).astype(np.float32)
    A = (A - A.mean()) / (A.std() + 1e-6)
    m = min(len(V), len(A))
    return V[:m], A[:m]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-gb", type=float, default=60.0,
                    help="download budget; the ISP allows ~200 GB/day")
    ap.add_argument("--films", type=int, default=14)
    ap.add_argument("--min-gb", type=float, default=1.0)
    ap.add_argument("--max-gb", type=float, default=6.0)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    man_p = f"{OUT}/manifest.json"
    man = json.load(open(man_p)) if os.path.exists(man_p) else {"films": [], "gb": 0.0}
    have = {f["id"] for f in man["films"]}

    cands = search(300, a.min_gb, a.max_gb)
    print(f"{len(cands)} candidate films in {a.min_gb}-{a.max_gb} GB", flush=True)
    used = 0.0
    for d in cands:
        if len(man["films"]) >= a.films or used >= a.budget_gb: break
        ident = d["identifier"]
        if ident in have: continue
        try:
            url = mp4_url(ident)
            if not url: continue
            tmp = f"/tmp/{ident[:60]}.mp4"
            print(f"  fetching {ident[:52]} ({(d.get('item_size') or 0)/1e9:.2f} GB)", flush=True)
            r = subprocess.run(["curl", "-sL", "--max-time", "2400", "-o", tmp, url])
            if r.returncode or not os.path.exists(tmp): continue
            gb = os.path.getsize(tmp) / 1e9
            used += gb
            V, A = extract(tmp)
            os.remove(tmp)                       # the source is not kept
            if V is None or len(V) < 500:
                print(f"    too short, skipped", flush=True); continue
            np.save(f"{OUT}/{ident}_frames.npy", V)
            np.save(f"{OUT}/{ident}_coch.npy", A)
            man["films"].append({"id": ident, "title": str(d.get("title", ""))[:80],
                                 "source_gb": round(gb, 2), "frames": int(len(V)),
                                 "minutes": round(len(V) / 25 / 60, 1),
                                 "url": f"https://archive.org/details/{ident}"})
            man["gb"] = round(man["gb"] + gb, 2)
            json.dump(man, open(man_p, "w"), indent=1)
            tot = sum(f["minutes"] for f in man["films"])
            print(f"    {len(V):,} frames = {len(V)/25/60:.1f} min "
                  f"| corpus now {tot/60:.2f} h from {len(man['films'])} films", flush=True)
        except Exception as e:
            print(f"    {type(e).__name__}: {e}", flush=True)
    tot = sum(f["minutes"] for f in man["films"])
    print(f"\ncorpus: {tot/60:.2f} hours over {len(man['films'])} films, "
          f"{used:.1f} GB downloaded and discarded", flush=True)


if __name__ == "__main__":
    main()

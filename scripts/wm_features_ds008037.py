"""delay-period EEG features for research step 2 (docs/LOG.md 2026-09-18 pre-registration).

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. .venv/bin/python scripts/wm_features_ds008037.py

per subject with both working-memory EEG and behaviour:
  G0  alignment: behavioural row i <-> the i-th `memory_array_onset` event; every pair's
      set size must agree.  a subject with more than 1% disagreement is EXCLUDED and
      printed.  this is CLAUDE.md's THINGS-EEG2 ordering trap -- an order rebuilt from
      something that merely has the right length matched 10 of 16,540 pairs there -- so the
      pairing is CHECKED against a field both files carry, never assumed.
  per trial: baseline -0.2..0 s before the array; window 0.3..1.0 s after
      `retention_interval_onset` (nothing on screen); 62 baseline-corrected mean voltages
      + 62 log alpha (8-13 Hz) powers; a trial over 300 uV peak-to-peak in the epoch is
      dropped; features z-scored within subject.
also kept per trial: subject, set size, target colour, all item colours (degrees), reported
colour -- for the stimulus descriptors and the behavioural signature.

one subject at a time, float32, so RSS stays ~1-2 GB.  writes out/wm_features_ds008037.npz
and a json summary (written as it goes: CLAUDE.md, write the cheap summary first).
"""
from __future__ import annotations

import glob, json, os, sys, time
import numpy as np

ROOT = "data/sources/ds008037/raw"
OUT_NPZ = "out/wm_features_ds008037.npz"
OUT_JSON = "out/wm_features_ds008037.json"
BASE = (-0.2, 0.0)          # relative to memory_array_onset
WIN = (0.3, 1.0)            # relative to retention_interval_onset
PTP_UV = 300.0


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist() if o.size <= 64 else f"<ndarray {o.shape}>"
    return str(o)


def parse_colours(s):
    s = str(s).strip()
    if s in ("", "n/a", "nan"):
        return []
    return [float(x) for x in s.split(",") if x.strip() not in ("", "n/a")]


def subject(sub):
    import mne
    import pandas as pd
    ev = pd.read_csv(f"{ROOT}/{sub}/eeg/{sub}_task-workingmemory_events.tsv", sep="\t")
    beh = pd.read_csv(f"{ROOT}/{sub}/beh/{sub}_task-workingmemory_beh.tsv", sep="\t")
    arr = ev[ev.trial_type == "memory_array_onset"].reset_index(drop=True)
    ret = ev[ev.trial_type == "retention_interval_onset"].reset_index(drop=True)
    info = {"sub": sub, "n_beh": len(beh), "n_array_events": len(arr), "n_retention_events": len(ret)}
    n = min(len(beh), len(arr))
    ss_ev = pd.to_numeric(arr.set_size[:n], errors="coerce").to_numpy()
    ss_bh = pd.to_numeric(beh.set_size[:n], errors="coerce").to_numpy()
    mismatch = float(np.mean(ss_ev != ss_bh)) if n else 1.0
    info.update(n_paired=int(n), set_size_mismatch=mismatch,
                count_mismatch=int(len(beh) != len(arr)))
    if n == 0 or mismatch > 0.01:
        info["excluded"] = f"G0 alignment: set-size disagreement {mismatch:.3%} over {n} pairs"
        return None, info
    raw = mne.io.read_raw_eeglab(f"{ROOT}/{sub}/eeg/{sub}_task-workingmemory_eeg.set",
                                 preload=True, verbose="ERROR")
    raw.pick("eeg")
    sf = raw.info["sfreq"]
    X = raw.get_data().astype(np.float32) * 1e6                    # V -> uV
    chans = raw.ch_names
    alpha = raw.copy().filter(8.0, 13.0, verbose="ERROR").get_data().astype(np.float32) * 1e6
    feats, keep, rows = [], [], []
    for i in range(n):
        t_arr = float(arr.onset[i])
        # the retention onset belonging to THIS array: the first one after it
        later = ret.onset[ret.onset > t_arr]
        if later.empty:
            continue
        t_ret = float(later.iloc[0])
        if t_ret - t_arr > 2.0:            # a missing retention marker would pair across trials
            continue
        b0, b1 = int((t_arr + BASE[0]) * sf), int((t_arr + BASE[1]) * sf)
        w0, w1 = int((t_ret + WIN[0]) * sf), int((t_ret + WIN[1]) * sf)
        if b0 < 0 or w1 > X.shape[1]:
            continue
        ep = X[:, b0:w1]
        if float((ep.max(1) - ep.min(1)).max()) > PTP_UV:
            continue
        base = X[:, b0:b1].mean(1)
        volt = X[:, w0:w1].mean(1) - base
        apow = np.log(np.mean(alpha[:, w0:w1] ** 2, 1) + 1e-6)
        feats.append(np.concatenate([volt, apow]))
        r = beh.iloc[i]
        rows.append({"set_size": int(r.set_size), "target_deg": float(r.target_color_degree),
                     "reported_deg": float(r.reported_color_degree),
                     "nontarget_deg": parse_colours(r.non_target_color_degree)})
        keep.append(i)
    info.update(n_kept=len(keep), n_dropped=int(n - len(keep)), sfreq=sf, n_channels=len(chans))
    if not feats:
        info["excluded"] = "no trial survived"
        return None, info
    F = np.stack(feats).astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)                          # within-subject z
    return {"F": F, "rows": rows, "chans": chans}, info


def main():
    subs = sorted(os.path.basename(os.path.dirname(os.path.dirname(p)))
                  for p in glob.glob(f"{ROOT}/sub-*/eeg/*task-workingmemory_eeg.set"))
    subs = [s for s in subs if os.path.exists(f"{ROOT}/{s}/beh/{s}_task-workingmemory_beh.tsv")]
    summary = {"preregistered": "docs/LOG.md 2026-09-18 research step 2", "n_candidates": len(subs),
               "window_after_retention_s": WIN, "baseline_before_array_s": BASE, "ptp_uv": PTP_UV,
               "subjects": []}
    allF, meta, chans0 = [], [], None
    t0 = time.time()
    for k, sub in enumerate(subs):
        try:
            got, info = subject(sub)
        except Exception as e:  # noqa: BLE001 -- recorded per subject, never silently skipped
            got, info = None, {"sub": sub, "excluded": f"{type(e).__name__}: {e}"}
        summary["subjects"].append(info)
        if got is not None:
            if chans0 is None:
                chans0 = got["chans"]
            if got["chans"] != chans0:
                info["excluded"] = "channel set differs from the first subject's"
            else:
                allF.append(got["F"])
                for r in got["rows"]:
                    meta.append({"sub": sub, **r})
        with open(OUT_JSON, "w") as fh:
            json.dump(summary, fh, indent=1, default=jdefault)
        print(f"[{k+1}/{len(subs)}] {sub}: {info.get('excluded', 'kept %d' % info.get('n_kept', 0))}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    F = np.concatenate(allF) if allF else np.zeros((0, 124), np.float32)
    maxn = max((len(m["nontarget_deg"]) for m in meta), default=0)
    np.savez_compressed(OUT_NPZ, F=F, chans=np.array(chans0 or []),
                        sub=np.array([m["sub"] for m in meta]),
                        set_size=np.array([m["set_size"] for m in meta]),
                        target_deg=np.array([m["target_deg"] for m in meta]),
                        reported_deg=np.array([m["reported_deg"] for m in meta]),
                        nontarget_deg=np.array([m["nontarget_deg"] + [np.nan] * (maxn - len(m["nontarget_deg"]))
                                                for m in meta], dtype=np.float32))
    summary.update(n_subjects_kept=len(allF), n_trials=int(len(F)),
                   excluded=[s["sub"] for s in summary["subjects"] if "excluded" in s])
    with open(OUT_JSON, "w") as fh:
        json.dump(summary, fh, indent=1, default=jdefault)
    print(f"done: {len(allF)} subjects, {len(F)} trials, {len(summary['excluded'])} excluded", flush=True)


if __name__ == "__main__":
    sys.exit(main())

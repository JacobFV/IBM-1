"""the fair comparison: select on one set, report on the other.

the dynamics-free control reaches 61.50% on the designated test set at step 500
and decays to 35% by 3000 -- ordinary overfitting on 13k pairs.  quoting its PEAK
is selecting a checkpoint on the evaluation set, which flatters it; quoting its
FINAL value flatters the model it is a control for.  neither is the comparison.

so this selects the control's checkpoint the same way the cortex model's was
selected -- best on the training-split holdout -- and reports what that
checkpoint scores on the designated test set.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util, os

D = "data/derived/things-paired"
ONSET, KEEP, DEC = 20, 50, 2
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

imgs = np.load(f"{D}/images_training.npy", mmap_mode="r")
ev = np.load(f"{D}/evoked_training_groupmean.npy", mmap_mode="r")
n = min(len(imgs), len(ev)); ntr = int(n * 0.8)
samp = np.asarray(ev[:ntr:7]).astype(np.float32)
med = np.median(samp, 0)
iqr = ((np.percentile(samp,75,0)-np.percentile(samp,25,0))/1.349).clip(1e-9)
def eeg(i):
    return np.clip((np.asarray(ev[i]).astype(np.float32)-med)/iqr,-6,6)[...,ONSET:ONSET+KEEP:DEC]
TI = np.load(f"{D}/images_test.npy")
TE = np.clip((np.load(f"{D}/evoked_test_groupmean.npy").astype(np.float32)-med)/iqr,-6,6)[...,ONSET:ONSET+KEEP:DEC]
T, C = eeg(np.arange(4)).shape[-1], ev.shape[1]

spec = importlib.util.spec_from_file_location("c", "scripts/control_eeg_retrieval.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
ie, ee = M.ImgEnc().to(dev), M.EegEnc(C, T).to(dev)
temp = nn.Parameter(torch.tensor(0.07, device=dev))
opt = torch.optim.AdamW(list(ie.parameters())+list(ee.parameters())+[temp], lr=3e-4, weight_decay=1e-4)

xT = torch.from_numpy(np.ascontiguousarray(TI)).to(dev)
xT = (xT.permute(0,3,1,2).float()/127.5)-1
yT = torch.from_numpy(TE).to(dev)

best_sel, best_des, best_step = 0.0, 0.0, -1
print(f"{'step':>6s} {'trainsplit(sel)':>16s} {'designated':>12s}")
for step in range(3001):
    i = np.random.randint(0, ntr, 256)
    x = torch.from_numpy(np.ascontiguousarray(imgs[i])).to(dev)
    x = (x.permute(0,3,1,2).float()/127.5)-1
    y = torch.from_numpy(eeg(i)).to(dev)
    lg = ie(x) @ ee(y).T / temp.clamp(0.01,1.0)
    lbl = torch.arange(len(i), device=dev)
    loss = 0.5*(F.cross_entropy(lg,lbl)+F.cross_entropy(lg.T,lbl))
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if step % 100 == 0:
        with torch.no_grad():
            # selection metric: 8 pools of the TRAINING-split holdout, exactly how
            # the cortex checkpoint was chosen
            accs=[]; rng=np.random.default_rng(90_000+step)
            for _ in range(8):
                j = rng.integers(ntr, n-1, 200)
                xt = torch.from_numpy(np.ascontiguousarray(imgs[j])).to(dev)
                xt = (xt.permute(0,3,1,2).float()/127.5)-1
                s_ = ie(xt) @ ee(torch.from_numpy(eeg(j)).to(dev)).T
                accs.append(float((s_.argmax(1)==torch.arange(200,device=dev)).float().mean()))
            sel = float(np.mean(accs))
            sd = ie(xT) @ ee(yT).T
            des = float((sd.argmax(1)==torch.arange(len(TI),device=dev)).float().mean())
        if sel > best_sel:
            best_sel, best_des, best_step = sel, des, step
        if step % 500 == 0:
            print(f"{step:6d} {100*sel:15.2f}% {100*des:11.2f}%", flush=True)
print(f"\nselected at step {best_step} by trainsplit ({100*best_sel:.2f}%)")
print(f"  -> DESIGNATED TEST SET: {100*best_des:.2f}%  ({best_des*len(TI):.1f}x chance)")
print(f"\n  cortex model (same selection rule): 63.50%")

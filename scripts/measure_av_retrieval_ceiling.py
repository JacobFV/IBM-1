#!/usr/bin/env python3
"""Does audio -> concurrent-frame retrieval have ANY signal on this corpus?

The AV bypass arm reads 1.09-1.27x chance -- essentially nothing. If the intact arm also lands at
chance, the comparison between them says nothing about the cortex: that is ledger row 19, where
three arms tied because the corpus had nothing to learn and the experiment measured nothing.

So the same instrument that established the MEG task's ceiling is applied here: a linear ridge
from the cochleagram context to the frame, then exact top-1 retrieval, with a shuffled-pairing
control that must sit at chance.

KNOWN ANSWERS, before any intact number:
  the shuffled control must land at 1/P by construction
  retrieving a frame against ITSELF must be 100%
"""
import numpy as np, math, sys
ROOT='/home/brandonin/Documents/IBM-1/'
X=np.load(ROOT+'data/derived/pd-film/Detour_66_coch.npy', mmap_mode='r')
F=np.load(ROOT+'data/derived/pd-film/Detour_66_frames.npy', mmap_mode='r')
n=min(len(X),len(F)); ctx=125; lo=int(n*0.9); P=32; chance=1.0/P
lag=np.linspace(0,ctx-1,25).astype(int)
rng=np.random.default_rng(7)
tr=np.sort(rng.choice(np.arange(ctx,lo),size=20000,replace=False))
ev=np.arange(lo+ctx, n-1, ctx)
print(f'train {len(tr):,} | eval {len(ev)} frames (the tail is only {n-lo:,} rows)')

def feats(j): return np.stack([np.asarray(X[q-ctx:q])[lag].ravel() for q in j]).astype(np.float64)
def frames(j):
    # 8x8x3 downsample: a frame's coarse appearance, which is what an envelope could plausibly predict
    f=np.stack([np.asarray(F[q]) for q in j]).astype(np.float64)/127.5-1.0
    return f.reshape(len(j),8,8,8,8,3).mean((2,4)).reshape(len(j),-1)

Ftr,Ttr=feats(tr),frames(tr); Fev,Tev=feats(ev),frames(ev)
mu,sd=Ftr.mean(0),Ftr.std(0)+1e-12
A=(Ftr-mu)/sd; B=(Fev-mu)/sd
W=np.linalg.solve(A.T@A+1e4*np.eye(A.shape[1]), A.T@(Ttr-Ttr.mean(0)))
Pd=B@W+Ttr.mean(0)
def norm(M):
    M=M-M.mean(1,keepdims=True); return M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-30)
Pn,Tn=norm(Pd),norm(Tev); N=len(ev)
def top1(S):
    r=(S>np.diag(S)[:,None]).sum(1)+1
    lg=lambda m,k:(math.lgamma(m+1)-math.lgamma(k+1)-math.lgamma(m-k+1) if m>=k>=0 else -math.inf)
    den=lg(N-1,P-1)
    return float(np.mean([math.exp(lg(N-x,P-1)-den) if N-x>=P-1 else 0.0 for x in r]))
print(f'\nKNOWN ANSWERS')
print(f'  frame retrieved against ITSELF: {top1(Tn@Tn.T):.1%}  (must be 100%)')
sh=rng.permutation(N); print(f'  SHUFFLED pairing:               {top1(Pn@Tn[sh].T):.2%}  (must be ~{chance:.2%})')
print(f'\n  RIDGE audio -> frame: top-1 {top1(Pn@Tn.T):.2%} = {top1(Pn@Tn.T)/chance:.2f}x chance')
corr=float(((Pd-Pd.mean(0)).ravel()@(Tev-Tev.mean(0)).ravel())/(np.linalg.norm(Pd-Pd.mean(0))*np.linalg.norm(Tev-Tev.mean(0))))
print(f'  per-pixel held-out correlation: {corr:+.4f}')

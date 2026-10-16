#!/usr/bin/env python3
"""Verify the dump features are correct by REPRODUCING the batch clustering
(filter -> AHC -> PLDA -> VBx) from dump embeddings, and comparing conf/DER
against the real online-vad30 run (conf 4.51%).

If reproduced conf ~= 4.51%, the dump (and thus the incremental-clustering
results built on it) faithfully matches the real pipeline.
"""
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from proto_incremental import parse_dump
from proto_plda import load_plda

SCORER = Path(__file__).with_name('score_der.py')
spec = importlib.util.spec_from_file_location('score_der', SCORER)
scoremod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoremod)

BASE = Path('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/asr-llm-tts-chain/build/3dspk/out')
DUMP = '/tmp/opencode/2p-feat-step10.bin'
STEP = 10.0
TOTAL = 1960.0
MIN_ACTIVE_RATIO = 0.2
SEG_FRAMES = 589
D = 256
PI = np.pi
Fa, Fb = 0.07, 0.8

P = load_plda()
mean1, mean2, lda, plda_mu, plda_tr, plda_psi = P


def plda_tf(train):
    """train (N,256) -> fea (N,128) PLDA xvectors."""
    tmp = train - mean1
    tmp = tmp / (np.linalg.norm(tmp, axis=1, keepdims=True) + 1e-30) * np.sqrt(256.0)
    tmp2 = tmp @ lda - mean2
    tmp2 = tmp2 / (np.linalg.norm(tmp2, axis=1, keepdims=True) + 1e-30) * np.sqrt(128.0) - plda_mu
    return tmp2 @ plda_tr.T


def vbx(ahc_labels, fea, Phi):
    N = fea.shape[0]
    S = ahc_labels.max() + 1
    q = np.zeros((N, S))
    for i in range(N):
        for s in range(S):
            q[i, s] = np.exp(7.0 if s == ahc_labels[i] else 0.0)
        q[i] /= q[i].sum()
    pi = np.full(S, 1.0 / S)
    G = -0.5 * (np.sum(fea * fea, axis=1) + 128.0 * np.log(2.0 * PI))
    V = np.sqrt(np.maximum(Phi, 0.0))
    rho = fea * V  # (N,128)
    for it in range(20):
        gsum = q.sum(axis=0)  # (S,)
        invL = 1.0 / (1.0 + Fa / Fb * gsum[:, None] * Phi[None, :])
        alpha = Fa / Fb * invL * (q.T @ rho)  # (S,128)
        lpi = np.log(pi + 1e-8)
        logp = Fa * (rho @ alpha.T - 0.5 * ((invL + alpha ** 2) @ Phi)) + G[:, None]
        mx = logp.max(axis=1, keepdims=True)
        log_px = mx + np.log(np.exp(logp - mx + lpi).sum(axis=1, keepdims=True))
        q = np.exp(logp + lpi - log_px)
        pi = q.sum(axis=0) / q.sum()
    return q, pi


def main():
    C, _, _, masks, embs = parse_dump(DUMP)
    # ---- filter ----
    keep = []
    for c in range(C):
        m = masks[c].astype(np.int32)
        for s in range(3):
            nclean = int(((m.sum(axis=1) == 1) & (m[:, s] == 1)).sum())
            if nclean >= MIN_ACTIVE_RATIO * SEG_FRAMES:
                keep.append(c * 3 + s)
    N = len(keep)
    print(f'filter: {N} embeddings kept (of {C*3})')
    train = np.vstack([embs[c][s] for c, s in ((k // 3, k % 3) for k in keep)]).astype(np.float64)

    # ---- AHC (centroid linkage, euclidean, 0.6) ----
    train_norm = train / np.linalg.norm(train, axis=1, keepdims=True)
    dendro = linkage(train_norm, method='centroid', metric='euclidean')
    ahc = fcluster(dendro, 0.6, criterion='distance') - 1
    _, ahc = np.unique(ahc, return_inverse=True)
    S = ahc.max() + 1
    print(f'AHC: {S} clusters (real pipeline 2P -> 2 speakers)')

    # ---- PLDA + VBx ----
    fea = plda_tf(train)
    Phi = plda_psi[:128]
    q, pi = vbx(ahc, fea, Phi)
    kept = np.where(pi > 1e-7)[0]
    K = len(kept)
    print(f'VBx: {K} speakers kept')
    centroids = (q[:, kept].T @ train) / q[:, kept].sum(axis=0, keepdims=True).T  # (K,256)

    # ---- soft = 1 + cos(all embs, centroids) + constrained argmax ----
    all_embs = np.concatenate(embs).astype(np.float64)  # (C*3, 256)
    e2k = cdist(all_embs, centroids, metric='cosine')  # (C*3, K)
    soft = (2 - e2k).reshape(C, 3, K)
    hard = np.full((C, 3), -1, dtype=np.int32)
    for c in range(C):
        spk, clu = linear_sum_assignment(soft[c], maximize=True)
        for s, k in zip(spk, clu):
            hard[c, s] = k

    # ---- reconstruct timeline (frame -> global speaker via hard + mask) ----
    runs = []
    for c in range(C):
        m = masks[c].astype(np.int32)
        fl = np.full(SEG_FRAMES, -1, dtype=np.int32)
        for t in range(SEG_FRAMES):
            for s in range(3):
                if m[t, s] and hard[c, s] >= 0:
                    fl[t] = int(hard[c, s])
                    break
        t = 0
        while t < SEG_FRAMES:
            if fl[t] < 0:
                t += 1
                continue
            spk = fl[t]
            t1 = t
            while t1 < SEG_FRAMES and fl[t1] == spk:
                t1 += 1
            st = c * STEP + 0.5 * 0.0619375 + t * 0.016875
            en = c * STEP + 0.5 * 0.0619375 + (t1 - 1) * 0.016875 + 0.016875
            runs.append((st, max(0.0, en - st), spk))
            t = t1
    hyp = '\n'.join(f'SPEAKER 2P 1 {b:.3f} {d:.3f} <NA> <NA> speaker-{spk} <NA> <NA>'
                    for b, d, spk in runs if d > 0) + '\n'

    # ---- score vs v8-recheck ----
    LINE = re.compile(r'^(\S+):\s*\[([\d.]+)\s+([\d.]+)\]\s*(.*)$')
    ref_lines = []
    for l in (BASE / 'label-2P-v8-recheck.txt').read_text().splitlines():
        mm = LINE.match(l.strip())
        if mm:
            spk = 'SPK' + mm.group(1).split('_')[-1]
            ref_lines.append(f'SPEAKER 2P 1 {mm.group(2)} '
                             f'{float(mm.group(3))-float(mm.group(2)):.3f} <NA> <NA> {spk} <NA> <NA>')
    ref = '\n'.join(ref_lines) + '\n'

    nf = int(TOTAL / scoremod.RES) + 3000
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(ref); rn = t.name
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(hyp); hn = t.name
    r, bd = scoremod.load_rttm(Path(rn), nf)
    h, _ = scoremod.load_rttm(Path(hn), nf)
    d = scoremod.der(r, h, bd, 0.25)
    t = d['total']
    print(f'\nREPRODUCED batch clustering (from dump):')
    print(f'  miss={d["miss"]/t:.2%} fa={d["fa"]/t:.2%} conf={d["conf"]/t:.2%} '
          f'DER={(d["miss"]+d["fa"]+d["conf"])/t:.2%}')
    print(f'  (real online-vad30: conf=4.51% DER=39.63%)')


if __name__ == '__main__':
    main()

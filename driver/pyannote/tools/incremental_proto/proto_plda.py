#!/usr/bin/env python3
"""Hypothesis test: does the PLDA discriminative projection (256->128) make a
single cosine threshold work across scenes (2P 2-speaker + T3 4-speaker)?

Batches use PLDA+VBx; incremental clustering with raw 256-D cosine needs
delta_new=1.0 for 2P but 0.5 for T3.  If clustering in the PLDA xvector space
(128-D) needs one threshold for both, that's the fix.
"""
import sys
from pathlib import Path
import numpy as np

from proto_incremental import parse_dump, timeline_to_rttm

PLDA = Path('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/'
            'asr-llm-tts-chain/models-for-asr-chain/pyannote/plda.bin')


def load_plda(path=PLDA):
    d = np.fromfile(path, dtype='<f4')
    p = 0
    mean1 = d[p:p + 256]; p += 256
    mean2 = d[p:p + 128]; p += 128
    lda = d[p:p + 256 * 128].reshape(256, 128); p += 256 * 128
    plda_mu = d[p:p + 128]; p += 128
    plda_tr = d[p:p + 128 * 128].reshape(128, 128); p += 128 * 128
    plda_psi = d[p:p + 128] if p + 128 <= d.size else d[p:]
    return mean1, mean2, lda, plda_mu, plda_tr, plda_psi


def plda_xvec_batch(embs, P):
    """embs: (N, 256) -> (N, 128) PLDA xvectors."""
    mean1, mean2, lda, plda_mu, plda_tr, _ = P
    tmp = embs - mean1  # (N,256)
    tmp = tmp / (np.linalg.norm(tmp, axis=1, keepdims=True) + 1e-30) * np.sqrt(256.0)
    tmp2 = tmp @ lda - mean2  # (N,128); lda is (256,128)
    tmp2 = tmp2 / (np.linalg.norm(tmp2, axis=1, keepdims=True) + 1e-30) * np.sqrt(128.0) - plda_mu
    return tmp2 @ plda_tr.T  # (N,128); plda_tr is (128,128) row=out


def incremental_cluster_plda(masks, embs, P, delta_new=0.5, max_speakers=20):
    """Same diart-style incremental clustering, but cosine over PLDA xvectors."""
    mean1, mean2, lda, plda_mu, plda_tr, _ = P
    centers = []   # running-sum of PLDA xvectors
    active = set()
    timeline = []

    def proj(e):  # e: (256,) -> (128,)
        tmp = e - mean1
        tmp = tmp / (np.linalg.norm(tmp) + 1e-30) * np.sqrt(256.0)
        t2 = lda.T @ tmp - mean2
        t2 = t2 / (np.linalg.norm(t2) + 1e-30) * np.sqrt(128.0) - plda_mu
        return plda_tr @ t2

    def cosd(a, b):
        return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for c in range(len(masks)):
        m = masks[c].astype(np.float32)
        active_local = [s for s in range(3) if m[:, s].max() >= 0.5]
        long_local = [s for s in range(3) if m[:, s].mean() >= 0.3]
        pe = [proj(embs[c][s]) for s in range(3)]
        assign = {}
        if not centers:
            for s in active_local:
                centers.append(pe[s].astype(np.float64).copy())
                active.add(len(centers) - 1)
                assign[s] = len(centers) - 1
        else:
            for s in active_local:
                best_g, best_d = None, float('inf')
                for g in active:
                    d = cosd(pe[s], centers[g])
                    if d < best_d:
                        best_d, best_g = d, g
                if best_g is not None and best_d <= delta_new:
                    assign[s] = best_g
            missed = [s for s in active_local if s not in assign]
            for s in missed:
                if len(active) < max_speakers and s in long_local:
                    centers.append(pe[s].astype(np.float64).copy())
                    active.add(len(centers) - 1)
                    assign[s] = len(centers) - 1
                elif active:
                    assign[s] = min(active, key=lambda g: cosd(pe[s], centers[g]))
            for s in long_local:
                if s in assign:
                    centers[assign[s]] += pe[s].astype(np.float64)
        frame_label = np.full(589, -1, dtype=np.int32)
        for t in range(589):
            if m[t].sum() == 0:
                continue
            for s in range(3):
                if m[t, s] >= 0.5 and s in assign:
                    frame_label[t] = assign[s]
                    break
        timeline.append(frame_label)
    return timeline, len(active)


def sweep(dump, label):
    C, ss, s0s, masks, embs = parse_dump(dump)
    P = load_plda()
    print(f'{label}:')
    for dn in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        tl, nspk = incremental_cluster_plda(masks, embs, P, delta_new=dn)
        print(f'  delta_new={dn:4.2f} -> final_speakers={nspk}')


if __name__ == '__main__':
    sweep('/tmp/opencode/2p-feat-step10.bin', '2P (2 speakers, want 2)')
    sweep('/tmp/opencode/t3-feat-step10.bin', 'T3 (4 speakers, want 4)')

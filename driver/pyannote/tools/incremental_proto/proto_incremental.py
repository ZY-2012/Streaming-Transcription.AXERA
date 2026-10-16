#!/usr/bin/env python3
"""Offline prototype: diart-style incremental online clustering on the 2P dump.

Reads the per-10s-sub-window segmentation masks + embeddings dumped by
diar-worker --dump-features, then runs diart's OnlineSpeakerClustering.identify()
logic (cosine distance + tau_active/rho_update/delta_new/max_speakers) in time
order to see (a) how early the first global speaker label can be emitted and
(b) how the speaker partition compares to the reference.

Reference for the comparison is label-2P-v8-recheck.txt (Speaker_0/Speaker_1
timeline), the closest-to-truth transcript we have.
"""
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

FRAME_STEP = 0.016875
FRAME_DUR = 0.0619375
SEG_FRAMES = 589
EMB_DIM = 256
N_LOCAL = 3
SAMPLE_RATE = 16000


def parse_dump(path):
    data = Path(path).read_bytes()
    assert data[:7] == b'C1DIAR1', 'bad magic'
    C, step_samples = struct.unpack_from('<II', data, 8)
    off = 16
    masks = []
    embs = []
    s0s = []
    for c in range(C):
        s0, = struct.unpack_from('<I', data, off)
        off += 4
        mask = np.frombuffer(data, dtype=np.uint8, count=SEG_FRAMES * N_LOCAL,
                             offset=off).reshape(SEG_FRAMES, N_LOCAL)
        off += SEG_FRAMES * N_LOCAL
        emb = np.frombuffer(data, dtype='<f4', count=N_LOCAL * EMB_DIM,
                            offset=off).reshape(N_LOCAL, EMB_DIM)
        off += N_LOCAL * EMB_DIM * 4
        masks.append(mask)
        embs.append(emb)
        s0s.append(s0)
    return C, step_samples, s0s, masks, embs


def incremental_cluster(masks, embs, tau_active=0.6, rho_update=0.3,
                        delta_new=1.0, max_speakers=20):
    """diart-style incremental clustering. Returns (timeline, centers_log).

    timeline: list of per-sub-window (frame -> global speaker or -1)
    centers_log: list of (num_known_speakers) after each sub-window
    """
    centers = []           # list of np arrays (running sum of embeddings)
    active = set()         # global speaker ids currently known
    timeline = []
    centers_log = []

    def cosd(a, b):
        # cosine distance between two vectors (normalized internally)
        return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    for c in range(len(masks)):
        m = masks[c].astype(np.float32)  # (589, 3)
        e = embs[c]                       # (3, 256)
        frame_act = m.sum(axis=1)
        # active local speakers: any frame active (hard mask -> max in {0,1})
        active_local = [s for s in range(N_LOCAL) if m[:, s].max() >= 0.5]
        # long local speakers: activation ratio >= rho_update
        long_local = [s for s in range(N_LOCAL) if m[:, s].mean() >= rho_update]

        assign = {}  # local -> global
        if not centers:
            for s in active_local:
                g = len(centers)
                centers.append(e[s].astype(np.float64).copy())
                active.add(g)
                assign[s] = g
        else:
            # match each active local to nearest known global under delta_new
            for s in active_local:
                best_g, best_d = None, float('inf')
                for g in active:
                    d = cosd(e[s], centers[g])
                    if d < best_d:
                        best_d, best_g = d, g
                if best_g is not None and best_d <= delta_new:
                    assign[s] = best_g
            missed = [s for s in active_local if s not in assign]
            for s in missed:
                if len(active) < max_speakers and s in long_local:
                    g = len(centers)
                    centers.append(e[s].astype(np.float64).copy())
                    active.add(g)
                    assign[s] = g
                elif active:
                    # force-assign to nearest known global
                    best_g = min(active, key=lambda g: cosd(e[s], centers[g]))
                    assign[s] = best_g
            # update centroids for matched long speakers (running sum)
            for s in long_local:
                if s in assign:
                    centers[assign[s]] += e[s].astype(np.float64)

        # rebuild this sub-window's frame -> global label
        frame_label = np.full(SEG_FRAMES, -1, dtype=np.int32)
        for t in range(SEG_FRAMES):
            if frame_act[t] == 0:
                continue
            # single-label: first active local speaker
            for s in range(N_LOCAL):
                if m[t, s] >= 0.5 and s in assign:
                    frame_label[t] = assign[s]
                    break
        timeline.append(frame_label)
        centers_log.append(len(active))
    return timeline, centers_log


def frame_time(c, t, step_sec):
    return c * step_sec + 0.5 * FRAME_DUR + t * FRAME_STEP


def timeline_to_rttm(timeline, step_sec):
    """Collapse per-frame labels into (start, dur, speaker) runs."""
    runs = []
    for c, fl in enumerate(timeline):
        t = 0
        while t < SEG_FRAMES:
            if fl[t] < 0:
                t += 1
                continue
            spk = fl[t]
            t1 = t
            while t1 < SEG_FRAMES and fl[t1] == spk:
                t1 += 1
            start = frame_time(c, t, step_sec)
            end = frame_time(c, t1 - 1, step_sec) + FRAME_STEP
            runs.append((start, max(0.0, end - start), spk))
            t = t1
    return runs


def load_ref(path):
    import re
    LINE = re.compile(r'^(\S+):\s*\[([\d.]+)\s+([\d.]+)\]\s*(.*)$')
    rows = []
    for l in Path(path).read_text().splitlines():
        m = LINE.match(l.strip())
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return rows


def speaker_conf(runs, ref, tolerance=0.25):
    """Confusion: fraction of reference speech assigned to the 'wrong' speaker.

    Uses a per-reference-segment winner-take-all overlap: each reference
    segment is labeled by whichever clustered speaker overlaps it most; conf
    counts reference speech whose label differs from the reference speaker
    id, after mapping the two label sets by total overlap.
    """
    # runs are (start, dur, spk); normalize to (start, end, spk)
    run_iv = [(s, s + d, int(spk)) for (s, d, spk) in runs]
    # map clustered speaker ids -> reference speaker ids by total overlap
    spk_overlap = {}
    for (rs, rb, re_) in ref:
        for (cb, ce, cs) in run_iv:
            ov = min(re_, ce) - max(rb, cb)
            if ov <= 0:
                continue
            spk_overlap.setdefault(rs, {}).setdefault(cs, 0.0)
            spk_overlap[rs][cs] += ov
    # winner mapping: each clustered id -> reference id it overlaps most
    clust_to_ref = {}
    for rs, d in spk_overlap.items():
        for cs, ov in d.items():
            if ov > clust_to_ref.get(cs, (0.0, None))[0]:
                clust_to_ref[cs] = (ov, rs)
    total = wrong = 0.0
    for (rs, rb, re_) in ref:
        dur = re_ - rb
        if dur <= 0:
            continue
        total += dur
        # which clustered speaker covers this segment most?
        best_cs, best_ov = None, 0.0
        for (cb, ce, cs) in run_iv:
            ov = min(re_, ce) - max(rb, cb)
            if ov > best_ov:
                best_ov, best_cs = ov, cs
        if best_cs is None:
            total -= dur  # uncovered -> missed, not conf; exclude from total
            continue
        mapped_ref = clust_to_ref.get(best_cs, (0.0, None))[1]
        if mapped_ref is not None and mapped_ref != rs:
            wrong += best_ov
    return wrong / total if total > 0 else 0.0, total


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else '/tmp/opencode/2p-feat-step10.bin'
    ref_path = sys.argv[2] if len(sys.argv) > 2 else (
        '/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/'
        'asr-llm-tts-chain/build/3dspk/out/label-2P-v8-recheck.txt')
    step_sec = 10.0

    C, step_samples, s0s, masks, embs = parse_dump(dump)
    print(f'C={C} step_samples={step_samples} '
          f'audio={C * step_samples / SAMPLE_RATE:.1f}s')

    ref = load_ref(ref_path)
    print(f'ref segments={len(ref)} speakers={sorted(set(r for r, _, _ in ref))}')

    for delta_new in (0.5, 0.75, 1.0, 1.25, 1.5):
        timeline, centers_log = incremental_cluster(
            masks, embs, tau_active=0.6, rho_update=0.3,
            delta_new=delta_new, max_speakers=20)
        runs = timeline_to_rttm(timeline, step_sec)
        # first label time
        first = None
        for c, fl in enumerate(timeline):
            if (fl >= 0).any():
                t = int(np.argmax(fl >= 0))
                first = frame_time(c, t, step_sec)
                break
        nspk = max((int(fl.max()) for fl in timeline), default=-1) + 1
        conf, total = speaker_conf(runs, ref)
        final_nspk = centers_log[-1]
        print(f'delta_new={delta_new:4.2f}  first_label={first:7.2f}s  '
              f'final_speakers={final_nspk:2d}  distinct_labels={nspk:2d}  '
              f'conf={conf:6.2%}  ref_speech={total:7.1f}s')


if __name__ == '__main__':
    main()

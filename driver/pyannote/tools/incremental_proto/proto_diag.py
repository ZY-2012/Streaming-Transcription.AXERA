#!/usr/bin/env python3
"""Deep diagnostics for the incremental-clustering prototype on 2P."""
import sys
from pathlib import Path
import numpy as np

from proto_incremental import (parse_dump, incremental_cluster, timeline_to_rttm,
                               load_ref, frame_time, SEG_FRAMES, FRAME_STEP,
                               FRAME_DUR)

DUMP = '/tmp/opencode/2p-feat-step10.bin'
REF = ('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/'
       'asr-llm-tts-chain/build/3dspk/out/label-2P-v8-recheck.txt')
STEP = 10.0

C, ss, s0s, masks, embs = parse_dump(DUMP)
ref = load_ref(REF)

# ---- speaker-discovery history (first 30 sub-windows = 0-300s) ----
timeline, centers_log = incremental_cluster(masks, embs, tau_active=0.6,
                                            rho_update=0.3, delta_new=1.0,
                                            max_speakers=20)
print('=== speaker discovery (per 10s sub-window) ===')
for c in range(0, min(40, C)):
    fl = timeline[c]
    n_local = int((masks[c].sum(axis=1) > 0).any(axis=0).sum())
    labels = sorted(set(int(x) for x in fl[fl >= 0]))
    # reference speakers active in this window
    wb, we = c * STEP, (c + 1) * STEP
    ref_spk = sorted(set(rs for (rs, rb, re_) in ref if re_ > wb and rb < we))
    print(f'  win{c:3d} [{wb:6.1f}-{we:6.1f}s] known={centers_log[c]:2d} '
          f'labels={labels} ref_active={ref_spk}')

# ---- early region: first 60s frame labels vs reference speaker ----
print('\n=== early 0-60s: incremental label vs reference ===')
# build frame->ref-speaker over 0-60s
def ref_speaker_at(t):
    for (rs, rb, re_) in ref:
        if rb <= t < re_:
            return rs
    return None
for c in range(6):
    fl = timeline[c]
    for t in range(0, SEG_FRAMES, 30):
        tt = frame_time(c, t, STEP)
        if tt > 60:
            break
        lab = int(fl[t])
        r = ref_speaker_at(tt)
        print(f'  t={tt:6.2f}s  incr_label={lab if lab >= 0 else "-"}  '
              f'ref={r or "-"}')

# ---- full DER-ish accounting (coverage / miss / fa / conf) ----
print('\n=== coverage / miss / fa / conf vs reference ===')
runs = timeline_to_rttm(timeline, STEP)
run_iv = [(s, s + d, int(spk)) for (s, d, spk) in runs]

# reference coverage by clustered speech (any speaker)
ref_total = sum(re_ - rb for (_, rb, re_) in ref)
covered = 0.0
for (_, rb, re_) in ref:
    for (cb, ce, _) in run_iv:
        ov = min(re_, ce) - max(rb, cb)
        if ov > 0:
            covered += ov
            # count overlap once per ref segment (break at first? no: sum is fine
            # only if runs are disjoint; they are per sub-window but windows are
            # non-overlapping so disjoint overall)
miss = ref_total - covered
clust_total = sum(ce - cb for (cb, ce, _) in run_iv)
fa = clust_total - covered
print(f'  ref_speech   = {ref_total:8.1f}s')
print(f'  covered      = {covered:8.1f}s  ({covered/ref_total:6.2%})')
print(f'  miss         = {miss:8.1f}s  ({miss/ref_total:6.2%})')
print(f'  clust_speech = {clust_total:8.1f}s')
print(f'  fa           = {fa:8.1f}s  ({fa/ref_total:6.2%})')

# conf with fixed identity mapping (first-seen order -> Speaker_0, Speaker_1)
print('\n=== conf with fixed (first-seen) identity mapping ===')
# determine first-seen order of global speakers
first_seen = {}
for c, fl in enumerate(timeline):
    for x in fl:
        if x >= 0 and int(x) not in first_seen:
            first_seen[int(x)] = frame_time(c, int(np.where(fl == x)[0][0]), STEP)
print('  first-seen:', {k: f'{v:.2f}s' for k, v in sorted(first_seen.items())})
# map global speaker k -> reference speaker by majority overlap (already done)
from proto_incremental import speaker_conf
conf, tot = speaker_conf(runs, ref)
print(f'  winner-take-all conf = {conf:6.2%}  over {tot:.1f}s')

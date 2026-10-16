#!/usr/bin/env python3
"""Fair head-to-head: incremental clustering vs the existing 120s-window batch
diarization, using the SAME overlap-matrix metric as report-2P-online-vad30.txt.

Existing batch result (from report-2P-online-vad30.txt):
    speaker-1(428s) -> Speaker_0:379s  Speaker_1:3s
    speaker-2(936s) -> Speaker_1:747s  Speaker_0:42s
    agreement 82.5%, misassign 17.5%
"""
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

# reference speaker speech time
ref_time = {}
for (rs, rb, re_) in ref:
    ref_time[rs] = ref_time.get(rs, 0.0) + (re_ - rb)
print(f'v8 reference speech: { {k: f"{v:.0f}s" for k, v in ref_time.items()} }')

timeline, clog = incremental_cluster(masks, embs, tau_active=0.6, rho_update=0.3,
                                     delta_new=1.0, max_speakers=20)
runs = timeline_to_rttm(timeline, STEP)
run_iv = [(s, s + d, int(spk)) for (s, d, spk) in runs]

# clustered speaker total speech time
clust_time = {}
for (cb, ce, cs) in run_iv:
    clust_time[cs] = clust_time.get(cs, 0.0) + (ce - cb)
print(f'clustered speech: { {k: f"{v:.0f}s" for k, v in sorted(clust_time.items())} }')

# overlap matrix: clustered speaker x reference speaker
overlap = {}
for (cb, ce, cs) in run_iv:
    for (rs, rb, re_) in ref:
        ov = min(ce, re_) - max(cb, rb)
        if ov <= 0:
            continue
        overlap.setdefault(cs, {}).setdefault(rs, 0.0)
        overlap[cs][rs] += ov

print('\noverlap matrix (clustered label x reference speaker):')
refs = sorted(ref_time)
print('        ' + '  '.join(f'{r[:9]:>10}' for r in refs))
for cs in sorted(overlap):
    row = [overlap[cs].get(r, 0.0) for r in refs]
    total = clust_time.get(cs, 0.0)
    print(f'label{cs}  ' + '  '.join(f'{v:>10.0f}' for v in row) +
          f'   (total {total:.0f}s)')

# agreement = for each clustered label, its majority reference assignment,
# summed; misassign = the rest (including reference time not covered).
agree = 0.0
misassign = 0.0
for cs in sorted(overlap):
    total = clust_time.get(cs, 0.0)
    majority = max(overlap[cs].values()) if overlap[cs] else 0.0
    agree += majority
    misassign += (total - majority)
clust_total = sum(clust_time.values())
print(f'\nagreement = {agree:.0f}s / {clust_total:.0f}s = {agree/clust_total:.1%}')
print(f'misassign = {misassign:.0f}s / {clust_total:.0f}s = {misassign/clust_total:.1%}')
print(f'\n  (existing batch: agreement 82.5%, misassign 17.5%)')

# also report the small "explicit swap" (majority-diagonal violation)
print('\n=== per-label majority assignment ===')
for cs in sorted(overlap):
    maj_ref = max(overlap[cs], key=overlap[cs].get)
    print(f'  label{cs} -> {maj_ref}  (majority {overlap[cs][maj_ref]:.0f}s)')

#!/usr/bin/env python3
"""Emit and persist the best 2P incremental-clustering RTTM (raw cosine,
delta_new=0.55, rho_update=0.3), for comparison against v8-recheck."""
import re
import sys
from pathlib import Path

import numpy as np

from proto_incremental import parse_dump
from proto_2p_optimize import incr_cluster, tl_to_rttm, transcript_to_rttm, score

BASE = Path('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/asr-llm-tts-chain/build/3dspk/out')
DUMP = sys.argv[1] if len(sys.argv) > 1 else '/tmp/opencode/2p-feat-step10.bin'
STEP = 10.0 if 'step10' in DUMP else 2.0
OUT = sys.argv[2] if len(sys.argv) > 2 else '/tmp/opencode/best-2p-incremental.rttm'

C, _, _, masks, embs = parse_dump(DUMP)
ref = transcript_to_rttm(BASE / 'label-2P-v8-recheck.txt', '2P')

# sweep around the best region for this dump
best = None
for dn in (0.5, 0.55, 0.6, 0.7, 0.8, 1.0):
    for rho in (0.2, 0.3, 0.4, 0.5):
        tl, nspk = incr_cluster(masks, embs, 'raw', dn, rho, 0.6)
        if nspk != 2:  # 2P must be exactly 2 speakers
            continue
        rttm = tl_to_rttm(tl, '2P')
        r = score(rttm, ref)
        cand = (r[2], r[3], dn, rho, rttm, r)
        if best is None or cand[0] < best[0]:
            best = cand

conf, der, dn, rho, rttm, r = best
Path(OUT).write_text(rttm)
print(f'BEST 2P incremental (step={STEP}): raw cosine dn={dn} rho={rho}')
print(f'  miss={r[0]:.2%} fa={r[1]:.2%} conf={conf:.2%} DER={der:.2%}')
print(f'  RTTM -> {OUT}')

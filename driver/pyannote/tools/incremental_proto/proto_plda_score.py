#!/usr/bin/env python3
"""Full DER scoring of PLDA-projected incremental clustering vs raw-cosine,
on both 2P and T3, with the canonical score_der.py (collar 0.25)."""
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from proto_incremental import parse_dump, timeline_to_rttm
from proto_plda import load_plda, incremental_cluster_plda

SCORER = Path(__file__).with_name('score_der.py')
spec = importlib.util.spec_from_file_location('score_der', SCORER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BASE = Path('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/asr-llm-tts-chain/build')


def v8_to_rttm(txt_path, uri):
    LINE = re.compile(r'^(\S+):\s*\[([\d.]+)\s+([\d.]+)\]\s*(.*)$')
    lines = []
    for l in Path(txt_path).read_text().splitlines():
        mm = LINE.match(l.strip())
        if mm:
            spk = 'SPK' + mm.group(1).split('_')[-1]
            lines.append(f'SPEAKER {uri} 1 {mm.group(2)} '
                         f'{float(mm.group(3))-float(mm.group(2)):.3f} '
                         f'<NA> <NA> {spk} <NA> <NA>')
    return '\n'.join(lines) + '\n'


def score(hyp_text, ref_text, total_sec):
    nf = int(total_sec / m.RES) + 3000
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(ref_text); ref_name = t.name
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(hyp_text); hyp_name = t.name
    r, bd = m.load_rttm(Path(ref_name), nf)
    h, _ = m.load_rttm(Path(hyp_name), nf)
    d = m.der(r, h, bd, 0.25)
    if d is None:
        return None
    t = d['total']
    return (d['miss'] / t, d['fa'] / t, d['conf'] / t,
            (d['miss'] + d['fa'] + d['conf']) / t)


def runs_to_rttm(runs, uri):
    return '\n'.join(f'SPEAKER {uri} 1 {b:.3f} {d:.3f} <NA> <NA> '
                     f'speaker-{int(s)} <NA> <NA>' for b, d, s in runs if d > 0) + '\n'


# ---- 2P ----
C2, _, _, masks2, embs2 = parse_dump('/tmp/opencode/2p-feat-step10.bin')
ref2 = v8_to_rttm(BASE / '3dspk/out/label-2P-v8-recheck.txt', '2P')
P = load_plda()

print('=== 2P (2 speakers) ===')
for dn in (0.5, 0.6, 0.8, 1.0):
    tl, nspk = incremental_cluster_plda(masks2, embs2, P, delta_new=dn)
    runs = timeline_to_rttm(tl, 10.0)
    r = score(runs_to_rttm(runs, '2P'), ref2, 1960.0)
    if r:
        print(f'  PLDA dn={dn} (spk={nspk}): miss={r[0]:.2%} fa={r[1]:.2%} '
              f'conf={r[2]:.2%} DER={r[3]:.2%}')

# ---- T3 ----
C3, _, _, masks3, embs3 = parse_dump('/tmp/opencode/t3-feat-step10.bin')
FID = 'R8001_M8004_MS801'
ref3_path = BASE / 'eval-alimeeting/ref' / f'{FID}.rttm'
lines = []
for l in ref3_path.read_text().splitlines():
    if not l.strip():
        continue
    f = l.split()
    b, d = float(f[3]), float(f[4])
    if b >= 240:
        continue
    d = min(d, 240 - b)
    if d <= 0:
        continue
    lines.append(' '.join(f[:4] + [f'{d:.3f}'] + f[5:]))
ref3 = '\n'.join(lines) + '\n'

print('\n=== T3 (4 speakers) ===')
for dn in (0.4, 0.5, 0.6):
    tl, nspk = incremental_cluster_plda(masks3, embs3, P, delta_new=dn)
    runs = timeline_to_rttm(tl, 10.0)
    r = score(runs_to_rttm(runs, FID), ref3, 240.0)
    if r:
        print(f'  PLDA dn={dn} (spk={nspk}): miss={r[0]:.2%} fa={r[1]:.2%} '
              f'conf={r[2]:.2%} DER={r[3]:.2%}')

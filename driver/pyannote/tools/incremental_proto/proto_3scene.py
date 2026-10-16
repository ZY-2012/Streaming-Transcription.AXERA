#!/usr/bin/env python3
"""Three-scene conf at the shared delta_new=0.55 (PLDA projection)."""
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


def window_rttm(path, win):
    lines = []
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        f = l.split()
        b, d = float(f[3]), float(f[4])
        if b >= win:
            continue
        d = min(d, win - b)
        if d <= 0:
            continue
        lines.append(' '.join(f[:4] + [f'{d:.3f}'] + f[5:]))
    return '\n'.join(lines) + '\n'


def v8_to_rttm(txt, uri):
    LINE = re.compile(r'^(\S+):\s*\[([\d.]+)\s+([\d.]+)\]\s*(.*)$')
    out = []
    for l in Path(txt).read_text().splitlines():
        mm = LINE.match(l.strip())
        if mm:
            spk = 'SPK' + mm.group(1).split('_')[-1]
            out.append(f'SPEAKER {uri} 1 {mm.group(2)} '
                       f'{float(mm.group(3))-float(mm.group(2)):.3f} <NA> <NA> {spk} <NA> <NA>')
    return '\n'.join(out) + '\n'


def score(hyp_text, ref_text, total_sec):
    nf = int(total_sec / m.RES) + 3000
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(ref_text); rn = t.name
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(hyp_text); hn = t.name
    r, bd = m.load_rttm(Path(rn), nf)
    h, _ = m.load_rttm(Path(hn), nf)
    d = m.der(r, h, bd, 0.25)
    t = d['total']
    return d['miss']/t, d['fa']/t, d['conf']/t, (d['miss']+d['fa']+d['conf'])/t


def runs_rttm(runs, uri):
    return '\n'.join(f'SPEAKER {uri} 1 {b:.3f} {d:.3f} <NA> <NA> speaker-{int(s)} <NA> <NA>'
                     for b, d, s in runs if d > 0) + '\n'


P = load_plda()
DN = 0.55

# 2P
C, _, _, m2, e2 = parse_dump('/tmp/opencode/2p-feat-step10.bin')
ref2 = v8_to_rttm(BASE / '3dspk/out/label-2P-v8-recheck.txt', '2P')
tl, n2 = incremental_cluster_plda(m2, e2, P, delta_new=DN)
r2 = score(runs_rttm(timeline_to_rttm(tl, 10.0), '2P'), ref2, 1960.0)
print(f'2P      (want 2): spk={n2}  miss={r2[0]:.2%} fa={r2[1]:.2%} conf={r2[2]:.2%} DER={r2[3]:.2%}')

# T3
C, _, _, m3, e3 = parse_dump('/tmp/opencode/t3-feat-step10.bin')
ref3 = window_rttm(BASE / 'eval-alimeeting/ref/R8001_M8004_MS801.rttm', 240.0)
tl, n3 = incremental_cluster_plda(m3, e3, P, delta_new=DN)
r3 = score(runs_rttm(timeline_to_rttm(tl, 10.0), 'R8001_M8004_MS801'), ref3, 240.0)
print(f'T3      (want 4): spk={n3}  miss={r3[0]:.2%} fa={r3[1]:.2%} conf={r3[2]:.2%} DER={r3[3]:.2%}')

# EN2002c
C, _, _, mc, ec = parse_dump('/tmp/opencode/en2002c-feat.bin')
refc = window_rttm(BASE / 'eval-ami/ref_nxt/EN2002c.rttm', 240.0)
tl, nc = incremental_cluster_plda(mc, ec, P, delta_new=DN)
rc = score(runs_rttm(timeline_to_rttm(tl, 10.0), 'EN2002c'), refc, 240.0)
print(f'EN2002c (want 3): spk={nc}  miss={rc[0]:.2%} fa={rc[1]:.2%} conf={rc[2]:.2%} DER={rc[3]:.2%}')

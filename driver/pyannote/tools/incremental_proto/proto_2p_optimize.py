#!/usr/bin/env python3
"""2P optimization: systematic sweep of incremental-clustering config against
the v8-recheck ground truth, with online-vad30 as the batch baseline.

Metrics via canonical score_der.py (collar 0.25). Finds the config that makes
the 2P speaker timeline closest to v8-recheck.
"""
import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

from proto_incremental import parse_dump
from proto_plda import load_plda

SCORER = Path(__file__).with_name('score_der.py')
spec = importlib.util.spec_from_file_location('score_der', SCORER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BASE = Path('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/asr-llm-tts-chain/build/3dspk/out')
DUMP = '/tmp/opencode/2p-feat-step10.bin'
STEP = 10.0
TOTAL = 1960.0
P = load_plda()
mean1, mean2, lda, plda_mu, plda_tr, _ = P


def proj(e):
    tmp = e - mean1
    tmp = tmp / (np.linalg.norm(tmp) + 1e-30) * np.sqrt(256.0)
    t2 = lda.T @ tmp - mean2
    t2 = t2 / (np.linalg.norm(t2) + 1e-30) * np.sqrt(128.0) - plda_mu
    return plda_tr @ t2


def cosd(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def incr_cluster(masks, embs, metric, delta_new, rho_update, tau_active,
                 max_speakers=20):
    centers = []
    active = set()
    timeline = []

    def feat(e):
        return proj(e) if metric == 'plda' else e.astype(np.float64)

    for c in range(len(masks)):
        m = masks[c].astype(np.float32)
        al = [s for s in range(3) if m[:, s].max() >= tau_active]
        ll = [s for s in range(3) if m[:, s].mean() >= rho_update]
        pe = [feat(embs[c][s]) for s in range(3)]
        assign = {}
        if not centers:
            for s in al:
                centers.append(pe[s].astype(np.float64).copy())
                active.add(len(centers) - 1)
                assign[s] = len(centers) - 1
        else:
            for s in al:
                bg, bd = None, float('inf')
                for g in active:
                    d = cosd(pe[s], centers[g])
                    if d < bd:
                        bd, bg = d, g
                if bg is not None and bd <= delta_new:
                    assign[s] = bg
            for s in [x for x in al if x not in assign]:
                if len(active) < max_speakers and s in ll:
                    centers.append(pe[s].astype(np.float64).copy())
                    active.add(len(centers) - 1)
                    assign[s] = len(centers) - 1
                elif active:
                    assign[s] = min(active, key=lambda g: cosd(pe[s], centers[g]))
            for s in ll:
                if s in assign:
                    centers[assign[s]] += pe[s].astype(np.float64)
        fl = np.full(589, -1, dtype=np.int32)
        for t in range(589):
            if m[t].sum() == 0:
                continue
            for s in range(3):
                if m[t, s] >= 0.5 and s in assign:
                    fl[t] = assign[s]
                    break
        timeline.append(fl)
    return timeline, len(active)


def tl_to_rttm(tl, uri):
    runs = []
    for c, fl in enumerate(tl):
        t = 0
        while t < 589:
            if fl[t] < 0:
                t += 1
                continue
            spk = int(fl[t])
            t1 = t
            while t1 < 589 and fl[t1] == spk:
                t1 += 1
            start = c * STEP + 0.5 * 0.0619375 + t * 0.016875
            end = c * STEP + 0.5 * 0.0619375 + (t1 - 1) * 0.016875 + 0.016875
            runs.append((start, max(0.0, end - start), spk))
            t = t1
    return '\n'.join(f'SPEAKER {uri} 1 {b:.3f} {d:.3f} <NA> <NA> speaker-{spk} <NA> <NA>'
                     for b, d, spk in runs if d > 0) + '\n'


def transcript_to_rttm(path, uri):
    LINE = re.compile(r'^(\S+):\s*\[([\d.]+)\s+([\d.]+)\]\s*(.*)$')
    out = []
    for l in Path(path).read_text().splitlines():
        mm = LINE.match(l.strip())
        if mm:
            out.append(f'SPEAKER {uri} 1 {mm.group(2)} '
                       f'{float(mm.group(3))-float(mm.group(2)):.3f} <NA> <NA> {mm.group(1)} <NA> <NA>')
    return '\n'.join(out) + '\n'


def score(hyp_text, ref_text):
    nf = int(TOTAL / m.RES) + 3000
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(ref_text); rn = t.name
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(hyp_text); hn = t.name
    r, bd = m.load_rttm(Path(rn), nf)
    h, _ = m.load_rttm(Path(hn), nf)
    d = m.der(r, h, bd, 0.25)
    t = d['total']
    return d['miss']/t, d['fa']/t, d['conf']/t, (d['miss']+d['fa']+d['conf'])/t


def main():
    C, _, _, masks, embs = parse_dump(DUMP)
    ref = transcript_to_rttm(BASE / 'label-2P-v8-recheck.txt', '2P')

    # baselines
    base = transcript_to_rttm(BASE / 'label-2P-online-vad30.txt', '2P')
    b = score(base, ref)
    print(f'{"BASELINE online-vad30":34s} miss={b[0]:.2%} fa={b[1]:.2%} conf={b[2]:.2%} DER={b[3]:.2%}')

    print()
    print(f'{"config":40s} {"spk":>3} {"miss":>7} {"fa":>7} {"conf":>7} {"DER":>7}')
    results = []
    for metric in ('raw', 'plda'):
        for dn in (0.5, 0.55, 0.6, 0.7, 0.8, 1.0, 1.2):
            for rho in (0.2, 0.3, 0.5):
                tl, nspk = incr_cluster(masks, embs, metric, dn, rho, 0.6)
                r = score(tl_to_rttm(tl, '2P'), ref)
                tag = f'{metric} dn={dn} rho={rho}'
                results.append((r[2], r[3], nspk, tag, r))
                print(f'{tag:40s} {nspk:3d} {r[0]:7.2%} {r[1]:7.2%} {r[2]:7.2%} {r[3]:7.2%}')

    results.sort(key=lambda x: (x[0], x[1]))
    print()
    print('=== TOP 5 by conf ===')
    for conf, der, nspk, tag, r in results[:5]:
        print(f'  {tag:38s} spk={nspk} conf={conf:.2%} DER={der:.2%} '
              f'(miss={r[0]:.2%} fa={r[1]:.2%})')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""T3 (4-speaker Ali 240s) validation of diart-style incremental clustering.

Dumps the per-10s sub-window masks+embeddings, runs incremental clustering,
emits an RTTM, and scores it against the human reference with the canonical
pyannote score_der.py (collar 0.25), so the numbers are directly comparable to
the existing batch runs in score_t3o.py.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

from proto_incremental import (parse_dump, incremental_cluster,
                               timeline_to_rttm)

SCORER = Path(__file__).with_name('score_der.py')
spec = importlib.util.spec_from_file_location('score_der', SCORER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BASE = Path('/ai_data/AX615/hy/workspace/ASR-LLM-TTS-Pipeline/asr-llm-tts-chain/build')
FID = 'R8001_M8004_MS801'
REF = BASE / 'eval-alimeeting/ref' / f'{FID}.rttm'
WIN = 240.0
STEP = 10.0


def window_rttm(path, win=WIN):
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


def runs_to_rttm(runs, uri):
    lines = []
    for (b, d, spk) in runs:
        if d <= 0:
            continue
        lines.append(f'SPEAKER {uri} 1 {b:.3f} {d:.3f} <NA> <NA> '
                     f'speaker-{int(spk)} <NA> <NA>')
    return '\n'.join(lines) + '\n'


def score_rttm(rttm_text, label):
    nf = int(WIN / m.RES) + 3000
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(window_rttm(REF))
        ref_name = t.name
    with tempfile.NamedTemporaryFile('w', suffix='.rttm', delete=False) as t:
        t.write(rttm_text)
        hyp_name = t.name
    r, bd = m.load_rttm(Path(ref_name), nf)
    h, _ = m.load_rttm(Path(hyp_name), nf)
    d = m.der(r, h, bd, 0.25)
    if d is None:
        print(f'{label:34s} -- no overlap --')
        return
    t = d['total']
    spks = {l.split()[7] for l in rttm_text.splitlines() if l.strip()}
    print(f'{label:34s} DER={((d["miss"]+d["fa"]+d["conf"])/t):7.2%} '
          f'miss={d["miss"]/t:7.2%} fa={d["fa"]/t:7.2%} '
          f'conf={d["conf"]/t:7.2%} spk={len(spks)}')


def main():
    dump = sys.argv[1] if len(sys.argv) > 1 else '/tmp/opencode/t3-feat-step10.bin'
    C, ss, s0s, masks, embs = parse_dump(dump)
    print(f'T3 dump: C={C} audio={C * ss / 16000:.1f}s  ref={FID}')

    print(f'\ncollar=0.25  window=[0,{WIN}s]')
    print('=== existing batch baselines (from score_t3o.py) ===')
    for dn in (1.0, 0.9, 1.1, 1.2):
        timeline, clog = incremental_cluster(masks, embs, tau_active=0.6,
                                             rho_update=0.3, delta_new=dn,
                                             max_speakers=20)
        runs = timeline_to_rttm(timeline, STEP)
        nspk = clog[-1]
        # discovery history
        hist = []
        for i, k in enumerate(clog):
            if not hist or k != hist[-1]:
                hist.append((i * STEP, k))
        score_rttm(runs_to_rttm(runs, FID),
                   f'incremental delta_new={dn} (final spk={nspk})')
        print(f'      discovery: ' +
              ', '.join(f'{t:.0f}s:{k}' for t, k in hist[:12]))


if __name__ == '__main__':
    main()

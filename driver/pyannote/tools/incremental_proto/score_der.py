#!/usr/bin/env python3
"""帧级 DER 评分器（与 pyannote 口径校验一致，10 ms 网格 + 匈牙利最优映射）。

用法：
  python score_der.py --ref ref_dir --hyp hyp_dir [--collar 0.25] [--only a,b]

- ref_dir/hyp_dir 下按文件名配对 {id}.rttm；--collar 为总宽度（0.25 = ±0.125 s）
- 输出逐场 DER 与加权 DER（miss/FA/conf 分解）
"""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

RES = 0.01  # 评分网格 10 ms


def load_rttm(path, n_frames):
    spks, segs = {}, []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if 'SPEAKER' not in f:
            continue
        i = f.index('SPEAKER')
        st, du, spk = float(f[i + 3]), float(f[i + 4]), f[i + 7]
        if du <= 0:
            continue
        spks.setdefault(spk, len(spks))
        segs.append((st, st + du, spks[spk]))
    grid = np.zeros((n_frames, max(len(spks), 1)), dtype=bool)
    bounds = []
    for st, en, s in segs:
        a, b = int(round(st / RES)), min(int(round(en / RES)), n_frames)
        if b > a:
            grid[a:b, s] = True
            bounds += [a, b]
    return grid, np.array(bounds, dtype=int)


def der(ref, hyp, bounds, collar):
    n = min(len(ref), len(hyp))
    ref, hyp = ref[:n], hyp[:n]
    keep = np.ones(n, dtype=bool)
    if collar > 0 and len(bounds):
        half = int(round(collar / 2 / RES))
        for b in bounds:
            keep[max(0, b - half):min(n, b + half)] = False
    ref, hyp = ref[keep], hyp[keep]
    n_ref, n_hyp = ref.sum(1), hyp.sum(1)
    total = n_ref.sum()
    if total == 0:
        return None
    ov = ref.astype(np.int32).T @ hyp.astype(np.int32)
    r_idx, h_idx = linear_sum_assignment(-ov)
    correct = (ref[:, r_idx] & hyp[:, h_idx]).sum(1)
    miss = np.maximum(0, n_ref - n_hyp).sum()
    fa = np.maximum(0, n_hyp - n_ref).sum()
    conf = (np.minimum(n_ref, n_hyp) - correct).sum()
    return dict(total=float(total) * RES, miss=float(miss) * RES,
                fa=float(fa) * RES, conf=float(conf) * RES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--hyp', required=True)
    ap.add_argument('--collar', type=float, default=0.25)
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    only = {x for x in args.only.split(',') if x}
    agg = dict(total=0.0, miss=0.0, fa=0.0, conf=0.0)
    rows = []
    for ref_p in sorted(Path(args.ref).glob('*.rttm')):
        fid = ref_p.name[:-5]
        if only and fid not in only:
            continue
        hyp_p = Path(args.hyp) / f'{fid}.rttm'
        if not hyp_p.exists():
            print(f'skip {fid}: no hyp')
            continue
        span = max(float(l.split()[3]) + float(l.split()[4])
                   for l in ref_p.read_text().splitlines())
        nf = int(span / RES) + 3000
        r, bd = load_rttm(ref_p, nf)
        h, _ = load_rttm(hyp_p, nf)
        d = der(r, h, bd, args.collar)
        if d is None:
            continue
        for k in agg:
            agg[k] += d[k]
        rows.append((fid, (d['miss'] + d['fa'] + d['conf']) / d['total'],
                     d['miss'] / d['total'], d['fa'] / d['total'], d['conf'] / d['total']))
    t = agg['total']
    for fid, e, ms, fa, cf in rows:
        print(f'{fid:12s} DER {e:7.2%}  miss {ms:6.2%} FA {fa:6.2%} conf {cf:6.2%}')
    if t:
        print(f'\n加权 ({len(rows)} files, collar={args.collar}): '
              f'DER {(agg["miss"] + agg["fa"] + agg["conf"]) / t:6.2%}  '
              f'miss {agg["miss"] / t:5.2%} FA {agg["fa"] / t:5.2%} '
              f'conf {agg["conf"] / t:5.2%}')


if __name__ == '__main__':
    main()

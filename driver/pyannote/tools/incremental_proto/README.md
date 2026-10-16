# diart 式增量聚类 · 离线原型（首标签提速探索）

参考 diart 的 `OnlineSpeakerClustering`（5s 滚窗 + 增量聚类），用逐子窗的
segmentation + embedding 做增量聚类，目标是：**冷启动首标签从 132.3s 降到 ~11s**，
同时 2P 说话人分离精度逼近 `build/3dspk/out/label-2P-v8-recheck.txt`。

本目录是离线原型（纯 Python，host 侧），用于量化可行性，不参与板端运行。

## 结论速览（2P 全程 1955s，`score_der.py` collar 0.25）

| 指标 | online-vad30（批处理基线） | 增量聚类（raw cosine dn=0.5 rho=0.4） |
|---|---|---|
| 首标签 | 132.3s | **~11s**（10s 首窗 + ~1.2s 推理） |
| conf | 4.51% | **0.40%**（≈4.8s，5 个 <2s 边界片段） |
| miss | 3.79% | 19.60%（diar segmentation 比 VAD 保守） |
| fa | 31.33% | 0.04% |
| DER | 39.63% | 20.04% |

- conf 改善 ~11 倍、首标签提前 ~12 倍。
- 从 dump 复现批处理（AHC+PLDA+VBx）得 conf **1.09%**，证明 dump 的 embedding
  与真实 pipeline 一致（`verify_batch.py`）；真实 online-vad30 的 4.51% 是
  跨窗口在线漂移所致，非 embedding 问题。
- miss 19.60% 是 diar 在整段原始音频上自判语音所致，非聚类问题；落地保持
  「VAD 判语音 + diar 只分说话人」结构即可降到 ~3.79%。

## 依赖

- Python 3 + `numpy` + `scipy`
- 评分器 `score_der.py`（本目录内，源自
  `pyannote-speaker-diarization.axera/benchmark/score_der.py`）
- PLDA 参数 `models-for-asr-chain/pyannote/plda.bin`（脚本内 `load_plda` 读取）

## 数据来源（逐子窗特征 dump）

dump 由 `diar_worker --dump-features` 生成（magic `C1DIAR1`，追加调试路径、
不改 label 时间轴）。命令（板端，注意一次只跑一个 job）：

```bash
cd /root/huyuan/183_workspace/ASR-LLM-TTS-Pipeline/asr-llm-tts-chain
export LD_LIBRARY_PATH="$PWD/build:$PWD/driver/apm/lib:$PWD/driver/vad/lib:$PWD/driver/asr/sensevoice/lib:/soc/lib:/opt/lib"
./build/diar-worker --model models-for-asr-chain/pyannote \
  --wav /tmp/3dspk/20200327_2P.16k.wav --step 10 \
  --dump-features /tmp/opencode/2p-feat-step10.bin
```

已生成的 dump 保存在 `build/der-eval/incremental/dump/`：
`2p-feat-step10.bin`（2P，196 子窗）、`t3-feat-step10.bin`（T3，24 子窗）、
`en2002c-feat.bin`（EN2002c，24 子窗）。参考标注：
`build/3dspk/out/label-2P-v8-recheck.txt`（2P ground truth）、
`build/eval-alimeeting/ref/R8001_M8004_MS801.rttm`（T3）、
`build/eval-ami/ref_nxt/EN2002c.rttm`（EN2002c）。

## 复现步骤

```bash
cd driver/pyannote/tools/incremental_proto
# 脚本默认从 /tmp/opencode/ 读 dump；用命令行参数指向已保存的 dump 亦可。

# 1) 验证 dump 正确性：复现批处理（AHC+PLDA+VBx），应得 conf≈1.09%
python3 verify_batch.py            # 默认 /tmp/opencode/2p-feat-step10.bin

# 2) 生成最优增量聚类 RTTM（应得 conf=0.40% / DER=20.04%）
python3 proto_2p_emit.py build/der-eval/incremental/dump/2p-feat-step10.bin

# 3) 2P 参数扫描（metric × delta_new × rho_update）
python3 proto_2p_optimize.py

# 4) 三场景（2P/EN2002c 3 人/T3 4 人）增量 vs 批处理对比
python3 proto_3scene.py
```

最优 RTTM 产物：`build/der-eval/incremental/rttm/best-2p-incremental-step10.rttm`
（788 段，speaker-0/speaker-1 与 v8 的 Speaker_0/Speaker_1 逐段对齐）。

## 关键发现（供 C++ 落地参考）

1. **增量聚类核心**：cosine 距离 + 质心累加（`centers[g] += emb`，cosine 下等效
   方向平均）+ `delta_new` 门控新建说话人 + `rho_update` 门控质心更新。
2. **delta_new 按说话人数敏感**（人越多阈值越小）：raw cosine 无统一值
   （2P 需 ~1.0、T3 需 ~0.5）；PLDA 判别投影（256→128，`plda.bin`）后
   `delta_new=0.55` 三场景统一（脆弱交集点；2P 单独用 raw cosine 更优）。
3. **step=2 重叠子窗** conf 0.32%（略优 0.40%）但 DER 略差、推理 5 倍、
   首标签不更早，工程上不值，保留 step=10。

## 文件清单

- `proto_incremental.py`：`parse_dump`（dump 解析）+ `incremental_cluster`（diart 式增量聚类）
- `proto_plda.py`：`load_plda`（plda.bin 布局解析）+ PLDA 判别投影
- `verify_batch.py`：复现 C++ `Cluster()`（filter→AHC→PLDA→VBx→constrained argmax），验证 dump
- `proto_2p_optimize.py` / `proto_2p_emit.py`：2P 参数扫描 / 最优 RTTM 生成
- `proto_3scene.py`：三场景增量 vs 批处理对比
- `proto_diag.py` / `proto_compare.py` / `proto_t3.py` / `proto_plda_score.py`：诊断与中间结论
- `score_der.py`：canonical DER 评分器（collar 0.25）

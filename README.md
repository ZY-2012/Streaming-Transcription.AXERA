# Streaming-Transcription.AXERA

AX650N 上的**真流式**会议转录 demo：与兄弟工程
[Progressive-Transcription.AXERA](https://github.com/ZY-2012/Progressive-Transcription.AXERA)
（渐进式：先出流式字幕、说话人标签等整段聚类后对账，标签延迟几十秒）不同，
本工程用端到端流式说话人分离模型
[Nemotron-3-Diarization](https://huggingface.co/HY-2012/Nemotron-3-Diarization.AXERA)
（streaming Sortformer）持续增量地产出 10 ms 分辨率说话人时间线，
**说话人标签延迟 ~1 s**，轮次稳定即输出 `Speaker_k [起 止] 文本`。

- [x] 端到端流式说话人分离（增量 10 ms 帧、最多 8 人、支持重叠语音）
- [x] **流式 ASR 可选**（`asr.mode: streaming`）：连续喂入流式 SenseVoice，
      切分只发生在停顿/换人处（不再按 VAD 段硬切、无词中断字）；
      `offline` 保持原逐段识别路径
- [x] 轮次式真流式转录：时间线覆盖到哪、带说话人的句子就出到哪
- [x] WAV 文件模拟麦克风（file-realtime）或 ALSA 实时采集
- [x] 流式 worker 与文件模式逐位对齐（对分测试 + 板端 A/B 验证）

## 1. 支持平台

- AX650N

## 2. 与渐进式工程的差异

| 维度 | Progressive（渐进式） | 本工程（真流式） |
|------|----------------------|-----------------|
| 说话人分离 | pyannote community-1，分块提交，标签延迟几十秒 | Nemotron-3 streaming Sortformer，~1 s |
| 时间线 | 说话人时间线随聚类窗推进（step 可调，最快 ~6 s 量级） | 每 0.48 s 提交一个 chunk 的 10 ms 帧流 |
| 转录形态 | 流式字幕（无说话人）先出，finals 权威稿事后对账 | 单一连续时间线，轮次稳定即出带说话人的整句 |
| 说话人身份 | 声纹表跨会话匹配（speakers.json） | session 内固定 8 槽位（speaker_0..7），无跨会话声纹 |

音频前端（ALSA 采集 / WebRTC APM / FastEnhance NPU 降噪 / TEN-VAD / 48k→16k）与
SenseVoice ASR、MP3 归档、live-demo 演示页全部复用 Progressive 工程。

## 3. 目录结构

```
Streaming-Transcription.AXERA/
|-- config/
|   |-- streaming-meeting.yaml  # 真流式示例（60s 双人会议 WAV）
|   `-- default-config.yaml     # 默认配置（ALSA 麦克风输入）
|-- driver/
|   |-- nemotron/               # 新增：Nemotron 流式分离（上游 vendor + 流式 worker）
|   |   |-- src/stream_worker.cpp     # 常驻流式 worker（PCM-in / 增量事件-out）
|   |   |-- src/nemotron_ax.cpp       # 上游文件模式（板端 A/B 的对齐基准）
|   |   |-- include/chunk_layout.hpp   # 文件/流式共享的 chunk 布局（同源保证一致）
|   |   |-- include/streaming_frontend.hpp  # 滚动 mel 前端（内存有界、位级对齐）
|   |   `-- tools/              # 主机端对分测试（无需 NPU）
|-- pyannote/ alsa_briage/ apm/ ns/ vad/ asr/   # 复用 Progressive
|-- framework/
|   |-- audio_pipeline.cpp      # 主管线（新增 NemotronStreamDiarizer 接入）
|   `-- configui/               # 配置解析（新增 diarization.engine / nemotron 段）
|-- scripts/                    # build.sh / download_toolchains.sh / run-target.sh
|-- samples/                    # 2P_meeting_60s.wav + reference.rttm
`-- tools/live-demo/            # 网页实时演示（复用 Progressive）
```

## 4. 相关模型及下载地址

| 模块 | 模型 | 下载 |
|------|------|------|
| ASR | SenseVoice（离线+流式 NPU2） | [AXERA-TECH/models-for-asr-chain](https://huggingface.co/AXERA-TECH/models-for-asr-chain) → `asr/sensevoice`；流式版 `streaming_sensevoice.axmodel` 来自 [HY-2012/Tmp_Models](https://huggingface.co/HY-2012/Tmp_Models)（配套 `driver/asr/sensevoice/bin/asr_server`） |
| VAD | TEN-VAD | 同上 → `vad/ten-vad` |
| 降噪 | FastEnhance | 同上 → `ns/fastenhance` |
| 说话人分离 | Nemotron-3（端到端流式） | [HY-2012/Nemotron-3-Diarization.AXERA](https://huggingface.co/HY-2012/Nemotron-3-Diarization.AXERA) → `models/` |

```bash
pip install -U "huggingface_hub[cli]"

hf download AXERA-TECH/models-for-asr-chain \
    --include "asr/*" "vad/*" "ns/*" \
    --local-dir models-for-asr-chain

hf download HY-2012/Nemotron-3-Diarization.AXERA \
    --include "models/*" \
    --local-dir nemotron-tmp
mkdir -p models-for-asr-chain/nemotron
cp -r nemotron-tmp/models/. models-for-asr-chain/nemotron/
rm -rf nemotron-tmp
```

完成后应存在 `models-for-asr-chain/nemotron/{preencode_lowlat,encoder_lowlat,
preencode,encoder}.axmodel`、`learnable_sil_emb.npy`、`feature_window.npy`、
`mel_filterbank.npy`。

> **NPU 模式注意**：HF 当前发布的是 Pulsar2 `npu_mode: NPU3` 编译的模型，
> 只能在 VNPU `DISABLE` 模式加载；而主管线（SenseVoice/FastEnhance）要求
> `BIG_LITTLE(2+1)`。因此**完整链路联调需要 NPU2 版本的 Nemotron 模型**
> （后续上传）。在此之前：
> - 单机验证流式 worker：板子切到 `DISABLE` 模式，直接跑
>   `build/stream-diar-worker`（见 §6.1）；
> - 主管线 + pyannote 仍可用（`diarization.engine: pyannote`）。

## 5. 交叉编译

```bash
bash scripts/download_toolchains.sh   # 或复用本机已有工具链
export CXX=toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
export AX650_SDK_ROOT=toolchains/ax650n_bsp_sdk/msp/out
export ALSA_INCLUDE_DIR=toolchains/alsa
export ALSA_LIBRARY=<板端 libasound.so.2 拷出来的路径>
bash scripts/build.sh
```

产物在 `build/`：`audio-pipeline`（主管线）、`stream-diar-worker`（真流式分离
worker）、`nemotron_ax650`（文件模式，worker 的对齐基准）、`run-alsa-apm`
（音频前端）、`diar-worker`（pyannote 回退）、`libfastenhance.so`。

主机端对分测试（无需 NPU / 板子）：

```bash
bash driver/nemotron/build_host.sh
driver/nemotron/build_host/test_chunk_layout   # 流式 chunk 序列 == 文件模式
driver/nemotron/build_host/test_stream_mel      # 滚动 mel == 整文件 mel（位级）
```

## 6. 板端运行

### 6.1 单机验证流式 worker（VNPU DISABLE 模式）

先做逐位对齐（gold test）：

```bash
export LD_LIBRARY_PATH=/soc/lib:${LD_LIBRARY_PATH:-}
M=models-for-asr-chain/nemotron
# 文件模式 vs 流式 worker：同一 wav 的 preds 必须逐字节一致
./build/nemotron_ax650 --preencode $M/preencode_lowlat.axmodel \
    --encoder $M/encoder_lowlat.axmodel --sil-emb $M/learnable_sil_emb.npy \
    --assets $M --wav samples/2P_meeting_60s.wav \
    --chunk-len 6 --chunk-right-context 4 --fifo-len 264 \
    --spkcache-update-period 222 --dump-preds /tmp/file.bin
./build/stream-diar-worker --preencode $M/preencode_lowlat.axmodel \
    --encoder $M/encoder_lowlat.axmodel --sil-emb $M/learnable_sil_emb.npy \
    --assets $M --wav samples/2P_meeting_60s.wav \
    --dump-preds /tmp/stream.bin
cmp /tmp/file.bin /tmp/stream.bin && echo PARITY-OK
```

再看真实流式时延（按 wall clock 喂音频，观察 'S' 帧到达节奏）：

```bash
./build/stream-diar-worker --preencode $M/preencode_lowlat.axmodel \
    --encoder $M/encoder_lowlat.axmodel --sil-emb $M/learnable_sil_emb.npy \
    --assets $M --wav samples/2P_meeting_60s.wav \
    --realtime --rttm /tmp/stream.rttm
```

### 6.2 完整链路（NPU2 模型就绪后）

把整个工程目录（含 `build/`、`models-for-asr-chain/`）拷到板端，在**工程根目录**执行：

```bash
# WAV 文件模拟麦克风（file-realtime，与真实采集同一时序）
CONFIG=config/streaming-meeting.yaml ./scripts/run-target.sh

# ALSA 实时采集（按板子实际声卡改 config/default-config.yaml 的 audio.capture.device）
CONFIG=config/default-config.yaml ./scripts/run-target.sh
```

或启动网页实时演示（python3 标准库，无需 pip 安装）：

```bash
python3 tools/live-demo/server.py 8090
# 浏览器打开 http://<板端IP>:8090，点击 Start
```

关键配置（`config/*.yaml` 的 `diarization:` 段）：

```yaml
diarization:
  engine: nemotron          # 真流式；pyannote 回退到渐进式行为
  nemotron:
    preset: lowlat          # ~0.8s 输入缓冲；offline = 30.4s（精度优先）
    onset: 0.5              # 帧概率滞回阈值（NeMo 默认）
    offset: 0.5
```

## 7. 流式 worker 协议（`build/stream-diar-worker`）

```
stdin    : uint32 payload_bytes + 16 kHz mono S16LE PCM；0 长度帧 = 冲刷收尾
response : 帧 = type + uint32 len + payload（默认 fd 3）
           'R' "ready"                      模型加载完成后一次
           'S' uint32 start_frame, uint32 n_frames,
               n_frames * 8 float32 概率     每步新增的 10 ms 说话人活动
                                             （帧号 = 从流开始的 10 ms 序号）
           'D' 统计 JSON                      冲刷完成后一次
           'E' 错误文本                       致命错误
```

管线侧 `NemotronStreamDiarizer` 持续喂 PCM，把 'S' 帧折叠成全局说话人时间线
（与 pyannote session 时间线同接口），轮次被时间线覆盖即刻调 SenseVoice
出句——这就是"真流式"的端到端路径。

## 8. 流式 ASR（`asr.mode: streaming`）

文本层从"按 VAD 段切好再识别"改为**连续流式**：

- 16 kHz 音频逐 hop 喂给流式 SenseVoice 会话（26 帧滚动窗，NPU2 版
  `streaming_sensevoice.axmodel`），识别不再看任何被硬切的窗口——
  切分处的糊字/丢字问题从根上消除；
- **三线程结构保证文字实时跟上音频**：音频线程零 I/O（只喂会话/切
  epoch），轮询线程异步 POST 增长窗口取增量文本，收尾线程做最终识别
  + 说话人归属。板端实测字幕滞后 **~0.24 s**（全程平稳无累积），
  权威行滞后中位 **~1.2 s**（其中 ~0.5 s 是说话人时间线的固有滞后）；
- 停顿 ≥ `asr.streaming.epoch_pause_seconds`（默认 0.3 s，词安全）成段为
  一行；每 ~0.8 s 取增量文本，标点宽松 LocalAgreement-2 确认稳定前缀
  （`stext` 事件，带临时说话人）；
- 行闭合时用 Nemotron 时间线归属说话人（整段主导说话人，或
  `split_turns: true` 时在轮次边界切开逐段归属），写入
  `transcript.txt` 并发 `final` 事件；
- `asr.refine: epoch` 可选：行闭合时用离线模型再识别一遍作权威稿。

配置要点：

```yaml
asr:
  mode: streaming          # streaming | offline（默认保持离线路径）
  streaming:
    backend: http          # http = 板载 asr_server（随包附带）；worker = 常驻
                           # 流式 worker（协议 docs/stream-asr-worker-protocol.md，
                           # 待配套二进制，延迟更低）
    poll_seconds: 0.8      # 增量文本节拍
    epoch_pause_seconds: 0.3
    split_turns: false
```

> **VNPU 模式**：启用流式 ASR 后主管线以 **VNPU DISABLE（统一 NPU）** 优先
> （失败自动回退 BIG_LITTLE），因为 asr_server 只会以 DISABLE 初始化；
> 全链模型均为 NPU1/NPU2 编译、DISABLE 下全部可加载，任意启动顺序均共存。
> 两个后端二进制（asr_server / test_sensevoice）已静态链接 AX 运行时。

## 9. 结果与验证

- AX650N 文件模式（上游实测）：lowlat 档 C++ RTF 0.179、85.9 ms/步，
  AMI DER 29.08%（vs NeMo FP32 +0.80pp）；worker 与文件模式输出逐位一致。
- 主机端：`test_chunk_layout` / `test_stream_mel` 任意到达模式下与文件模式
  全部一致（保证流式重构无偏差）。
- 待板端实测（NPU2 模型到位后）：说话人标签端到端延迟、轮次出句延迟、
  NPU 占用、DER/轮次与 pyannote online 模式对比。

## 10. 参考

- [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)（NVIDIA Open Model License）
- [Nemotron-3-Diarization.AXERA](https://github.com/ZY-2012/Nemotron-3-Diarization.AXERA)
  （模型转换与推理源码；`driver/nemotron` 上游来源）
- [Progressive-Transcription.AXERA](https://github.com/ZY-2012/Progressive-Transcription.AXERA)
  （本工程音频链路的派生来源）
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)、[TEN-VAD](https://github.com/TEN-framework/ten-vad)

## 11. 技术讨论

- GitHub issues
- QQ 群: 139953715

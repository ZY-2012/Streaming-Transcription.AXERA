# Streaming-Transcription.AXERA

AX650N 上的**真流式**会议转录：说话的同时字幕逐字出现（滞后 ~0.1 s）、
说话人标签随时间线增量产出（~0.5 s）、每轮结束后输出带标点的权威定稿
（0.6–1.0 s）。端到端流式分离模型
[Nemotron-3-Diarization](https://huggingface.co/HY-2012/Nemotron-3-Diarization.AXERA)
+ 流式 SenseVoice，附网页实时演示，照本文步骤可直接复现。

## 1. 核心能力

- 连续流式 ASR：26 帧滚动窗口持续解码，不等句子说完，切分只发生在停顿处
- 端到端流式说话人分离：10 ms 帧分辨率、最多 8 人、支持重叠语音
- 轮次归属：换人边界处按字符—时间映射切分，各自标注说话人
- 定稿精修：轮次闭合后用离线模型重识别，补全标点、修正同音字
- 网页实时演示：字幕打字机、说话人时间线、支持上传自定义音频

## 2. 模块说明

| 模块 | 位置 | 说明 |
|------|------|------|
| 音频前端 | `driver/{alsa_briage,apm,ns,vad}` | ALSA 或 WAV 文件模拟麦克风 → WebRTC APM + FastEnhance 降噪 → TEN-VAD → 16 kHz hop 流 |
| 流式说话人分离 | `driver/nemotron` → `build/stream-diar-worker` | Nemotron Sortformer 常驻进程，每 10 ms 输出 8 槽位概率；管线按 `onset/offset`(0.5) 迟滞折叠成说话人时间线 |
| 流式 ASR | `driver/stream-asr-worker` → `build/stream-asr-worker` | 流式 SenseVoice 常驻进程：26 帧（1.56 s）窗口每 6 帧（360 ms）解码一次，partial 持续推送；协议见 `docs/stream-asr-worker-protocol.md` |
| 主管线与定稿 | `framework/audio_pipeline.cpp` → `build/audio-pipeline` | 三线程结构：音频线程零 I/O 喂音；轮询线程消费 partial 出字幕；收尾线程 flush 定稿、离线精修、说话人归属 |
| 网页演示 | `tools/live-demo/` | python3 标准库 HTTP + SSE，页面播放音频并与事件流同步，端到端延迟即肉眼可见 |

**转写流程**：停顿 ≥ 0.3 s（或满 12 s）时一轮闭合 → 同步 flush 拿到权威
文本 → 离线模型整段重识别（精修）→ 按分离时间线归属说话人 → 写入
`transcript.txt` 并推送 `final` 事件。行首弱起音（"嗯"等）由预起振环形
缓冲回补，避免吞字。

## 3. 目录结构

```
Streaming-Transcription.AXERA/
|-- config/                   # streaming-meeting.yaml（演示）/ default-config.yaml（ALSA）
|-- driver/
|   |-- nemotron/             # 流式分离 worker 与模型推理源码
|   |-- stream-asr-worker/    # 流式 ASR worker（C++，从零实现）
|   `-- asr/ns/vad/apm/...    # ASR 运行库、降噪、VAD、音频前端
|-- framework/                # 主管线 + 配置解析（text_agreement.hpp 文本对齐/切分工具）
|-- docs/                      # worker 协议文档
|-- scripts/                   # build.sh / download_toolchains.sh / run-target.sh
|-- samples/                   # 演示音频（16 kHz vad_example.wav、8 kHz 20200327_2P.wav）
`-- tools/live-demo/           # 网页演示（server.py + index.html）
```

## 4. 模型下载

| 模块 | 模型 | 下载 |
|------|------|------|
| ASR | SenseVoice（离线） | [AXERA-TECH/models-for-asr-chain](https://huggingface.co/AXERA-TECH/models-for-asr-chain) → `asr/sensevoice` |
| ASR | SenseVoice（流式） | [HY-2012/Tmp_Models](https://huggingface.co/HY-2012/Tmp_Models) → `streaming_sensevoice.axmodel`（放入 `asr/sensevoice/`） |
| VAD / 降噪 | TEN-VAD / FastEnhance | 同 models-for-asr-chain → `vad/` `ns/` |
| 说话人分离 | Nemotron-3（NPU2） | [HY-2012/Nemotron-3-Diarization.AXERA](https://huggingface.co/HY-2012/Nemotron-3-Diarization.AXERA) |

```bash
pip install -U "huggingface_hub[cli]"

hf download AXERA-TECH/models-for-asr-chain \
    --include "asr/*" "vad/*" "ns/*" \
    --local-dir models-for-asr-chain
# 再把 streaming_sensevoice.axmodel 放入 models-for-asr-chain/asr/sensevoice/

hf download HY-2012/Nemotron-3-Diarization.AXERA \
    --include "models/*" \
    --local-dir nemotron-tmp
mkdir -p models-for-asr-chain/nemotron
cp -r nemotron-tmp/models/. models-for-asr-chain/nemotron/
rm -rf nemotron-tmp
```

## 5. 交叉编译

```bash
bash scripts/download_toolchains.sh        # 或复用本机工具链
export CXX=toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++
export AX650_SDK_ROOT=toolchains/ax650n_bsp_sdk/msp/out
bash scripts/build.sh
```

产物在 `build/`：`audio-pipeline`（主管线）、`stream-asr-worker`、
`stream-diar-worker`（流式分离）、`nemotron_ax650`（文件模式对齐基准）。

## 6. 复现网页结果

把整个工程目录（含 `build/` 与 `models-for-asr-chain/`）放到板端（NFS
挂载或拷贝均可），在**工程根目录**执行：

```bash
python3 tools/live-demo/server.py 8090
```

浏览器打开 `http://<板端IP>:8090`：

1. 顶部选择输入：`vad_example.wav`（70 s，16 kHz）或 `20200327_2P.wav`
   （33 min，8 kHz），也可上传自定义音频（16 kHz 单声道 WAV）；
2. 点击**开始**——页面同步播放音频，字幕逐字滚动、说话人时间线随音频
   前沿推进，每轮结束落一行带说话人的定稿；
3. 播放结束自动收尾；也可随时点击停止。

结果文件在 `runs/`：

| 文件 | 内容 |
|------|------|
| `transcript.txt` | 定稿文本（`speaker_k [起 止] 文本` 逐行） |
| `events.jsonl` | 全部事件流（`stext` 字幕 / `final` 定稿 / `diar` 时间线），含精确时间戳 |
| `run.log` | 运行日志（`[asr-final]` 延迟分解、`[asr-split]` 切分明细） |

命令行直跑（无网页）：`CONFIG=config/streaming-meeting.yaml ./scripts/run-target.sh`。

## 7. 关键配置（`config/streaming-meeting.yaml`）

```yaml
diarization:
  engine: nemotron
  nemotron:
    preset: lowlat          # ~0.8 s 输入缓冲（offline = 30.4 s，精度优先）
    onset: 0.5              # 帧概率迟回阈值；短应答/重叠语音标为 "?" 即源于此
    offset: 0.5
asr:
  mode: streaming
  refine: epoch             # 轮次闭合后离线模型重识别作权威稿
  streaming:
    backend: worker          # 常驻流式 worker；http = 板载 asr_server（延迟较高）
    epoch_pause_seconds: 0.3 # 停顿成行阈值
    epoch_cap_seconds: 12    # 单轮音频上限
    split_turns: true        # 换人边界切开逐段归属
    turn_shift_seconds: 0.55 # 分离边界前移补偿（标签滞后 ~0.5 s）
```

## 8. 预期指标（板端实测，vad_example / 2P 开头）

| 指标 | 数值 |
|------|------|
| 字幕滞后（音频到出字） | ~0.1 s 中位 |
| 定稿滞后（轮次闭合到落行） | 0.6–1.0 s |
| 说话人标签滞后 | ~0.5 s |
| "?" 的含义 | 分离模型无法判定（短应答、轻声、重叠语音） |

## 9. 常见问题

- **启动失败 / NPU 报错**：板端残留进程占用 NPU，`pkill -f audio-pipeline`
  等清理后重试；演示页的"开始"会自动替换上一轮运行。
- **多模型共存**：管线优先以 VNPU `DISABLE` 模式初始化（失败自动回退
  `BIG_LITTLE`），流式 ASR、离线精修、分离、降噪模型可同时驻留。
- **上传音频**：需 16 kHz 单声道 WAV（页面端自动转码），1 MB–300 MB。
- **安全提示**：演示页无鉴权，仅限可信局域网使用。

## 10. 参考

- [Nemotron-3-Diarization.AXERA](https://github.com/ZY-2012/Nemotron-3-Diarization.AXERA)（模型转换与推理源码）
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) / [TEN-VAD](https://github.com/TEN-framework/ten-vad)
- [Progressive-Transcription.AXERA](https://github.com/ZY-2012/Progressive-Transcription.AXERA)（音频链路的派生来源）
- 流式 worker 协议：`docs/stream-asr-worker-protocol.md`
- 技术讨论：GitHub issues / QQ 群 139953715

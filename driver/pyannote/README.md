# pyannote community-1 说话人分离（板端 C++）

本目录是 [pyannote-speaker-diarization.axera](https://github.com/ZY-2012/pyannote-speaker-diarization.axera)
（MIT，见 `LICENSE`）的板端推理源码裁剪版，用于 `build/diar-worker`：

- `src/diar_engine.*`：社区模型流水线（sincnet 分割、4 层 BiLSTM、嵌入、PLDA/VBx 聚类）。
  本地改动：`Run()` 额外输出非重叠样本区间（ASR 切片用）和每簇 256 维聚类中心。
- `src/EngineWrapper_axera.cpp`：AX Engine 封装。本地改动：NPU 模式自动回退
  （先按 VNPU DISABLE 初始化，失败再按 BIG_LITTLE），诊断输出到 stderr；
  句柄仍尝试绑定 affinity 0x1。
- `src/speaker_table.*`：跨段说话人表，JSON 格式与旧 CampPlus 表兼容
  （`speaker_id`/`sample_count`/`last_seen_time`/`embedding_center`），
  256 维中心余弦匹配；每个说话人同时保留最近 8 个分段质心并取最大余弦，
  `speaker-N` 编号跨段保持稳定。默认匹配阈值 0.55（多人长会议建议 0.5–0.65 间微调）。
  `AssignOnline()` 提供 diart 式在线聚类决策层：块内全部簇与说话人表做
  匈牙利整体最优分配（带哑列补齐），新说话人创建/质心更新带语音量门控
  （对应 diart 的 rho_update/delta_new 思想），说话人总数封顶。
- `src/diar_worker.cpp`：常驻 worker。协议（stdin 收请求长度+WAV 路径，
  fd 3 回 `R`/`E` 帧；启动回 `R ready`）。
  也支持单机调试：`diar-worker --model DIR --wav in.wav [--out out.rttm]`。
  `--assign greedy|online` 选择决策层（默认 greedy，即原逐簇贪心，segment/session
  模式行为不变）；online 模式额外参数：`--min-new-sec`（新说话人语音门控，默认 2.0）、
  `--update-min-sec`（质心更新门控，默认 1.0）、`--max-speakers`（封顶，默认 20）。
  逐段回退请求（pipeline 命名的 `*.raw.wav`）在 online 模式下只识别不建新、
  不更新质心（混合质心无出生权）。
- `tools/label_transcript.cpp`：**独立评测工具** `build/label-transcript`，
  输入一段 WAV，输出带秒级时间戳与说话人标签的可读转录：

  ```text
  Speaker_0: [0.031 9.700] 设定一下嗯比较适合的年龄段啊。...
  Speaker_1: [9.700 12.029] 感觉可以说是有一种。
  ```

  行为：fork `diar-worker`（整文件一次推理）拿到说话人时间轴与重叠 RTTM，
  与 RTTM 活动区间求交后合并/拆分轮次（默认 0.5 s 合并），逐轮调用 SenseVoice
  （`AX_ASR_RunFile`，BIG_LITTLE 模式）；ASR 切片在 turn 首尾外扩 `--pad` 秒
  （默认 0.3，仅钳制到音频范围；**不要**加"不跨相邻说话人"钳制——实测该钳制
  会把 R8003 的 equal 从 79.3% 拉低到 77.3%，因为邻说话人方向的 0.3 s 上下文
  更多时候是本句的字头而非串音）；
  可选 `--chunk-search`/`--inner-pad`/`--dedupe` 启用静音切点与接缝处理；
  可选 `--rttm out.rttm` 输出同一套 `Speaker_k` 标签用于 DER 评分。

  编译（开发机交叉）：

  ```bash
  aarch64-linux-gnu-g++ -std=c++17 -O2 -I driver/asr/sensevoice/include \
    driver/pyannote/tools/label_transcript.cpp \
    -ldl -pthread -Wl,--export-dynamic -o build/label-transcript
  ```

  板端运行（16 kHz 单声道 WAV）：

  ```bash
  export LD_LIBRARY_PATH="$PWD/build:$PWD/driver/apm/lib:$PWD/driver/vad/lib:$PWD/driver/asr/sensevoice/lib:/soc/lib"
  export AX_SENSEVOICE_SHARE_DIR="$PWD/driver/asr/sensevoice"
  ./build/label-transcript --wav in.wav \
    --model /tmp/diar-models-npu2 \
    --asr-model models-for-asr-chain/asr --language zh \
    --out out.txt --rttm out.rttm
  ```

  主要参数：`--model`（pyannote 模型目录）、`--threshold`（跨窗身份阈值，默认 0.55）、
  `--max-chunk`（ASR 单块上限秒数，默认 30）、`--merge-gap`（同说话人合并间隔，默认 0.5）、
  `--pad`（turn 首尾外扩秒数，默认 0.3，不跨相邻说话人）、
  `--chunk-search`（>0 时在到达上限前的该秒数窗口内寻找静音切点，默认 0 = 固定切块。
  实测 R8003 GT：固定切块 equal 79.3% / delete 12.4%；静音切 77.4% / 13.5%。
  静音切可修复超长 turn 尾部被 30 s 切口截断的个别句子，按需开启）、
  `--inner-pad`/`--dedupe`（静音切模式下块间左上下文 0/1 与接缝去重 0/1，默认 0/0；
  注意去重在硬切模式也会误删真实接缝重复，默认关闭）、
  `--language`（auto/zh/en/yue/ja/ko）。
- `third_party/kissfft`：fbank 所需的 FFT 实现。

## NPU 模式要求（重要）

社区模型的 axmodel 由 Pulsar2 以 `npu_mode` 编译，模式必须与 AX650 设备当前的
VNPU 模式一致：

| 模型编译模式 | `AX_ENGINE_GetHandleModelType` | 可在哪些设备模式加载 |
| --- | --- | --- |
| NPU1（如原 CampPlus） | 0 | DISABLE(0) / BIG_LITTLE(2) / STD(1) |
| NPU2（如链路 SenseVoice/说话人分离模型） | 1 | DISABLE(0) / BIG_LITTLE(2) |
| NPU3（当前 `_hf.axera` 包） | 2 | 仅 DISABLE(0) |

链路必须运行在 BIG_LITTLE：`driver/ns` 的 FastEnhance 会主动校验该模式，
SenseVoice 与说话人分离模型也按 NPU2 分区编译。AX650 的 VNPU 模式是设备级、跨进程
互斥：链路运行在 BIG_LITTLE 时，另一个进程无法再以 DISABLE 初始化
（`AX_ENGINE_Init` 返回 `0x80060087`），反之亦然。

因此随链路部署的 pyannote 模型必须使用 `npu_mode: "NPU2"` 重新编译
（改动 `model_convert/pulsar2_config_*.json` 后重新执行 Pulsar2 量化编译流程）。
`models-for-asr-chain/pyannote/` 部署的就是 NPU2 版本（来源：
`pyannote-speaker-diarization_hf.axera/models_npu2/`）；原始 NPU3 模型保留在同级
`models/` 目录，只能在单机 VNPU-DISABLE 模式下运行，用于精度/性能对照。
`diar-worker` 已实现模式自动回退（先 DISABLE 再 BIG_LITTLE），模型换成 NPU2 后无需
改代码即可随链路运行。

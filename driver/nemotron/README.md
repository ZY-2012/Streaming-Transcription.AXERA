# cpp/ — Nemotron-3-Diarization on AX650N

C++ 板端推理，wav → RTTM。推理逻辑与 `../python/nemotron_sdk` 逐项对齐（同两张
拆图、同样的打包、同样的 10 ms 高分辨率切片 + 80 ms 降采样状态机、同样的后处理）。

## 本工程的本地新增（Streaming-Transcription.AXERA）

在上游文件模式之外，本工程为真流式转录新增/改造了以下部分（其余为上游原样）：

| 文件 | 说明 |
|---|---|
| `src/stream_worker.cpp` | **新增**：常驻流式 worker。stdin 收帧式 16 kHz PCM，`StreamingState` 常驻，每 ~0.48 s 吐一步新增的 10 ms 说话人概率帧（协议见文件头注释）；`--wav` 单机调试模式经与文件模式完全相同的流式路径，`--dump-preds` 可与 `nemotron_ax650` 逐字节比对，`--realtime` 按墙钟喂音可实测标签延迟。 |
| `include/chunk_layout.hpp` | **新增**：文件模式与流式共享同一份 chunk 布局定义（`BuildChunkLayout` / `NextChunk`），保证两条路径的 chunk 边界、左右上下文完全一致。 |
| `include/streaming_frontend.hpp` | **新增**：滚动预加重样本缓冲 + 滚动 mel 帧缓冲（绝对帧号索引，内存有界）；帧只依赖 `[f*160-200, f*160+200)` 样本，裁剪后仍与整文件计算位级一致。 |
| `src/mel_frontend.cpp` | **改造**：抽出 `compute_range_view()`（外部滚动缓冲视图），原 `compute_range()` 委托给它，文件模式行为不变。 |
| `src/nemotron_diarizer.cpp` | **改造**：chunk 布局改用共享的 `chunk_layout.hpp`（语义不变）。 |
| `tools/test_chunk_layout.cpp` | **新增**：主机端测试——任意 PCM 到达模式 + 冲刷尾部产生的 chunk 序列与文件模式完全一致。 |
| `tools/test_stream_mel.cpp` | **新增**：主机端测试——滚动 mel（含裁剪）与整文件 mel 位级一致。 |

## 目录

| 文件 | 作用 |
|---|---|
| `nemotron_ax.cpp` | 主程序（命令行解析 + 流程编排） |
| `src/audio_reader.cpp` | WAV 读取 + 重采样到 16 kHz |
| `src/npy_reader.cpp` | 读取检查点导出的 `.npy` 资产（窗/滤波器组/静音嵌入） |
| `src/mel_frontend.cpp` | NeMo 兼容 log-mel（KissFFT，128 mel，pad_to 16），多线程 FFT |
| `src/ax_engine_runner.cpp` | AX Engine 命名多路 IO 封装（双模型共享 init/deinit 引用计数） |
| `src/nemotron_state.cpp` | FIFO/spkcache 状态机（learnable 静音嵌入 / top-k + AOSC 压缩） |
| `src/nemotron_postprocess.cpp` | preds → RTTM（10 ms 帧长） |
| `src/nemotron_diarizer.cpp` | 流式循环 + 生产者/消费者流水线（`--threads`） |
| `tools/check_mel.cpp` | 主机端 mel 对分（`--assets <dir>` 复用检查点窗/滤波器组） |
| `tools/test_state.cpp` | 主机端单步状态机对分（可选 `sil_emb.npy` 参数） |
| `tools/test_compress.cpp` | 主机端 spkcache 压缩对分（可选 `sil_emb.npy` 参数） |
| `third_party/kissfft/` | FFT（随仓提交） |

## 工具链

不随仓提交，一键下载（aarch64 gcc 9.2 + AX650N BSP SDK）：

```bash
bash download_toolchains.sh      # -> cpp/toolchains/
```

已有本地副本时用环境变量覆盖默认路径：

```bash
TOOLCHAIN_ROOT=/path/to/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu \
BSP_MSP_DIR=/path/to/ax650n_bsp_sdk/msp/out \
bash build_ax650.sh
```

## 编译

```bash
bash build_host.sh      # 主机端对分工具 -> build_host/
bash build_ax650.sh     # 交叉编译      -> bin/nemotron_ax650, bin/bench_encoder
```

预编译可执行文件可参考 HuggingFace 的 `bin/nemotron_ax650`。

## 运行（板端）

```bash
export LD_LIBRARY_PATH=/soc/lib:$LD_LIBRARY_PATH
./bin/nemotron_ax650 \
    --preencode models/preencode.axmodel \
    --encoder   models/encoder.axmodel \
    --sil-emb   models/learnable_sil_emb.npy \
    --assets    models \
    --wav meeting.wav --rttm out.rttm --threads 8

# 低延迟档（~0.8 s 输入缓冲）：
./bin/nemotron_ax650 --preencode models/preencode_lowlat.axmodel \
    --encoder models/encoder_lowlat.axmodel --sil-emb models/learnable_sil_emb.npy \
    --assets models --wav meeting.wav --rttm out.rttm --threads 8 \
    --chunk-len 6 --chunk-right-context 4 --fifo-len 264 --spkcache-update-period 222
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--preencode` / `--encoder` | 必填 | 两张 axmodel |
| `--wav` / `--rttm` | 必填 / 可选 | 输入 / 输出 |
| `--sil-emb` | 建议 | AOSC 压缩用 learnable silence embedding（缺失会退化为运行均值并警告） |
| `--assets` | 可选 | `feature_window.npy` / `mel_filterbank.npy` 所在目录（必须复用检查点内量化值） |
| `--dump-preds` | 可选 | 导出帧级概率（float32, T×8，10 ms/帧）用于对分 |
| `--threads` | 8 | `1`=串行；`>1` 时并行 mel FFT + preencode 与 encoder 流水线重叠 |

## 主机对分（C++ vs numpy）

```bash
python ../model_convert/state_math_parity.py      # 200 步状态机：全部一致
python ../model_convert/compress_math_parity.py   # 首次压缩：选帧零差异
../cpp/build_host/check_mel --wav meeting.wav --out mel.bin --assets ../export
```

## 性能（AX650N 实测）

| 实现 | 配置 | RTF | ms/步 |
|---|---|---|---|
| Python（axengine） | offline | 0.0096 | 258.6 |
| **C++ `--threads 8`** | **offline** | **0.0066** | **177~180** |
| Python（axengine） | low latency | 0.189 | 90.6 |
| **C++ `--threads 8`** | **low latency** | **0.179** | **85.9** |

ES2004a（1049 s）：C++ offline 总耗时 6.97 s（其中 NPU encoder 6.76 s、mel 2.99 s
与 NPU 重叠），编译期 `max_cycle = 168.9 M`（≈211 ms @800 MHz）与实测吻合。

优化：mel 前端按 chunk 增量计算并与上一 chunk 的 encoder 推理并行；`seq` 直接写入
axengine 输入缓冲（去掉中间拷贝）。C++ preds 与 Python SDK 逐帧 cosine 0.99991。

## 精度（collar=0，pyannote.metrics，本地重构参考）

| 数据集 | NeMo FP32 | 主机 ONNX | Python 板端 | C++ 板端 | C++ ΔDER |
|---|---|---|---|---|---|
| AMI-SDM 5 场 | 29.82% | 29.83% | 29.99% | **29.97%** | **+0.14pp** |
| AliMeeting 4 场（远场） | 17.92% | 17.93% | 18.23% | **18.19%** | **+0.26pp** |

（C++ 与 Python 的 DER 在 ±0.05pp 内波动，属状态机浮点噪声。）

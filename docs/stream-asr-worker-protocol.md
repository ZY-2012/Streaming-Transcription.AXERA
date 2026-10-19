# stream-asr-worker 协议（流式 SenseVoice 常驻 worker）

`asr.streaming.backend: worker` 时，主管线 fork 常驻 worker 并按本协议交换数据。
协议与 `stream-diar-worker` 同构（帧式 PCM 进 / 帧式事件出），worker 内部持有
流式 SenseVoice 模型（`streaming_sensevoice.axmodel`，26 帧滚动窗口）与单一识别
会话（`AX_ASR_StreamInit/Feed/Result/Finish`）。

## 启动

```
stream-asr-worker --model-dir <dir>     # 读取 <dir>/sensevoice/{streaming_}sensevoice.axmodel + assets
                 [--language zh]        # auto/zh/en/yue/ja/ko
                 [--threads N]
                 [--response-fd 3]
```

模型加载完成后回一个 `'R'` 帧（payload = `ready`）。加载失败向 response fd 回
`'E'` 帧并退出。

## 请求（stdin，帧式）

```
uint32 payload_bytes + payload（16 kHz 单声道 S16LE PCM）
payload_bytes == 0  →  冲刷收尾（StreamFinish）
```

建议每 >= 100 ms 音频写一帧（管线侧按 hop 到达自动聚合）。

## 响应（默认 fd 3，帧式：type + uint32 len + payload）

| type | payload | 说明 |
|---|---|---|
| `'R'` | `ready` | 就绪（每次会话重置后同样回一帧） |
| `'T'` | `uint32 flags` + UTF-8 文本 | `flags bit0`：partial——**当前会话从开始到已喂入位置的累计识别文本**（文本有变化才发）；`flags bit1`：final——收尾后（`StreamFinish`）的最终文本，发完后 worker 自动 `StreamInit` 开始新会话 |
| `'E'` | 错误文本 | 致命错误，worker 退出 |

## 语义要点

1. **partial 是累计文本**（非增量），调用方用 LocalAgreement-2 取稳定前缀；
   文本回退（个别字符修正）是允许的，由调用方处理。
2. `payload_bytes == 0` 之后不要再写 PCM，直到收到 final `'T'`；
   随后即可直接喂下一个会话的音频（无需重新握手）。
3. worker 对 PCM 的消费速率即背压：喂入过快时 stdin pipe 写满阻塞，
   调用方据此自然限速。
4. NPU 归属：worker 必须以 **VNPU DISABLE 初始化失败时自动回退 BIG_LITTLE**（与
   `driver/nemotron` 的 `ax_engine_runner` 相同的策略），从而与主管线（DISABLE
   优先、BIG_LITTLE 回退）在任意启动顺序下共存；BIG_LITTLE 下建议 affinity 0x1。
   与管线其它 NPU2 使用方的协调由调用方控制喂入节奏实现。

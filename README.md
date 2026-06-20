# silero-vad-go

Go 语言实现的 Silero VAD 语音活动检测，基于 ONNX Runtime 推理。

将 Python [silero-vad](https://github.com/snakers4/silero-vad) 的 `OnnxWrapper` 完整移植到 Go，支持实时流式语音唤醒检测。

## 特性

- 纯 Go 实现，无需 Python 环境
- 基于 ONNX Runtime 推理，性能接近原生
- 完整移植 Python `OnnxWrapper` 的状态管理（LSTM state + context）
- 实时流式检测，支持语音段自动分割
- Windows waveIn API 录音（跨平台录音可自行替换）

## 前置准备

### ONNX 模型

仓库已内置 `silero_vad.onnx` 模型文件，无需额外下载。

如需替换为其他版本，可从 [Releases](https://github.com/snakers4/silero-vad/releases) 下载覆盖。

### ONNX Runtime 库

仓库已内置 Windows 版 `onnxruntime.dll`。如需其他平台，从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载：

| 平台 | 文件 | 放置位置 |
|------|------|----------|
| Windows | `onnxruntime.dll` | 项目根目录或 PATH |
| Linux | `libonnxruntime.so` | 项目根目录或 ldconfig 路径 |

**版本要求**: ONNX Runtime >= 1.17.0

## 安装

```bash
go get github.com/moxin1044/silero-vad-go
```

## 快速开始

```go
package main

import (
    "fmt"
    "os/signal"
    "syscall"

    vad "github.com/moxin1044/silero-vad-go"
)

func main() {
    listener, err := vad.NewListener("silero_vad.onnx", "onnxruntime.dll")
    if err != nil {
        panic(err)
    }
    defer listener.Close()

    listener.SetThreshold(0.5)       // 语音检测阈值
    listener.SetSilenceDuration(0.8)  // 静音 0.8s 认为说话结束
    listener.SetMinSpeechDuration(0.3) // 最短 0.3s
    listener.SetMaxSpeechDuration(5.0) // 最长 5s

    listener.Start(func(audio []float32, duration float64) {
        fmt.Printf("[唤醒] 检测到语音! 时长=%.2fs\n", duration)
        // 接入 ASR 或其他逻辑
    })

    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    <-sig
    listener.Stop()
}
```

运行示例：

```bash
cd example
go run main.go
```

## API

### VAD 核心接口

```go
// 创建 VAD 实例
v, err := vad.New("silero_vad.onnx", "onnxruntime.dll")

// 对 512 样本的音频块预测语音概率 (0-1)
prob, err := v.Predict(audio []float32)

// 直接判断是否为语音
isSpeech, err := v.IsSpeech(audio []float32, threshold float32)

// 重置模型状态（开始新会话时调用）
v.ResetStates()

// 释放资源
v.Close()
```

### Listener 实时监听器

```go
listener, _ := vad.NewListener("silero_vad.onnx", "onnxruntime.dll")

// 配置参数
listener.SetThreshold(0.5)        // 语音概率阈值
listener.SetSilenceDuration(0.8)   // 静音多少秒后认为说话结束
listener.SetMinSpeechDuration(0.3) // 最短语音时长，短于此忽略
listener.SetMaxSpeechDuration(5.0)  // 最大语音时长，超过自动截断
listener.SetSaveDir("tmp")          // 语音片段保存目录

// 开始监听
listener.Start(func(audio []float32, duration float64) {
    // 检测到完整语音段时回调
})
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Threshold | 0.5 | 语音概率阈值，越高越严格 |
| SilenceDuration | 0.8s | 静音持续多久后认为说话结束 |
| MinSpeechDuration | 0.3s | 短于此长度的语音视为噪声忽略 |
| MaxSpeechDuration | 5.0s | 单次语音最大时长，超过自动截断 |

## 工作原理

1. **录音**: Windows waveIn API 以 16kHz 单声道 16-bit 采集音频
2. **分块**: 每 512 样本（32ms）为一个处理窗口
3. **推理**: 将 context(64) + audio(512) = 576 样本送入 ONNX 模型
4. **状态管理**: 维护 LSTM 隐状态 `[2, 1, 128]` 和 64 样本上下文
5. **语音段**: 检测到语音开始缓冲，静音超过阈值时输出完整段

## 许可证

MIT

## 致谢

- [Silero VAD](https://github.com/snakers4/silero-vad) - 原始模型
- [onnxruntime_go](https://github.com/yalue/onnxruntime_go) - Go ONNX Runtime 绑定

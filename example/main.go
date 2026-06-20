// 实时语音唤醒检测示例
// 用法: go run example/main.go
package main

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"

	vad "github.com/moxin1044/silero-vad-go"
)

func main() {
	// 确定可执行文件所在目录（用于查找模型和 DLL）
	exePath, _ := os.Executable()
	baseDir := filepath.Dir(exePath)
	// 兼容 go run 的情况（exe 在临时目录）
	if _, err := os.Stat(filepath.Join(baseDir, "silero_vad.onnx")); os.IsNotExist(err) {
		baseDir, _ = os.Getwd()
	}

	// 查找模型文件
	modelPath := filepath.Join(baseDir, "silero_vad.onnx")
	if _, err := os.Stat(modelPath); err != nil {
		fmt.Println("未找到 silero_vad.onnx 模型文件")
		fmt.Println()
		fmt.Println("获取方式（任选其一）：")
		fmt.Println("  1. pip install silero_vad && 复制 silero_vad/data/silero_vad.onnx")
		fmt.Println("  2. 从 https://github.com/snakers4/silero-vad/releases 下载")
		os.Exit(1)
	}

	// ONNX Runtime 库路径（显式指定，避免加载到系统旧版本）
	var onnxLibPath string
	switch runtime.GOOS {
	case "windows":
		onnxLibPath = filepath.Join(baseDir, "onnxruntime.dll")
		if _, err := os.Stat(onnxLibPath); err != nil {
			onnxLibPath = "" // 回退到系统默认搜索
		}
	case "linux":
		onnxLibPath = filepath.Join(baseDir, "libonnxruntime.so")
		if _, err := os.Stat(onnxLibPath); err != nil {
			onnxLibPath = ""
		}
	default:
		onnxLibPath = ""
	}

	fmt.Println("[*] 初始化语音唤醒检测器...")

	// 创建监听器
	listener, err := vad.NewListener(modelPath, onnxLibPath)
	if err != nil {
		fmt.Printf("[!] 初始化失败: %v\n", err)
		os.Exit(1)
	}
	defer listener.Close()

	// 配置参数（可选，使用默认值即可）
	listener.SetThreshold(0.5)         // 语音检测阈值
	listener.SetSilenceDuration(0.8)   // 静音 0.8s 认为说话结束
	listener.SetMinSpeechDuration(0.3) // 最短 0.3s
	listener.SetMaxSpeechDuration(5.0) // 最长 5s
	listener.SetSaveDir("tmp")         // 保存到 tmp 目录

	// 语音回调函数
	onSpeech := func(audio []float32, duration float64) {
		fmt.Printf("[唤醒] 检测到语音! 时长=%.2fs, 样本=%d\n", duration, len(audio))
		// 这里可以接入 ASR、触发其他逻辑等
	}

	// 开始监听
	if err := listener.Start(onSpeech); err != nil {
		fmt.Printf("[!] 启动失败: %v\n", err)
		os.Exit(1)
	}

	// 等待 Ctrl+C 信号
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	// 停止
	listener.Stop()
	fmt.Println("[*] 已退出")
}

// Package vad 基于 Silero VAD ONNX 模型的语音活动检测
// 对应 Python silero_vad 库的 OnnxWrapper
package vad

import (
	"fmt"
	"sync"

	"github.com/yalue/onnxruntime_go"
)

const (
	// SampleRate 采样率（16kHz）
	SampleRate = 16000
	// WindowSize 每次处理的样本数（silero VAD v4 要求 512）
	WindowSize = 512
	// ContextSize 上下文样本数（silero VAD 要求 64）
	ContextSize = 64
	// StateSize LSTM 状态大小（2 层 × 128 维）
	StateSize = 128
	// InputSize 实际送入模型的样本数 = ContextSize + WindowSize
	InputSize = ContextSize + WindowSize // 576
)

// VAD Silero VAD 语音活动检测器
type VAD struct {
	mu      sync.Mutex
	session *onnxruntime_go.DynamicAdvancedSession

	// 模型状态（对应 Python OnnxWrapper 的 _state 和 _context）
	state   []float32 // shape: [2, 1, 128], 展平为 256
	context []float32 // shape: [64], 上一帧的最后 64 个样本

	// 预分配的输入输出缓冲区
	inputData  []float32 // [576]
	stateData  []float32 // [256]
	srData     []int64   // [1] = 16000
	outputData []float32 // [1]
	outState   []float32 // [256]

	// 张量
	inputTensor  *onnxruntime_go.Tensor[float32]
	stateTensor  *onnxruntime_go.Tensor[float32]
	srTensor     *onnxruntime_go.Tensor[int64]
	outputTensor *onnxruntime_go.Tensor[float32]
	outStateT    *onnxruntime_go.Tensor[float32]
}

// New 创建 VAD 检测器
// modelPath: silero_vad.onnx 文件路径
// onnxLibPath: onnxruntime.dll (Windows) / libonnxruntime.so (Linux) 路径，
//              传空字符串则使用系统默认搜索路径
func New(modelPath, onnxLibPath string) (*VAD, error) {
	// 初始化 ONNX Runtime 共享库
	if onnxLibPath != "" {
		onnxruntime_go.SetSharedLibraryPath(onnxLibPath)
	}
	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("初始化 ONNX Runtime 环境失败: %w", err)
	}

	v := &VAD{
		state:       make([]float32, 2*1*StateSize), // [2, 1, 128] = 256
		context:     make([]float32, ContextSize),  // [64]
		inputData:   make([]float32, InputSize),     // [576]
		stateData:   make([]float32, 2*1*StateSize),
		srData:      []int64{SampleRate},
		outputData:  make([]float32, 1),
		outState:    make([]float32, 2*1*StateSize),
	}

	var err error

	// 创建输入张量
	v.inputTensor, err = onnxruntime_go.NewTensor([]int64{1, InputSize}, v.inputData)
	if err != nil {
		return nil, fmt.Errorf("创建 input 张量失败: %w", err)
	}
	v.stateTensor, err = onnxruntime_go.NewTensor([]int64{2, 1, StateSize}, v.stateData)
	if err != nil {
		return nil, fmt.Errorf("创建 state 张量失败: %w", err)
	}
	v.srTensor, err = onnxruntime_go.NewTensor([]int64{1}, v.srData)
	if err != nil {
		return nil, fmt.Errorf("创建 sr 张量失败: %w", err)
	}

	// 创建输出张量
	v.outputTensor, err = onnxruntime_go.NewTensor([]int64{1, 1}, v.outputData)
	if err != nil {
		return nil, fmt.Errorf("创建 output 张量失败: %w", err)
	}
	v.outStateT, err = onnxruntime_go.NewTensor([]int64{2, 1, StateSize}, v.outState)
	if err != nil {
		return nil, fmt.Errorf("创建 outState 张量失败: %w", err)
	}

	// 创建动态会话（支持混合类型输入：float32 + int64）
	v.session, err = onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		nil, // 默认选项
	)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}

	v.ResetStates()
	return v, nil
}

// ResetStates 重置模型状态（对应 Python reset_states）
func (v *VAD) ResetStates() {
	v.mu.Lock()
	defer v.mu.Unlock()

	for i := range v.state {
		v.state[i] = 0
	}
	for i := range v.context {
		v.context[i] = 0
	}
	copy(v.stateData, v.state)
}

// Predict 对一个 512 样本的音频块进行 VAD 预测
// audio: float32 数组，长度必须为 512，值范围 [-1, 1]
// 返回：语音概率 (0-1)
func (v *VAD) Predict(audio []float32) (float32, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if len(audio) != WindowSize {
		return 0, fmt.Errorf("音频块长度必须为 %d, 实际 %d", WindowSize, len(audio))
	}

	// 拼接 context (64) + audio (512) = 576
	copy(v.inputData[:ContextSize], v.context)
	copy(v.inputData[ContextSize:], audio)

	// 同步 state 到 stateData
	copy(v.stateData, v.state)

	// 执行推理
	inputs := []onnxruntime_go.Value{v.inputTensor, v.stateTensor, v.srTensor}
	outputs := []onnxruntime_go.Value{v.outputTensor, v.outStateT}
	if err := v.session.Run(inputs, outputs); err != nil {
		return 0, fmt.Errorf("ONNX 推理失败: %w", err)
	}

	// 读取输出
	prob := v.outputData[0]

	// 更新状态
	copy(v.state, v.outState)

	// 更新 context：取当前输入的最后 64 个样本
	copy(v.context, v.inputData[WindowSize:])

	return prob, nil
}

// IsSpeech 判断当前音频块是否为语音
// threshold: 语音概率阈值（推荐 0.5）
func (v *VAD) IsSpeech(audio []float32, threshold float32) (bool, error) {
	prob, err := v.Predict(audio)
	if err != nil {
		return false, err
	}
	return prob >= threshold, nil
}

// Close 释放资源
func (v *VAD) Close() {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.session != nil {
		v.session.Destroy()
		v.session = nil
	}
	if v.inputTensor != nil {
		v.inputTensor.Destroy()
		v.inputTensor = nil
	}
	if v.stateTensor != nil {
		v.stateTensor.Destroy()
		v.stateTensor = nil
	}
	if v.srTensor != nil {
		v.srTensor.Destroy()
		v.srTensor = nil
	}
	if v.outputTensor != nil {
		v.outputTensor.Destroy()
		v.outputTensor = nil
	}
	if v.outStateT != nil {
		v.outStateT.Destroy()
		v.outStateT = nil
	}
}

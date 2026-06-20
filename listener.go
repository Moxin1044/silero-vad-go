// Package vad - 实时语音唤醒监听器
// 使用 Windows waveIn API 流式录音 + Silero VAD 检测
package vad

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// Windows waveIn API 相关
var (
	winmm               = syscall.NewLazyDLL("winmm.dll")
	waveInOpenProc      = winmm.NewProc("waveInOpen")
	waveInPrepareProc   = winmm.NewProc("waveInPrepareHeader")
	waveInUnprepareProc = winmm.NewProc("waveInUnprepareHeader")
	waveInAddBufferProc = winmm.NewProc("waveInAddBuffer")
	waveInStartProc     = winmm.NewProc("waveInStart")
	waveInStopProc      = winmm.NewProc("waveInStop")
	waveInResetProc     = winmm.NewProc("waveInReset")
	waveInCloseProc     = winmm.NewProc("waveInClose")
)

const (
	waveFormatPCM   = 1
	mmsyserrNoError = 0
	whdrDone        = 1
	whdrPrepared    = 2
	waveMapper      = 0xFFFFFFFF
)

type waveFormatEx struct {
	wFormatTag      uint16
	nChannels       uint16
	nSamplesPerSec  uint32
	nAvgBytesPerSec uint32
	nBlockAlign     uint16
	wBitsPerSample  uint16
	cbSize          uint16
}

type waveHdr struct {
	lpData          *byte
	dwBufferLength  uint32
	dwBytesRecorded uint32
	dwUser          uintptr
	dwFlags         uint32
	dwLoops         uint32
	lpNext          uintptr
	reserved        uintptr
}

// SpeechCallback 语音段回调函数
// audio: 语音片段的 float32 数据 [-1, 1]
// duration: 语音时长（秒）
type SpeechCallback func(audio []float32, duration float64)

// Listener 实时语音唤醒监听器
type Listener struct {
	mu           sync.Mutex
	vad          *VAD
	threshold    float32
	silenceSec   float64
	minSpeechSec float64
	maxSpeechSec float64

	// 录音状态
	hwaveIn    uintptr
	headers    []waveHdr
	buffers    [][]byte
	bufferSize int // 每个缓冲区的字节数
	numBuffers int
	running    bool
	stopCh     chan struct{}

	// VAD 状态
	isSpeaking    bool
	speechBuffer  []float32
	silenceFrames int
	speechStart   time.Time
	sampleCounter int

	// 预缓冲环形队列：保存最近 preBufferSec 秒的音频
	// 检测到语音开始时回填，避免第一个字被截断
	preBuffer    [][]float32
	preBufferCap int // 环形队列容量（块数）
	preBufferLen int // 当前已填充块数
	preBufferIdx int // 下一个写入位置

	// 保存目录
	saveDir  string
	callback SpeechCallback
}

// NewListener 创建监听器
// modelPath: silero_vad.onnx 路径
// onnxLibPath: onnxruntime.dll 路径
func NewListener(modelPath, onnxLibPath string) (*Listener, error) {
	v, err := New(modelPath, onnxLibPath)
	if err != nil {
		return nil, err
	}

	return &Listener{
		vad:          v,
		threshold:    0.5,
		silenceSec:   0.8,
		minSpeechSec: 0.3,
		maxSpeechSec: 5.0,
		bufferSize:   WindowSize * 2, // 512 samples × 2 bytes = 1024 bytes
		numBuffers:   4,
		stopCh:       make(chan struct{}),
		saveDir:      "tmp",
		// 预缓冲 0.5 秒：0.5 * 16000 / 512 ≈ 16 块
		preBufferCap: 16,
		preBuffer:    make([][]float32, 16),
	}, nil
}

// SetThreshold 设置语音检测阈值（默认 0.5）
func (l *Listener) SetThreshold(t float32) { l.threshold = t }

// SetSilenceDuration 设置静音结束时长（秒，默认 0.8）
func (l *Listener) SetSilenceDuration(sec float64) { l.silenceSec = sec }

// SetMinSpeechDuration 设置最短语音时长（秒，默认 0.3）
func (l *Listener) SetMinSpeechDuration(sec float64) { l.minSpeechSec = sec }

// SetMaxSpeechDuration 设置最大语音时长（秒，默认 5.0）
func (l *Listener) SetMaxSpeechDuration(sec float64) { l.maxSpeechSec = sec }

// SetSaveDir 设置语音片段保存目录
func (l *Listener) SetSaveDir(dir string) { l.saveDir = dir }

// Start 开始持续监听
// callback: 检测到语音段时的回调函数
func (l *Listener) Start(callback SpeechCallback) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.running {
		return fmt.Errorf("已在监听中")
	}

	l.callback = callback
	l.running = true
	l.stopCh = make(chan struct{})

	// 初始化录音
	if err := l.initRecording(); err != nil {
		l.running = false
		return err
	}

	// 启动采集循环
	go l.captureLoop()

	fmt.Printf("[VAD] 开始监听 (采样率=%d, 窗口=%d样本, 阈值=%.2f)\n",
		SampleRate, WindowSize, l.threshold)
	fmt.Printf("[VAD] 静音结束=%.1fs, 最短语音=%.1fs, 最长语音=%.1fs\n",
		l.silenceSec, l.minSpeechSec, l.maxSpeechSec)
	fmt.Println("[VAD] 按 Ctrl+C 停止")

	return nil
}

// Stop 停止监听
func (l *Listener) Stop() {
	l.mu.Lock()
	defer l.mu.Unlock()

	if !l.running {
		return
	}
	l.running = false
	close(l.stopCh)

	// 停止录音
	waveInStopProc.Call(l.hwaveIn)
	waveInResetProc.Call(l.hwaveIn)
	time.Sleep(50 * time.Millisecond)

	// 释放缓冲区
	for i := range l.headers {
		if l.headers[i].dwFlags&whdrPrepared != 0 {
			waveInUnprepareProc.Call(l.hwaveIn,
				uintptr(unsafe.Pointer(&l.headers[i])),
				uintptr(unsafe.Sizeof(l.headers[i])))
		}
	}
	waveInCloseProc.Call(l.hwaveIn)
	l.hwaveIn = 0

	// 处理残留语音
	if l.isSpeaking && len(l.speechBuffer) > 0 {
		l.finalizeSpeech()
	}

	fmt.Println("[VAD] 已停止监听")
}

// initRecording 初始化 waveIn 录音
func (l *Listener) initRecording() error {
	format := waveFormatEx{
		wFormatTag:      waveFormatPCM,
		nChannels:       1,
		nSamplesPerSec:  SampleRate,
		nAvgBytesPerSec: SampleRate * 2,
		nBlockAlign:     2,
		wBitsPerSample:  16,
	}

	// 打开录音设备
	ret, _, _ := waveInOpenProc.Call(
		uintptr(unsafe.Pointer(&l.hwaveIn)),
		uintptr(waveMapper),
		uintptr(unsafe.Pointer(&format)),
		0, 0, 0,
	)
	if ret != mmsyserrNoError {
		return fmt.Errorf("打开录音设备失败: %d", ret)
	}

	// 初始化缓冲区
	l.headers = make([]waveHdr, l.numBuffers)
	l.buffers = make([][]byte, l.numBuffers)
	for i := 0; i < l.numBuffers; i++ {
		l.buffers[i] = make([]byte, l.bufferSize)
		l.headers[i].lpData = &l.buffers[i][0]
		l.headers[i].dwBufferLength = uint32(l.bufferSize)

		ret, _, _ = waveInPrepareProc.Call(l.hwaveIn,
			uintptr(unsafe.Pointer(&l.headers[i])),
			uintptr(unsafe.Sizeof(l.headers[i])))
		if ret != mmsyserrNoError {
			waveInCloseProc.Call(l.hwaveIn)
			return fmt.Errorf("准备缓冲区失败: %d", ret)
		}

		ret, _, _ = waveInAddBufferProc.Call(l.hwaveIn,
			uintptr(unsafe.Pointer(&l.headers[i])),
			uintptr(unsafe.Sizeof(l.headers[i])))
		if ret != mmsyserrNoError {
			waveInCloseProc.Call(l.hwaveIn)
			return fmt.Errorf("添加缓冲区失败: %d", ret)
		}
	}

	// 开始录音
	ret, _, _ = waveInStartProc.Call(l.hwaveIn)
	if ret != mmsyserrNoError {
		waveInCloseProc.Call(l.hwaveIn)
		return fmt.Errorf("开始录音失败: %d", ret)
	}

	return nil
}

// captureLoop 采集循环
func (l *Listener) captureLoop() {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-l.stopCh:
			return
		case <-ticker.C:
			if !l.running {
				return
			}
			l.processBuffers()
		}
	}
}

// processBuffers 处理已完成的缓冲区
func (l *Listener) processBuffers() {
	for i := range l.headers {
		if l.headers[i].dwFlags&whdrDone == 0 {
			continue
		}

		n := int(l.headers[i].dwBytesRecorded)
		if n > 0 && n <= l.bufferSize {
			// 将 16-bit PCM 转换为 float32 并送入 VAD
			pcmData := l.buffers[i][:n]
			samples := pcm16ToFloat32(pcmData)
			l.processSamples(samples)
		}

		// 回收缓冲区
		waveInUnprepareProc.Call(l.hwaveIn,
			uintptr(unsafe.Pointer(&l.headers[i])),
			uintptr(unsafe.Sizeof(l.headers[i])))

		l.headers[i].dwBytesRecorded = 0
		l.headers[i].dwFlags = 0
		l.headers[i].dwUser = 0
		l.headers[i].dwLoops = 0
		l.headers[i].lpNext = 0
		l.headers[i].reserved = 0

		ret, _, _ := waveInPrepareProc.Call(l.hwaveIn,
			uintptr(unsafe.Pointer(&l.headers[i])),
			uintptr(unsafe.Sizeof(l.headers[i])))
		if ret != mmsyserrNoError {
			fmt.Printf("[VAD] 重新 prepare 缓冲区失败: %d\n", ret)
			continue
		}

		ret, _, _ = waveInAddBufferProc.Call(l.hwaveIn,
			uintptr(unsafe.Pointer(&l.headers[i])),
			uintptr(unsafe.Sizeof(l.headers[i])))
		if ret != mmsyserrNoError {
			fmt.Printf("[VAD] 重新 add 缓冲区失败: %d\n", ret)
		}
	}
}

// processSamples 处理音频样本（每次必须 512 样本）
func (l *Listener) processSamples(samples []float32) {
	for len(samples) >= WindowSize {
		chunk := samples[:WindowSize]
		samples = samples[WindowSize:]

		l.sampleCounter += WindowSize

		// VAD 检测
		prob, err := l.vad.Predict(chunk)
		if err != nil {
			fmt.Printf("[VAD] 推理失败: %v\n", err)
			continue
		}

		hasSpeech := prob >= l.threshold

		if hasSpeech {
			if !l.isSpeaking {
				// 语音开始：回填预缓冲区中最近 0.5 秒的音频
				l.isSpeaking = true
				l.speechStart = time.Now()
				l.speechBuffer = make([]float32, 0, WindowSize*40)
				l.silenceFrames = 0
				// 从环形队列按时间顺序取出已缓冲的音频块
				if l.preBufferLen > 0 {
					start := (l.preBufferIdx - l.preBufferLen + l.preBufferCap) % l.preBufferCap
					for i := 0; i < l.preBufferLen; i++ {
						idx := (start + i) % l.preBufferCap
						if l.preBuffer[idx] != nil {
							l.speechBuffer = append(l.speechBuffer, l.preBuffer[idx]...)
						}
					}
				}
				fmt.Printf("[VAD] 检测到人声开始 (prob=%.3f, 回填%.1fs) %s\n",
					prob, float64(l.preBufferLen*WindowSize)/float64(SampleRate),
					time.Now().Format("15:04:05"))
			}
			l.speechBuffer = append(l.speechBuffer, chunk...)
			l.silenceFrames = 0

			// 检查最大时长
			duration := time.Since(l.speechStart).Seconds()
			if duration >= l.maxSpeechSec {
				fmt.Printf("[VAD] 达到最大时长 %.1fs, 截断\n", l.maxSpeechSec)
				l.finalizeSpeech()
			}
		} else {
			if l.isSpeaking {
				l.speechBuffer = append(l.speechBuffer, chunk...)
				l.silenceFrames++

				// 静音持续超过阈值，认为说话结束
				silenceDuration := float64(l.silenceFrames) * float64(WindowSize) / float64(SampleRate)
				if silenceDuration >= l.silenceSec {
					l.finalizeSpeech()
				}
			} else {
				// 非语音状态：写入预缓冲环形队列
				chunkCopy := make([]float32, WindowSize)
				copy(chunkCopy, chunk)
				l.preBuffer[l.preBufferIdx] = chunkCopy
				l.preBufferIdx = (l.preBufferIdx + 1) % l.preBufferCap
				if l.preBufferLen < l.preBufferCap {
					l.preBufferLen++
				}
			}
		}
	}
}

// finalizeSpeech 完成一段语音
func (l *Listener) finalizeSpeech() {
	if len(l.speechBuffer) == 0 {
		l.isSpeaking = false
		return
	}

	audio := l.speechBuffer
	duration := float64(len(audio)) / float64(SampleRate)

	// 重置状态
	l.isSpeaking = false
	l.speechBuffer = nil
	l.silenceFrames = 0

	// 过短忽略
	if duration < l.minSpeechSec {
		fmt.Printf("[VAD] 语音过短 (%.2fs < %.2fs), 忽略\n", duration, l.minSpeechSec)
		return
	}

	fmt.Printf("[VAD] 语音结束, 时长 %.2fs, 样本 %d\n", duration, len(audio))

	// 保存 WAV
	os.MkdirAll(l.saveDir, 0755)
	timestamp := time.Now().Format("20060102_150405")
	wavPath := filepath.Join(l.saveDir, fmt.Sprintf("wake_%s.wav", timestamp))
	if err := saveWAV(wavPath, audio); err != nil {
		fmt.Printf("[VAD] 保存 WAV 失败: %v\n", err)
	} else {
		fmt.Printf("[VAD] 已保存: %s\n", wavPath)
	}

	// 触发回调
	if l.callback != nil {
		go l.callback(audio, duration)
	}
}

// Close 释放资源
func (l *Listener) Close() {
	l.Stop()
	l.vad.Close()
}

// pcm16ToFloat32 将 16-bit PCM 转换为 float32 [-1, 1]
func pcm16ToFloat32(data []byte) []float32 {
	numSamples := len(data) / 2
	samples := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		val := int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
		samples[i] = float32(val) / 32768.0
	}
	return samples
}

// saveWAV 将 float32 音频保存为 16-bit PCM WAV
func saveWAV(path string, audio []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// 转换为 int16
	pcm := make([]byte, len(audio)*2)
	for i, s := range audio {
		val := int16(math.MaxInt16 * math.Min(1.0, math.Max(-1.0, float64(s))))
		binary.LittleEndian.PutUint16(pcm[i*2:i*2+2], uint16(val))
	}

	dataSize := uint32(len(pcm))
	// RIFF header
	binary.Write(f, binary.LittleEndian, []byte("RIFF"))
	binary.Write(f, binary.LittleEndian, uint32(36+dataSize))
	binary.Write(f, binary.LittleEndian, []byte("WAVE"))
	// fmt chunk
	binary.Write(f, binary.LittleEndian, []byte("fmt "))
	binary.Write(f, binary.LittleEndian, uint32(16))
	binary.Write(f, binary.LittleEndian, uint16(waveFormatPCM))
	binary.Write(f, binary.LittleEndian, uint16(1)) // mono
	binary.Write(f, binary.LittleEndian, uint32(SampleRate))
	binary.Write(f, binary.LittleEndian, uint32(SampleRate*2))
	binary.Write(f, binary.LittleEndian, uint16(2))  // block align
	binary.Write(f, binary.LittleEndian, uint16(16)) // bits per sample
	// data chunk
	binary.Write(f, binary.LittleEndian, []byte("data"))
	binary.Write(f, binary.LittleEndian, dataSize)
	_, err = f.Write(pcm)
	return err
}

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFileDialog,
                              QSlider, QComboBox, QTextEdit, QProgressBar,
                              QGroupBox, QRadioButton, QSpinBox, QTabWidget,
                              QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import pyaudio
import numpy as np
import wave
from audio_tape_storage import AudioTapeEncoder
from pathlib import Path

class AudioRecorderThread(QThread):
    """오디오 녹음 스레드"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, output_file, sample_rate=44100, device_index=None):
        super().__init__()
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.is_recording = False
        self.frames = []
        
    def run(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=self.sample_rate,
                          input=True,
                          input_device_index=self.device_index,
                          frames_per_buffer=1024)
            
            self.is_recording = True
            self.frames = []
            
            while self.is_recording:
                data = stream.read(1024)
                self.frames.append(data)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # WAV 파일로 저장
            wf = wave.open(self.output_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            self.finished.emit(self.output_file)
        except Exception as e:
            self.error.emit(str(e))
    
    def stop_recording(self):
        self.is_recording = False


class AudioPlayerThread(QThread):
    """오디오 재생 스레드"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, audio_file, device_index=None):
        super().__init__()
        self.audio_file = audio_file
        self.device_index = device_index
        self.is_playing = False
        
    def run(self):
        try:
            wf = wave.open(self.audio_file, 'rb')
            p = pyaudio.PyAudio()
            
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                          channels=wf.getnchannels(),
                          rate=wf.getframerate(),
                          output=True,
                          output_device_index=self.device_index)
            
            self.is_playing = True
            data = wf.readframes(1024)
            
            while data and self.is_playing:
                stream.write(data)
                data = wf.readframes(1024)
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
    
    def stop_playing(self):
        self.is_playing = False


class EncodingThread(QThread):
    """인코딩 작업 스레드"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, encoder, input_file, output_file):
        super().__init__()
        self.encoder = encoder
        self.input_file = input_file
        self.output_file = output_file
        
    def run(self):
        try:
            self.progress.emit(30)
            stats = self.encoder.file_to_audio(self.input_file, self.output_file)
            self.progress.emit(100)
            self.finished.emit(stats)
        except Exception as e:
            self.error.emit(str(e))


class DecodingThread(QThread):
    """디코딩 작업 스레드"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, encoder, input_file, output_file):
        super().__init__()
        self.encoder = encoder
        self.input_file = input_file
        self.output_file = output_file
        
    def run(self):
        try:
            self.progress.emit(30)
            result = self.encoder.audio_to_file(self.input_file, self.output_file)
            self.progress.emit(100)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AudioTapeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.recorder_thread = None
        self.player_thread = None
        self.encoding_thread = None
        self.decoding_thread = None
        self.input_device_index = None
        self.output_device_index = None
        
        self.initUI()
        self.update_encoder()
        
    def initUI(self):
        self.setWindowTitle('오디오 테이프 데이터 저장 시스템')
        self.setGeometry(100, 100, 900, 700)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 탭 위젯
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # 인코딩 탭
        encode_tab = self.create_encode_tab()
        tab_widget.addTab(encode_tab, "인코딩 (파일 → 오디오)")
        
        # 디코딩 탭
        decode_tab = self.create_decode_tab()
        tab_widget.addTab(decode_tab, "디코딩 (오디오 → 파일)")
        
        # 오디오 입출력 탭
        audio_tab = self.create_audio_tab()
        tab_widget.addTab(audio_tab, "오디오 입/출력")
        
        # 설정 탭
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "설정")
        
        # 로그 영역
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.log("프로그램이 시작되었습니다.")
        
        # 오디오 장치 목록 초기화 (log_text가 생성된 후)
        self.refresh_audio_devices()
        
    def create_encode_tab(self):
        """인코딩 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 입력 파일 선택
        input_group = QGroupBox("입력 파일 선택")
        input_layout = QHBoxLayout()
        self.encode_input_label = QLabel("파일을 선택하세요")
        self.encode_input_btn = QPushButton("파일 선택")
        self.encode_input_btn.clicked.connect(self.select_encode_input)
        input_layout.addWidget(self.encode_input_label)
        input_layout.addWidget(self.encode_input_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 출력 파일 선택
        output_group = QGroupBox("출력 오디오 파일")
        output_layout = QHBoxLayout()
        self.encode_output_label = QLabel("output.wav")
        self.encode_output_btn = QPushButton("저장 위치 선택")
        self.encode_output_btn.clicked.connect(self.select_encode_output)
        output_layout.addWidget(self.encode_output_label)
        output_layout.addWidget(self.encode_output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 인코딩 버튼
        encode_btn_layout = QHBoxLayout()
        self.encode_btn = QPushButton("인코딩 시작")
        self.encode_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.encode_btn.clicked.connect(self.start_encoding)
        encode_btn_layout.addWidget(self.encode_btn)
        
        # 인코딩 후 자동 재생 체크박스
        self.auto_play_checkbox = QRadioButton("인코딩 후 자동 재생")
        self.auto_play_checkbox.setChecked(True)
        encode_btn_layout.addWidget(self.auto_play_checkbox)
        
        layout.addLayout(encode_btn_layout)
        
        # 프로그레스 바
        self.encode_progress = QProgressBar()
        layout.addWidget(self.encode_progress)
        
        # 통계 정보
        stats_group = QGroupBox("인코딩 통계")
        stats_layout = QVBoxLayout()
        self.encode_stats_label = QLabel("통계 정보가 여기 표시됩니다.")
        stats_layout.addWidget(self.encode_stats_label)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return widget
    
    def create_decode_tab(self):
        """디코딩 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 입력 오디오 파일 선택
        input_group = QGroupBox("입력 오디오 파일")
        input_layout = QHBoxLayout()
        self.decode_input_label = QLabel("오디오 파일을 선택하세요")
        self.decode_input_btn = QPushButton("파일 선택")
        self.decode_input_btn.clicked.connect(self.select_decode_input)
        input_layout.addWidget(self.decode_input_label)
        input_layout.addWidget(self.decode_input_btn)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 출력 파일 선택
        output_group = QGroupBox("복원될 파일")
        output_layout = QHBoxLayout()
        self.decode_output_label = QLabel("자동으로 설정됨 (메타데이터에서)")
        self.decode_output_btn = QPushButton("저장 위치 변경")
        self.decode_output_btn.clicked.connect(self.select_decode_output)
        output_layout.addWidget(self.decode_output_label)
        output_layout.addWidget(self.decode_output_btn)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # 디코딩 버튼
        decode_btn_layout = QHBoxLayout()
        self.decode_btn = QPushButton("디코딩 시작")
        self.decode_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.decode_btn.clicked.connect(self.start_decoding)
        decode_btn_layout.addWidget(self.decode_btn)
        
        # 녹음 후 자동 디코딩 체크박스
        self.auto_decode_checkbox = QRadioButton("녹음 완료 후 자동 디코딩")
        self.auto_decode_checkbox.setChecked(True)
        decode_btn_layout.addWidget(self.auto_decode_checkbox)
        
        layout.addLayout(decode_btn_layout)
        
        # 프로그레스 바
        self.decode_progress = QProgressBar()
        layout.addWidget(self.decode_progress)
        
        # 결과 정보
        result_group = QGroupBox("디코딩 결과")
        result_layout = QVBoxLayout()
        self.decode_result_label = QLabel("결과 정보가 여기 표시됩니다.")
        result_layout.addWidget(self.decode_result_label)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        layout.addStretch()
        return widget
    
    def create_audio_tab(self):
        """오디오 입출력 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 녹음 섹션
        record_group = QGroupBox("테이프에서 녹음")
        record_layout = QVBoxLayout()
        
        record_file_layout = QHBoxLayout()
        self.record_file_label = QLabel("recorded.wav")
        record_file_btn = QPushButton("저장 위치 선택")
        record_file_btn.clicked.connect(self.select_record_file)
        record_file_layout.addWidget(QLabel("저장 파일:"))
        record_file_layout.addWidget(self.record_file_label)
        record_file_layout.addWidget(record_file_btn)
        record_layout.addLayout(record_file_layout)
        
        record_btn_layout = QHBoxLayout()
        self.record_start_btn = QPushButton("녹음 시작")
        self.record_start_btn.clicked.connect(self.start_recording)
        self.record_stop_btn = QPushButton("녹음 중지")
        self.record_stop_btn.clicked.connect(self.stop_recording)
        self.record_stop_btn.setEnabled(False)
        record_btn_layout.addWidget(self.record_start_btn)
        record_btn_layout.addWidget(self.record_stop_btn)
        record_layout.addLayout(record_btn_layout)
        
        record_group.setLayout(record_layout)
        layout.addWidget(record_group)
        
        # 재생 섹션
        play_group = QGroupBox("테이프로 재생")
        play_layout = QVBoxLayout()
        
        play_file_layout = QHBoxLayout()
        self.play_file_label = QLabel("재생할 파일을 선택하세요")
        play_file_btn = QPushButton("파일 선택")
        play_file_btn.clicked.connect(self.select_play_file)
        play_file_layout.addWidget(self.play_file_label)
        play_file_layout.addWidget(play_file_btn)
        play_layout.addLayout(play_file_layout)
        
        play_btn_layout = QHBoxLayout()
        self.play_start_btn = QPushButton("재생 시작")
        self.play_start_btn.clicked.connect(self.start_playing)
        self.play_stop_btn = QPushButton("재생 중지")
        self.play_stop_btn.clicked.connect(self.stop_playing)
        self.play_stop_btn.setEnabled(False)
        play_btn_layout.addWidget(self.play_start_btn)
        play_btn_layout.addWidget(self.play_stop_btn)
        play_layout.addLayout(play_btn_layout)
        
        play_group.setLayout(play_layout)
        layout.addWidget(play_group)
        
        # 오디오 장치 정보
        device_group = QGroupBox("오디오 장치 정보")
        device_layout = QVBoxLayout()
        self.device_info_label = QLabel()
        self.update_audio_devices()
        device_layout.addWidget(self.device_info_label)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        layout.addStretch()
        return widget
    
    def create_settings_tab(self):
        """설정 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 주파수 개수 설정
        freq_group = QGroupBox("주파수 개수 (대역폭 설정)")
        freq_layout = QVBoxLayout()
        
        freq_label = QLabel("사용할 주파수 개수 (2: 기본, 8: 최대 대역폭)")
        freq_layout.addWidget(freq_label)
        
        freq_slider_layout = QHBoxLayout()
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setMinimum(2)
        self.freq_slider.setMaximum(8)
        self.freq_slider.setValue(2)
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.setTickInterval(1)
        self.freq_slider.valueChanged.connect(self.update_frequency_label)
        
        self.freq_value_label = QLabel("2")
        self.freq_value_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        
        freq_slider_layout.addWidget(self.freq_slider)
        freq_slider_layout.addWidget(self.freq_value_label)
        freq_layout.addLayout(freq_slider_layout)
        
        self.freq_info_label = QLabel()
        self.update_frequency_info(2)
        freq_layout.addWidget(self.freq_info_label)
        
        freq_group.setLayout(freq_layout)
        layout.addWidget(freq_group)
        
        # 주파수 범위 설정
        freq_range_group = QGroupBox("주파수 범위 (Hz)")
        freq_range_layout = QVBoxLayout()
        
        # 최소 주파수
        min_freq_layout = QHBoxLayout()
        min_freq_layout.addWidget(QLabel("최소 주파수:"))
        self.min_freq_spin = QSpinBox()
        self.min_freq_spin.setRange(800, 20000)
        self.min_freq_spin.setValue(1400)
        self.min_freq_spin.setSingleStep(100)
        self.min_freq_spin.setSuffix(" Hz")
        min_freq_layout.addWidget(self.min_freq_spin)
        freq_range_layout.addLayout(min_freq_layout)
        
        # 최대 주파수
        max_freq_layout = QHBoxLayout()
        max_freq_layout.addWidget(QLabel("최대 주파수:"))
        self.max_freq_spin = QSpinBox()
        self.max_freq_spin.setRange(1000, 22000)
        self.max_freq_spin.setValue(11025)
        self.max_freq_spin.setSingleStep(100)
        self.max_freq_spin.setSuffix(" Hz")
        max_freq_layout.addWidget(self.max_freq_spin)
        freq_range_layout.addLayout(max_freq_layout)
        
        freq_range_info = QLabel("테이프 음역을 최대한 활용합니다\n권장: 1400-11025 Hz (CD 품질의 절반)")
        freq_range_info.setWordWrap(True)
        freq_range_layout.addWidget(freq_range_info)
        
        freq_range_group.setLayout(freq_range_layout)
        layout.addWidget(freq_range_group)
        
        # 심볼 지속시간 설정
        symbol_dur_group = QGroupBox("심볼 지속시간")
        symbol_dur_layout = QVBoxLayout()
        
        symbol_dur_hlayout = QHBoxLayout()
        symbol_dur_hlayout.addWidget(QLabel("심볼 지속시간:"))
        self.symbol_duration_spin = QSpinBox()
        self.symbol_duration_spin.setRange(5, 100)
        self.symbol_duration_spin.setValue(20)
        self.symbol_duration_spin.setSingleStep(5)
        self.symbol_duration_spin.setSuffix(" ms")
        symbol_dur_hlayout.addWidget(self.symbol_duration_spin)
        symbol_dur_layout.addLayout(symbol_dur_hlayout)
        
        symbol_dur_info = QLabel(
            "• 짧을수록: 빠른 전송, 노이즈에 약함\n"
            "• 길수록: 느린 전송, 안정적"
        )
        symbol_dur_info.setWordWrap(True)
        symbol_dur_layout.addWidget(symbol_dur_info)
        
        symbol_dur_group.setLayout(symbol_dur_layout)
        layout.addWidget(symbol_dur_group)
        
        # 압축 설정
        compression_group = QGroupBox("압축 설정")
        compression_layout = QVBoxLayout()
        
        compression_label = QLabel("압축 강도 (0: 압축 안함, 9: 최대 압축)")
        compression_layout.addWidget(compression_label)
        
        slider_layout = QHBoxLayout()
        self.compression_slider = QSlider(Qt.Orientation.Horizontal)
        self.compression_slider.setMinimum(0)
        self.compression_slider.setMaximum(9)
        self.compression_slider.setValue(6)
        self.compression_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.compression_slider.setTickInterval(1)
        self.compression_slider.valueChanged.connect(self.update_compression_label)
        
        self.compression_value_label = QLabel("6")
        self.compression_value_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        
        slider_layout.addWidget(self.compression_slider)
        slider_layout.addWidget(self.compression_value_label)
        compression_layout.addLayout(slider_layout)
        
        compression_info = QLabel(
            "• 압축 강도가 높을수록: 오디오 길이가 짧아지지만, 인코딩/디코딩 시간 증가\n"
            "• 압축 강도가 낮을수록: 처리 속도가 빠르지만, 오디오 길이 증가"
        )
        compression_info.setWordWrap(True)
        compression_layout.addWidget(compression_info)
        
        compression_group.setLayout(compression_layout)
        layout.addWidget(compression_group)
        
        # 샘플링 레이트 설정
        samplerate_group = QGroupBox("샘플링 레이트")
        samplerate_layout = QVBoxLayout()
        
        self.samplerate_combo = QComboBox()
        self.samplerate_combo.addItems(["44100 Hz (CD 품질)", "22050 Hz (중간)", "11025 Hz (낮음)"])
        self.samplerate_combo.setCurrentIndex(0)
        samplerate_layout.addWidget(self.samplerate_combo)
        
        samplerate_info = QLabel(
            "• 44100 Hz: 고품질, 안정적 (권장)\n"
            "• 22050 Hz: 중간 품질, 더 짧은 오디오\n"
            "• 11025 Hz: 낮은 품질, 가장 짧은 오디오"
        )
        samplerate_info.setWordWrap(True)
        samplerate_layout.addWidget(samplerate_info)
        
        samplerate_group.setLayout(samplerate_layout)
        layout.addWidget(samplerate_group)
        
        # 오디오 장치 선택
        device_group = QGroupBox("오디오 장치 선택")
        device_layout = QVBoxLayout()
        
        # 입력 장치
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("입력 장치 (녹음):"))
        self.input_device_combo = QComboBox()
        input_layout.addWidget(self.input_device_combo)
        device_layout.addLayout(input_layout)
        
        # 출력 장치
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("출력 장치 (재생):"))
        self.output_device_combo = QComboBox()
        output_layout.addWidget(self.output_device_combo)
        device_layout.addLayout(output_layout)
        
        # 장치 새로고침 버튼
        refresh_btn = QPushButton("장치 목록 새로고침")
        refresh_btn.clicked.connect(self.refresh_audio_devices)
        device_layout.addWidget(refresh_btn)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # 설정 적용 버튼
        apply_btn = QPushButton("설정 적용")
        apply_btn.clicked.connect(self.update_encoder)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        return widget
    
    def update_compression_label(self, value):
        """압축 강도 라벨 업데이트"""
        self.compression_value_label.setText(str(value))
    
    def update_frequency_label(self, value):
        """주파수 개수 라벨 업데이트"""
        self.freq_value_label.setText(str(value))
        self.update_frequency_info(value)
    
    def update_frequency_info(self, num_freq):
        """주파수 정보 업데이트"""
        frequencies = [800 + i * 600 for i in range(num_freq)]
        bands = [(f - 300, f + 300) for f in frequencies]
        
        info_text = f"주파수 수: {num_freq}개\n"
        info_text += f"한 심볼당 비트 수: {int(np.log2(num_freq))}비트\n"
        info_text += f"대역폭 증가율: {num_freq/2 * 100:.0f}%\n\n"
        info_text += "사용 주파수 및 대역:\n"
        
        for i, (freq, (lower, upper)) in enumerate(zip(frequencies, bands)):
            info_text += f"  {i}: {freq}Hz (대역: {lower}-{upper}Hz)\n"
        
        self.freq_info_label.setText(info_text)
    
    def refresh_audio_devices(self):
        """오디오 장치 목록 새로고침"""
        try:
            p = pyaudio.PyAudio()
            
            # 입력 장치 목록
            self.input_device_combo.clear()
            self.output_device_combo.clear()
            
            default_input = p.get_default_input_device_info()
            default_output = p.get_default_output_device_info()
            
            for i in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(i)
                device_name = device_info['name']
                
                # 입력 장치
                if device_info['maxInputChannels'] > 0:
                    display_name = f"{i}: {device_name}"
                    if i == default_input['index']:
                        display_name += " (기본)"
                    self.input_device_combo.addItem(display_name, i)
                    if i == default_input['index']:
                        self.input_device_combo.setCurrentIndex(self.input_device_combo.count() - 1)
                
                # 출력 장치
                if device_info['maxOutputChannels'] > 0:
                    display_name = f"{i}: {device_name}"
                    if i == default_output['index']:
                        display_name += " (기본)"
                    self.output_device_combo.addItem(display_name, i)
                    if i == default_output['index']:
                        self.output_device_combo.setCurrentIndex(self.output_device_combo.count() - 1)
            
            p.terminate()
            self.log("오디오 장치 목록을 새로고침했습니다.")
            
        except Exception as e:
            self.log(f"오디오 장치 목록 불러오기 실패: {str(e)}")
    
    def update_encoder(self):
        """인코더 설정 업데이트"""
        compression = self.compression_slider.value()
        sample_rate_text = self.samplerate_combo.currentText()
        sample_rate = int(sample_rate_text.split()[0])
        num_frequencies = self.freq_slider.value()
        min_freq = self.min_freq_spin.value()
        max_freq = self.max_freq_spin.value()
        symbol_duration = self.symbol_duration_spin.value() / 1000.0  # ms -> s
        
        # 선택된 오디오 장치 인덱스 저장
        if self.input_device_combo.currentData() is not None:
            self.input_device_index = self.input_device_combo.currentData()
        if self.output_device_combo.currentData() is not None:
            self.output_device_index = self.output_device_combo.currentData()
        
        self.encoder = AudioTapeEncoder(
            sample_rate=sample_rate, 
            compression_level=compression,
            num_frequencies=num_frequencies,
            min_freq=min_freq,
            max_freq=max_freq,
            symbol_duration=symbol_duration
        )
        self.log(f"설정 업데이트: 압축 {compression}, 샘플링 {sample_rate}Hz, 주파수 {num_frequencies}개")
        self.log(f"주파수 범위: {min_freq}-{max_freq}Hz, 심볼 지속시간: {symbol_duration*1000:.1f}ms")
        
        # 주파수 정보 표시
        freq_info = f"주파수: {[f'{f:.0f}Hz' for f in self.encoder.frequencies]}"
        self.log(freq_info)
    
    def update_audio_devices(self):
        """오디오 장치 정보 업데이트"""
        try:
            p = pyaudio.PyAudio()
            info_text = f"사용 가능한 오디오 장치 수: {p.get_device_count()}\n\n"
            
            default_input = p.get_default_input_device_info()
            default_output = p.get_default_output_device_info()
            
            info_text += f"기본 입력 장치: {default_input['name']}\n"
            info_text += f"기본 출력 장치: {default_output['name']}"
            
            p.terminate()
            self.device_info_label.setText(info_text)
        except Exception as e:
            self.device_info_label.setText(f"오디오 장치 정보를 가져올 수 없습니다: {str(e)}")
    
    def select_encode_input(self):
        """인코딩 입력 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(self, "입력 파일 선택", "", "모든 파일 (*.*)")
        if file_path:
            self.encode_input_label.setText(file_path)
            
            # 기본 출력 파일명 설정
            base_name = Path(file_path).stem
            default_output = str(Path.home() / f"{base_name}_encoded.wav")
            self.encode_output_label.setText(default_output)
    
    def select_encode_output(self):
        """인코딩 출력 파일 선택"""
        file_path, _ = QFileDialog.getSaveFileName(self, "출력 파일 선택", "", "WAV 파일 (*.wav)")
        if file_path:
            self.encode_output_label.setText(file_path)
    
    def select_decode_input(self):
        """디코딩 입력 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(self, "오디오 파일 선택", "", "WAV 파일 (*.wav)")
        if file_path:
            self.decode_input_label.setText(file_path)
    
    def select_decode_output(self):
        """디코딩 출력 파일 선택"""
        file_path, _ = QFileDialog.getSaveFileName(self, "복원 파일 저장", "", "모든 파일 (*.*)")
        if file_path:
            self.decode_output_label.setText(file_path)
    
    def select_record_file(self):
        """녹음 파일 선택"""
        file_path, _ = QFileDialog.getSaveFileName(self, "녹음 파일 저장", "", "WAV 파일 (*.wav)")
        if file_path:
            self.record_file_label.setText(file_path)
    
    def select_play_file(self):
        """재생 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(self, "재생 파일 선택", "", "WAV 파일 (*.wav)")
        if file_path:
            self.play_file_label.setText(file_path)
    
    def start_encoding(self):
        """인코딩 시작"""
        input_file = self.encode_input_label.text()
        output_file = self.encode_output_label.text()
        
        if input_file == "파일을 선택하세요" or not Path(input_file).exists():
            QMessageBox.warning(self, "경고", "입력 파일을 선택하세요.")
            return
        
        self.encode_btn.setEnabled(False)
        self.encode_progress.setValue(0)
        self.log(f"인코딩 시작: {input_file} → {output_file}")
        
        self.encoding_thread = EncodingThread(self.encoder, input_file, output_file)
        self.encoding_thread.finished.connect(self.encoding_finished)
        self.encoding_thread.error.connect(self.encoding_error)
        self.encoding_thread.progress.connect(self.encode_progress.setValue)
        self.encoding_thread.start()
    
    def encoding_finished(self, stats):
        """인코딩 완료"""
        self.encode_btn.setEnabled(True)
        self.encode_progress.setValue(100)
        
        stats_text = f"""
인코딩 완료!

원본 크기: {stats['original_size']:,} 바이트
압축 후 크기: {stats['compressed_size']:,} 바이트
압축률: {stats['compression_ratio']:.2f}%
오디오 길이: {stats['duration_seconds']:.2f}초
심볼 레이트: {stats['symbol_rate']:.2f} symbols/sec
비트 레이트: {stats['bit_rate']:.2f} bps
주파수 개수: {stats['num_frequencies']}개
        """
        
        self.encode_stats_label.setText(stats_text)
        self.log("인코딩 완료!")
        
        # 자동 재생 체크 시
        if self.auto_play_checkbox.isChecked():
            output_file = self.encode_output_label.text()
            self.play_file_label.setText(output_file)
            self.log("자동 재생을 시작합니다...")
            QMessageBox.information(self, "완료", "인코딩이 완료되었습니다!\n자동으로 재생을 시작합니다.")
            self.start_playing()
        else:
            QMessageBox.information(self, "완료", "인코딩이 완료되었습니다!")
    
    def encoding_error(self, error_msg):
        """인코딩 에러"""
        self.encode_btn.setEnabled(True)
        self.log(f"인코딩 에러: {error_msg}")
        QMessageBox.critical(self, "에러", f"인코딩 중 오류가 발생했습니다:\n{error_msg}")
    
    def start_decoding(self):
        """디코딩 시작"""
        input_file = self.decode_input_label.text()
        output_file = self.decode_output_label.text()
        
        if input_file == "오디오 파일을 선택하세요" or not Path(input_file).exists():
            QMessageBox.warning(self, "경고", "입력 오디오 파일을 선택하세요.")
            return
        
        # 출력 파일이 기본값이면 None으로 (자동 설정)
        if output_file == "자동으로 설정됨 (메타데이터에서)":
            output_file = None
        
        self.decode_btn.setEnabled(False)
        self.decode_progress.setValue(0)
        self.log(f"디코딩 시작: {input_file}")
        
        self.decoding_thread = DecodingThread(self.encoder, input_file, output_file)
        self.decoding_thread.finished.connect(self.decoding_finished)
        self.decoding_thread.error.connect(self.decoding_error)
        self.decoding_thread.progress.connect(self.decode_progress.setValue)
        self.decoding_thread.start()
    
    def decoding_finished(self, result):
        """디코딩 완료"""
        self.decode_btn.setEnabled(True)
        self.decode_progress.setValue(100)
        
        result_text = f"""
디코딩 완료!

복원된 파일: {result['filename']}
파일 크기: {result['size']:,} 바이트
원본 파일명: {result['metadata']['filename']}
압축 강도: {result['metadata']['compression_level']}
        """
        
        self.decode_result_label.setText(result_text)
        self.log(f"디코딩 완료: {result['filename']}")
        QMessageBox.information(self, "완료", f"디코딩이 완료되었습니다!\n파일: {result['filename']}")
    
    def decoding_error(self, error_msg):
        """디코딩 에러"""
        self.decode_btn.setEnabled(True)
        self.log(f"디코딩 에러: {error_msg}")
        QMessageBox.critical(self, "에러", f"디코딩 중 오류가 발생했습니다:\n{error_msg}")
    
    def start_recording(self):
        """녹음 시작"""
        output_file = self.record_file_label.text()
        
        self.record_start_btn.setEnabled(False)
        self.record_stop_btn.setEnabled(True)
        self.log("녹음 시작...")
        
        sample_rate_text = self.samplerate_combo.currentText()
        sample_rate = int(sample_rate_text.split()[0])
        
        self.recorder_thread = AudioRecorderThread(output_file, sample_rate, self.input_device_index)
        self.recorder_thread.finished.connect(self.recording_finished)
        self.recorder_thread.error.connect(self.recording_error)
        self.recorder_thread.start()
    
    def stop_recording(self):
        """녹음 중지"""
        if self.recorder_thread:
            self.recorder_thread.stop_recording()
    
    def recording_finished(self, file_path):
        """녹음 완료"""
        self.record_start_btn.setEnabled(True)
        self.record_stop_btn.setEnabled(False)
        self.log(f"녹음 완료: {file_path}")
        
        # 자동 디코딩 체크 시
        if self.auto_decode_checkbox.isChecked():
            self.decode_input_label.setText(file_path)
            self.log("자동 디코딩을 시작합니다...")
            QMessageBox.information(self, "완료", f"녹음이 완료되었습니다!\n자동으로 디코딩을 시작합니다.\n파일: {file_path}")
            self.start_decoding()
        else:
            QMessageBox.information(self, "완료", f"녹음이 완료되었습니다!\n파일: {file_path}")
    
    def recording_error(self, error_msg):
        """녹음 에러"""
        self.record_start_btn.setEnabled(True)
        self.record_stop_btn.setEnabled(False)
        self.log(f"녹음 에러: {error_msg}")
        QMessageBox.critical(self, "에러", f"녹음 중 오류가 발생했습니다:\n{error_msg}")
    
    def start_playing(self):
        """재생 시작"""
        file_path = self.play_file_label.text()
        
        if file_path == "재생할 파일을 선택하세요" or not Path(file_path).exists():
            QMessageBox.warning(self, "경고", "재생할 파일을 선택하세요.")
            return
        
        self.play_start_btn.setEnabled(False)
        self.play_stop_btn.setEnabled(True)
        self.log(f"재생 시작: {file_path}")
        
        self.player_thread = AudioPlayerThread(file_path, self.output_device_index)
        self.player_thread.finished.connect(self.playing_finished)
        self.player_thread.error.connect(self.playing_error)
        self.player_thread.start()
    
    def stop_playing(self):
        """재생 중지"""
        if self.player_thread:
            self.player_thread.stop_playing()
    
    def playing_finished(self):
        """재생 완료"""
        self.play_start_btn.setEnabled(True)
        self.play_stop_btn.setEnabled(False)
        self.log("재생 완료")
    
    def playing_error(self, error_msg):
        """재생 에러"""
        self.play_start_btn.setEnabled(True)
        self.play_stop_btn.setEnabled(False)
        self.log(f"재생 에러: {error_msg}")
        QMessageBox.critical(self, "에러", f"재생 중 오류가 발생했습니다:\n{error_msg}")
    
    def log(self, message):
        """로그 메시지 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        # log_text가 아직 생성되지 않았으면 무시
        if hasattr(self, 'log_text'):
            self.log_text.append(f"[{timestamp}] {message}")
        else:
            print(f"[{timestamp}] {message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioTapeGUI()
    window.show()
    sys.exit(app.exec())

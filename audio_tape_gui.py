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
    
    def __init__(self, output_file, sample_rate=44100):
        super().__init__()
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.is_recording = False
        self.frames = []
        
    def run(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=self.sample_rate,
                          input=True,
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
    
    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file
        self.is_playing = False
        
    def run(self):
        try:
            wf = wave.open(self.audio_file, 'rb')
            p = pyaudio.PyAudio()
            
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                          channels=wf.getnchannels(),
                          rate=wf.getframerate(),
                          output=True)
            
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
        self.encode_btn = QPushButton("인코딩 시작")
        self.encode_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.encode_btn.clicked.connect(self.start_encoding)
        layout.addWidget(self.encode_btn)
        
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
        self.decode_btn = QPushButton("디코딩 시작")
        self.decode_btn.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.decode_btn.clicked.connect(self.start_decoding)
        layout.addWidget(self.decode_btn)
        
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
        
        # 설정 적용 버튼
        apply_btn = QPushButton("설정 적용")
        apply_btn.clicked.connect(self.update_encoder)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        return widget
    
    def update_compression_label(self, value):
        """압축 강도 라벨 업데이트"""
        self.compression_value_label.setText(str(value))
    
    def update_encoder(self):
        """인코더 설정 업데이트"""
        compression = self.compression_slider.value()
        sample_rate_text = self.samplerate_combo.currentText()
        sample_rate = int(sample_rate_text.split()[0])
        
        self.encoder = AudioTapeEncoder(sample_rate=sample_rate, compression_level=compression)
        self.log(f"설정 업데이트: 압축 강도 {compression}, 샘플링 레이트 {sample_rate} Hz")
    
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
비트레이트: {stats['bitrate']:.2f} bps
        """
        
        self.encode_stats_label.setText(stats_text)
        self.log("인코딩 완료!")
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
        
        self.recorder_thread = AudioRecorderThread(output_file, sample_rate)
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
        
        self.player_thread = AudioPlayerThread(file_path)
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
        self.log_text.append(f"[{timestamp}] {message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioTapeGUI()
    window.show()
    sys.exit(app.exec())

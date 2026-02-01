import numpy as np
import wave
import struct
import zlib
from pathlib import Path
import json

class AudioTapeEncoder:
    """오디오 테이프에 데이터를 저장하기 위한 인코더"""
    
    def __init__(self, sample_rate=44100, compression_level=6, num_frequencies=2, 
                 min_freq=1400, max_freq=11025, symbol_duration=0.02):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            compression_level: 압축 강도 (0-9, 9가 최대 압축)
            num_frequencies: 사용할 주파수 개수 (2-8)
            min_freq: 최소 주파수 (Hz)
            max_freq: 최대 주파수 (Hz) - 테이프 음역 최대 활용
            symbol_duration: 심볼 지속 시간 (초)
        """
        self.sample_rate = sample_rate
        self.compression_level = compression_level
        self.num_frequencies = max(2, min(8, num_frequencies))
        self.symbol_duration = symbol_duration
        
        # 주파수 범위를 전체 대역에 골고루 분배
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # 주파수를 전체 대역에 균등 배치
        if self.num_frequencies == 1:
            self.frequencies = [(min_freq + max_freq) / 2]
        else:
            freq_step = (max_freq - min_freq) / (self.num_frequencies - 1)
            self.frequencies = [min_freq + i * freq_step for i in range(self.num_frequencies)]
        
        # 클럭 주파수 (데이터 주파수보다 훨씬 높게 설정)
        self.clock_freq = max_freq + 1000  # 데이터 범위 밖
        
        # 주파수 대역 설정 (각 주파수를 중심으로 범위 설정)
        self.frequency_bands = []
        for i, freq in enumerate(self.frequencies):
            if i == 0:
                # 첫 번째 주파수
                if len(self.frequencies) > 1:
                    upper = (self.frequencies[0] + self.frequencies[1]) / 2
                else:
                    upper = freq + 500
                lower = min_freq - 200
            elif i == len(self.frequencies) - 1:
                # 마지막 주파수
                lower = (self.frequencies[i-1] + self.frequencies[i]) / 2
                upper = max_freq + 200
            else:
                # 중간 주파수들
                lower = (self.frequencies[i-1] + self.frequencies[i]) / 2
                upper = (self.frequencies[i] + self.frequencies[i+1]) / 2
            
            self.frequency_bands.append((lower, upper))
        
        self.bits_per_symbol = int(np.log2(self.num_frequencies))
        
        # 시작 마커용 고정 주파수 (저주파와 고주파)
        self.marker_freq_low = 800   # 저주파 마커
        self.marker_freq_high = 2400  # 고주파 마커
        self.marker_duration = 0.5    # 마커 지속 시간 (긴 시간)
        
        # 기준 진폭 (디코딩 시 사용)
        self.reference_amplitude = None
    
    def file_to_audio(self, input_file, output_wav, title="", notes=""):
        """파일을 오디오 신호로 변환"""
        from datetime import datetime
        
        # 1. 파일 읽기
        with open(input_file, 'rb') as f:
            file_data = f.read()
        
        # 2. 데이터 압축
        if self.compression_level > 0:
            compressed_data = zlib.compress(file_data, level=self.compression_level)
        else:
            compressed_data = file_data
        
        # 3. 시작 마커 헤더 정보 생성
        marker_header = {
            'record_time': datetime.now().isoformat(),
            'filename': Path(input_file).name,
            'original_size': len(file_data),
            'compressed_size': len(compressed_data),
            'compression_level': self.compression_level,
            'num_frequencies': self.num_frequencies,
            'frequencies': self.frequencies,
            'symbol_duration': self.symbol_duration,
            'sample_rate': self.sample_rate,
            'title': title,
            'notes': notes,
            'min_freq': self.min_freq,
            'max_freq': self.max_freq
        }
        
        marker_header_bytes = json.dumps(marker_header, ensure_ascii=False).encode('utf-8')
        marker_header_length = len(marker_header_bytes)
        
        # 4. 전체 데이터 구성
        full_data = struct.pack('<I', marker_header_length) + marker_header_bytes + compressed_data
        
        # 5. 데이터를 심볼 스트림으로 변환
        symbol_stream = self._bytes_to_symbols(full_data)
        
        # 6. 시작 마커 생성 (긴 지속시간의 고정 주파수 패턴)
        start_marker_audio = self._create_start_marker()
        
        # 7. 클럭 신호와 데이터를 결합한 오디오 생성
        data_audio = self._symbols_to_audio_with_clock(symbol_stream)
        
        # 8. 전체 오디오 구성: [침묵] + [시작 마커] + [침묵] + [데이터]
        silence_start = np.zeros(int(self.sample_rate * 0.5))
        silence_mid = np.zeros(int(self.sample_rate * 0.3))
        
        full_audio = np.concatenate([
            silence_start,
            start_marker_audio,
            silence_mid,
            data_audio
        ])
        
        # 9. WAV 파일로 저장
        self._save_wav(full_audio, output_wav)
        
        # 통계 정보 반환
        compression_ratio = len(compressed_data) / len(file_data) * 100
        duration = len(full_audio) / self.sample_rate
        
        return {
            'original_size': len(file_data),
            'compressed_size': len(compressed_data),
            'compression_ratio': compression_ratio,
            'duration_seconds': duration,
            'symbol_rate': len(symbol_stream) / (len(data_audio) / self.sample_rate),
            'bit_rate': len(symbol_stream) * self.bits_per_symbol / (len(data_audio) / self.sample_rate),
            'num_frequencies': self.num_frequencies,
            'frequency_range': f"{self.min_freq:.0f}-{self.max_freq:.0f} Hz",
            'marker_header': marker_header
        }
    
    def audio_to_file(self, input_wav, output_file=None):
        """오디오 신호를 파일로 복원"""
        # 1. WAV 파일 읽기
        audio_signal = self._load_wav(input_wav)
        
        # 2. 시작 마커 찾기 및 기준 진폭 설정
        marker_end_pos = self._find_start_marker_in_audio(audio_signal)
        
        if marker_end_pos == -1:
            raise ValueError("시작 마커를 찾을 수 없습니다")
        
        # 3. 시작 마커 이후의 오디오에서 데이터 추출
        data_audio = audio_signal[marker_end_pos:]
        
        # 4. 클럭과 데이터를 분리하여 심볼 디코딩
        symbol_stream = self._audio_to_symbols_with_clock(data_audio)
        
        # 5. 심볼을 바이트로 변환
        data_bytes = self._symbols_to_bytes(symbol_stream)
        
        # 6. 헤더 길이 읽기
        if len(data_bytes) < 4:
            raise ValueError("데이터가 너무 짧습니다")
        
        header_length = struct.unpack('<I', data_bytes[:4])[0]
        
        if header_length > 50000 or header_length <= 0:
            raise ValueError(f"잘못된 헤더 길이: {header_length}")
        
        # 7. 헤더 읽기
        if len(data_bytes) < 4 + header_length:
            raise ValueError(f"데이터가 헤더를 포함하기에 충분하지 않습니다")
        
        header_bytes = data_bytes[4:4+header_length]
        
        try:
            header = json.loads(header_bytes.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"헤더 파싱 실패: {str(e)}")
        
        # 헤더에서 설정 복원
        if 'num_frequencies' in header:
            self.num_frequencies = header['num_frequencies']
            self.frequencies = header['frequencies']
            self.bits_per_symbol = int(np.log2(self.num_frequencies))
            if 'symbol_duration' in header:
                self.symbol_duration = header['symbol_duration']
            if 'min_freq' in header:
                self.min_freq = header['min_freq']
            if 'max_freq' in header:
                self.max_freq = header['max_freq']
        
        # 8. 압축된 데이터 추출
        compressed_data = data_bytes[4+header_length:]
        
        # 9. 데이터 압축 해제
        if header['compression_level'] > 0:
            try:
                file_data = zlib.decompress(compressed_data)
            except zlib.error as e:
                raise ValueError(f"압축 해제 실패: {str(e)}")
        else:
            file_data = compressed_data
        
        # 10. 파일로 저장
        if output_file is None:
            output_file = header['filename']
        
        with open(output_file, 'wb') as f:
            f.write(file_data)
        
        return {
            'filename': output_file,
            'size': len(file_data),
            'header': header
        }
    
    def _create_start_marker(self):
        """시작 마커 오디오 생성 - 고정 주파수 긴 지속시간"""
        marker_samples = int(self.sample_rate * self.marker_duration)
        
        # 패턴: 저-고-저-고 (각 0.5초)
        pattern = [
            (self.marker_freq_low, marker_samples),
            (self.marker_freq_high, marker_samples),
            (self.marker_freq_low, marker_samples),
            (self.marker_freq_high, marker_samples)
        ]
        
        marker_audio = []
        for freq, samples in pattern:
            t = np.linspace(0, self.marker_duration, samples, endpoint=False)
            signal = np.sin(2 * np.pi * freq * t) * 0.9  # 높은 진폭
            marker_audio.append(signal)
        
        return np.concatenate(marker_audio)
    
    def _find_start_marker_in_audio(self, audio):
        """오디오에서 시작 마커 찾기 및 기준 진폭 설정"""
        # 시작 마커의 각 구간 길이
        marker_samples = int(self.sample_rate * self.marker_duration)
        search_window = marker_samples // 2
        
        # 오디오를 작은 윈도우로 스캔
        best_score = 0
        best_pos = -1
        
        for i in range(0, len(audio) - marker_samples * 4, search_window):
            # 4개 구간 추출
            seg1 = audio[i:i+marker_samples]
            seg2 = audio[i+marker_samples:i+marker_samples*2]
            seg3 = audio[i+marker_samples*2:i+marker_samples*3]
            seg4 = audio[i+marker_samples*3:i+marker_samples*4]
            
            if len(seg4) < marker_samples:
                break
            
            # FFT로 주파수 분석
            score1 = self._check_frequency_match(seg1, self.marker_freq_low)
            score2 = self._check_frequency_match(seg2, self.marker_freq_high)
            score3 = self._check_frequency_match(seg3, self.marker_freq_low)
            score4 = self._check_frequency_match(seg4, self.marker_freq_high)
            
            total_score = score1 + score2 + score3 + score4
            
            if total_score > best_score:
                best_score = total_score
                best_pos = i + marker_samples * 4
                
                # 기준 진폭 설정 (마커의 평균 진폭)
                self.reference_amplitude = np.mean([
                    np.max(np.abs(seg1)), np.max(np.abs(seg2)),
                    np.max(np.abs(seg3)), np.max(np.abs(seg4))
                ])
        
        # 충분히 높은 점수일 때만 마커로 인정
        if best_score > 2.0:  # 4개 구간 중 평균 0.5 이상
            return best_pos
        else:
            return -1
    
    def _check_frequency_match(self, segment, target_freq):
        """세그먼트가 목표 주파수를 포함하는지 확인"""
        if len(segment) == 0:
            return 0
        
        # FFT
        fft = np.fft.fft(segment)
        freqs = np.fft.fftfreq(len(segment), 1/self.sample_rate)
        
        # 양의 주파수만
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        # 목표 주파수 ±100Hz 범위의 파워
        freq_mask = (positive_freqs >= target_freq - 100) & (positive_freqs <= target_freq + 100)
        target_power = np.sum(positive_fft[freq_mask])
        
        # 전체 파워 대비 비율
        total_power = np.sum(positive_fft)
        if total_power > 0:
            return target_power / total_power
        else:
            return 0
    
    def _bytes_to_symbols(self, data):
        """바이트를 심볼 리스트로 변환"""
        # 먼저 비트로 변환
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        
        # 비트를 심볼로 변환 (bits_per_symbol 비트씩 묶음)
        symbols = []
        for i in range(0, len(bits), self.bits_per_symbol):
            symbol_value = 0
            for j in range(self.bits_per_symbol):
                if i + j < len(bits):
                    symbol_value = (symbol_value << 1) | bits[i + j]
            symbols.append(symbol_value)
        
        return symbols
    
    def _symbols_to_bytes(self, symbols):
        """심볼 리스트를 바이트로 변환"""
        # 심볼을 비트로 변환
        bits = []
        for symbol in symbols:
            for i in range(self.bits_per_symbol - 1, -1, -1):
                bits.append((symbol >> i) & 1)
        
        # 비트를 바이트로 변환
        bytes_data = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_data.append(byte)
        
        return bytes(bytes_data)
    
    def _symbols_to_audio_with_clock(self, symbols):
        """심볼을 클럭 신호와 함께 오디오로 변환"""
        samples_per_symbol = int(self.sample_rate * self.symbol_duration)
        total_samples = samples_per_symbol * len(symbols)
        audio = np.zeros(total_samples)
        
        for i, symbol in enumerate(symbols):
            # 심볼 값에 해당하는 주파수 선택
            if symbol < len(self.frequencies):
                freq = self.frequencies[symbol]
            else:
                freq = self.frequencies[0]
            
            t = np.linspace(0, self.symbol_duration, samples_per_symbol, endpoint=False)
            
            # 데이터 신호 (60% 진폭)
            data_signal = np.sin(2 * np.pi * freq * t) * 0.6
            
            # 클럭 신호 (30% 진폭)
            clock_signal = np.sin(2 * np.pi * self.clock_freq * t) * 0.3
            
            # 합성
            signal = data_signal + clock_signal
            
            # 부드러운 전환
            if i < len(symbols) - 1:
                window = np.ones(samples_per_symbol)
                fade_samples = samples_per_symbol // 10
                window[-fade_samples:] = np.linspace(1, 0.5, fade_samples)
                signal *= window
            
            start_idx = i * samples_per_symbol
            audio[start_idx:start_idx + samples_per_symbol] = signal
        
        # 정규화
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio
    
    def _audio_to_symbols_with_clock(self, audio):
        """클럭 신호를 이용해 오디오에서 심볼 추출"""
        samples_per_symbol = int(self.sample_rate * self.symbol_duration)
        num_symbols = len(audio) // samples_per_symbol
        symbols = []
        
        for i in range(num_symbols):
            start_idx = i * samples_per_symbol
            segment = audio[start_idx:start_idx + samples_per_symbol]
            
            if len(segment) < samples_per_symbol:
                break
            
            # FFT 분석
            fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), 1/self.sample_rate)
            
            # 양의 주파수만
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # 각 주파수 대역의 파워 측정 (기준 진폭 고려)
            band_powers = []
            for lower, upper in self.frequency_bands:
                band_mask = (positive_freqs >= lower) & (positive_freqs <= upper)
                band_power = np.sum(positive_fft[band_mask])
                
                # 기준 진폭이 있으면 정규화
                if self.reference_amplitude is not None and self.reference_amplitude > 0:
                    band_power = band_power / (self.reference_amplitude * len(segment))
                
                band_powers.append(band_power)
            
            # 가장 강한 대역의 심볼 선택
            if band_powers:
                symbol = np.argmax(band_powers)
            else:
                symbol = 0
            
            symbols.append(symbol)
        
        return symbols
    
    def _save_wav(self, audio, filename):
        """오디오 신호를 WAV 파일로 저장"""
        # 16비트 정수로 변환
        audio_int16 = np.int16(audio * 32767)
        
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # 모노
            wav_file.setsampwidth(2)  # 16비트
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def _load_wav(self, filename):
        """WAV 파일에서 오디오 신호 읽기"""
        with wave.open(filename, 'r') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio_int16 = np.frombuffer(frames, dtype=np.int16)
            # 정규화
            audio = audio_int16.astype(np.float32) / 32767.0
            return audio


if __name__ == "__main__":
    # 테스트
    print("=" * 70)
    print("향상된 오디오 테이프 시스템 테스트")
    print("=" * 70)
    
    # 테스트 데이터
    test_data = b"Hello, Enhanced Audio Tape System!" * 20
    with open('/home/claude/test_file.txt', 'wb') as f:
        f.write(test_data)
    
    # 2주파수 테스트 (전체 대역 활용)
    print("\n=== 2주파수 (1400-11025 Hz 전체 활용) ===")
    encoder = AudioTapeEncoder(
        compression_level=6,
        num_frequencies=2,
        min_freq=1400,
        max_freq=11025,
        symbol_duration=0.02
    )
    
    print(f"사용 주파수: {[f'{f:.0f}Hz' for f in encoder.frequencies]}")
    print(f"주파수 대역: {[(f'{l:.0f}Hz', f'{u:.0f}Hz') for l, u in encoder.frequency_bands]}")
    
    print("\n인코딩 중...")
    stats = encoder.file_to_audio('/home/claude/test_file.txt', '/home/claude/output_enhanced.wav',
                                   title="테스트", notes="향상된 시스템")
    
    print(f"원본 크기: {stats['original_size']:,} bytes")
    print(f"압축률: {stats['compression_ratio']:.2f}%")
    print(f"오디오 길이: {stats['duration_seconds']:.2f}초")
    print(f"주파수 범위: {stats['frequency_range']}")
    print(f"비트레이트: {stats['bit_rate']:.2f} bps")
    
    print("\n디코딩 중...")
    try:
        result = encoder.audio_to_file('/home/claude/output_enhanced.wav', '/home/claude/restored_enhanced.txt')
        print(f"복원 파일: {result['filename']}")
        print(f"파일 크기: {result['size']:,} bytes")
        
        # 검증
        with open('/home/claude/test_file.txt', 'rb') as f:
            original = f.read()
        with open('/home/claude/restored_enhanced.txt', 'rb') as f:
            restored = f.read()
        
        if original == restored:
            print("✓ 검증 성공!")
        else:
            print("✗ 검증 실패")
    except Exception as e:
        print(f"디코딩 에러: {e}")
    
    print("\n" + "=" * 70)

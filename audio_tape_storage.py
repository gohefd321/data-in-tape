import numpy as np
import wave
import struct
import zlib
from pathlib import Path
import json

class AudioTapeEncoder:
    """오디오 테이프에 데이터를 저장하기 위한 인코더"""
    
    def __init__(self, sample_rate=44100, compression_level=6, num_frequencies=2):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            compression_level: 압축 강도 (0-9, 9가 최대 압축)
            num_frequencies: 사용할 주파수 개수 (2-8)
        """
        self.sample_rate = sample_rate
        self.compression_level = compression_level
        self.num_frequencies = max(2, min(8, num_frequencies))
        
        # 다중 주파수 설정 (800Hz부터 시작, 600Hz 간격)
        self.frequencies = [800 + i * 600 for i in range(self.num_frequencies)]
        
        # 주파수 대역 설정 (각 주파수 ±300Hz)
        self.frequency_bands = []
        for freq in self.frequencies:
            lower = freq - 300
            upper = freq + 300
            self.frequency_bands.append((lower, upper))
        
        self.bit_duration = 0.01  # 각 비트의 지속 시간 (초)
        self.bits_per_symbol = int(np.log2(self.num_frequencies))  # 한 심볼당 비트 수
        
    def file_to_audio(self, input_file, output_wav):
        """파일을 오디오 신호로 변환"""
        # 1. 파일 읽기
        with open(input_file, 'rb') as f:
            file_data = f.read()
        
        # 2. 메타데이터 생성
        metadata = {
            'filename': Path(input_file).name,
            'original_size': len(file_data),
            'compression_level': self.compression_level,
            'num_frequencies': self.num_frequencies,
            'frequencies': self.frequencies
        }
        metadata_bytes = json.dumps(metadata).encode('utf-8')
        metadata_length = len(metadata_bytes)
        
        # 3. 데이터 압축
        if self.compression_level > 0:
            compressed_data = zlib.compress(file_data, level=self.compression_level)
        else:
            compressed_data = file_data
        
        # 4. 전체 데이터 구성: [메타데이터 길이(4바이트)] + [메타데이터] + [압축된 데이터]
        full_data = struct.pack('<I', metadata_length) + metadata_bytes + compressed_data
        
        # 5. 데이터를 심볼 스트림으로 변환
        symbol_stream = self._bytes_to_symbols(full_data)
        
        # 6. 동기화 프리앰블 추가 (주파수 순환 패턴)
        preamble = [i % self.num_frequencies for i in range(100)]
        
        # 7. 시작 마커 추가 (역순 패턴 - 프리앰블과 구별)
        start_marker = []
        for _ in range(4):
            for i in range(self.num_frequencies -1, -1, -1):
                start_marker.append(i)
        
        # 8. 전체 심볼 스트림 구성
        full_symbol_stream = preamble + start_marker + symbol_stream
        
        # 9. 심볼을 오디오 신호로 변환
        audio_signal = self._symbols_to_audio(full_symbol_stream)
        
        # 10. WAV 파일로 저장
        self._save_wav(audio_signal, output_wav)
        
        # 통계 정보 반환
        compression_ratio = len(compressed_data) / len(file_data) * 100
        duration = len(audio_signal) / self.sample_rate
        
        return {
            'original_size': len(file_data),
            'compressed_size': len(compressed_data),
            'compression_ratio': compression_ratio,
            'duration_seconds': duration,
            'symbol_rate': len(symbol_stream) / duration,
            'bit_rate': len(symbol_stream) * self.bits_per_symbol / duration,
            'num_frequencies': self.num_frequencies
        }
    
    def audio_to_file(self, input_wav, output_file=None):
        """오디오 신호를 파일로 복원"""
        # 1. WAV 파일 읽기
        audio_signal = self._load_wav(input_wav)
        
        # 2. 오디오 신호를 심볼로 디코딩
        symbol_stream = self._audio_to_symbols(audio_signal)
        
        # 3. 프리앰블과 시작 마커 찾기
        start_marker = []
        for _ in range(4):
            for i in range(self.num_frequencies -1, -1, -1):
                start_marker.append(i)
        
        data_start = self._find_start_marker(symbol_stream, start_marker)
        
        if data_start == -1:
            raise ValueError("시작 마커를 찾을 수 없습니다")
        
        # 4. 데이터 심볼 추출
        data_symbols = symbol_stream[data_start:]
        
        # 5. 심볼을 바이트로 변환
        data_bytes = self._symbols_to_bytes(data_symbols)
        
        # 6. 메타데이터 길이 읽기
        if len(data_bytes) < 4:
            raise ValueError("데이터가 너무 짧습니다")
        
        metadata_length = struct.unpack('<I', data_bytes[:4])[0]
        
        # 메타데이터 길이가 비정상적으로 크면 에러
        if metadata_length > 10000 or metadata_length <= 0:
            raise ValueError(f"잘못된 메타데이터 길이: {metadata_length}")
        
        # 7. 메타데이터 읽기
        if len(data_bytes) < 4 + metadata_length:
            raise ValueError(f"데이터가 메타데이터를 포함하기에 충분하지 않습니다. 필요: {4 + metadata_length}, 실제: {len(data_bytes)}")
        
        metadata_bytes = data_bytes[4:4+metadata_length]
        
        try:
            metadata = json.loads(metadata_bytes.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"메타데이터 파싱 실패: {str(e)}")
        
        # 메타데이터에서 주파수 정보 복원
        if 'num_frequencies' in metadata:
            self.num_frequencies = metadata['num_frequencies']
            self.frequencies = metadata['frequencies']
            self.bits_per_symbol = int(np.log2(self.num_frequencies))
        
        # 8. 압축된 데이터 추출
        compressed_data = data_bytes[4+metadata_length:]
        
        # 9. 데이터 압축 해제
        if metadata['compression_level'] > 0:
            try:
                file_data = zlib.decompress(compressed_data)
            except zlib.error as e:
                raise ValueError(f"압축 해제 실패: {str(e)}")
        else:
            file_data = compressed_data
        
        # 10. 파일로 저장
        if output_file is None:
            output_file = metadata['filename']
        
        with open(output_file, 'wb') as f:
            f.write(file_data)
        
        return {
            'filename': output_file,
            'size': len(file_data),
            'metadata': metadata
        }
    
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
    
    def _symbols_to_audio(self, symbols):
        """심볼 스트림을 다중 주파수 FSK 오디오 신호로 변환"""
        samples_per_symbol = int(self.sample_rate * self.bit_duration)
        total_samples = samples_per_symbol * len(symbols)
        audio = np.zeros(total_samples)
        
        for i, symbol in enumerate(symbols):
            # 심볼 값에 해당하는 주파수 선택
            if symbol < len(self.frequencies):
                freq = self.frequencies[symbol]
            else:
                freq = self.frequencies[0]  # 오류 방지
            
            t = np.linspace(0, self.bit_duration, samples_per_symbol, endpoint=False)
            
            # 사인파 생성
            signal = np.sin(2 * np.pi * freq * t)
            
            # 클릭 노이즈 방지를 위한 부드러운 전환
            if i < len(symbols) - 1:
                window = np.ones(samples_per_symbol)
                fade_samples = samples_per_symbol // 10
                window[-fade_samples:] = np.linspace(1, 0, fade_samples)
                signal *= window
            
            start_idx = i * samples_per_symbol
            audio[start_idx:start_idx + samples_per_symbol] = signal
        
        # 정규화
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def _audio_to_symbols(self, audio):
        """다중 주파수 FSK 오디오 신호를 심볼 스트림으로 디코딩 (대역 판별)"""
        samples_per_symbol = int(self.sample_rate * self.bit_duration)
        num_symbols = len(audio) // samples_per_symbol
        symbols = []
        
        for i in range(num_symbols):
            start_idx = i * samples_per_symbol
            segment = audio[start_idx:start_idx + samples_per_symbol]
            
            # FFT를 사용하여 주파수 분석
            fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), 1/self.sample_rate)
            
            # 양의 주파수만 사용
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # 각 주파수 대역의 파워 측정
            band_powers = []
            for lower, upper in self.frequency_bands:
                # 대역 내 주파수 인덱스 찾기
                band_mask = (positive_freqs >= lower) & (positive_freqs <= upper)
                band_power = np.sum(positive_fft[band_mask])
                band_powers.append(band_power)
            
            # 가장 강한 대역에 해당하는 심볼 선택
            if band_powers:
                symbol = np.argmax(band_powers)
            else:
                symbol = 0
            
            symbols.append(symbol)
        
        return symbols
    
    def _find_start_marker(self, stream, marker):
        """스트림에서 시작 마커 찾기"""
        marker_len = len(marker)
        for i in range(len(stream) - marker_len):
            # 심볼 스트림 비교 (numpy int도 처리)
            match = True
            for j in range(marker_len):
                if int(stream[i + j]) != int(marker[j]):
                    match = False
                    break
            if match:
                return i + marker_len
        return -1
    
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
    # 다양한 주파수 개수로 테스트
    print("=" * 70)
    print("다중 주파수 테스트")
    print("=" * 70)
    
    test_data = b"Hello, Audio Tape Storage with Multi-Frequency!" * 50
    with open('/home/claude/test_file.txt', 'wb') as f:
        f.write(test_data)
    
    for num_freq in [2, 4, 8]:
        print(f"\n{'='*70}")
        print(f"주파수 개수: {num_freq}")
        print(f"{'='*70}")
        
        encoder = AudioTapeEncoder(compression_level=6, num_frequencies=num_freq)
        
        print(f"사용 주파수: {encoder.frequencies}")
        print(f"주파수 대역: {encoder.frequency_bands}")
        print(f"심볼당 비트 수: {encoder.bits_per_symbol}")
        
        print("\n=== 인코딩 중 ===")
        stats = encoder.file_to_audio('/home/claude/test_file.txt', 
                                       f'/home/claude/output_{num_freq}freq.wav')
        print(f"원본 크기: {stats['original_size']:,} bytes")
        print(f"압축 후 크기: {stats['compressed_size']:,} bytes")
        print(f"압축률: {stats['compression_ratio']:.2f}%")
        print(f"오디오 길이: {stats['duration_seconds']:.2f}초")
        print(f"심볼 레이트: {stats['symbol_rate']:.2f} symbols/sec")
        print(f"비트 레이트: {stats['bit_rate']:.2f} bps")
        print(f"대역폭 증가율: {num_freq/2 * 100:.0f}%")
        
        print("\n=== 디코딩 중 ===")
        result = encoder.audio_to_file(f'/home/claude/output_{num_freq}freq.wav',
                                        f'/home/claude/restored_{num_freq}freq.txt')
        print(f"복원된 파일: {result['filename']}")
        print(f"파일 크기: {result['size']:,} bytes")
        
        # 검증
        with open('/home/claude/test_file.txt', 'rb') as f:
            original = f.read()
        with open(f'/home/claude/restored_{num_freq}freq.txt', 'rb') as f:
            restored = f.read()
        
        if original == restored:
            print("✓ 검증 성공! 원본과 복원된 파일이 동일합니다.")
        else:
            print("✗ 검증 실패! 파일이 다릅니다.")
    
    print("\n" + "=" * 70)

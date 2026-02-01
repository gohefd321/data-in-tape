import numpy as np
import wave
import struct
import zlib
from pathlib import Path
import json

class AudioTapeEncoder:
    """오디오 테이프에 데이터를 저장하기 위한 인코더"""
    
    def __init__(self, sample_rate=44100, compression_level=6):
        """
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            compression_level: 압축 강도 (0-9, 9가 최대 압축)
        """
        self.sample_rate = sample_rate
        self.compression_level = compression_level
        
        # FSK 파라미터 (테이프에 적합한 주파수)
        self.freq_0 = 1200  # 0 비트를 나타내는 주파수 (Hz)
        self.freq_1 = 2400  # 1 비트를 나타내는 주파수 (Hz)
        self.bit_duration = 0.01  # 각 비트의 지속 시간 (초)
        
    def file_to_audio(self, input_file, output_wav):
        """파일을 오디오 신호로 변환"""
        # 1. 파일 읽기
        with open(input_file, 'rb') as f:
            file_data = f.read()
        
        # 2. 메타데이터 생성
        metadata = {
            'filename': Path(input_file).name,
            'original_size': len(file_data),
            'compression_level': self.compression_level
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
        
        # 5. 데이터를 비트 스트림으로 변환
        bit_stream = self._bytes_to_bits(full_data)
        
        # 6. 동기화 프리앰블 추가 (1010... 패턴 100비트)
        preamble = [i % 2 for i in range(100)]
        
        # 7. 시작 마커 추가 (특별한 패턴)
        start_marker = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        
        # 8. 전체 비트 스트림 구성
        full_bit_stream = preamble + start_marker + bit_stream
        
        # 9. 비트를 오디오 신호로 변환
        audio_signal = self._bits_to_audio(full_bit_stream)
        
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
            'bitrate': len(bit_stream) / duration
        }
    
    def audio_to_file(self, input_wav, output_file=None):
        """오디오 신호를 파일로 복원"""
        # 1. WAV 파일 읽기
        audio_signal = self._load_wav(input_wav)
        
        # 2. 오디오 신호를 비트로 디코딩
        bit_stream = self._audio_to_bits(audio_signal)
        
        # 3. 프리앰블과 시작 마커 찾기
        start_marker = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
        data_start = self._find_start_marker(bit_stream, start_marker)
        
        if data_start == -1:
            raise ValueError("시작 마커를 찾을 수 없습니다")
        
        # 4. 데이터 비트 추출
        data_bits = bit_stream[data_start:]
        
        # 5. 비트를 바이트로 변환
        data_bytes = self._bits_to_bytes(data_bits)
        
        # 6. 메타데이터 길이 읽기
        if len(data_bytes) < 4:
            raise ValueError("데이터가 너무 짧습니다")
        
        metadata_length = struct.unpack('<I', data_bytes[:4])[0]
        
        # 7. 메타데이터 읽기
        metadata_bytes = data_bytes[4:4+metadata_length]
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        
        # 8. 압축된 데이터 추출
        compressed_data = data_bytes[4+metadata_length:]
        
        # 9. 데이터 압축 해제
        if metadata['compression_level'] > 0:
            file_data = zlib.decompress(compressed_data)
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
    
    def _bytes_to_bits(self, data):
        """바이트를 비트 리스트로 변환"""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits
    
    def _bits_to_bytes(self, bits):
        """비트 리스트를 바이트로 변환"""
        # 8비트씩 묶어서 바이트로 변환
        bytes_data = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_data.append(byte)
        return bytes(bytes_data)
    
    def _bits_to_audio(self, bits):
        """비트 스트림을 FSK 오디오 신호로 변환"""
        samples_per_bit = int(self.sample_rate * self.bit_duration)
        total_samples = samples_per_bit * len(bits)
        audio = np.zeros(total_samples)
        
        for i, bit in enumerate(bits):
            freq = self.freq_1 if bit == 1 else self.freq_0
            t = np.linspace(0, self.bit_duration, samples_per_bit, endpoint=False)
            
            # 사인파 생성
            signal = np.sin(2 * np.pi * freq * t)
            
            # 클릭 노이즈 방지를 위한 부드러운 전환
            if i < len(bits) - 1:
                window = np.ones(samples_per_bit)
                fade_samples = samples_per_bit // 10
                window[-fade_samples:] = np.linspace(1, 0, fade_samples)
                signal *= window
            
            start_idx = i * samples_per_bit
            audio[start_idx:start_idx + samples_per_bit] = signal
        
        # 정규화
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def _audio_to_bits(self, audio):
        """FSK 오디오 신호를 비트 스트림으로 디코딩"""
        samples_per_bit = int(self.sample_rate * self.bit_duration)
        num_bits = len(audio) // samples_per_bit
        bits = []
        
        for i in range(num_bits):
            start_idx = i * samples_per_bit
            segment = audio[start_idx:start_idx + samples_per_bit]
            
            # FFT를 사용하여 주파수 분석
            fft = np.fft.fft(segment)
            freqs = np.fft.fftfreq(len(segment), 1/self.sample_rate)
            
            # 양의 주파수만 사용
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # freq_0과 freq_1 근처의 파워 측정
            freq_0_idx = np.argmin(np.abs(positive_freqs - self.freq_0))
            freq_1_idx = np.argmin(np.abs(positive_freqs - self.freq_1))
            
            power_0 = positive_fft[freq_0_idx]
            power_1 = positive_fft[freq_1_idx]
            
            # 더 강한 주파수에 따라 비트 결정
            bits.append(1 if power_1 > power_0 else 0)
        
        return bits
    
    def _find_start_marker(self, bits, marker):
        """비트 스트림에서 시작 마커 찾기"""
        marker_len = len(marker)
        for i in range(len(bits) - marker_len):
            if bits[i:i+marker_len] == marker:
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
    # 테스트 코드
    encoder = AudioTapeEncoder(compression_level=6)
    
    # 테스트 파일 생성
    test_data = b"Hello, Audio Tape Storage! This is a test message." * 100
    with open('/home/claude/test_file.txt', 'wb') as f:
        f.write(test_data)
    
    print("=== 인코딩 중 ===")
    stats = encoder.file_to_audio('/home/claude/test_file.txt', '/home/claude/output.wav')
    print(f"원본 크기: {stats['original_size']} 바이트")
    print(f"압축 후 크기: {stats['compressed_size']} 바이트")
    print(f"압축률: {stats['compression_ratio']:.2f}%")
    print(f"오디오 길이: {stats['duration_seconds']:.2f}초")
    print(f"비트레이트: {stats['bitrate']:.2f} bps")
    
    print("\n=== 디코딩 중 ===")
    result = encoder.audio_to_file('/home/claude/output.wav', '/home/claude/restored_file.txt')
    print(f"복원된 파일: {result['filename']}")
    print(f"파일 크기: {result['size']} 바이트")
    
    # 검증
    with open('/home/claude/test_file.txt', 'rb') as f:
        original = f.read()
    with open('/home/claude/restored_file.txt', 'rb') as f:
        restored = f.read()
    
    if original == restored:
        print("\n✓ 검증 성공! 원본과 복원된 파일이 동일합니다.")
    else:
        print("\n✗ 검증 실패! 파일이 다릅니다.")

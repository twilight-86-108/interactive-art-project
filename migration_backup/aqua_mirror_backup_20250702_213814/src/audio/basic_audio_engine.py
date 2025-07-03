"""
基本音響システム - Week 2版
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import threading
import time
import math

# sounddevice がインストールされていない場合のフォールバック
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class BasicAudioEngine:
    """基本音響システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("BasicAudioEngine")
        self.config = config
        
        # 音響設定
        self.sample_rate = config.get('audio.sample_rate', 44100)
        self.buffer_size = config.get('audio.buffer_size', 1024)
        self.master_volume = config.get('audio.master_volume', 0.5)
        
        # 音響ストリーム
        self.audio_stream: Optional[sd.OutputStream] = None
        self.playing = False
        
        # 感情音響設定
        self.emotion_frequencies = {
            'HAPPY': 440.0,     # A4
            'SAD': 220.0,       # A3
            'ANGRY': 110.0,     # A2
            'SURPRISED': 880.0, # A5
            'FEAR': 330.0,      # E4
            'NEUTRAL': 261.63   # C4
        }
        
        # 現在の音響状態
        self.current_emotion = 'NEUTRAL'
        self.current_intensity = 0.0
        self.wave_intensity = 0.0
        
        # 音生成パラメータ
        self.time_offset = 0.0
        
        if AUDIO_AVAILABLE:
            self.logger.info("🔊 基本音響システム初期化完了")
        else:
            self.logger.warning("⚠️ sounddevice未インストール - 音響機能無効")
    
    def initialize(self) -> bool:
        """音響システム初期化"""
        try:
            if not AUDIO_AVAILABLE:
                self.logger.warning("⚠️ 音響システム利用不可")
                return False
            
            # 音響デバイス確認
            devices = sd.query_devices()
            self.logger.info(f"利用可能音響デバイス数: {len(devices)}")
            
            # 音響ストリーム作成
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                callback=self._audio_callback,
                blocksize=self.buffer_size
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 音響システム初期化失敗: {e}")
            return False
    
    def start(self):
        """音響再生開始"""
        try:
            if self.audio_stream and not self.playing:
                self.audio_stream.start()
                self.playing = True
                self.logger.info("🎵 音響再生開始")
        except Exception as e:
            self.logger.error(f"❌ 音響再生開始失敗: {e}")
    
    def stop(self):
        """音響再生停止"""
        try:
            if self.audio_stream and self.playing:
                self.audio_stream.stop()
                self.playing = False
                self.logger.info("⏹️ 音響再生停止")
        except Exception as e:
            self.logger.error(f"❌ 音響再生停止失敗: {e}")
    
    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """音響コールバック"""
        try:
            # 感情音生成
            emotion_audio = self._generate_emotion_audio(frames)
            
            # 水音生成
            water_audio = self._generate_water_audio(frames)
            
            # ミックス
            mixed_audio = (emotion_audio + water_audio) * 0.5
            
            # 音量適用
            outdata[:] = mixed_audio * self.master_volume
            
        except Exception:
            # エラー時は無音
            outdata.fill(0)
    
    def _generate_emotion_audio(self, frames: int) -> np.ndarray:
        """感情音生成"""
        try:
            if self.current_intensity < 0.1:
                return np.zeros((frames, 2))
            
            # 基本周波数
            base_freq = self.emotion_frequencies.get(self.current_emotion, 261.63)
            
            # 時間配列
            t = np.linspace(self.time_offset, 
                          self.time_offset + frames / self.sample_rate, 
                          frames, endpoint=False)
            
            # 基本波形生成
            wave = np.sin(2 * np.pi * base_freq * t)
            
            # ハーモニクス追加
            wave += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)  # 2倍音
            wave += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)  # 3倍音
            
            # エンベロープ適用
            wave *= self.current_intensity * 0.3
            
            # ステレオ変換
            stereo_audio = np.column_stack([wave, wave])
            
            # 時間オフセット更新
            self.time_offset += frames / self.sample_rate
            
            return stereo_audio.astype(np.float32)
            
        except Exception:
            return np.zeros((frames, 2))
    
    def _generate_water_audio(self, frames: int) -> np.ndarray:
        """水音生成"""
        try:
            if self.wave_intensity < 0.1:
                return np.zeros((frames, 2))
            
            # ホワイトノイズベース
            noise = np.random.normal(0, 0.1, frames)
            
            # ローパスフィルター（水音らしく）
            filtered_noise = noise * 0.5
            
            # 波の強度に応じた音量
            volume = self.wave_intensity * 0.2
            
            # ステレオ変換
            water_audio = np.column_stack([filtered_noise, filtered_noise]) * volume
            
            return water_audio.astype(np.float32)
            
        except Exception:
            return np.zeros((frames, 2))
    
    def update_emotion_audio(self, emotion_result):
        """感情音響更新"""
        if emotion_result:
            self.current_emotion = emotion_result.emotion
            self.current_intensity = emotion_result.confidence
        else:
            self.current_emotion = 'NEUTRAL'
            self.current_intensity = 0.0
    
    def update_wave_audio(self, wave_sources):
        """波音更新"""
        if wave_sources:
            total_intensity = sum(source.intensity for source in wave_sources)
            self.wave_intensity = min(total_intensity, 1.0)
        else:
            self.wave_intensity = 0.0
    
    def set_master_volume(self, volume: float):
        """マスター音量設定"""
        self.master_volume = max(0.0, min(volume, 1.0))
    
    def cleanup(self):
        """リソース解放"""
        try:
            self.stop()
            if self.audio_stream:
                self.audio_stream.close()
            self.logger.info("✅ 音響システムリソース解放完了")
        except Exception as e:
            self.logger.error(f"❌ 音響システムリソース解放失敗: {e}")

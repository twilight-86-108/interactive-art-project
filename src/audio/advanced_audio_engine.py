"""
高品質音響システム - 感情・水面連動
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import threading
import time
import math
from collections import deque

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class AdvancedAudioEngine:
    """高品質音響システム - 感情・水面連動"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("AdvancedAudioEngine")
        self.config = config
        
        # 音響設定
        self.sample_rate = config.get('audio.sample_rate', 44100)
        self.buffer_size = config.get('audio.buffer_size', 1024)
        self.channels = 2
        self.master_volume = config.get('audio.master_volume', 0.5)
        
        # 音響ストリーム
        self.audio_stream: Optional = None
        self.playing = False
        
        # 高度感情音響マッピング
        self.emotion_audio_configs = {
            'HAPPY': {
                'base_frequency': 440.0,  # A4
                'harmonics': [1.0, 0.6, 0.4, 0.2, 0.1],
                'wave_type': 'sine_major',
                'reverb': 0.4,
                'brightness': 0.9,
                'modulation_rate': 3.0,
                'envelope': 'smooth'
            },
            'SAD': {
                'base_frequency': 220.0,  # A3
                'harmonics': [1.0, 0.4, 0.2, 0.1],
                'wave_type': 'sine_minor',
                'reverb': 0.8,
                'brightness': 0.3,
                'modulation_rate': 1.0,
                'envelope': 'slow'
            },
            'ANGRY': {
                'base_frequency': 110.0,  # A2
                'harmonics': [1.0, 0.8, 0.6, 0.4, 0.3],
                'wave_type': 'sawtooth',
                'reverb': 0.2,
                'brightness': 1.0,
                'modulation_rate': 8.0,
                'envelope': 'sharp'
            },
            'SURPRISED': {
                'base_frequency': 880.0,  # A5
                'harmonics': [1.0, 0.7, 0.5, 0.3, 0.2],
                'wave_type': 'sine_bright',
                'reverb': 0.5,
                'brightness': 1.0,
                'modulation_rate': 6.0,
                'envelope': 'attack'
            },
            'FEAR': {
                'base_frequency': 330.0,  # E4
                'harmonics': [1.0, 0.3, 0.7, 0.2, 0.5],
                'wave_type': 'tremolo',
                'reverb': 0.9,
                'brightness': 0.4,
                'modulation_rate': 12.0,
                'envelope': 'nervous'
            },
            'DISGUST': {
                'base_frequency': 160.0,  # 低め
                'harmonics': [1.0, 0.2, 0.8, 0.1],
                'wave_type': 'distorted',
                'reverb': 0.3,
                'brightness': 0.6,
                'modulation_rate': 2.0,
                'envelope': 'harsh'
            },
            'NEUTRAL': {
                'base_frequency': 261.63,  # C4
                'harmonics': [1.0, 0.5, 0.25, 0.125],
                'wave_type': 'sine',
                'reverb': 0.3,
                'brightness': 0.5,
                'modulation_rate': 0.5,
                'envelope': 'gentle'
            }
        }
        
        # 現在の音響状態
        self.current_emotion = 'NEUTRAL'
        self.current_intensity = 0.0
        self.target_intensity = 0.0
        
        # 水音システム
        self.water_audio = WaterAudioGenerator(self.sample_rate)
        
        # 環境音システム
        self.ambient_audio = AmbientAudioGenerator(self.sample_rate)
        
        # 音響履歴（平滑化用）
        self.emotion_history = deque(maxlen=30)  # 1秒分
        
        # 時間管理
        self.time_offset = 0.0
        self.last_update_time = time.time()
        
        if AUDIO_AVAILABLE:
            self.logger.info("🔊 高品質音響システム初期化完了")
        else:
            self.logger.warning("⚠️ sounddevice未インストール - 音響機能無効")
    
    def initialize(self) -> bool:
        """音響システム初期化"""
        try:
            if not AUDIO_AVAILABLE:
                return False
            
            # 音響デバイス確認
            devices = sd.query_devices()
            self.logger.info(f"利用可能音響デバイス数: {len(devices)}")
            
            # 音響ストリーム作成
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
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
                self.logger.info("🎵 高品質音響再生開始")
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
        """高品質音響コールバック"""
        try:
            # 感情音響生成
            emotion_audio = self._generate_advanced_emotion_audio(frames)
            
            # 水音生成
            water_audio = self.water_audio.generate(frames, self.channels)
            
            # 環境音生成
            ambient_audio = self.ambient_audio.generate(frames, self.channels)
            
            # 高品質ミックス
            mixed_audio = self._advanced_mix_audio([emotion_audio, water_audio, ambient_audio])
            
            # マスタリング
            final_audio = self._apply_mastering(mixed_audio)
            
            # 出力
            outdata[:] = final_audio * self.master_volume
            
        except Exception as e:
            outdata.fill(0)
            self.logger.error(f"❌ 音響コールバックエラー: {e}")
    
    def _generate_advanced_emotion_audio(self, frames: int) -> np.ndarray:
        """高度感情音響生成"""
        try:
            if self.current_intensity < 0.05:
                return np.zeros((frames, self.channels))
            
            # 感情設定取得
            config = self.emotion_audio_configs.get(self.current_emotion, 
                                                   self.emotion_audio_configs['NEUTRAL'])
            
            # 時間配列
            dt = 1.0 / self.sample_rate
            t = np.arange(frames) * dt + self.time_offset
            
            # 基本周波数（感情による変調）
            base_freq = config['base_frequency']
            modulation_rate = config['modulation_rate']
            
            # 周波数変調
            freq_modulation = 1.0 + 0.05 * np.sin(2 * np.pi * modulation_rate * t)
            instantaneous_freq = base_freq * freq_modulation
            
            # 基本波形生成
            if config['wave_type'] == 'sine_major':
                # メジャースケール和音
                audio = (np.sin(2 * np.pi * instantaneous_freq * t) +
                        0.6 * np.sin(2 * np.pi * instantaneous_freq * 1.25 * t) +  # 3度
                        0.4 * np.sin(2 * np.pi * instantaneous_freq * 1.5 * t))    # 5度
                        
            elif config['wave_type'] == 'sine_minor':
                # マイナースケール和音
                audio = (np.sin(2 * np.pi * instantaneous_freq * t) +
                        0.6 * np.sin(2 * np.pi * instantaneous_freq * 1.2 * t) +   # 短3度
                        0.4 * np.sin(2 * np.pi * instantaneous_freq * 1.5 * t))    # 5度
                        
            elif config['wave_type'] == 'sawtooth':
                # ノコギリ波（倍音豊富）
                audio = np.zeros_like(t)
                for harmonic in range(1, 8):
                    audio += np.sin(2 * np.pi * instantaneous_freq * harmonic * t) / harmonic
                    
            elif config['wave_type'] == 'tremolo':
                # トレモロ効果
                tremolo_freq = 8.0
                tremolo = 1.0 + 0.5 * np.sin(2 * np.pi * tremolo_freq * t)
                audio = np.sin(2 * np.pi * instantaneous_freq * t) * tremolo
                
            else:  # デフォルト
                audio = np.sin(2 * np.pi * instantaneous_freq * t)
            
            # ハーモニクス追加
            harmonics = config['harmonics']
            harmonic_audio = np.zeros_like(audio)
            for i, strength in enumerate(harmonics[1:], 2):
                harmonic_audio += strength * np.sin(2 * np.pi * instantaneous_freq * i * t)
            
            audio += harmonic_audio * 0.5
            
            # エンベロープ適用
            envelope = self._generate_envelope(frames, config['envelope'])
            audio *= envelope
            
            # 強度・ブライトネス適用
            brightness = config['brightness']
            if brightness > 0.7:
                # 高音強調
                audio = self._apply_brightness_filter(audio, brightness)
            
            # 強度適用
            intensity = self._get_smoothed_intensity()
            audio *= intensity * 0.3
            
            # ステレオ処理
            stereo_audio = self._apply_stereo_effects(audio, config)
            
            # 時間オフセット更新
            self.time_offset += frames * dt
            
            return stereo_audio.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"❌ 高度感情音響生成失敗: {e}")
            return np.zeros((frames, self.channels))
    
    def _generate_envelope(self, frames: int, envelope_type: str) -> np.ndarray:
        """エンベロープ生成"""
        t = np.linspace(0, 1, frames)
        
        if envelope_type == 'smooth':
            return 0.5 + 0.5 * np.sin(2 * np.pi * t - np.pi/2)
        elif envelope_type == 'attack':
            return np.minimum(t * 4, 1.0)
        elif envelope_type == 'slow':
            return np.power(t, 0.3)
        elif envelope_type == 'sharp':
            return np.where(t < 0.1, t * 10, 1.0)
        elif envelope_type == 'nervous':
            return 0.7 + 0.3 * np.sin(t * 20 * np.pi)
        elif envelope_type == 'harsh':
            return np.where(t < 0.05, 1.0, 0.8 + 0.2 * np.random.random(frames - int(0.05 * frames)))
        else:  # gentle
            return np.ones(frames)
    
    def _apply_brightness_filter(self, audio: np.ndarray, brightness: float) -> np.ndarray:
        """ブライトネスフィルター適用"""
        try:
            # 簡易ハイパスフィルター
            if len(audio) < 3:
                return audio
            
            filtered = np.copy(audio)
            for i in range(1, len(audio)):
                filtered[i] = audio[i] - 0.3 * audio[i-1] * brightness
            
            return filtered
            
        except Exception:
            return audio
    
    def _apply_stereo_effects(self, mono_audio: np.ndarray, config: Dict) -> np.ndarray:
        """ステレオエフェクト適用"""
        try:
            left_channel = mono_audio.copy()
            right_channel = mono_audio.copy()
            
            # 感情による位相・遅延効果
            if config['wave_type'] == 'tremolo':
                # 左右で位相をずらす
                delay_samples = int(0.001 * self.sample_rate)  # 1ms遅延
                if len(right_channel) > delay_samples:
                    right_channel[delay_samples:] = mono_audio[:-delay_samples]
                    right_channel[:delay_samples] = 0
            
            elif config['wave_type'] == 'sine_major':
                # 左右で若干音程をずらしてコーラス効果
                right_channel *= 1.01  # 微妙な音程差
            
            # リバーブシミュレーション（簡易）
            reverb_strength = config['reverb']
            if reverb_strength > 0.3:
                reverb_delay = int(0.03 * self.sample_rate)  # 30ms
                if len(left_channel) > reverb_delay:
                    left_channel[reverb_delay:] += mono_audio[:-reverb_delay] * reverb_strength * 0.3
                    right_channel[reverb_delay:] += mono_audio[:-reverb_delay] * reverb_strength * 0.3
            
            return np.column_stack([left_channel, right_channel])
            
        except Exception:
            return np.column_stack([mono_audio, mono_audio])
    
    def _get_smoothed_intensity(self) -> float:
        """平滑化された強度取得"""
        # 目標強度への滑らかな変化
        smoothing_factor = 0.05
        self.current_intensity += (self.target_intensity - self.current_intensity) * smoothing_factor
        return self.current_intensity
    
    def _advanced_mix_audio(self, audio_sources: List[np.ndarray]) -> np.ndarray:
        """高品質音響ミックス"""
        try:
            if not audio_sources:
                return np.zeros((self.buffer_size, self.channels))
            
            # 全音源を同じサイズに調整
            max_frames = max(len(audio) for audio in audio_sources if audio is not None)
            mixed_audio = np.zeros((max_frames, self.channels))
            
            # ダイナミックミックス
            for i, audio in enumerate(audio_sources):
                if audio is not None and len(audio) > 0:
                    # サイズ調整
                    if len(audio) != max_frames:
                        padded_audio = np.zeros((max_frames, self.channels))
                        min_len = min(len(audio), max_frames)
                        padded_audio[:min_len] = audio[:min_len]
                        audio = padded_audio
                    
                    # 音源別重み
                    if i == 0:  # 感情音響
                        weight = 0.6
                    elif i == 1:  # 水音
                        weight = 0.3
                    else:  # 環境音
                        weight = 0.1
                    
                    mixed_audio += audio * weight
            
            return mixed_audio
            
        except Exception as e:
            self.logger.error(f"❌ 高品質音響ミックス失敗: {e}")
            return np.zeros((self.buffer_size, self.channels))
    
    def _apply_mastering(self, audio: np.ndarray) -> np.ndarray:
        """マスタリング処理"""
        try:
            # ソフトクリッピング
            audio = np.tanh(audio)
            
            # 正規化
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0.8:
                audio = audio * 0.8 / max_amplitude
            
            return audio
            
        except Exception:
            return audio
    
    def update_emotion_audio(self, emotion_result):
        """感情音響更新"""
        try:
            if emotion_result:
                # 感情履歴に追加
                self.emotion_history.append({
                    'emotion': emotion_result.emotion,
                    'confidence': emotion_result.confidence,
                    'timestamp': time.time()
                })
                
                # 最新感情設定
                self.current_emotion = emotion_result.emotion
                self.target_intensity = emotion_result.confidence * 0.8
            else:
                self.current_emotion = 'NEUTRAL'
                self.target_intensity = 0.0
                
        except Exception as e:
            self.logger.error(f"❌ 感情音響更新失敗: {e}")
    
    def update_water_audio(self, wave_sources):
        """水音更新"""
        try:
            self.water_audio.update_wave_sources(wave_sources)
        except Exception as e:
            self.logger.error(f"❌ 水音更新失敗: {e}")
    
    def set_master_volume(self, volume: float):
        """マスター音量設定"""
        self.master_volume = np.clip(volume, 0.0, 1.0)
    
    def get_audio_statistics(self) -> Dict[str, Any]:
        """音響統計取得"""
        return {
            'current_emotion': self.current_emotion,
            'intensity': self.current_intensity,
            'target_intensity': self.target_intensity,
            'master_volume': self.master_volume,
            'emotion_history_length': len(self.emotion_history),
            'water_sources': len(self.water_audio.wave_sources) if hasattr(self.water_audio, 'wave_sources') else 0
        }
    
    def cleanup(self):
        """リソース解放"""
        try:
            self.stop()
            if self.audio_stream:
                self.audio_stream.close()
            self.logger.info("✅ 高品質音響システムリソース解放完了")
        except Exception as e:
            self.logger.error(f"❌ 音響システムリソース解放失敗: {e}")


class WaterAudioGenerator:
    """水音生成器"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.wave_sources = []
    
    def update_wave_sources(self, wave_sources):
        """波源更新"""
        self.wave_sources = wave_sources
    
    def generate(self, frames: int, channels: int) -> np.ndarray:
        """水音生成"""
        try:
            if not self.wave_sources:
                return np.zeros((frames, channels))
            
            # ベースノイズ
            noise = np.random.normal(0, 0.1, frames)
            
            # ローパスフィルター（水音らしく）
            if frames > 2:
                filtered_noise = np.copy(noise)
                for i in range(1, frames):
                    filtered_noise[i] = 0.7 * filtered_noise[i] + 0.3 * filtered_noise[i-1]
                noise = filtered_noise
            
            # 波源の強度に応じて音量調整
            total_intensity = sum(source.get('intensity', 0) for source in self.wave_sources)
            volume = min(total_intensity * 0.15, 0.4)
            
            water_audio = noise * volume
            
            # ステレオ変換
            stereo_audio = np.column_stack([water_audio, water_audio])
            
            return stereo_audio.astype(np.float32)
            
        except Exception:
            return np.zeros((frames, channels))


class AmbientAudioGenerator:
    """環境音生成器"""
    
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.time_offset = 0.0
    
    def generate(self, frames: int, channels: int) -> np.ndarray:
        """環境音生成"""
        try:
            # 静寂な環境音（微細なホワイトノイズ）
            ambient = np.random.normal(0, 0.02, frames)
            
            # 時間更新
            self.time_offset += frames / self.sample_rate
            
            # ステレオ変換
            stereo_ambient = np.column_stack([ambient, ambient])
            
            return stereo_ambient.astype(np.float32)
            
        except Exception:
            return np.zeros((frames, channels))

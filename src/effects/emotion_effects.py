"""
感情エフェクト管理システム
シェーダーパラメータ制御・視覚エフェクト統合
"""

import moderngl
import numpy as np
import logging
import time
import math
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from src.emotion.emotion_analyzer import EmotionType

class EffectIntensity(Enum):
    """エフェクト強度レベル"""
    SUBTLE = 0.3
    NORMAL = 0.6
    INTENSE = 1.0

class EmotionEffectManager:
    """
    感情エフェクト管理システム
    感情認識→シェーダーパラメータ変換
    """
    
    def __init__(self, ctx: moderngl.Context, config):
        self.ctx = ctx
        self.config = config
        self.logger = logging.getLogger("EmotionEffectManager")
        
        # エフェクト設定
        self.effect_intensity = EffectIntensity.NORMAL
        self.enable_ripples = True
        self.enable_glow = True
        self.enable_color_blend = True
        
        # エフェクトパラメータ
        self.current_params = {
            'emotion_type': 4,  # NEUTRAL
            'emotion_intensity': 0.0,
            'emotion_color': (0.5, 0.5, 0.5),
            'emotion_confidence': 0.0,
            'ripple_strength': 0.5,
            'color_blend_factor': 0.3,
            'glow_intensity': 0.4
        }
        
        # アニメーション状態
        self.animation_time = 0.0
        self.transition_speed = 2.0  # パラメータ変化速度
        self.target_params = self.current_params.copy()
        
        # 感情固有設定
        self.emotion_settings = {
            EmotionType.HAPPY: {
                'ripple_strength': 0.6,
                'color_blend_factor': 0.4,
                'glow_intensity': 0.7,
                'animation_speed': 1.5
            },
            EmotionType.SAD: {
                'ripple_strength': 0.3,
                'color_blend_factor': 0.5,
                'glow_intensity': 0.3,
                'animation_speed': 0.8
            },
            EmotionType.ANGRY: {
                'ripple_strength': 0.8,
                'color_blend_factor': 0.6,
                'glow_intensity': 0.9,
                'animation_speed': 2.5
            },
            EmotionType.SURPRISED: {
                'ripple_strength': 0.7,
                'color_blend_factor': 0.3,
                'glow_intensity': 0.8,
                'animation_speed': 3.0
            },
            EmotionType.NEUTRAL: {
                'ripple_strength': 0.2,
                'color_blend_factor': 0.1,
                'glow_intensity': 0.2,
                'animation_speed': 1.0
            },
            EmotionType.FEAR: {
                'ripple_strength': 0.4,
                'color_blend_factor': 0.4,
                'glow_intensity': 0.5,
                'animation_speed': 2.0
            },
            EmotionType.DISGUST: {
                'ripple_strength': 0.5,
                'color_blend_factor': 0.4,
                'glow_intensity': 0.4,
                'animation_speed': 1.2
            }
        }
        
        self.logger.info("✅ 感情エフェクトマネージャー初期化完了")
    
    def update_emotion(self, emotion_result: Dict[str, Any]):
        """
        感情認識結果からエフェクトパラメータ更新
        
        Args:
            emotion_result: 感情認識結果
        """
        try:
            emotion = emotion_result.get('emotion', EmotionType.NEUTRAL)
            intensity = emotion_result.get('intensity', 0.0)
            confidence = emotion_result.get('confidence', 0.0)
            color = emotion_result.get('color', (0.5, 0.5, 0.5))
            
            # エフェクト強度調整
            adjusted_intensity = self._adjust_intensity(intensity, confidence)
            
            # 感情固有設定取得
            emotion_config = self.emotion_settings.get(emotion, self.emotion_settings[EmotionType.NEUTRAL])
            
            # ターゲットパラメータ更新
            self.target_params.update({
                'emotion_type': emotion.value,
                'emotion_intensity': adjusted_intensity,
                'emotion_color': color,
                'emotion_confidence': confidence,
                'ripple_strength': emotion_config['ripple_strength'] * self.effect_intensity.value,
                'color_blend_factor': emotion_config['color_blend_factor'] * self.effect_intensity.value,
                'glow_intensity': emotion_config['glow_intensity'] * self.effect_intensity.value
            })
            
            # アニメーション速度調整
            self.transition_speed = emotion_config['animation_speed']
            
        except Exception as e:
            self.logger.error(f"感情エフェクト更新エラー: {e}")
    
    def _adjust_intensity(self, intensity: float, confidence: float) -> float:
        """強度調整・信頼度考慮"""
        # 信頼度による減衰
        confidence_factor = min(1.0, confidence / 0.7)  # 0.7以上で最大効果
        
        # 強度調整
        adjusted = intensity * confidence_factor
        
        # 最小値保証
        if adjusted < 0.05:
            adjusted = 0.0
        
        return adjusted
    
    def update_animation(self, delta_time: float):
        """アニメーション更新・パラメータ補間"""
        self.animation_time += delta_time
        
        # パラメータ補間
        interpolation_factor = min(1.0, delta_time * self.transition_speed)
        
        for key in self.current_params:
            if isinstance(self.current_params[key], tuple):
                # RGB色の補間
                current = np.array(self.current_params[key])
                target = np.array(self.target_params[key])
                self.current_params[key] = tuple(
                    current + (target - current) * interpolation_factor
                )
            else:
                # スカラー値の補間
                current = self.current_params[key]
                target = self.target_params[key]
                self.current_params[key] = current + (target - current) * interpolation_factor
    
    def apply_to_shader(self, program: moderngl.Program):
        """シェーダープログラムにパラメータ適用"""
        try:
            # 時間パラメータ
            program['u_time'] = self.animation_time
            
            # 感情パラメータ
            program['u_emotion_type'] = int(self.current_params['emotion_type'])
            program['u_emotion_intensity'] = float(self.current_params['emotion_intensity'])
            program['u_emotion_color'] = self.current_params['emotion_color']
            program['u_emotion_confidence'] = float(self.current_params['emotion_confidence'])
            
            # エフェクトパラメータ
            if self.enable_ripples:
                program['u_ripple_strength'] = float(self.current_params['ripple_strength'])
            else:
                program['u_ripple_strength'] = 0.0
            
            if self.enable_color_blend:
                program['u_color_blend_factor'] = float(self.current_params['color_blend_factor'])
            else:
                program['u_color_blend_factor'] = 0.0
            
            if self.enable_glow:
                program['u_glow_intensity'] = float(self.current_params['glow_intensity'])
            else:
                program['u_glow_intensity'] = 0.0
                
        except Exception as e:
            self.logger.error(f"シェーダーパラメータ適用エラー: {e}")
    
    def set_effect_intensity(self, intensity: EffectIntensity):
        """エフェクト強度設定"""
        self.effect_intensity = intensity
        self.logger.info(f"エフェクト強度変更: {intensity.name}")
    
    def toggle_ripples(self):
        """波紋エフェクト切り替え"""
        self.enable_ripples = not self.enable_ripples
        self.logger.info(f"波紋エフェクト: {'ON' if self.enable_ripples else 'OFF'}")
    
    def toggle_glow(self):
        """グローエフェクト切り替え"""
        self.enable_glow = not self.enable_glow
        self.logger.info(f"グローエフェクト: {'ON' if self.enable_glow else 'OFF'}")
    
    def toggle_color_blend(self):
        """色彩ブレンド切り替え"""
        self.enable_color_blend = not self.enable_color_blend
        self.logger.info(f"色彩ブレンド: {'ON' if self.enable_color_blend else 'OFF'}")
    
    def get_current_params(self) -> Dict[str, Any]:
        """現在のパラメータ取得"""
        return self.current_params.copy()
    
    def reset_effects(self):
        """エフェクトリセット"""
        self.target_params = {
            'emotion_type': 4,  # NEUTRAL
            'emotion_intensity': 0.0,
            'emotion_color': (0.5, 0.5, 0.5),
            'emotion_confidence': 0.0,
            'ripple_strength': 0.2,
            'color_blend_factor': 0.1,
            'glow_intensity': 0.2
        }
        self.logger.info("エフェクトリセット完了")

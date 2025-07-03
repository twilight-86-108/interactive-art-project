"""
基本インタラクション管理 - Week 2版
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class InteractionMode(Enum):
    """インタラクションモード"""
    CALM = "calm"
    ACTIVE = "active"
    PLAYFUL = "playful"

@dataclass
class InteractionState:
    """インタラクション状態"""
    mode: InteractionMode = InteractionMode.CALM
    energy_level: float = 0.5
    user_engagement: float = 0.0
    session_duration: float = 0.0

class BasicInteractionManager:
    """基本インタラクション管理システム"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("BasicInteractionManager")
        self.config = config
        
        # インタラクション状態
        self.current_state = InteractionState()
        self.session_start_time = time.time()
        
        # 統計
        self.interaction_count = 0
        self.gesture_count = 0
        self.emotion_changes = 0
        
        self.logger.info("🎭 基本インタラクション管理システム初期化完了")
    
    def process_interaction(self, emotion_result, gesture_results) -> Dict[str, Any]:
        """インタラクション処理"""
        try:
            # 状態更新
            self._update_state(emotion_result, gesture_results)
            
            # エフェクトパラメータ計算
            effect_params = self._calculate_effects(emotion_result, gesture_results)
            
            # 統計更新
            self._update_statistics(emotion_result, gesture_results)
            
            return {
                'state': self.current_state,
                'effects': effect_params,
                'statistics': self._get_statistics()
            }
            
        except Exception as e:
            self.logger.error(f"❌ インタラクション処理失敗: {e}")
            return {}
    
    def _update_state(self, emotion_result, gesture_results):
        """状態更新"""
        try:
            # セッション時間更新
            self.current_state.session_duration = time.time() - self.session_start_time
            
            # エネルギーレベル計算
            emotion_energy = 0.0
            if emotion_result:
                emotion_energy = emotion_result.confidence
            
            gesture_energy = len(gesture_results) * 0.3
            
            target_energy = (emotion_energy + gesture_energy) / 2
            
            # 滑らかな変化
            self.current_state.energy_level = (
                self.current_state.energy_level * 0.8 + 
                target_energy * 0.2
            )
            
            # エンゲージメント計算
            if gesture_results or (emotion_result and emotion_result.confidence > 0.5):
                self.current_state.user_engagement = min(
                    self.current_state.user_engagement + 0.05, 1.0
                )
            else:
                self.current_state.user_engagement = max(
                    self.current_state.user_engagement - 0.02, 0.0
                )
            
            # モード決定
            if self.current_state.energy_level > 0.7:
                self.current_state.mode = InteractionMode.PLAYFUL
            elif self.current_state.energy_level > 0.4:
                self.current_state.mode = InteractionMode.ACTIVE
            else:
                self.current_state.mode = InteractionMode.CALM
            
        except Exception as e:
            self.logger.error(f"❌ 状態更新失敗: {e}")
    
    def _calculate_effects(self, emotion_result, gesture_results) -> Dict[str, Any]:
        """エフェクトパラメータ計算"""
        try:
            effects = {
                'water_intensity': 1.0,
                'audio_volume': 0.5,
                'color_intensity': 0.5,
                'wave_sources': []
            }
            
            # モード別調整
            if self.current_state.mode == InteractionMode.PLAYFUL:
                effects['water_intensity'] = 1.5
                effects['audio_volume'] = 0.8
                effects['color_intensity'] = 1.0
            elif self.current_state.mode == InteractionMode.ACTIVE:
                effects['water_intensity'] = 1.2
                effects['audio_volume'] = 0.6
                effects['color_intensity'] = 0.8
            
            # ジェスチャーから波源生成
            for gesture in gesture_results:
                # 手の位置を水面座標に変換
                water_x = (gesture.position[0] - 0.5) * 2.0
                water_y = (gesture.position[1] - 0.5) * 2.0
                
                # ジェスチャータイプに応じた強度
                intensity = gesture.confidence
                if gesture.type.value == 'fist':
                    intensity *= 2.0
                elif gesture.type.value == 'point':
                    intensity *= 1.2
                elif gesture.type.value == 'open_palm':
                    intensity *= 0.8
                
                effects['wave_sources'].append({
                    'position': (water_x, water_y),
                    'intensity': intensity
                })
            
            # 感情による色調整
            if emotion_result:
                emotion_colors = {
                    'HAPPY': (1.0, 0.8, 0.0),
                    'SAD': (0.3, 0.5, 0.8),
                    'ANGRY': (0.9, 0.1, 0.1),
                    'SURPRISED': (1.0, 0.1, 0.6),
                    'NEUTRAL': (0.5, 0.5, 0.5)
                }
                
                effects['emotion_color'] = emotion_colors.get(
                    emotion_result.emotion, (0.5, 0.5, 0.5)
                )
                effects['emotion_intensity'] = emotion_result.confidence
            
            return effects
            
        except Exception as e:
            self.logger.error(f"❌ エフェクト計算失敗: {e}")
            return {}
    
    def _update_statistics(self, emotion_result, gesture_results):
        """統計更新"""
        try:
            if emotion_result or gesture_results:
                self.interaction_count += 1
            
            if gesture_results:
                self.gesture_count += len(gesture_results)
            
            # 感情変化検出（実装簡略化）
            if emotion_result and emotion_result.emotion != 'NEUTRAL':
                self.emotion_changes += 1
                
        except Exception as e:
            self.logger.error(f"❌ 統計更新失敗: {e}")
    
    def _get_statistics(self) -> Dict[str, Any]:
        """統計取得"""
        return {
            'session_duration': self.current_state.session_duration,
            'interaction_count': self.interaction_count,
            'gesture_count': self.gesture_count,
            'emotion_changes': self.emotion_changes,
            'engagement_level': self.current_state.user_engagement,
            'energy_level': self.current_state.energy_level,
            'current_mode': self.current_state.mode.value
        }
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("🧹 基本インタラクション管理システムリソース解放完了")

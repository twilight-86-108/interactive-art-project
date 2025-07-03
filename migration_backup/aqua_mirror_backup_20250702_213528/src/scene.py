# src/scene.py
import pygame
import math
import random
import numpy as np
import logging
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional

class EffectType(Enum):
    """エフェクトタイプの定義"""
    RIPPLE = "ripple"
    BUBBLE = "bubble"
    PARTICLE = "particle"
    GLOW = "glow"

class SceneManager:
    """シーン描画・管理クラス（エラー修正版）"""
    
    def __init__(self, width: int, height: int, config: Dict[str, Any]):
        self.width = width
        self.height = height
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        try:
            # エフェクト管理
            self.active_effects = []
            self.particles = []
            self.ripples = []
            
            # 状態管理
            self.current_emotion = "neutral"
            self.emotion_intensity = 0.5
            
            # パースペクティブパラメータ
            self.perspective_offset = [0.0, 0.0]
            self.perspective_scale = 1.0
            
            # リソース読み込み
            self._load_assets()
            
            # パフォーマンス管理
            self.max_particles = config.get('visual_effects', {}).get('particles', {}).get('max_count', 500)
            self.max_ripples = 10
            
            self.logger.info("シーンマネージャーが初期化されました")
            
        except Exception as e:
            self.logger.error(f"シーンマネージャー初期化エラー: {e}")
            raise
    
    def _load_assets(self):
        """アセット読み込み"""
        try:
            assets_config = self.config.get('assets', {})
            
            # 背景画像の読み込み
            try:
                bg_path = assets_config.get('images', {}).get('background', 'assets/images/underwater_scene.jpg')
                self.background_image = pygame.image.load(bg_path)
                self.background_image = pygame.transform.scale(
                    self.background_image, (self.width, self.height)
                )
                self.logger.info(f"背景画像を読み込みました: {bg_path}")
            except Exception as e:
                self.logger.warning(f"背景画像読み込み失敗: {e}")
                # フォールバック：グラデーション背景作成
                self.background_image = self._create_gradient_background()
            
            # 魚の画像読み込み（オプション）
            self.fish_images = []
            fish_paths = assets_config.get('images', {}).get('fish', [])
            for fish_path in fish_paths:
                try:
                    fish_img = pygame.image.load(fish_path)
                    self.fish_images.append(fish_img)
                except Exception as e:
                    self.logger.warning(f"魚画像読み込み失敗 {fish_path}: {e}")
            
            # フォント初期化
            pygame.font.init()
            self.font = pygame.font.Font(None, 36)
            
        except Exception as e:
            self.logger.error(f"アセット読み込みエラー: {e}")
            # 最小限のフォールバック
            self.background_image = self._create_gradient_background()
            self.fish_images = []
            self.font = pygame.font.Font(None, 36)
    
    def _create_gradient_background(self) -> pygame.Surface:
        """グラデーション背景の作成"""
        try:
            surface = pygame.Surface((self.width, self.height))
            
            # 深い青のグラデーション
            for y in range(self.height):
                ratio = y / self.height
                # 上から下へ：暗い青から明るい青へ
                blue = int(50 + ratio * 100)
                green = int(ratio * 50)
                color = (0, green, blue)
                pygame.draw.line(surface, color, (0, y), (self.width, y))
            
            return surface
            
        except Exception as e:
            self.logger.error(f"グラデーション背景作成エラー: {e}")
            # 最終フォールバック：単色
            surface = pygame.Surface((self.width, self.height))
            surface.fill((0, 50, 100))
            return surface
    
    def update(self, detection_result: Dict[str, Any], current_state: Any):
        """シーン更新"""
        try:
            # 感情データの取得
            emotion = getattr(current_state, 'value', 'neutral') if hasattr(current_state, 'value') else 'neutral'
            
            # ヘッドトラッキングによるパースペクティブ更新
            if detection_result.get('face_detected') and emotion != 'standby':
                self._update_perspective(detection_result.get('face_center'))
            else:
                # 待機状態では中央にリセット
                self._reset_perspective()
            
            # インタラクション処理
            if emotion == 'interaction' or detection_result.get('hands_detected'):
                self._handle_interactions(detection_result)
            
            # エフェクト更新
            self._update_effects(1/30)  # 30FPS想定
            
            # 感情状態更新
            self.current_emotion = emotion
            
        except Exception as e:
            self.logger.error(f"シーン更新エラー: {e}")
    
    def _update_perspective(self, face_center: Optional[Tuple[float, float, float]]):
        """パースペクティブ更新"""
        try:
            if face_center:
                # 顔の位置に基づいてオフセットを計算
                offset_x = (face_center[0] - 0.5) * 100  # スケール調整
                offset_y = (face_center[1] - 0.5) * 100
                
                # スムージング（急激な変化を抑制）
                smooth_factor = 0.1
                self.perspective_offset[0] += (offset_x - self.perspective_offset[0]) * smooth_factor
                self.perspective_offset[1] += (offset_y - self.perspective_offset[1]) * smooth_factor
                
                # 距離に基づくスケール変更
                if len(face_center) > 2:
                    distance = face_center[2]
                    target_scale = 1.0 + distance * 0.1  # 距離による拡大縮小
                    self.perspective_scale += (target_scale - self.perspective_scale) * smooth_factor
                    
        except Exception as e:
            self.logger.error(f"パースペクティブ更新エラー: {e}")
    
    def _reset_perspective(self):
        """パースペクティブリセット"""
        try:
            smooth_factor = 0.05
            self.perspective_offset[0] *= (1 - smooth_factor)
            self.perspective_offset[1] *= (1 - smooth_factor)
            self.perspective_scale += (1.0 - self.perspective_scale) * smooth_factor
            
        except Exception as e:
            self.logger.error(f"パースペクティブリセットエラー: {e}")
    
    def _handle_interactions(self, detection_result: Dict[str, Any]):
        """インタラクション処理"""
        try:
            # 手の位置による波紋エフェクト
            hand_positions = detection_result.get('hand_positions', [])
            
            for hand_pos in hand_positions:
                # 確率的エフェクト生成
                if random.random() < 0.1:  # 10%の確率
                    screen_x = hand_pos[0] * self.width
                    screen_y = hand_pos[1] * self.height
                    
                    # 画面範囲内チェック
                    if 0 <= screen_x <= self.width and 0 <= screen_y <= self.height:
                        self._add_effect(EffectType.RIPPLE, (screen_x, screen_y))
            
            # ジェスチャーによる特別エフェクト
            gestures = detection_result.get('hand_gestures', [])
            for gesture in gestures:
                self._handle_gesture_effect(gesture)
            
        except Exception as e:
            self.logger.error(f"インタラクション処理エラー: {e}")
    
    def _handle_gesture_effect(self, gesture: Dict[str, Any]):
        """ジェスチャー別エフェクト処理"""
        try:
            gesture_type = gesture.get('type', 'unknown')
            position = gesture.get('position', (0.5, 0.5, 0))
            
            screen_x = position[0] * self.width
            screen_y = position[1] * self.height
            
            if gesture_type == 'point':
                self._add_effect(EffectType.GLOW, (screen_x, screen_y))
            elif gesture_type == 'peace':
                self._add_effect(EffectType.PARTICLE, (screen_x, screen_y))
            elif gesture_type == 'open':
                self._add_effect(EffectType.BUBBLE, (screen_x, screen_y))
                
        except Exception as e:
            self.logger.error(f"ジェスチャーエフェクト処理エラー: {e}")
    
    def _add_effect(self, effect_type: EffectType, position: Tuple[float, float]):
        """エフェクト追加"""
        try:
            current_time = pygame.time.get_ticks()
            
            effect = {
                'type': effect_type,
                'position': position,
                'start_time': current_time,
                'duration': 2000,  # 2秒
                'data': {}
            }
            
            if effect_type == EffectType.RIPPLE:
                effect['data'] = {
                    'radius': 0,
                    'max_radius': random.uniform(50, 150),
                    'color': self._get_emotion_color()
                }
                self.ripples.append(effect)
                
                # 最大数制限
                if len(self.ripples) > self.max_ripples:
                    self.ripples.pop(0)
                    
            elif effect_type == EffectType.PARTICLE:
                # パーティクル群生成
                for _ in range(random.randint(5, 15)):
                    self._create_particle(position)
                    
            elif effect_type == EffectType.BUBBLE:
                effect['data'] = {
                    'size': random.uniform(10, 30),
                    'velocity_y': random.uniform(-2, -1),
                    'color': (255, 255, 255, 100)
                }
                self.active_effects.append(effect)
                
            elif effect_type == EffectType.GLOW:
                effect['data'] = {
                    'radius': random.uniform(30, 60),
                    'intensity': 0.8,
                    'color': self._get_emotion_color()
                }
                self.active_effects.append(effect)
            
        except Exception as e:
            self.logger.error(f"エフェクト追加エラー: {e}")
    
    def _create_particle(self, center_pos: Tuple[float, float]):
        """パーティクル生成"""
        try:
            if len(self.particles) >= self.max_particles:
                # 古いパーティクルを削除
                self.particles.pop(0)
            
            # ランダムな初期位置・速度
            offset_x = random.uniform(-30, 30)
            offset_y = random.uniform(-30, 30)
            
            particle = {
                'x': center_pos[0] + offset_x,
                'y': center_pos[1] + offset_y,
                'vx': random.uniform(-2, 2),
                'vy': random.uniform(-3, -1),
                'life': random.uniform(1.0, 3.0),
                'max_life': random.uniform(1.0, 3.0),
                'size': random.uniform(2, 8),
                'color': self._get_emotion_color(),
                'alpha': 255
            }
            
            self.particles.append(particle)
            
        except Exception as e:
            self.logger.error(f"パーティクル生成エラー: {e}")
    
    def _get_emotion_color(self) -> Tuple[int, int, int]:
        """現在の感情に基づく色彩取得"""
        try:
            emotion_colors = {
                'happy': (255, 223, 0),      # 黄色
                'sad': (70, 130, 180),       # 青
                'angry': (220, 20, 60),      # 赤
                'surprised': (255, 20, 147), # ピンク
                'neutral': (72, 209, 204)    # ターコイズ
            }
            
            return emotion_colors.get(self.current_emotion, emotion_colors['neutral'])
            
        except Exception as e:
            self.logger.error(f"感情色彩取得エラー: {e}")
            return (72, 209, 204)  # デフォルト色
    
    def _update_effects(self, delta_time: float):
        """エフェクト更新"""
        try:
            current_time = pygame.time.get_ticks()
            
            # 波紋更新
            for ripple in self.ripples[:]:
                elapsed = current_time - ripple['start_time']
                progress = elapsed / ripple['duration']
                
                if progress >= 1.0:
                    self.ripples.remove(ripple)
                    continue
                
                ripple['data']['radius'] = ripple['data']['max_radius'] * progress
            
            # パーティクル更新
            for particle in self.particles[:]:
                particle['x'] += particle['vx']
                particle['y'] += particle['vy']
                particle['vy'] += 0.1  # 重力
                particle['life'] -= delta_time
                
                # 透明度計算
                life_ratio = max(0, particle['life'] / particle['max_life'])
                particle['alpha'] = int(255 * life_ratio)
                
                if particle['life'] <= 0:
                    self.particles.remove(particle)
            
            # その他エフェクト更新
            for effect in self.active_effects[:]:
                elapsed = current_time - effect['start_time']
                progress = elapsed / effect['duration']
                
                if progress >= 1.0:
                    self.active_effects.remove(effect)
                    continue
                
                # エフェクト種別に応じた更新
                if effect['type'] == EffectType.BUBBLE:
                    effect['position'] = (
                        effect['position'][0],
                        effect['position'][1] + effect['data']['velocity_y']
                    )
                    
        except Exception as e:
            self.logger.error(f"エフェクト更新エラー: {e}")
    
    def draw(self, screen: pygame.Surface):
        """描画処理"""
        try:
            # 背景描画（パースペクティブ適用）
            self._draw_background(screen)
            
            # エフェクト描画
            self._draw_effects(screen)
            
            # インタラクティブオブジェクト描画
            self._draw_interactive_objects(screen)
            
        except Exception as e:
            self.logger.error(f"描画処理エラー: {e}")
    
    def _draw_background(self, screen: pygame.Surface):
        """背景描画"""
        try:
            if self.background_image:
                # パースペクティブ変換を適用した背景描画
                scaled_width = int(self.width * self.perspective_scale)
                scaled_height = int(self.height * self.perspective_scale)
                
                if scaled_width > 0 and scaled_height > 0:
                    scaled_bg = pygame.transform.scale(
                        self.background_image, (scaled_width, scaled_height)
                    )
                    
                    # 描画位置計算
                    draw_x = int(-self.perspective_offset[0] - (scaled_width - self.width) // 2)
                    draw_y = int(-self.perspective_offset[1] - (scaled_height - self.height) // 2)
                    
                    screen.blit(scaled_bg, (draw_x, draw_y))
                else:
                    # スケールが無効な場合はオリジナルサイズで描画
                    screen.blit(self.background_image, (0, 0))
            else:
                # 背景画像がない場合は単色背景
                screen.fill((0, 50, 100))
                
        except Exception as e:
            self.logger.error(f"背景描画エラー: {e}")
            screen.fill((0, 50, 100))  # フォールバック
    
    def _draw_effects(self, screen: pygame.Surface):
        """エフェクト描画"""
        try:
            # 波紋描画
            for ripple in self.ripples:
                self._draw_ripple(screen, ripple)
            
            # パーティクル描画
            for particle in self.particles:
                self._draw_particle(screen, particle)
            
            # その他エフェクト描画
            for effect in self.active_effects:
                self._draw_effect(screen, effect)
                
        except Exception as e:
            self.logger.error(f"エフェクト描画エラー: {e}")
    
    def _draw_ripple(self, screen: pygame.Surface, ripple: Dict[str, Any]):
        """波紋描画"""
        try:
            radius = int(ripple['data']['radius'])
            if radius <= 0:
                return
                
            color = ripple['data']['color']
            alpha = max(0, 255 - int(255 * (ripple['data']['radius'] / ripple['data']['max_radius'])))
            
            # 透明度付きサーフェス作成
            if radius > 0 and alpha > 0:
                ripple_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
                ripple_color = (*color, alpha)
                pygame.draw.circle(ripple_surface, ripple_color, (radius + 5, radius + 5), radius, 3)
                
                draw_x = int(ripple['position'][0] - radius - 5)
                draw_y = int(ripple['position'][1] - radius - 5)
                screen.blit(ripple_surface, (draw_x, draw_y))
                
        except Exception as e:
            self.logger.error(f"波紋描画エラー: {e}")
    
    def _draw_particle(self, screen: pygame.Surface, particle: Dict[str, Any]):
        """パーティクル描画"""
        try:
            if particle['alpha'] <= 0:
                return
                
            size = max(1, int(particle['size']))
            x = int(particle['x'])
            y = int(particle['y'])
            
            # 画面範囲チェック
            if 0 <= x <= self.width and 0 <= y <= self.height:
                # 透明度付きサーフェス作成
                particle_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                color_with_alpha = (*particle['color'], particle['alpha'])
                pygame.draw.circle(particle_surface, color_with_alpha, (size, size), size)
                
                screen.blit(particle_surface, (x - size, y - size))
                
        except Exception as e:
            self.logger.error(f"パーティクル描画エラー: {e}")
    
    def _draw_effect(self, screen: pygame.Surface, effect: Dict[str, Any]):
        """その他エフェクト描画"""
        try:
            effect_type = effect['type']
            position = effect['position']
            
            if effect_type == EffectType.BUBBLE:
                size = int(effect['data']['size'])
                color = effect['data']['color']
                
                if size > 0:
                    bubble_surface = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(bubble_surface, color, (size, size), size, 2)
                    
                    x = int(position[0] - size)
                    y = int(position[1] - size)
                    screen.blit(bubble_surface, (x, y))
                    
            elif effect_type == EffectType.GLOW:
                radius = int(effect['data']['radius'])
                color = effect['data']['color']
                intensity = effect['data']['intensity']
                
                if radius > 0:
                    glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    alpha = int(100 * intensity)
                    glow_color = (*color, alpha)
                    pygame.draw.circle(glow_surface, glow_color, (radius, radius), radius)
                    
                    x = int(position[0] - radius)
                    y = int(position[1] - radius)
                    screen.blit(glow_surface, (x, y))
                    
        except Exception as e:
            self.logger.error(f"エフェクト描画エラー: {e}")
    
    def _draw_interactive_objects(self, screen: pygame.Surface):
        """インタラクティブオブジェクト描画"""
        try:
            # 魚や他のオブジェクトの描画
            if self.fish_images:
                current_time = pygame.time.get_ticks()
                
                for i, fish_img in enumerate(self.fish_images[:3]):  # 最大3匹
                    x = 300 + i * 200 + int(math.sin(current_time / 1000 + i) * 50)
                    y = 400 + int(math.cos(current_time / 1500 + i) * 30)
                    
                    # 画面範囲チェック
                    if 0 <= x <= self.width - fish_img.get_width() and 0 <= y <= self.height - fish_img.get_height():
                        screen.blit(fish_img, (x, y))
            else:
                # 魚の画像がない場合、色付きの円で代用
                current_time = pygame.time.get_ticks()
                
                for i in range(3):
                    x = 300 + i * 200 + int(math.sin(current_time / 1000 + i) * 50)
                    y = 400 + int(math.cos(current_time / 1500 + i) * 30)
                    color = (255, 100 + i * 50, 50)
                    
                    # 画面範囲チェック
                    if 0 <= x <= self.width and 0 <= y <= self.height:
                        pygame.draw.circle(screen, color, (x, y), 20)
                        
        except Exception as e:
            self.logger.error(f"インタラクティブオブジェクト描画エラー: {e}")
    
    def clear_effects(self):
        """全エフェクトクリア"""
        try:
            self.active_effects.clear()
            self.particles.clear()
            self.ripples.clear()
            self.logger.info("全エフェクトをクリアしました")
        except Exception as e:
            self.logger.error(f"エフェクトクリアエラー: {e}")
    
    def get_effect_count(self) -> Dict[str, int]:
        """エフェクト数取得"""
        try:
            return {
                'particles': len(self.particles),
                'ripples': len(self.ripples),
                'other_effects': len(self.active_effects),
                'total': len(self.particles) + len(self.ripples) + len(self.active_effects)
            }
        except Exception as e:
            self.logger.error(f"エフェクト数取得エラー: {e}")
            return {'particles': 0, 'ripples': 0, 'other_effects': 0, 'total': 0}

    def apply_quality_settings(self, settings: Dict[str, Any]):
        """品質設定を適用"""
        try:
            self.max_particles = settings.get('particle_count', self.max_particles)
            # 他の品質関連パラメータもここで更新できる
            #例: self.effect_quality = settings.get('effect_quality', 'high')
            self.logger.info(f"シーン品質設定を適用: パーティクル数={self.max_particles}")
        except Exception as e:
            self.logger.error(f"品質設定適用エラー: {e}")
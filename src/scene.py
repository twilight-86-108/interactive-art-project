# src/scene.py
import pygame
import math
import random
from enum import Enum

class EffectType(Enum):
    """エフェクトタイプの定義"""
    RIPPLE = "ripple"
    BUBBLE = "bubble"
    FISH_APPEAR = "fish_appear"
    FISH_HIDE = "fish_hide"

class SceneManager:
    """シーン描画・管理クラス"""
    
    def __init__(self, width, height, config):
        self.width = width
        self.height = height
        self.config = config
        
        # エフェクト管理
        self.active_effects = []
        self.fish_positions = []
        self.fish_visible = True
        
        # リソース読み込み
        self._load_assets()
        
        # パースペクティブパラメータ
        self.perspective_offset = [0, 0]
        self.perspective_scale = 1.0
    
    def _load_assets(self):
        """アセット読み込み"""
        try:
            # 背景画像
            bg_path = self.config['assets']['background_image']
            self.background_image = pygame.image.load(bg_path)
            self.background_image = pygame.transform.scale(
                self.background_image, (self.width, self.height)
            )
            
            # 魚の画像（オプション）
            self.fish_images = []
            for fish_path in self.config['assets'].get('fish_images', []):
                try:
                    fish_img = pygame.image.load(fish_path)
                    self.fish_images.append(fish_img)
                except:
                    pass  # 画像がない場合はスキップ
            
            print("アセットが読み込まれました")
            
        except Exception as e:
            print(f"アセット読み込みエラー: {e}")
            # フォールバック：単色背景
            self.background_image = pygame.Surface((self.width, self.height))
            self.background_image.fill((0, 50, 100))  # 深い青色
            self.fish_images = []
    
    def update(self, detection_result, current_state):
        """シーン更新"""
        # ヘッドトラッキングによるパースペクティブ更新
        if detection_result.get('face_detected') and current_state.value != 'standby':
            self._update_perspective(detection_result['face_center'])
        else:
            # 待機状態では中央にリセット
            self._reset_perspective()
        
        # インタラクション処理
        if current_state.value == 'interaction':
            self._handle_interactions(detection_result)
        
        # エフェクト更新
        self._update_effects()
    
    def _update_perspective(self, face_center):
        """パースペクティブ更新"""
        if face_center:
            # 顔の位置に基づいてオフセットを計算
            # 中央を(0.5, 0.5)として、オフセットを計算
            offset_x = (face_center[0] - 0.5) * 100  # スケール調整
            offset_y = (face_center[1] - 0.5) * 100
            
            # スムージング（急激な変化を抑制）
            smooth_factor = 0.1
            self.perspective_offset[0] += (offset_x - self.perspective_offset[0]) * smooth_factor
            self.perspective_offset[1] += (offset_y - self.perspective_offset[1]) * smooth_factor
            
            # 距離に基づくスケール変更
            distance = face_center[2] if len(face_center) > 2 else 0
            target_scale = 1.0 + distance * 0.1  # 距離による拡大縮小
            self.perspective_scale += (target_scale - self.perspective_scale) * smooth_factor
    
    def _reset_perspective(self):
        """パースペクティブリセット"""
        smooth_factor = 0.05
        self.perspective_offset[0] *= (1 - smooth_factor)
        self.perspective_offset[1] *= (1 - smooth_factor)
        self.perspective_scale += (1.0 - self.perspective_scale) * smooth_factor
    
    def _handle_interactions(self, detection_result):
        """インタラクション処理"""
        # 手の接触判定
        hand_positions = detection_result.get('hand_positions', [])
        interaction_region = self.config['interaction']['interaction_regions']['water_surface']
        
        for hand_pos in hand_positions:
            # 正規化座標を画面座標に変換
            screen_x = hand_pos[0] * self.width
            screen_y = hand_pos[1] * self.height
            
            # 接触領域内かチェック
            if (interaction_region['x'] <= screen_x <= interaction_region['x'] + interaction_region['width'] and
                interaction_region['y'] <= screen_y <= interaction_region['y'] + interaction_region['height']):
                
                # 波紋エフェクト追加
                self._add_effect(EffectType.RIPPLE, (screen_x, screen_y))
        
        # 顔の接近判定
        face_distance = detection_result.get('face_distance', float('inf'))
        approach_threshold = self.config['interaction']['approach_threshold_z']
        
        if face_distance < approach_threshold:
            if self.fish_visible:
                self.fish_visible = False
                self._add_effect(EffectType.FISH_HIDE, None)
        else:
            if not self.fish_visible:
                self.fish_visible = True
                self._add_effect(EffectType.FISH_APPEAR, None)
    
    def _add_effect(self, effect_type, position):
        """エフェクト追加"""
        effect = {
            'type': effect_type,
            'position': position,
            'start_time': pygame.time.get_ticks(),
            'duration': 2000,  # 2秒
            'data': {}
        }
        
        if effect_type == EffectType.RIPPLE:
            effect['data'] = {'radius': 0, 'max_radius': 100}
        
        self.active_effects.append(effect)
    
    def _update_effects(self):
        """エフェクト更新"""
        current_time = pygame.time.get_ticks()
        
        # アクティブエフェクトの更新
        for effect in self.active_effects[:]:  # コピーして反復
            elapsed = current_time - effect['start_time']
            progress = elapsed / effect['duration']
            
            if progress >= 1.0:
                self.active_effects.remove(effect)
                continue
            
            # エフェクト固有の更新
            if effect['type'] == EffectType.RIPPLE:
                effect['data']['radius'] = effect['data']['max_radius'] * progress
    
    def draw(self, screen):
        """描画処理"""
        # 背景描画（パースペクティブ適用）
        self._draw_background(screen)
        
        # インタラクティブオブジェクト描画
        self._draw_interactive_objects(screen)
        
        # エフェクト描画
        self._draw_effects(screen)
    
    def _draw_background(self, screen):
        """背景描画"""
        # パースペクティブ変換を適用した背景描画
        scaled_bg = pygame.transform.scale(
            self.background_image,
            (int(self.width * self.perspective_scale),
             int(self.height * self.perspective_scale))
        )
        
        # 描画位置計算
        draw_x = -self.perspective_offset[0] - (scaled_bg.get_width() - self.width) // 2
        draw_y = -self.perspective_offset[1] - (scaled_bg.get_height() - self.height) // 2
        
        screen.blit(scaled_bg, (draw_x, draw_y))
    
    def _draw_interactive_objects(self, screen):
        """インタラクティブオブジェクト描画"""
        if self.fish_visible and self.fish_images:
            # 魚を描画（簡易版）
            for i, fish_img in enumerate(self.fish_images[:3]):  # 最大3匹
                x = 300 + i * 200 + int(math.sin(pygame.time.get_ticks() / 1000 + i) * 50)
                y = 400 + int(math.cos(pygame.time.get_ticks() / 1500 + i) * 30)
                screen.blit(fish_img, (x, y))
        else:
            # 魚の画像がない場合、色付きの円で代用
            if self.fish_visible:
                for i in range(3):
                    x = 300 + i * 200 + int(math.sin(pygame.time.get_ticks() / 1000 + i) * 50)
                    y = 400 + int(math.cos(pygame.time.get_ticks() / 1500 + i) * 30)
                    color = (255, 100 + i * 50, 50)
                    pygame.draw.circle(screen, color, (x, y), 20)
    
    def _draw_effects(self, screen):
        """エフェクト描画"""
        for effect in self.active_effects:
            if effect['type'] == EffectType.RIPPLE and effect['position']:
                radius = int(effect['data']['radius'])
                alpha = max(0, 255 - int(255 * (radius / effect['data']['max_radius'])))
                
                # 波紋を描画（透明度付きサーフェス使用）
                if radius > 0:
                    ripple_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(ripple_surface, (255, 255, 255, alpha), (radius, radius), radius, 3)
                    screen.blit(ripple_surface, 
                              (effect['position'][0] - radius, effect['position'][1] - radius))
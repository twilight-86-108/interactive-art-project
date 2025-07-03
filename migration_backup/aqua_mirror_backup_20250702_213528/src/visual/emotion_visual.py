import pygame
import math
import time
import numpy as np
from typing import Dict, Tuple, List, Optional
from enum import Enum

class ColorTransitionMode(Enum):
    INSTANT = "instant"
    LINEAR = "linear"
    SMOOTH = "smooth"
    PULSE = "pulse"

class EmotionVisualEngine:
    """感情連動視覚表現システム（Day 4版）"""
    
    def __init__(self, width: int, height: int, config: dict):
        self.width = width
        self.height = height
        self.config = config
        
        # 感情別カラーパレット
        self.emotion_palettes = {
            'happy': {
                'primary': (255, 223, 0),        # 明るい黄色
                'secondary': (255, 165, 0),      # オレンジ
                'accent': (255, 99, 71),         # コーラル
                'background': (255, 248, 220),   # コーンシルク
                'text': (139, 69, 19)            # サドルブラウン
            },
            'sad': {
                'primary': (70, 130, 180),       # スチールブルー
                'secondary': (135, 206, 235),    # スカイブルー
                'accent': (176, 196, 222),       # ライトスチールブルー
                'background': (230, 230, 250),   # ラベンダー
                'text': (25, 25, 112)            # ミッドナイトブルー
            },
            'angry': {
                'primary': (220, 20, 60),        # クリムゾン
                'secondary': (255, 69, 0),       # レッドオレンジ
                'accent': (255, 140, 0),         # ダークオレンジ
                'background': (255, 228, 225),   # ミスティローズ
                'text': (139, 0, 0)              # ダークレッド
            },
            'surprised': {
                'primary': (255, 20, 147),       # ディープピンク
                'secondary': (255, 105, 180),    # ホットピンク
                'accent': (255, 182, 193),       # ライトピンク
                'background': (255, 240, 245),   # ラベンダーブラッシュ
                'text': (199, 21, 133)           # ミディアムバイオレットレッド
            },
            'neutral': {
                'primary': (72, 209, 204),       # ミディアムターコイズ
                'secondary': (175, 238, 238),    # パウダーブルー
                'accent': (64, 224, 208),        # ターコイズ
                'background': (240, 248, 255),   # アリスブルー
                'text': (47, 79, 79)             # ダークスレートグレー
            }
        }
        
        # 現在の色彩状態
        self.current_emotion = 'neutral'
        self.current_palette = self.emotion_palettes['neutral'].copy()
        self.target_palette = self.emotion_palettes['neutral'].copy()
        
        # 色彩遷移設定
        visual_config = config.get('visual_effects', {}).get('colors', {})
        self.transition_speed = visual_config.get('transition_speed', 0.5)
        self.saturation_boost = visual_config.get('saturation_boost', 1.2)
        self.brightness_adjustment = visual_config.get('brightness_adjustment', 1.0)
        
        # アニメーション状態
        self.animation_time = 0.0
        self.pulse_phase = 0.0
        self.transition_progress = 1.0
        
        # グラデーション・エフェクト
        self.gradient_cache = {}
        self.effect_surfaces = {}
        
        print("✅ Emotion Visual Engine 初期化完了")
    
    def update_emotion(self, emotion: str, confidence: float, transition_mode: ColorTransitionMode = ColorTransitionMode.SMOOTH):
        """感情更新・色彩遷移開始"""
        if emotion != self.current_emotion:
            self.current_emotion = emotion
            self.target_palette = self.emotion_palettes.get(emotion, self.emotion_palettes['neutral']).copy()
            
            # 信頼度に応じた色彩強度調整
            self._adjust_palette_intensity(self.target_palette, confidence)
            
            # 遷移開始
            if transition_mode == ColorTransitionMode.INSTANT:
                self.current_palette = self.target_palette.copy()
                self.transition_progress = 1.0
            else:
                self.transition_progress = 0.0
    
    def update(self, delta_time: float):
        """視覚エフェクト更新"""
        self.animation_time += delta_time
        self.pulse_phase += delta_time * 2.0  # 2Hz
        
        # 色彩遷移更新
        if self.transition_progress < 1.0:
            self.transition_progress += delta_time * self.transition_speed
            self.transition_progress = min(1.0, self.transition_progress)
            
            # 色彩補間
            self._interpolate_palette()
    
    def _adjust_palette_intensity(self, palette: Dict[str, Tuple[int, int, int]], confidence: float):
        """信頼度に応じたパレット強度調整"""
        intensity = max(0.3, min(1.0, confidence))  # 0.3-1.0の範囲
        
        for key, color in palette.items():
            # HSV変換で彩度調整
            h, s, v = self._rgb_to_hsv(color)
            s *= intensity * self.saturation_boost
            v *= self.brightness_adjustment
            
            palette[key] = self._hsv_to_rgb(h, min(1.0, s), min(1.0, v))
    
    def _interpolate_palette(self):
        """パレット色彩補間"""
        # スムース補間関数
        t = self._smooth_step(self.transition_progress)
        
        for key in self.current_palette:
            current_color = self.current_palette[key]
            target_color = self.target_palette[key]
            
            # RGB線形補間
            interpolated_color = (
                int(current_color[0] + (target_color[0] - current_color[0]) * t),
                int(current_color[1] + (target_color[1] - current_color[1]) * t),
                int(current_color[2] + (target_color[2] - current_color[2]) * t)
            )
            
            self.current_palette[key] = interpolated_color
    
    def _smooth_step(self, t: float) -> float:
        """スムースステップ補間"""
        return t * t * (3.0 - 2.0 * t)
    
    def render_background(self, screen: pygame.Surface):
        """背景描画"""
        # グラデーション背景作成
        gradient_key = f"{self.current_palette['primary']}_{self.current_palette['secondary']}"
        
        if gradient_key not in self.gradient_cache:
            self.gradient_cache[gradient_key] = self._create_gradient_surface(
                self.current_palette['primary'], 
                self.current_palette['secondary']
            )
        
        screen.blit(self.gradient_cache[gradient_key], (0, 0))
        
        # パルスエフェクト（感情が強い場合）
        if self.current_emotion != 'neutral':
            self._render_pulse_effect(screen)
    
    def _create_gradient_surface(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> pygame.Surface:
        """グラデーションサーフェス作成"""
        surface = pygame.Surface((self.width, self.height))
        
        for y in range(self.height):
            # 非線形グラデーション
            ratio = math.sin(y / self.height * math.pi / 2)
            
            blended_color = (
                int(color1[0] + (color2[0] - color1[0]) * ratio),
                int(color1[1] + (color2[1] - color1[1]) * ratio),
                int(color1[2] + (color2[2] - color1[2]) * ratio)
            )
            
            pygame.draw.line(surface, blended_color, (0, y), (self.width, y))
        
        return surface
    
    def _render_pulse_effect(self, screen: pygame.Surface):
        """パルスエフェクト描画"""
        # パルスの強度計算
        pulse_intensity = (math.sin(self.pulse_phase) + 1.0) / 2.0  # 0-1の範囲
        
        if pulse_intensity > 0.7:  # 強いパルスのみ描画
            alpha = int((pulse_intensity - 0.7) * 255 / 0.3)
            
            # 透明サーフェス作成
            pulse_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # 中心から外側へのグラデーション
            center_x, center_y = self.width // 2, self.height // 2
            max_radius = math.sqrt(center_x**2 + center_y**2)
            
            for radius in range(0, int(max_radius), 10):
                color_alpha = int(alpha * (1.0 - radius / max_radius))
                if color_alpha > 0:
                    color = (*self.current_palette['accent'], color_alpha)
                    pygame.draw.circle(pulse_surface, color, (center_x, center_y), radius, 2)
            
            screen.blit(pulse_surface, (0, 0))
    
    def render_emotion_aura(self, screen: pygame.Surface, position: Tuple[int, int], intensity: float = 1.0):
        """感情オーラ描画"""
        x, y = position
        
        # オーラサイズ
        aura_size = int(100 * intensity)
        
        # 透明サーフェス作成
        aura_surface = pygame.Surface((aura_size * 2, aura_size * 2), pygame.SRCALPHA)
        
        # グラデーション円描画
        primary_color = self.current_palette['primary']
        
        for i in range(aura_size, 0, -5):
            alpha = int(100 * (1.0 - i / aura_size) * intensity)
            color = (*primary_color, alpha)
            pygame.draw.circle(aura_surface, color, (aura_size, aura_size), i)
        
        # 画面に描画
        screen.blit(aura_surface, (x - aura_size, y - aura_size))
    
    def get_current_color(self, color_type: str = 'primary') -> Tuple[int, int, int]:
        """現在の色取得"""
        return self.current_palette.get(color_type, (255, 255, 255))
    
    def create_emotion_text_surface(self, text: str, font_size: int = 36) -> pygame.Surface:
        """感情色彩テキストサーフェス作成"""
        font = pygame.font.Font(None, font_size)
        text_color = self.current_palette['text']
        background_color = self.current_palette['background']
        
        # テキスト描画
        text_surface = font.render(text, True, text_color)
        
        # 背景付きサーフェス作成
        padding = 10
        bg_surface = pygame.Surface((
            text_surface.get_width() + padding * 2,
            text_surface.get_height() + padding * 2
        ), pygame.SRCALPHA)
        
        # 背景描画（角丸風）
        bg_rect = bg_surface.get_rect()
        pygame.draw.rect(bg_surface, (*background_color, 200), bg_rect, border_radius=5)
        
        # テキスト描画
        bg_surface.blit(text_surface, (padding, padding))
        
        return bg_surface
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """RGB to HSV変換"""
        r, g, b = [x / 255.0 for x in rgb]
        
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val
        
        # Value
        v = max_val
        
        # Saturation
        s = 0 if max_val == 0 else diff / max_val
        
        # Hue
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        return h / 360.0, s, v
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """HSV to RGB変換"""
        h *= 360
        
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c
        
        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
    
    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計"""
        return {
            'current_emotion': self.current_emotion,
            'transition_progress': self.transition_progress,
            'gradient_cache_size': len(self.gradient_cache),
            'animation_time': self.animation_time
        }

# テスト実行用
if __name__ == "__main__":
    print("🔍 Emotion Visual Engine テスト開始...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Emotion Visual Test")
    clock = pygame.time.Clock()
    
    config = {
        'visual_effects': {
            'colors': {
                'transition_speed': 1.0,
                'saturation_boost': 1.2,
                'brightness_adjustment': 1.0
            }
        }
    }
    
    engine = EmotionVisualEngine(800, 600, config)
    
    emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised']
    current_emotion_index = 0
    
    print("🎨 感情色彩テスト開始（SPACE: 感情切替, ESC: 終了）...")
    
    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # 感情切替
                    current_emotion_index = (current_emotion_index + 1) % len(emotions)
                    emotion = emotions[current_emotion_index]
                    engine.update_emotion(emotion, 0.8, ColorTransitionMode.SMOOTH)
                    print(f"感情切替: {emotion}")
        
        # 更新
        engine.update(delta_time)
        
        # 描画
        screen.fill((0, 0, 0))
        engine.render_background(screen)
        
        # マウス位置にオーラ
        mouse_pos = pygame.mouse.get_pos()
        engine.render_emotion_aura(screen, mouse_pos, 1.0)
        
        # 感情テキスト
        emotion_text = engine.create_emotion_text_surface(f"Emotion: {engine.current_emotion}")
        screen.blit(emotion_text, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()
    
    stats = engine.get_performance_stats()
    print(f"📊 性能統計: {stats}")
    
    print("✅ Emotion Visual Engine テスト完了")

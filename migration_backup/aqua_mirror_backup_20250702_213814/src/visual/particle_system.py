import pygame
import numpy as np
import random
import math
import time
from typing import List, Tuple, Dict, Optional
from enum import Enum

class ParticleType(Enum):
    BASIC = "basic"
    EMOTION = "emotion"
    GESTURE = "gesture"
    RIPPLE = "ripple"

class Particle:
    """個別パーティクルクラス"""
    
    def __init__(self, x: float, y: float, vx: float, vy: float, 
                 color: Tuple[int, int, int], size: float, life: float,
                 particle_type: ParticleType = ParticleType.BASIC):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.max_life = life
        self.life = life
        self.particle_type = particle_type
        
        # エフェクト用パラメータ
        self.alpha = 255
        self.gravity = 0.1
        self.fade_rate = 1.0
        self.size_decay = 0.98
        
        # 追加データ
        self.data = {}
    
    def update(self, delta_time: float):
        """パーティクル更新"""
        # 位置更新
        self.x += self.vx * delta_time * 60  # 60FPS基準
        self.y += self.vy * delta_time * 60
        
        # 重力適用
        self.vy += self.gravity * delta_time * 60
        
        # ライフ減少
        self.life -= delta_time * self.fade_rate
        
        # アルファ値更新
        life_ratio = max(0, self.life / self.max_life)
        self.alpha = int(255 * life_ratio)
        
        # サイズ減少
        self.size *= self.size_decay
    
    def is_alive(self) -> bool:
        """生存確認"""
        return self.life > 0 and self.size > 0.1
    
    def render(self, screen: pygame.Surface):
        """パーティクル描画"""
        if not self.is_alive():
            return
        
        try:
            # 透明度付きサーフェス作成
            particle_surface = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
            
            # 色にアルファ値適用
            color_with_alpha = (*self.color, self.alpha)
            
            # パーティクルタイプに応じた描画
            if self.particle_type == ParticleType.BASIC:
                pygame.draw.circle(particle_surface, color_with_alpha, 
                                 (self.size, self.size), self.size)
            elif self.particle_type == ParticleType.EMOTION:
                self._render_emotion_particle(particle_surface, color_with_alpha)
            elif self.particle_type == ParticleType.GESTURE:
                self._render_gesture_particle(particle_surface, color_with_alpha)
            elif self.particle_type == ParticleType.RIPPLE:
                self._render_ripple_particle(particle_surface, color_with_alpha)
            
            # 画面に描画
            screen.blit(particle_surface, 
                       (self.x - self.size, self.y - self.size))
            
        except Exception as e:
            # 描画エラーは無視して続行
            pass
    
    def _render_emotion_particle(self, surface: pygame.Surface, color: Tuple[int, int, int, int]):
        """感情パーティクル描画"""
        # 星形パーティクル
        center = (self.size, self.size)
        points = []
        
        for i in range(10):
            angle = i * math.pi / 5
            if i % 2 == 0:
                radius = self.size
            else:
                radius = self.size * 0.5
            
            x = center[0] + math.cos(angle) * radius
            y = center[1] + math.sin(angle) * radius
            points.append((x, y))
        
        if len(points) >= 3:
            pygame.draw.polygon(surface, color, points)
    
    def _render_gesture_particle(self, surface: pygame.Surface, color: Tuple[int, int, int, int]):
        """ジェスチャーパーティクル描画"""
        # 四角形パーティクル
        rect = pygame.Rect(0, 0, self.size * 2, self.size * 2)
        pygame.draw.rect(surface, color, rect)
    
    def _render_ripple_particle(self, surface: pygame.Surface, color: Tuple[int, int, int, int]):
        """波紋パーティクル描画"""
        # リング状パーティクル
        center = (self.size, self.size)
        pygame.draw.circle(surface, color, center, self.size, 2)

class ParticleSystem:
    """パーティクルシステム（Day 4版）"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 設定パラメータ
        visual_config = config.get('visual_effects', {})
        particle_config = visual_config.get('particles', {})
        
        self.max_particles = particle_config.get('max_count', 500)
        self.emission_rate = particle_config.get('emission_rate', 30)
        self.gravity = particle_config.get('gravity', 0.1)
        
        # パーティクル管理
        self.particles: List[Particle] = []
        self.emission_accumulator = 0.0
        
        # パフォーマンス統計
        self.update_times = []
        self.render_times = []
        
        print("✅ Particle System 初期化完了")
    
    def emit_particles(self, position: Tuple[float, float], count: int,
                      particle_type: ParticleType = ParticleType.BASIC,
                      color: Tuple[int, int, int] = (255, 255, 255),
                      velocity_range: Tuple[float, float] = (-2, 2),
                      size_range: Tuple[float, float] = (2, 6),
                      life_range: Tuple[float, float] = (1.0, 3.0)):
        """パーティクル放出"""
        x, y = position
        
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            
            # ランダムパラメータ生成
            vx = random.uniform(*velocity_range)
            vy = random.uniform(*velocity_range)
            size = random.uniform(*size_range)
            life = random.uniform(*life_range)
            
            # 色の微調整
            color_variation = 30
            varied_color = (
                max(0, min(255, color[0] + random.randint(-color_variation, color_variation))),
                max(0, min(255, color[1] + random.randint(-color_variation, color_variation))),
                max(0, min(255, color[2] + random.randint(-color_variation, color_variation)))
            )
            
            particle = Particle(x, y, vx, vy, varied_color, size, life, particle_type)
            self.particles.append(particle)
    
    def emit_emotion_particles(self, position: Tuple[float, float], emotion: str, intensity: float = 1.0):
        """感情パーティクル放出"""
        emotion_configs = {
            'happy': {
                'color': (255, 223, 0),      # 黄色
                'count': int(20 * intensity),
                'velocity_range': (-3, 3),
                'size_range': (3, 8),
                'life_range': (2.0, 4.0)
            },
            'sad': {
                'color': (70, 130, 180),     # 青
                'count': int(10 * intensity),
                'velocity_range': (-1, 1),
                'size_range': (2, 5),
                'life_range': (3.0, 5.0)
            },
            'angry': {
                'color': (220, 20, 60),      # 赤
                'count': int(30 * intensity),
                'velocity_range': (-4, 4),
                'size_range': (4, 10),
                'life_range': (1.0, 2.0)
            },
            'surprised': {
                'color': (255, 20, 147),     # ピンク
                'count': int(40 * intensity),
                'velocity_range': (-5, 5),
                'size_range': (2, 6),
                'life_range': (0.5, 1.5)
            },
            'neutral': {
                'color': (200, 200, 200),    # グレー
                'count': int(5 * intensity),
                'velocity_range': (-1, 1),
                'size_range': (3, 6),
                'life_range': (2.0, 3.0)
            }
        }
        
        config = emotion_configs.get(emotion, emotion_configs['neutral'])
        
        self.emit_particles(
            position=position,
            count=config['count'],
            particle_type=ParticleType.EMOTION,
            color=config['color'],
            velocity_range=config['velocity_range'],
            size_range=config['size_range'],
            life_range=config['life_range']
        )
    
    def emit_gesture_particles(self, position: Tuple[float, float], gesture_type: str, intensity: float = 1.0):
        """ジェスチャーパーティクル放出"""
        gesture_configs = {
            'wave': {
                'color': (0, 255, 255),      # シアン
                'count': int(15 * intensity),
                'velocity_pattern': 'wave'
            },
            'circle': {
                'color': (255, 165, 0),      # オレンジ
                'count': int(25 * intensity),
                'velocity_pattern': 'circular'
            },
            'point': {
                'color': (0, 255, 0),        # 緑
                'count': int(10 * intensity),
                'velocity_pattern': 'directional'
            },
            'clap': {
                'color': (255, 255, 0),      # 黄色
                'count': int(50 * intensity),
                'velocity_pattern': 'explosion'
            }
        }
        
        config = gesture_configs.get(gesture_type, gesture_configs['point'])
        
        # パターンに応じた速度設定
        if config['velocity_pattern'] == 'wave':
            velocity_range = (-3, 3)
        elif config['velocity_pattern'] == 'circular':
            velocity_range = (-2, 2)
        elif config['velocity_pattern'] == 'directional':
            velocity_range = (-1, 5)  # 上向き
        elif config['velocity_pattern'] == 'explosion':
            velocity_range = (-6, 6)
        else:
            velocity_range = (-2, 2)
        
        self.emit_particles(
            position=position,
            count=config['count'],
            particle_type=ParticleType.GESTURE,
            color=config['color'],
            velocity_range=velocity_range,
            size_range=(3, 7),
            life_range=(1.5, 3.0)
        )
    
    def update(self, delta_time: float):
        """パーティクルシステム更新"""
        start_time = time.time()
        
        # 全パーティクル更新
        for particle in self.particles[:]:  # コピーして反復
            particle.update(delta_time)
            
            # 死んだパーティクルを削除
            if not particle.is_alive():
                self.particles.remove(particle)
        
        # 更新時間記録
        update_time = time.time() - start_time
        self.update_times.append(update_time)
        if len(self.update_times) > 60:
            self.update_times.pop(0)
    
    def render(self, screen: pygame.Surface):
        """パーティクル描画"""
        start_time = time.time()
        
        for particle in self.particles:
            particle.render(screen)
        
        # 描画時間記録
        render_time = time.time() - start_time
        self.render_times.append(render_time)
        if len(self.render_times) > 60:
            self.render_times.pop(0)
    
    def clear_particles(self):
        """全パーティクルクリア"""
        self.particles.clear()
    
    def get_particle_count(self) -> int:
        """現在のパーティクル数取得"""
        return len(self.particles)
    
    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計取得"""
        return {
            'particle_count': len(self.particles),
            'max_particles': self.max_particles,
            'avg_update_time': sum(self.update_times) / len(self.update_times) if self.update_times else 0,
            'avg_render_time': sum(self.render_times) / len(self.render_times) if self.render_times else 0
        }

# テスト実行用
if __name__ == "__main__":
    print("🔍 Particle System テスト開始...")
    
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Particle System Test")
    clock = pygame.time.Clock()
    
    config = {
        'visual_effects': {
            'particles': {
                'max_count': 500,
                'emission_rate': 30,
                'gravity': 0.1
            }
        }
    }
    
    particle_system = ParticleSystem(config)
    
    print("🎆 パーティクルテスト開始（クリック: パーティクル放出, ESC: 終了）...")
    
    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0  # 60FPS, 秒単位
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # マウス位置にパーティクル放出
                mouse_pos = pygame.mouse.get_pos()
                
                if event.button == 1:  # 左クリック: 感情パーティクル
                    emotions = ['happy', 'sad', 'angry', 'surprised']
                    emotion = random.choice(emotions)
                    particle_system.emit_emotion_particles(mouse_pos, emotion, 1.0)
                elif event.button == 3:  # 右クリック: ジェスチャーパーティクル
                    gestures = ['wave', 'circle', 'point', 'clap']
                    gesture = random.choice(gestures)
                    particle_system.emit_gesture_particles(mouse_pos, gesture, 1.0)
        
        # 更新
        particle_system.update(delta_time)
        
        # 描画
        screen.fill((0, 0, 0))  # 黒背景
        particle_system.render(screen)
        
        # 情報表示
        font = pygame.font.Font(None, 36)
        info_text = font.render(f"Particles: {particle_system.get_particle_count()}", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))
        
        pygame.display.flip()
    
    pygame.quit()
    
    # 統計表示
    stats = particle_system.get_performance_stats()
    print(f"📊 性能統計: {stats}")
    
    print("✅ Particle System テスト完了")

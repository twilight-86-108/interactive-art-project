import pygame
import sys
import time
import cv2
from typing import Optional

from .camera_manager import CameraManager
from .gpu_processor import GPUProcessor
from .error_manager import ErrorManager, ErrorSeverity
from .performance_monitor import PerformanceMonitor

class AquaMirrorApp:
    """Aqua Mirror メインアプリケーション（Day 2版）"""
    
    def __init__(self, config: dict):
        self.config = config
        self.running = True
        self.demo_mode = False
        
        # コンポーネント
        self.camera_manager = None
        self.gpu_processor = None
        self.error_manager = None
        self.performance_monitor = None
        
        # Pygame関連
        self.screen = None
        self.clock = None
        self.font = None
        
        # 状態管理
        self.current_frame = None
        self.frame_count = 0
        self.debug_mode = config.get('debug_mode', True)
        
        print("🌊 Aqua Mirror App 初期化開始...")
        self._initialize()
    
    def _initialize(self):
        """アプリケーション初期化"""
        try:
            # エラーマネージャー初期化（最優先）
            self.error_manager = ErrorManager()
            
            # パフォーマンスモニター初期化
            self.performance_monitor = PerformanceMonitor()
            self.performance_monitor.start_monitoring()
            
            # GPU処理初期化
            self.gpu_processor = GPUProcessor()
            
            # Pygame初期化
            self._initialize_pygame()
            
            # カメラ初期化
            self._initialize_camera()
            
            print("✅ アプリケーション初期化完了")
            
        except Exception as e:
            print(f"❌ 初期化エラー: {e}")
            if self.error_manager:
                self.error_manager.handle_error(e, ErrorSeverity.CRITICAL, self)
            raise
    
    def _initialize_pygame(self):
        """Pygame初期化"""
        pygame.init()
        
        # 画面設定
        width = self.config['hardware']['display']['width']
        height = self.config['hardware']['display']['height']
        fullscreen = self.config['hardware']['display']['fullscreen']
        
        flags = pygame.DOUBLEBUF | pygame.HWSURFACE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("Aqua Mirror - Day 2")
        
        # クロック・フォント初期化
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        print("✅ Pygame初期化完了")
    
    def _initialize_camera(self):
        """カメラ初期化"""
        try:
            device_id = self.config['hardware']['camera']['device_id']
            self.camera_manager = CameraManager(device_id)
            
            if not self.camera_manager.initialize():
                raise RuntimeError("カメラ初期化失敗")
            
            print("✅ カメラ初期化完了")
            
        except Exception as e:
            print(f"⚠️  カメラ初期化エラー: {e}")
            self.error_manager.handle_error(e, ErrorSeverity.ERROR, self)
    
    def enable_demo_mode(self):
        """デモモード有効化（カメラなし動作）"""
        self.demo_mode = True
        print("🎭 デモモード有効化")
    
    def reduce_quality(self):
        """品質設定削減"""
        print("📉 品質設定を下げました")
        # 実装: 解像度下げ、FPS目標下げなど
    
    def adjust_performance(self):
        """パフォーマンス調整"""
        print("⚡ パフォーマンス調整実行")
        # 実装: フレームスキップ、品質調整など
    
    def run(self):
        """メインループ実行"""
        print("🚀 メインループ開始...")
        
        try:
            while self.running:
                frame_start = time.time()
                
                # イベント処理
                self._handle_events()
                
                # フレーム更新
                self._update_frame()
                
                # 描画
                self._render()
                
                # パフォーマンス記録
                frame_time = time.time() - frame_start
                self.performance_monitor.record_frame_time(frame_time)
                
                # FPS制御
                target_fps = self.config['performance']['target_fps']
                self.clock.tick(target_fps)
                
                self.frame_count += 1
                
        except Exception as e:
            print(f"❌ メインループエラー: {e}")
            self.error_manager.handle_error(e, ErrorSeverity.CRITICAL, self)
        finally:
            self._cleanup()
    
    def _handle_events(self):
        """イベント処理"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_key_event(event.key)
    
    def _handle_key_event(self, key):
        """キーイベント処理"""
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_d:
            self.debug_mode = not self.debug_mode
            print(f"🐛 デバッグモード: {'ON' if self.debug_mode else 'OFF'}")
        elif key == pygame.K_f:
            # フルスクリーン切替（簡易版）
            print("🖥️  フルスクリーン切替（未実装）")
        elif key == pygame.K_r:
            # カメラリセット
            if self.camera_manager:
                try:
                    self.camera_manager.cleanup()
                    self.camera_manager.initialize()
                    print("📹 カメラリセット完了")
                except Exception as e:
                    self.error_manager.handle_error(e, ErrorSeverity.ERROR, self)
    
    def _update_frame(self):
        """フレーム更新"""
        try:
            if self.demo_mode:
                # デモモード: ランダムカラー生成
                import numpy as np
                width = self.config['hardware']['display']['width']
                height = self.config['hardware']['display']['height']
                self.current_frame = np.random.randint(0, 255, (height//4, width//4, 3), dtype=np.uint8)
            
            elif self.camera_manager:
                # カメラフレーム取得
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # GPU処理でリサイズ
                    processed_frame = self.gpu_processor.resize_frame(
                        frame, 
                        (self.config['hardware']['display']['width'] // 2,
                         self.config['hardware']['display']['height'] // 2)
                    )
                    self.current_frame = processed_frame
                    
        except Exception as e:
            self.error_manager.handle_error(e, ErrorSeverity.WARNING, self)
    
    def _render(self):
        """描画処理"""
        # 背景クリア
        self.screen.fill((0, 20, 40))  # 深い青色
        
        # カメラ画像表示
        if self.current_frame is not None:
            self._render_camera_frame()
        
        # デバッグ情報表示
        if self.debug_mode:
            self._render_debug_info()
        
        # 画面更新
        pygame.display.flip()
    
    def _render_camera_frame(self):
        """カメラフレーム描画"""
        try:
            # OpenCV BGR -> Pygame RGB 変換
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            
            # Pygame サーフェス作成
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # 画面中央に配置
            screen_rect = self.screen.get_rect()
            frame_rect = frame_surface.get_rect()
            frame_rect.center = screen_rect.center
            
            self.screen.blit(frame_surface, frame_rect)
            
        except Exception as e:
            self.error_manager.handle_error(e, ErrorSeverity.WARNING, self)
    
    def _render_debug_info(self):
        """デバッグ情報描画"""
        try:
            y_offset = 10
            line_height = 30
            
            # FPS表示
            fps = self.clock.get_fps()
            fps_text = self.font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, y_offset))
            y_offset += line_height
            
            # フレーム数表示
            frame_text = self.font.render(f"Frames: {self.frame_count}", True, (255, 255, 255))
            self.screen.blit(frame_text, (10, y_offset))
            y_offset += line_height
            
            # モード表示
            mode = "Demo" if self.demo_mode else "Camera"
            mode_text = self.font.render(f"Mode: {mode}", True, (255, 255, 255))
            self.screen.blit(mode_text, (10, y_offset))
            y_offset += line_height
            
            # GPU状態
            gpu_status = "GPU" if self.gpu_processor.gpu_available else "CPU"
            gpu_text = self.font.render(f"Processing: {gpu_status}", True, (255, 255, 255))
            self.screen.blit(gpu_text, (10, y_offset))
            y_offset += line_height
            
            # パフォーマンス警告
            warnings = self.performance_monitor.check_performance_warnings()
            if warnings:
                warning_text = self.font.render(f"⚠️ {', '.join(warnings)}", True, (255, 255, 0))
                self.screen.blit(warning_text, (10, y_offset))
            
            # キー操作ガイド
            guide_y = self.screen.get_height() - 100
            guides = [
                "ESC: 終了",
                "D: デバッグ切替",
                "R: カメラリセット"
            ]
            
            for i, guide in enumerate(guides):
                guide_text = self.font.render(guide, True, (200, 200, 200))
                self.screen.blit(guide_text, (10, guide_y + i * 25))
                
        except Exception as e:
            # デバッグ情報描画でエラーが出ても継続
            pass
    
    def _cleanup(self):
        """リソース解放"""
        print("🧹 リソース解放中...")
        
        components = [
            (self.performance_monitor, 'stop_monitoring'),
            (self.camera_manager, 'cleanup'),
            (self.gpu_processor, 'cleanup')
        ]
        
        for component, method_name in components:
            if component and hasattr(component, method_name):
                try:
                    getattr(component, method_name)()
                except Exception as e:
                    print(f"⚠️  {component.__class__.__name__} 解放エラー: {e}")
        
        pygame.quit()
        print("✅ リソース解放完了")

# テスト実行用
if __name__ == "__main__":
    # テスト設定
    test_config = {
        'hardware': {
            'camera': {'device_id': 0},
            'display': {'width': 1280, 'height': 720, 'fullscreen': False}
        },
        'performance': {'target_fps': 30},
        'debug_mode': True
    }
    
    app = AquaMirrorApp(test_config)
    app.run()

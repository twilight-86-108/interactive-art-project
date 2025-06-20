# src/core/app.py を完全修正
import pygame
import sys
import time
import cv2
import numpy as np
from typing import Optional, Any

from .camera_manager import CameraManager
from .gpu_processor import GPUProcessor
from .error_manager import ErrorManager, ErrorSeverity
from .performance_monitor import PerformanceMonitor

class AquaMirrorApp:
    """Aqua Mirror メインアプリケーション（エラー修正版）"""
    
    def __init__(self, config: dict):
        self.config = config
        self.running = True
        self.demo_mode = False
        self.initialization_failed = False
        
        # コンポーネント（初期値をNoneに設定）
        self.camera_manager: Optional[CameraManager] = None
        self.gpu_processor: Optional[GPUProcessor] = None
        self.error_manager: Optional[ErrorManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # Pygame関連（初期値をNoneに設定）
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        
        # 状態管理
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.debug_mode = config.get('debug_mode', True)
        
        print("🌊 Aqua Mirror App 初期化開始...")
        self._safe_initialize()
    
    def _safe_initialize(self):
        """安全な初期化（エラー処理付き）"""
        try:
            # エラーマネージャー初期化（最優先）
            self._initialize_error_manager()
            
            # パフォーマンスモニター初期化
            self._initialize_performance_monitor()
            
            # GPU処理初期化
            self._initialize_gpu_processor()
            
            # Pygame初期化
            self._initialize_pygame()
            
            # カメラ初期化
            self._initialize_camera()
            
            print("✅ アプリケーション初期化完了")
            
        except Exception as e:
            print(f"❌ 重大な初期化エラー: {e}")
            self.initialization_failed = True
            self._enable_safe_mode()
    
    def _initialize_error_manager(self):
        """エラーマネージャー初期化"""
        try:
            self.error_manager = ErrorManager()
        except Exception as e:
            print(f"⚠️  エラーマネージャー初期化失敗: {e}")
            self.error_manager = None
    
    def _initialize_performance_monitor(self):
        """パフォーマンスモニター初期化"""
        try:
            self.performance_monitor = PerformanceMonitor()
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
        except Exception as e:
            print(f"⚠️  パフォーマンスモニター初期化失敗: {e}")
            self.performance_monitor = None
    
    def _initialize_gpu_processor(self):
        """GPU処理初期化"""
        try:
            self.gpu_processor = GPUProcessor()
        except Exception as e:
            print(f"⚠️  GPU処理初期化失敗: {e}")
            self.gpu_processor = None
    
    def _safe_error_handle(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR):
        """安全なエラーハンドリング"""
        if self.error_manager:
            try:
                self.error_manager.handle_error(error, severity, self)
            except:
                print(f"⚠️  エラーハンドリング失敗: {error}")
        else:
            print(f"⚠️  エラー（マネージャー無効）: {error}")
    
    def _get_camera_config(self) -> dict:
        """カメラ設定取得（フォールバック対応）"""
        try:
            # 新構造の場合
            if 'hardware' in self.config and 'camera' in self.config['hardware']:
                return self.config['hardware']['camera']
            # 旧構造の場合
            elif 'camera' in self.config:
                return self.config['camera']
            # デフォルト設定
            else:
                return {'device_id': 0}
        except:
            return {'device_id': 0}
    
    def _get_display_config(self) -> dict:
        """ディスプレイ設定取得（フォールバック対応）"""
        try:
            # 新構造の場合
            if 'hardware' in self.config and 'display' in self.config['hardware']:
                return self.config['hardware']['display']
            # 旧構造の場合
            elif 'display' in self.config:
                return self.config['display']
            # デフォルト設定
            else:
                return {'width': 1280, 'height': 720, 'fullscreen': False}
        except:
            return {'width': 1280, 'height': 720, 'fullscreen': False}
    
    def _get_performance_config(self) -> dict:
        """パフォーマンス設定取得（フォールバック対応）"""
        try:
            if 'performance' in self.config:
                return self.config['performance']
            else:
                return {'target_fps': 30}
        except:
            return {'target_fps': 30}
    
    def _initialize_pygame(self):
        """Pygame初期化（安全版）"""
        try:
            pygame.init()
            
            # 画面設定（フォールバック対応）
            display_config = self._get_display_config()
            width = display_config.get('width', 1280)
            height = display_config.get('height', 720)
            fullscreen = display_config.get('fullscreen', False)
            
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE
            if fullscreen:
                flags |= pygame.FULLSCREEN
            
            self.screen = pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption("Aqua Mirror - Day 2")
            
            # クロック・フォント初期化
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            
            print(f"✅ Pygame初期化完了 ({width}x{height})")
            
        except Exception as e:
            print(f"❌ Pygame初期化エラー: {e}")
            self._safe_error_handle(e, ErrorSeverity.CRITICAL)
            # フォールバック：最小設定で再試行
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((800, 600))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 24)
                print("✅ Pygame最小設定で初期化完了")
            except Exception as fallback_error:
                print(f"❌ Pygame最小設定も失敗: {fallback_error}")
                self.screen = None
                self.clock = None
                self.font = None
    
    def _initialize_camera(self):
        """カメラ初期化（安全版）"""
        try:
            camera_config = self._get_camera_config()
            device_id = camera_config.get('device_id', 0)
            
            self.camera_manager = CameraManager(device_id)
            
            if not self.camera_manager.initialize():
                raise RuntimeError("カメラ初期化失敗")
            
            print("✅ カメラ初期化完了")
            
        except Exception as e:
            print(f"⚠️  カメラ初期化エラー: {e}")
            self._safe_error_handle(e, ErrorSeverity.ERROR)
            self.camera_manager = None
            self.enable_demo_mode()
    
    def _enable_safe_mode(self):
        """セーフモード有効化"""
        print("🛡️  セーフモード有効化")
        self.demo_mode = True
        
        # 最小限のPygame設定
        if not self.screen:
            try:
                pygame.init()
                self.screen = pygame.display.set_mode((800, 600))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 24)
            except:
                pass
    
    def enable_demo_mode(self):
        """デモモード有効化（カメラなし動作）"""
        self.demo_mode = True
        print("🎭 デモモード有効化")
    
    def reduce_quality(self):
        """品質設定削減"""
        print("📉 品質設定を下げました")
    
    def adjust_performance(self):
        """パフォーマンス調整"""
        print("⚡ パフォーマンス調整実行")
    
    def run(self):
        """メインループ実行（安全版）"""
        if self.initialization_failed and not self.screen:
            print("❌ 初期化が完全に失敗しました。アプリケーションを終了します。")
            return
        
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
                if self.performance_monitor:
                    try:
                        self.performance_monitor.record_frame_time(frame_time)
                    except:
                        pass
                
                # FPS制御
                if self.clock:
                    try:
                        performance_config = self._get_performance_config()
                        target_fps = performance_config.get('target_fps', 30)
                        self.clock.tick(target_fps)
                    except:
                        time.sleep(1.0 / 30)  # フォールバック：30FPS相当の待機
                else:
                    time.sleep(1.0 / 30)  # クロックがない場合のフォールバック
                
                self.frame_count += 1
                
        except Exception as e:
            print(f"❌ メインループエラー: {e}")
            self._safe_error_handle(e, ErrorSeverity.CRITICAL)
        finally:
            self._cleanup()
    
    def _handle_events(self):
        """イベント処理（安全版）"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key_event(event.key)
        except:
            # イベント処理エラーでも継続
            pass
    
    def _handle_key_event(self, key):
        """キーイベント処理（安全版）"""
        try:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_d:
                self.debug_mode = not self.debug_mode
                print(f"🐛 デバッグモード: {'ON' if self.debug_mode else 'OFF'}")
            elif key == pygame.K_f:
                print("🖥️  フルスクリーン切替（未実装）")
            elif key == pygame.K_r:
                # カメラリセット
                if self.camera_manager:
                    try:
                        self.camera_manager.cleanup()
                        self.camera_manager.initialize()
                        print("📹 カメラリセット完了")
                    except Exception as e:
                        self._safe_error_handle(e, ErrorSeverity.ERROR)
        except:
            pass
    
    def _update_frame(self):
        """フレーム更新（安全版）"""
        try:
            if self.demo_mode:
                # デモモード: ランダムカラー生成
                display_config = self._get_display_config()
                width = display_config.get('width', 1280)
                height = display_config.get('height', 720)
                self.current_frame = np.random.randint(0, 255, (height//4, width//4, 3), dtype=np.uint8)
            
            elif self.camera_manager:
                # カメラフレーム取得
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    # GPU処理でリサイズ
                    display_config = self._get_display_config()
                    target_width = display_config.get('width', 1280) // 2
                    target_height = display_config.get('height', 720) // 2
                    
                    if self.gpu_processor:
                        try:
                            processed_frame = self.gpu_processor.resize_frame(
                                frame, (target_width, target_height)
                            )
                            self.current_frame = processed_frame
                        except Exception as gpu_error:
                            # GPU処理失敗時はそのまま使用
                            self.current_frame = cv2.resize(frame, (target_width, target_height))
                    else:
                        # GPU処理なしの場合
                        self.current_frame = cv2.resize(frame, (target_width, target_height))
                        
        except Exception as e:
            self._safe_error_handle(e, ErrorSeverity.WARNING)
    
    def _render(self):
        """描画処理（安全版）"""
        if not self.screen:
            return
        
        try:
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
            
        except Exception as e:
            # 描画エラーでも継続
            try:
                if self.screen:
                    self.screen.fill((100, 0, 0))  # エラー時は赤背景
                    pygame.display.flip()
            except:
                pass
    
    def _render_camera_frame(self):
        """カメラフレーム描画（安全版）"""
        if not self.screen or self.current_frame is None:
            return
        
        try:
            # None チェック
            if self.current_frame is None:
                return
            
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
            self._safe_error_handle(e, ErrorSeverity.WARNING)
    
    def _render_debug_info(self):
        """デバッグ情報描画（安全版）"""
        if not self.screen or not self.font:
            return
        
        try:
            y_offset = 10
            line_height = 30
            
            # 基本情報
            fps = self.clock.get_fps() if self.clock else 0.0
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
            gpu_status = "GPU" if (self.gpu_processor and self.gpu_processor.gpu_available) else "CPU"
            gpu_text = self.font.render(f"Processing: {gpu_status}", True, (255, 255, 255))
            self.screen.blit(gpu_text, (10, y_offset))
            y_offset += line_height
            
            # 初期化状態
            if self.initialization_failed:
                error_text = self.font.render("⚠️ セーフモード", True, (255, 255, 0))
                self.screen.blit(error_text, (10, y_offset))
                y_offset += line_height
            
            # パフォーマンス警告
            if self.performance_monitor:
                try:
                    warnings = self.performance_monitor.check_performance_warnings()
                    if warnings:
                        warning_text = self.font.render(f"⚠️ {', '.join(warnings)}", True, (255, 255, 0))
                        self.screen.blit(warning_text, (10, y_offset))
                except:
                    pass
            
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
        """リソース解放（安全版）"""
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
        
        try:
            pygame.quit()
        except:
            pass
        
        print("✅ リソース解放完了")

# テスト実行用
if __name__ == "__main__":
    # テスト設定（両方の構造に対応）
    test_config = {
        'camera': {'device_id': 0},
        'display': {'width': 1280, 'height': 720, 'fullscreen': False},
        'performance': {'target_fps': 30},
        'debug_mode': True
    }
    
    app = AquaMirrorApp(test_config)
    app.run()
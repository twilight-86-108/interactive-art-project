# src/app.py - 統合版メインアプリケーション
import pygame
import time
import sys
import logging
from enum import Enum
from typing import Dict, Any, Optional
from collections import deque

# コンポーネントのインポート
from core.config_loader import ConfigLoader

# エラーハンドリング・パフォーマンス管理
try:
    from core.error_manager import ErrorManager, ErrorSeverity
except ImportError:
    print("⚠️ ErrorManager not found, using fallback")
    ErrorManager = None
    ErrorSeverity = None

try:
    from core.performance_monitor import PerformanceMonitor
except ImportError:
    print("⚠️ PerformanceMonitor not found, using fallback")
    PerformanceMonitor = None

try:
    from core.quality_manager import AdaptiveQualityManager
except ImportError:
    print("⚠️ AdaptiveQualityManager not found, using fallback")
    AdaptiveQualityManager = None

try:
    from core.gpu_processor import GPUProcessor
except ImportError:
    print("⚠️ GPUProcessor not found, using fallback")
    GPUProcessor = None

# AI処理コンポーネント
try:
    from vision.vision_processor import VisionProcessor
except ImportError:
    print("⚠️ VisionProcessor not found, using fallback")
    VisionProcessor = None

class _EmotionFallback(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"

try:
    from emotion.emotion_analyzer import EmotionAnalyzer, Emotion
except ImportError:
    print("⚠️ EmotionAnalyzer not found, using fallback")
    EmotionAnalyzer = None
    Emotion = _EmotionFallback

try:
    from scene import SceneManager
except ImportError:
    print("⚠️ SceneManager not found, using fallback")
    SceneManager = None

class AppState(Enum):
    """アプリケーション状態の定義"""
    STANDBY = "standby"
    RECOGNITION = "recognition"
    INTERACTION = "interaction"
    EXPERIENCE_END = "experience_end"
    ERROR_RECOVERY = "error_recovery"
    DEMO_MODE = "demo_mode"
    ERROR = "error"

class AquaMirrorApp:
    """統合版メインアプリケーションクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # ログ設定
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # アプリケーション状態
        self.running = True
        self.clock = pygame.time.Clock()
        self.current_state = AppState.STANDBY
        self.state_timer = 0
        self.last_detection_time = 0
        
        # パフォーマンス管理
        self.target_fps = config.get('performance', {}).get('target_fps', 30)
        self.frame_times = deque(maxlen=60)
        
        # デバッグ・デモモード
        self.debug_mode = config.get('debug_mode', False)
        self.demo_mode = config.get('demo_mode', False)
        
        # 高度な管理コンポーネント
        self.error_manager = None
        self.performance_monitor = None
        self.quality_manager = None
        
        # AI処理コンポーネント
        self.vision_processor = None
        self.emotion_analyzer = None
        self.scene_manager = None
        self.camera = None
        
        # デモ用データ
        self.demo_data = self._create_demo_data() if self.demo_mode else None
        self.demo_index = 0
        
        # 統合初期化
        self._integrated_initialization()
        
        self.logger.info("🌊 Aqua Mirror アプリケーションが初期化されました")
    
    def _setup_logging(self):
        """ログシステム設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log'),
                logging.StreamHandler()
            ]
        )
    
    def _integrated_initialization(self):
        """統合初期化処理"""
        try:
            self.logger.info("🚀 統合初期化を開始します...")
            
            # 高度管理コンポーネント初期化
            self._init_advanced_managers()
            
            # Pygame初期化
            self._init_pygame()
            
            # カメラ初期化（デモモードでない場合）
            if not self.demo_mode:
                self._init_camera()
            else:
                self.logger.info("📺 デモモードが有効です")
            
            # AI処理コンポーネント初期化
            self._init_ai_components()
            
            # シーンマネージャー初期化
            self._init_scene_manager()
            
            # 品質管理初期化
            self._init_quality_management()
            
            # GPU処理確認
            self._check_gpu_status()
            
            self.logger.info("✅ 統合初期化完了")
            
        except Exception as e:
            if self.error_manager and ErrorSeverity:
                self.error_manager.handle_error(e, ErrorSeverity.CRITICAL, self)
            else:
                self.logger.error(f"初期化エラー: {e}")
            raise
    
    def _init_advanced_managers(self):
        """高度管理コンポーネント初期化"""
        try:
            # エラーマネージャー
            if ErrorManager:
                self.error_manager = ErrorManager(self.config)
                self.logger.info("✅ ErrorManager初期化完了")
            else:
                self.logger.warning("⚠️ ErrorManager利用不可、基本エラーハンドリングを使用")
            
            # パフォーマンスモニター
            if PerformanceMonitor:
                self.performance_monitor = PerformanceMonitor()
                self.logger.info("✅ PerformanceMonitor初期化完了")
            else:
                self.logger.warning("⚠️ PerformanceMonitor利用不可")
            
        except Exception as e:
            self.logger.error(f"高度管理コンポーネント初期化エラー: {e}")
    
    def _init_pygame(self):
        """Pygame初期化"""
        try:
            pygame.init()
            pygame.mixer.init()
            pygame.font.init()
            
            display_config = self.config.get('display', {})
            width = display_config.get('width', 1920)
            height = display_config.get('height', 1080)
            fullscreen = display_config.get('fullscreen', False)
            
            # 最適化されたフラグ
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE
            if fullscreen:
                flags |= pygame.FULLSCREEN
            
            self.screen = pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption("Aqua Mirror - Interactive Art Experience")
            
            self.logger.info(f"🖥️ Pygame初期化完了: {width}x{height}, fullscreen={fullscreen}")
            
        except Exception as e:
            self.logger.error(f"Pygame初期化エラー: {e}")
            raise
    
    def _init_camera(self):
        """カメラ初期化"""
        try:
            self.camera = CameraManager(self.config)
            
            if not self.camera.initialize():
                raise RuntimeError("カメラの初期化に失敗しました")
            
            self.camera.start_capture()
            self.logger.info("📹 カメラが初期化されました")
            
        except Exception as e:
            self.logger.error(f"カメラ初期化エラー: {e}")
            self.logger.warning("📺 デモモードに切り替えます")
            self.demo_mode = True
            self.demo_data = self._create_demo_data()
    
    def _init_ai_components(self):
        """AI処理コンポーネント初期化"""
        try:
            # Vision Processor
            if VisionProcessor:
                self.vision_processor = VisionProcessor(self.config)
                self.logger.info("🤖 VisionProcessor初期化完了")
            else:
                self.logger.warning("⚠️ VisionProcessor利用不可")
            
            # Emotion Analyzer
            if EmotionAnalyzer:
                self.emotion_analyzer = EmotionAnalyzer()
                self.logger.info("💭 EmotionAnalyzer初期化完了")
            else:
                self.logger.warning("⚠️ EmotionAnalyzer利用不可")
            
        except Exception as e:
            self.logger.error(f"AI処理コンポーネント初期化エラー: {e}")
            if self.error_manager and ErrorSeverity:
                self.error_manager.handle_error(e, ErrorSeverity.ERROR, self)
    
    def _init_scene_manager(self):
        """シーンマネージャー初期化"""
        try:
            if SceneManager:
                display_config = self.config.get('display', {})
                self.scene_manager = SceneManager(
                    display_config.get('width', 1920),
                    display_config.get('height', 1080),
                    self.config
                )
                self.logger.info("🎨 SceneManager初期化完了")
            else:
                self.logger.warning("⚠️ SceneManager利用不可")
                
        except Exception as e:
            self.logger.error(f"SceneManager初期化エラー: {e}")
            if self.error_manager and ErrorSeverity:
                self.error_manager.handle_error(e, ErrorSeverity.ERROR, self)
    
    def _init_quality_management(self):
        """品質管理初期化"""
        try:
            if AdaptiveQualityManager and self.performance_monitor:
                self.quality_manager = AdaptiveQualityManager(self.performance_monitor)
                self.logger.info("⚙️ 品質管理初期化完了")
            else:
                self.logger.warning("⚠️ 品質管理機能利用不可")
                
        except Exception as e:
            self.logger.error(f"品質管理初期化エラー: {e}")
    
    def _check_gpu_status(self):
        """GPU処理確認"""
        try:
            if (self.vision_processor and 
                hasattr(self.vision_processor, 'gpu_processor') and
                hasattr(self.vision_processor.gpu_processor, 'is_gpu_available')):
                gpu_status = "有効" if self.vision_processor.gpu_processor.is_gpu_available() else "無効"
                self.logger.info(f"🖥️ GPU加速: {gpu_status}")
            else:
                self.logger.info("🖥️ GPU加速: 状態不明")
        except Exception as e:
            self.logger.error(f"GPU状態確認エラー: {e}")
    
    def _create_demo_data(self) -> list:
        """デモ用データ作成"""
        return [
            {
                'face_detected': True,
                'face_center': (0.5, 0.4, 0.5),
                'hands_detected': True,
                'hand_positions': [(0.3, 0.6), (0.7, 0.6)],
                'emotion': Emotion.HAPPY,
                'duration': 3.0,
                'face_landmarks': {'multi_face_landmarks': [{}]}
            },
            {
                'face_detected': True,
                'face_center': (0.6, 0.3, 0.3),
                'hands_detected': False,
                'hand_positions': [],
                'emotion': Emotion.SURPRISED,
                'duration': 2.5,
                'face_landmarks': {'multi_face_landmarks': [{}]}
            },
            {
                'face_detected': True,
                'face_center': (0.4, 0.5, 0.7),
                'hands_detected': True,
                'hand_positions': [(0.5, 0.5)],
                'emotion': Emotion.SAD,
                'duration': 2.0,
                'face_landmarks': {'multi_face_landmarks': [{}]}
            },
            {
                'face_detected': False,
                'face_center': None,
                'hands_detected': False,
                'hand_positions': [],
                'emotion': Emotion.NEUTRAL,
                'duration': 1.5,
                'face_landmarks': None
            }
        ]
    
    def run(self):
        """最適化メインループ"""
        self.logger.info("🚀 メインループを開始します")
        
        frame_count = 0
        last_quality_check = 0
        last_stats_time = 0
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # エラーハンドリング付きメイン処理
                try:
                    self._safe_handle_events()
                    self._safe_update()
                    self._safe_render()
                    
                except Exception as e:
                    if self.error_manager and ErrorSeverity:
                        recovery_success = self.error_manager.handle_error(e, ErrorSeverity.ERROR, self)
                        if not recovery_success:
                            self.current_state = AppState.ERROR_RECOVERY
                    else:
                        self.logger.error(f"メインループエラー: {e}")
                        self._handle_basic_error(e)
                
                # パフォーマンス記録
                frame_time = time.time() - frame_start_time
                self.frame_times.append(frame_time)
                
                if self.performance_monitor:
                    self.performance_monitor.record_frame_time(frame_time)
                
                # 品質管理（3秒間隔）
                if self.quality_manager and time.time() - last_quality_check > 3.0:
                    if self.quality_manager.update():
                        self._apply_quality_settings()
                    last_quality_check = time.time()
                
                # FPS制御
                self.clock.tick(self.target_fps)
                frame_count += 1
                
                # デバッグ情報（100フレームごと）
                if self.debug_mode and frame_count % 100 == 0:
                    self._print_debug_stats()
                
                # パフォーマンス統計（5秒ごと）
                if time.time() - last_stats_time > 5.0:
                    self._log_performance_stats()
                    last_stats_time = time.time()
                    
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ ユーザーによって停止されました")
        except Exception as e:
            if self.error_manager and ErrorSeverity:
                self.error_manager.handle_error(e, ErrorSeverity.CRITICAL, self)
            else:
                self.logger.error(f"致命的エラー: {e}")
                self.logger.exception("詳細なエラー情報:")
        finally:
            self._cleanup()
    
    def _safe_handle_events(self):
        """安全なイベント処理"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key_event(event.key)
        except Exception as e:
            self.logger.error(f"イベント処理エラー: {e}")
    
    def _handle_key_event(self, key):
        """拡張キーイベント処理"""
        try:
            if key == pygame.K_ESCAPE:
                self.running = False
            elif key == pygame.K_F1:
                self.debug_mode = not self.debug_mode
                self.logger.info(f"デバッグモード: {'ON' if self.debug_mode else 'OFF'}")
            elif key == pygame.K_F2:
                self.demo_mode = not self.demo_mode
                if self.demo_mode and not self.demo_data:
                    self.demo_data = self._create_demo_data()
                self.logger.info(f"デモモード: {'ON' if self.demo_mode else 'OFF'}")
            elif key == pygame.K_F3:
                # シーンエフェクトクリア
                if self.scene_manager and hasattr(self.scene_manager, 'clear_effects'):
                    self.scene_manager.clear_effects()
                    self.logger.info("エフェクトをクリアしました")
            elif key == pygame.K_F4:
                # 品質レベル切り替え
                if self.quality_manager:
                    self.quality_manager.cycle_quality_level()
                    self.logger.info("品質レベルを切り替えました")
            elif key == pygame.K_SPACE:
                # 状態リセット
                self.current_state = AppState.STANDBY
                self.logger.info("状態をリセットしました")
            elif key == pygame.K_r:
                # 強制再起動
                self._restart_components()
            elif key == pygame.K_p:
                # パフォーマンス統計表示
                stats = self.get_performance_stats()
                self.logger.info(f"パフォーマンス統計: {stats}")
                
        except Exception as e:
            self.logger.error(f"キーイベント処理エラー: {e}")
    
    def _safe_update(self):
        """安全な状態更新"""
        try:
            current_time = time.time()
            
            # 検出データ取得
            if self.demo_mode or (self.error_manager and self.error_manager.is_demo_mode()):
                detection_result = self._get_demo_detection_result()
            else:
                detection_result = self._get_real_detection_result()
            
            # 感情分析
            current_emotion, emotion_confidence = self._analyze_emotion(detection_result)
            
            # 状態遷移の管理
            self._update_state(detection_result, current_emotion, current_time)
            
            # シーンマネージャーに検出結果を渡す
            if self.scene_manager:
                self.scene_manager.update(detection_result, self.current_state)
            
        except Exception as e:
            self.logger.error(f"状態更新エラー: {e}")
            if self.error_manager and ErrorSeverity:
                self.error_manager.handle_error(e, ErrorSeverity.WARNING, self)
    
    def _get_real_detection_result(self):
        """実際のカメラからの検出結果取得"""
        try:
            if not self.camera:
                return {}
            
            # フレーム取得
            frame = self.camera.get_frame()
            if frame is None:
                return {}
            
            # AI処理
            if self.vision_processor:
                return self.vision_processor.process_frame(frame)
            else:
                return {}
            
        except Exception as e:
            self.logger.error(f"リアル検出エラー: {e}")
            return {}
    
    def _get_demo_detection_result(self):
        """デモ用検出結果取得"""
        try:
            if not self.demo_data:
                return {}
            
            # 時間ベースでデモデータを切り替え
            current_time = time.time()
            if not hasattr(self, 'demo_start_time'):
                self.demo_start_time = current_time
            
            elapsed = current_time - self.demo_start_time
            current_demo = self.demo_data[self.demo_index % len(self.demo_data)]
            
            if elapsed > current_demo['duration']:
                self.demo_index += 1
                self.demo_start_time = current_time
                current_demo = self.demo_data[self.demo_index % len(self.demo_data)]
            
            # デモデータを検出結果形式に変換
            return {
                'face_detected': current_demo['face_detected'],
                'face_center': current_demo['face_center'],
                'face_landmarks': current_demo.get('face_landmarks'),
                'hands_detected': current_demo['hands_detected'],
                'hand_positions': current_demo['hand_positions'],
                'hand_gestures': [],
                'face_distance': current_demo['face_center'][2] if current_demo['face_center'] else float('inf'),
                'timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error(f"デモ検出エラー: {e}")
            return {}
    
    def _analyze_emotion(self, detection_result):
        """感情分析"""
        try:
            if (detection_result.get('face_detected') and 
                detection_result.get('face_landmarks') and 
                self.emotion_analyzer):
                return self.emotion_analyzer.analyze_emotion(detection_result['face_landmarks'])
            else:
                return Emotion.NEUTRAL, 0.0
                
        except Exception as e:
            self.logger.error(f"感情分析エラー: {e}")
            return Emotion.NEUTRAL, 0.0
    
    def _update_state(self, detection_result, current_emotion, current_time):
        """状態遷移の管理"""
        try:
            face_detected = detection_result.get('face_detected', False)
            hands_detected = detection_result.get('hands_detected', False)
            face_distance = detection_result.get('face_distance', float('inf'))
            
            if self.current_state == AppState.STANDBY:
                if face_detected:
                    self.current_state = AppState.RECOGNITION
                    self.state_timer = current_time
                    self.logger.debug("状態変更: STANDBY -> RECOGNITION")
            
            elif self.current_state == AppState.RECOGNITION:
                if not face_detected:
                    if current_time - self.last_detection_time > 3.0:
                        self.current_state = AppState.STANDBY
                        self.logger.debug("状態変更: RECOGNITION -> STANDBY")
                else:
                    self.last_detection_time = current_time
                    
                    # インタラクション条件の確認
                    approach_threshold = self.config.get('interaction', {}).get('approach_threshold_z', 0.8)
                    if face_distance < approach_threshold or hands_detected:
                        self.current_state = AppState.INTERACTION
                        self.logger.debug("状態変更: RECOGNITION -> INTERACTION")
            
            elif self.current_state == AppState.INTERACTION:
                if not face_detected and not hands_detected:
                    self.current_state = AppState.EXPERIENCE_END
                    self.state_timer = current_time
                    self.logger.debug("状態変更: INTERACTION -> EXPERIENCE_END")
            
            elif self.current_state == AppState.EXPERIENCE_END:
                if current_time - self.state_timer > 3.0:
                    self.current_state = AppState.STANDBY
                    self.logger.debug("状態変更: EXPERIENCE_END -> STANDBY")
            
            elif self.current_state == AppState.ERROR_RECOVERY:
                # エラー復旧状態から自動復帰
                if current_time - self.state_timer > 5.0:
                    self.current_state = AppState.STANDBY
                    self.logger.info("エラー復旧完了: ERROR_RECOVERY -> STANDBY")
            
        except Exception as e:
            self.logger.error(f"状態遷移エラー: {e}")
    
    def _safe_render(self):
        """安全な描画処理"""
        try:
            # 背景クリア
            self.screen.fill((0, 0, 0))
            
            # シーン描画
            if self.scene_manager:
                self.scene_manager.draw(self.screen)
            else:
                self._render_fallback_scene()
            
            # UI描画
            if self.debug_mode:
                self._render_debug_info()
            
            if self.demo_mode:
                self._render_demo_overlay()
            
            if self.current_state == AppState.ERROR_RECOVERY:
                self._render_error_overlay()
            
            # 画面更新
            pygame.display.flip()
            
        except Exception as e:
            self.logger.error(f"描画エラー: {e}")
            # 最小限の描画でも継続
            try:
                self.screen.fill((255, 0, 0))  # 赤い画面でエラー表示
                pygame.display.flip()
            except:
                pass
    
    def _render_fallback_scene(self):
        """フォールバック描画"""
        try:
            # シンプルなグラデーション背景
            for y in range(self.screen.get_height()):
                ratio = y / self.screen.get_height()
                color = (int(50 * (1 - ratio)), int(100 * (1 - ratio)), int(150 * (1 - ratio)))
                pygame.draw.line(self.screen, color, (0, y), (self.screen.get_width(), y))
            
            # 中央にタイトル表示
            font = pygame.font.Font(None, 72)
            title = font.render("Aqua Mirror", True, (255, 255, 255))
            title_rect = title.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2))
            self.screen.blit(title, title_rect)
            
        except Exception as e:
            self.logger.error(f"フォールバック描画エラー: {e}")
    
    def _render_debug_info(self):
        """デバッグ情報の描画"""
        try:
            font = pygame.font.Font(None, 20)
            y_offset = 10
            
            debug_info = [
                f"FPS: {self.clock.get_fps():.1f}",
                f"State: {self.current_state.value}",
                f"Demo Mode: {self.demo_mode}",
            ]
            
            # AI処理情報
            if self.vision_processor and hasattr(self.vision_processor, 'get_debug_info'):
                ai_info = self.vision_processor.get_debug_info()
                debug_info.extend([f"{k}: {v}" for k, v in ai_info.items()])
            
            # エフェクト情報
            if self.scene_manager and hasattr(self.scene_manager, 'get_effect_count'):
                effect_count = self.scene_manager.get_effect_count()
                debug_info.append(f"Effects: {effect_count.get('total', 0)}")
            
            # パフォーマンス情報
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                debug_info.append(f"Frame Time: {avg_frame_time*1000:.1f}ms")
            
            # 描画
            for i, info in enumerate(debug_info):
                text = font.render(info, True, (255, 255, 255))
                self.screen.blit(text, (10, y_offset + i * 22))
            
        except Exception as e:
            self.logger.error(f"デバッグ情報描画エラー: {e}")
    
    def _render_demo_overlay(self):
        """デモモードオーバーレイ"""
        try:
            font = pygame.font.Font(None, 32)
            demo_text = font.render("DEMO MODE", True, (255, 255, 0))
            
            # 右上に表示
            text_rect = demo_text.get_rect()
            x = self.screen.get_width() - text_rect.width - 20
            y = 20
            
            # 半透明背景
            bg_rect = pygame.Rect(x - 10, y - 5, text_rect.width + 20, text_rect.height + 10)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 128))
            self.screen.blit(bg_surface, bg_rect)
            
            self.screen.blit(demo_text, (x, y))
            
        except Exception as e:
            self.logger.error(f"デモオーバーレイ描画エラー: {e}")
    
    def _render_error_overlay(self):
        """エラー復旧オーバーレイ"""
        try:
            font = pygame.font.Font(None, 28)
            error_text = font.render("ERROR RECOVERY MODE", True, (255, 100, 100))
            
            # 中央上部に表示
            text_rect = error_text.get_rect()
            x = (self.screen.get_width() - text_rect.width) // 2
            y = 50
            
            # 背景
            bg_rect = pygame.Rect(x - 15, y - 10, text_rect.width + 30, text_rect.height + 20)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((255, 0, 0, 100))
            self.screen.blit(bg_surface, bg_rect)
            
            self.screen.blit(error_text, (x, y))
            
        except Exception as e:
            self.logger.error(f"エラーオーバーレイ描画エラー: {e}")
    
    def _apply_quality_settings(self):
        """品質設定適用"""
        try:
            if not self.quality_manager:
                return
            
            settings = self.quality_manager.get_current_settings()
            self.logger.debug(f"品質設定適用: {settings}")
            
            # MediaPipe設定更新（次回初期化時に適用）
            if self.vision_processor:
                if hasattr(self.vision_processor, '_quality_settings'):
                    self.vision_processor._quality_settings = settings
            
            # シーンマネージャー設定更新
            if self.scene_manager and hasattr(self.scene_manager, 'apply_quality_settings'):
                self.scene_manager.apply_quality_settings(settings)
            
        except Exception as e:
            self.logger.error(f"品質設定適用エラー: {e}")
    
    def _print_debug_stats(self):
        """デバッグ統計表示"""
        try:
            if self.performance_monitor:
                metrics = self.performance_monitor.get_current_metrics()
                if metrics:
                    self.logger.info(f"📊 FPS: {metrics.fps:.1f} | "
                                   f"CPU: {metrics.cpu_usage:.1f}% | "
                                   f"メモリ: {metrics.memory_usage:.1f}% | "
                                   f"処理時間: {metrics.processing_time*1000:.1f}ms")
            
            # エラー統計
            if self.error_manager:
                error_stats = self.error_manager.get_error_statistics()
                if error_stats and error_stats.get('total_errors', 0) > 0:
                    self.logger.info(f"🚨 総エラー数: {error_stats['total_errors']}")
            
        except Exception as e:
            self.logger.error(f"デバッグ統計表示エラー: {e}")
    
    def _log_performance_stats(self):
        """パフォーマンス統計ログ"""
        try:
            stats = self.get_performance_stats()
            if stats:
                self.logger.debug(f"Performance: {stats}")
        except Exception as e:
            self.logger.error(f"パフォーマンス統計ログエラー: {e}")
    
    def _handle_basic_error(self, error):
        """基本エラーハンドリング（フォールバック）"""
        try:
            self.logger.error(f"基本エラーハンドリング: {error}")
            
            # デモモードに切り替え
            if not self.demo_mode:
                self.demo_mode = True
                self.demo_data = self._create_demo_data()
                self.logger.info("エラー復旧のためデモモードに切り替えました")
            
            # 状態をエラー復旧モードに
            self.current_state = AppState.ERROR_RECOVERY
            self.state_timer = time.time()
            
        except Exception as e:
            self.logger.error(f"基本エラーハンドリング失敗: {e}")
    
    def _restart_components(self):
        """コンポーネント再起動"""
        try:
            self.logger.info("🔄 コンポーネントを再起動しています...")
            
            # AI処理コンポーネント再初期化
            self._init_ai_components()
            
            # カメラ再初期化
            if not self.demo_mode and self.camera:
                self.camera.cleanup()
                self._init_camera()
            
            self.logger.info("✅ コンポーネント再起動完了")
            
        except Exception as e:
            self.logger.error(f"コンポーネント再起動エラー: {e}")
    
    def get_performance_stats(self):
        """パフォーマンス統計取得"""
        try:
            stats = {
                'fps': self.clock.get_fps(),
                'current_state': self.current_state.value,
                'demo_mode': self.demo_mode,
                'error_recovery_mode': self.current_state == AppState.ERROR_RECOVERY
            }
            
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                stats['fps_estimate'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                stats['avg_frame_time'] = avg_frame_time
            
            if self.performance_monitor:
                pm_stats = self.performance_monitor.get_current_metrics()
                if pm_stats:
                    stats.update(pm_stats.__dict__)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"パフォーマンス統計取得エラー: {e}")
            return {}
    
    def _cleanup(self):
        """統合クリーンアップ処理"""
        try:
            self.logger.info("🧹 アプリケーションをクリーンアップしています...")
            
            # コンポーネントクリーンアップ
            cleanup_components = [
                ('vision_processor', self.vision_processor),
                ('camera', self.camera),
                ('performance_monitor', self.performance_monitor),
                ('error_manager', self.error_manager)
            ]
            
            for name, component in cleanup_components:
                if component and hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                        self.logger.debug(f"✅ {name} クリーンアップ完了")
                    except Exception as e:
                        self.logger.error(f"❌ {name} クリーンアップエラー: {e}")
            
            # Pygame終了
            pygame.quit()
            
            self.logger.info("🌊 Aqua Mirror を終了しました")
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")

# CameraManager クラス（統合版）
class CameraManager:
    """統合版カメラマネージャー"""
    
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.logger = logging.getLogger(__name__)
        self.capture_active = False
        
    def initialize(self):
        """カメラ初期化"""
        try:
            import cv2
            
            camera_config = self.config.get('camera', {})
            device_id = camera_config.get('device_id', 0)
            
            self.camera = cv2.VideoCapture(device_id)
            
            if not self.camera.isOpened():
                return False
            
            # カメラ設定
            width = camera_config.get('width', 1920)
            height = camera_config.get('height', 1080)
            fps = camera_config.get('fps', 30)
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 追加最適化設定
            auto_settings = camera_config.get('auto_settings', {})
            if not auto_settings.get('autofocus', True):
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            self.logger.info(f"📹 カメラ初期化成功: {width}x{height}@{fps}fps")
            return True
            
        except ImportError:
            self.logger.error("OpenCV がインストールされていません")
            return False
        except Exception as e:
            self.logger.error(f"カメラ初期化エラー: {e}")
            return False
    
    def start_capture(self):
        """フレーム取得開始"""
        try:
            if self.camera and self.camera.isOpened():
                self.capture_active = True
                self.logger.info("📹 カメラキャプチャを開始しました")
                return True
            else:
                self.logger.warning("カメラが初期化されていません")
                return False
        except Exception as e:
            self.logger.error(f"カメラキャプチャ開始エラー: {e}")
            return False
    
    def stop_capture(self):
        """フレーム取得停止"""
        try:
            self.capture_active = False
            self.logger.info("📹 カメラキャプチャを停止しました")
        except Exception as e:
            self.logger.error(f"カメラキャプチャ停止エラー: {e}")
    
    def get_frame(self):
        """フレーム取得"""
        try:
            if not self.camera or not self.camera.isOpened() or not self.capture_active:
                return None
            
            ret, frame = self.camera.read()
            return frame if ret else None
            
        except Exception as e:
            self.logger.error(f"フレーム取得エラー: {e}")
            return None
    
    def cleanup(self):
        """リソース解放"""
        try:
            self.stop_capture()
            if self.camera:
                self.camera.release()
            self.logger.info("📹 カメラがクリーンアップされました")
        except Exception as e:
            self.logger.error(f"カメラクリーンアップエラー: {e}")

# メイン実行部分
if __name__ == "__main__":
    import argparse
    
    # コマンドライン引数解析
    parser = argparse.ArgumentParser(description='Aqua Mirror Interactive Art')
    parser.add_argument('--config', default='config/config.json', help='設定ファイルパス')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--demo', action='store_true', help='デモモード')
    
    args = parser.parse_args()
    
    try:
        # 設定読み込み
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
        
        # コマンドライン引数で設定を上書き
        if args.debug:
            config['debug_mode'] = True
        if args.demo:
            config['demo_mode'] = True
        
        # アプリケーション実行
        app = AquaMirrorApp(config)
        app.run()
        
    except Exception as e:
        print(f"❌ 起動エラー: {e}")
        sys.exit(1)
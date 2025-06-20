# src/app.py
import pygame
import time
import sys
import logging
from enum import Enum
from typing import Dict, Any, Optional
from collections import deque

# コンポーネントのインポート
from core.config_loader import ConfigLoader
from vision.vision_processor import VisionProcessor
from emotion.emotion_analyzer import EmotionAnalyzer, Emotion
from scene import SceneManager

class AppState(Enum):
    """アプリケーション状態の定義"""
    STANDBY = "standby"
    RECOGNITION = "recognition"
    INTERACTION = "interaction"
    EXPERIENCE_END = "experience_end"
    ERROR = "error"

class AquaMirrorApp:
    """メインアプリケーションクラス（エラー修正版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
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
        
        # コンポーネント参照
        self.vision_processor = None
        self.emotion_analyzer = None
        self.scene_manager = None
        self.camera = None
        
        # デモ用データ
        self.demo_data = self._create_demo_data() if self.demo_mode else None
        self.demo_index = 0
        
        # 初期化
        self._initialize_components()
        
        self.logger.info("Aqua Mirror アプリケーションが初期化されました")
    
    def _initialize_components(self):
        """コンポーネント初期化"""
        try:
            # Pygame初期化
            self._init_pygame()
            
            # カメラ初期化（デモモードでない場合）
            if not self.demo_mode:
                self._init_camera()
            
            # AI処理コンポーネント初期化
            try:
                self.vision_processor = VisionProcessor(self.config)
                self.logger.info("VisionProcessor初期化完了")
            except Exception as e:
                self.logger.error(f"VisionProcessor初期化エラー: {e}")
                self.vision_processor = None
            
            try:
                self.emotion_analyzer = EmotionAnalyzer()
                self.logger.info("EmotionAnalyzer初期化完了")
            except Exception as e:
                self.logger.error(f"EmotionAnalyzer初期化エラー: {e}")
                self.emotion_analyzer = None
            
            # シーンマネージャー初期化
            try:
                display_config = self.config.get('display', {})
                self.scene_manager = SceneManager(
                    display_config.get('width', 1920),
                    display_config.get('height', 1080),
                    self.config
                )
                self.logger.info("SceneManager初期化完了")
            except Exception as e:
                self.logger.error(f"SceneManager初期化エラー: {e}")
                self.scene_manager = None
            
            self.logger.info("コンポーネント初期化が完了しました")
            
        except Exception as e:
            self.logger.error(f"コンポーネント初期化エラー: {e}")
            raise
    
    def _init_pygame(self):
        """Pygame初期化"""
        try:
            pygame.init()
            pygame.mixer.init()
            
            display_config = self.config.get('display', {})
            width = display_config.get('width', 1920)
            height = display_config.get('height', 1080)
            fullscreen = display_config.get('fullscreen', False)
            
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE
            if fullscreen:
                flags |= pygame.FULLSCREEN
            
            self.screen = pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption("Aqua Mirror - Interactive Art Experience")
            
            # フォント初期化
            pygame.font.init()
            
            self.logger.info(f"Pygame初期化完了: {width}x{height}, fullscreen={fullscreen}")
            
        except Exception as e:
            self.logger.error(f"Pygame初期化エラー: {e}")
            raise
    
    def _init_camera(self):
        """カメラ初期化"""
        try:
            import cv2
            
            # カメラマネージャー作成（configオブジェクトを渡す）
            self.camera = CameraManager(self.config)
            
            if not self.camera.initialize():
                raise RuntimeError("カメラの初期化に失敗しました")
            
            self.camera.start_capture()
            self.logger.info("カメラが初期化されました")
            
        except ImportError:
            self.logger.error("CameraManagerのインポートに失敗しました。内蔵版を使用します")
            self.camera = CameraManager(self.config)
            if not self.camera.initialize():
                self.logger.warning("デモモードに切り替えます")
                self.demo_mode = True
                self.demo_data = self._create_demo_data()
        except Exception as e:
            self.logger.error(f"カメラ初期化エラー: {e}")
            self.logger.warning("デモモードに切り替えます")
            self.demo_mode = True
            self.demo_data = self._create_demo_data()
    
    def _create_demo_data(self) -> list:
        """デモ用データ作成"""
        return [
            {
                'face_detected': True,
                'face_center': (0.5, 0.4, 0.5),
                'hands_detected': True,
                'hand_positions': [(0.3, 0.6), (0.7, 0.6)],
                'emotion': Emotion.HAPPY,
                'duration': 3.0
            },
            {
                'face_detected': True,
                'face_center': (0.6, 0.3, 0.3),
                'hands_detected': False,
                'hand_positions': [],
                'emotion': Emotion.SURPRISED,
                'duration': 2.5
            },
            {
                'face_detected': True,
                'face_center': (0.4, 0.5, 0.7),
                'hands_detected': True,
                'hand_positions': [(0.5, 0.5)],
                'emotion': Emotion.SAD,
                'duration': 2.0
            },
            {
                'face_detected': False,
                'face_center': None,
                'hands_detected': False,
                'hand_positions': [],
                'emotion': Emotion.NEUTRAL,
                'duration': 1.5
            }
        ]
    
    def run(self):
        """メインループ実行"""
        self.logger.info("メインループを開始します")
        
        try:
            while self.running:
                frame_start = time.time()
                
                # イベント処理
                self._handle_events()
                
                # 状態更新
                self._update()
                
                # 描画
                self._render()
                
                # パフォーマンス管理
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time)
                
                # FPS制御
                self.clock.tick(self.target_fps)
                
        except Exception as e:
            self.logger.error(f"メインループエラー: {e}")
            self.logger.exception("詳細なエラー情報:")
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
        elif key == pygame.K_F1:
            self.debug_mode = not self.debug_mode
            self.logger.info(f"デバッグモード: {'ON' if self.debug_mode else 'OFF'}")
        elif key == pygame.K_F2:
            self.demo_mode = not self.demo_mode
            self.logger.info(f"デモモード: {'ON' if self.demo_mode else 'OFF'}")
        elif key == pygame.K_F3:
            # シーンエフェクトクリア
            if self.scene_manager:
                self.scene_manager.clear_effects()
                self.logger.info("エフェクトをクリアしました")
        elif key == pygame.K_SPACE:
            # 状態リセット
            self.current_state = AppState.STANDBY
            self.logger.info("状態をリセットしました")
    
    def _update(self):
        """状態更新"""
        try:
            current_time = time.time()
            
            # 検出データ取得
            if self.demo_mode:
                detection_result = self._get_demo_detection_result()
            else:
                detection_result = self._get_real_detection_result()
            
            # 感情分析
            current_emotion = Emotion.NEUTRAL
            emotion_confidence = 0.0
            
            if (detection_result.get('face_detected') and 
                detection_result.get('face_landmarks') and 
                self.emotion_analyzer is not None):
                try:
                    current_emotion, emotion_confidence = self.emotion_analyzer.analyze_emotion(
                        detection_result['face_landmarks']
                    )
                except Exception as e:
                    self.logger.error(f"感情分析エラー: {e}")
                    current_emotion = Emotion.NEUTRAL
                    emotion_confidence = 0.0
            
            # 状態遷移の管理
            self._update_state(detection_result, current_emotion, current_time)
            
            # シーンマネージャーに検出結果を渡す
            if self.scene_manager:
                self.scene_manager.update(detection_result, self.current_state)
            
        except Exception as e:
            self.logger.error(f"状態更新エラー: {e}")
    
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
            if self.vision_processor is not None:
                return self.vision_processor.process_frame(frame)
            else:
                self.logger.warning("VisionProcessorが利用できません")
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
                'face_landmarks': {'multi_face_landmarks': []} if current_demo['face_detected'] else None,
                'hands_detected': current_demo['hands_detected'],
                'hand_positions': current_demo['hand_positions'],
                'hand_gestures': [],
                'timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error(f"デモ検出エラー: {e}")
            return {}
    
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
                    # 顔が見つからなくなった場合、少し待ってからSTANDBYに戻る
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
                # 3秒後にSTANDBYに戻る
                if current_time - self.state_timer > 3.0:
                    self.current_state = AppState.STANDBY
                    self.logger.debug("状態変更: EXPERIENCE_END -> STANDBY")
            
        except Exception as e:
            self.logger.error(f"状態遷移エラー: {e}")
    
    def _render(self):
        """描画処理"""
        try:
            # 背景クリア
            self.screen.fill((0, 0, 0))
            
            # シーン描画
            if self.scene_manager:
                self.scene_manager.draw(self.screen)
            
            # デバッグ情報の描画
            if self.debug_mode:
                self._render_debug_info()
            
            # デモモード表示
            if self.demo_mode:
                self._render_demo_overlay()
            
            # 画面更新
            pygame.display.flip()
            
        except Exception as e:
            self.logger.error(f"描画エラー: {e}")
    
    def _render_debug_info(self):
        """デバッグ情報の描画"""
        try:
            font = pygame.font.Font(None, 24)
            y_offset = 10
            
            # FPS表示
            fps = self.clock.get_fps()
            fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
            self.screen.blit(fps_text, (10, y_offset))
            y_offset += 25
            
            # 現在の状態表示
            state_text = font.render(f"State: {self.current_state.value}", True, (255, 255, 255))
            self.screen.blit(state_text, (10, y_offset))
            y_offset += 25
            
            # 検出情報表示
            if self.vision_processor is not None:
                debug_info = self.vision_processor.get_debug_info()
                for key, value in debug_info.items():
                    info_text = font.render(f"{key}: {value}", True, (255, 255, 255))
                    self.screen.blit(info_text, (10, y_offset))
                    y_offset += 25
            else:
                no_vision_text = font.render("VisionProcessor: Not Available", True, (255, 255, 0))
                self.screen.blit(no_vision_text, (10, y_offset))
                y_offset += 25
            
            # エフェクト数表示
            if self.scene_manager:
                effect_count = self.scene_manager.get_effect_count()
                effect_text = font.render(f"Effects: {effect_count['total']}", True, (255, 255, 255))
                self.screen.blit(effect_text, (10, y_offset))
                y_offset += 25
            
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
            
            # 背景
            bg_rect = pygame.Rect(x - 10, y - 5, text_rect.width + 20, text_rect.height + 10)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            
            self.screen.blit(demo_text, (x, y))
            
        except Exception as e:
            self.logger.error(f"デモオーバーレイ描画エラー: {e}")
    
    def _cleanup(self):
        """クリーンアップ処理"""
        try:
            self.logger.info("アプリケーションをクリーンアップしています...")
            
            # コンポーネントのクリーンアップ
            if self.vision_processor:
                self.vision_processor.cleanup()
            
            if self.camera:
                self.camera.cleanup()
            
            # Pygame終了
            pygame.quit()
            
            self.logger.info("クリーンアップが完了しました")
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
    
    def get_performance_stats(self):
        """パフォーマンス統計取得"""
        try:
            if not self.frame_times:
                return {}
            
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps_estimate = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            return {
                'fps': self.clock.get_fps(),
                'fps_estimate': fps_estimate,
                'avg_frame_time': avg_frame_time,
                'current_state': self.current_state.value,
                'demo_mode': self.demo_mode
            }
            
        except Exception as e:
            self.logger.error(f"パフォーマンス統計エラー: {e}")
            return {}

# CameraManager クラス（app.py内で定義）
class CameraManager:
    """簡易カメラマネージャー"""
    
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
            
            self.logger.info(f"カメラ初期化成功: {width}x{height}@{fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"カメラ初期化エラー: {e}")
            return False
    
    def start_capture(self):
        """フレーム取得開始"""
        try:
            if self.camera and self.camera.isOpened():
                self.capture_active = True
                self.logger.info("カメラキャプチャを開始しました")
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
            self.logger.info("カメラキャプチャを停止しました")
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
            self.logger.info("カメラがクリーンアップされました")
        except Exception as e:
            self.logger.error(f"カメラクリーンアップエラー: {e}")
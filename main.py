"""
Aqua Mirror Week 2 - 高度統合アプリケーション
Windows Native 高品質版
"""

import sys
import logging
import time
import numpy as np
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent))

from src.core.config_loader import ConfigLoader
from src.core.moderngl_app import ModernGLApp
from src.vision.camera_manager import CameraManager
from src.emotion.advanced_emotion_engine import AdvancedEmotionEngine
from src.gesture.gesture_recognizer import GestureRecognizer
from src.effects.advanced_water_simulation import AdvancedWaterSimulation
from src.audio.advanced_audio_engine import AdvancedAudioEngine
from src.interaction.basic_interaction_manager import BasicInteractionManager

class AquaMirrorAdvancedApp:
    """Aqua Mirror Week 2 高度統合アプリケーション"""
    
    def __init__(self):
        self.logger = logging.getLogger("AquaMirrorAdvancedApp")
        
        # 設定読み込み
        self.config = ConfigLoader()
        
        
        # コンポーネント
        self.moderngl_app = None
        self.camera_manager = None
        self.emotion_engine = None
        self.gesture_recognizer = None
        self.water_simulation = None
        self.audio_engine = None
        self.interaction_manager = None
        
        # カメラテクスチャ
        self.camera_texture = None
        
        # 状態管理
        self.running = False
        
        # パフォーマンス統計
        self.performance_stats = {
            'frame_count': 0,
            'last_fps_time': time.time(),
            'current_fps': 0.0,
            'avg_frame_time': 0.0,
            'gpu_memory_usage': 0.0
        }
        
        # 品質管理
        self.quality_level = 3  # 1-5
        self.last_quality_adjustment = time.time()
        
        self.logger.info("🌊 Aqua Mirror Week 2 高度統合アプリケーション初期化")
    
    def initialize(self) -> bool:
        """アプリケーション初期化"""
        try:
            # ModernGL基盤初期化
            self.moderngl_app = ModernGLApp(self.config)
            
            if not self.moderngl_app.initialize():
                self.logger.error("❌ ModernGL初期化失敗")
                return False
            
            # カメラテクスチャ作成
            self._create_camera_texture()
            
           # ModernGLAppがすでに初期化したカメラマネージャーを取得する
            self.camera_manager = self.moderngl_app.camera
            if self.camera_manager and self.camera_manager.is_available:
                self.logger.info("✅ 既存のカメラマネージャーの参照を取得完了")
            else:
                self.logger.warning("⚠️ カメラが利用できません - デモモードで継続")
            
            # AI エンジン初期化
            if self.config.get('week2.enable_advanced_emotion', True):
                self.emotion_engine = AdvancedEmotionEngine(self.config.get_all())
            
            if self.config.get('week2.enable_gesture_recognition', True):
                self.gesture_recognizer = GestureRecognizer(self.config.get_all())
            
            # 高品質水面シミュレーション初期化
            if self.config.get('week2.enable_water_physics', True):
                self.water_simulation = AdvancedWaterSimulation(
                    self.moderngl_app.ctx, 
                    self.config.get_all()
                )
                if not self.water_simulation.initialize():
                    self.logger.warning("⚠️ 高品質水面シミュレーション初期化失敗")
                    self.water_simulation = None
            
            # 高品質音響システム初期化
            if self.config.get('week2.enable_audio_engine', True):
                self.audio_engine = AdvancedAudioEngine(self.config.get_all())
                if self.audio_engine.initialize():
                    self.audio_engine.start()
                else:
                    self.logger.warning("⚠️ 音響システム初期化失敗")
                    self.audio_engine = None
            
            # インタラクション管理初期化
            if self.config.get('week2.enable_interaction_manager', True):
                self.interaction_manager = BasicInteractionManager(self.config.get_all())
            
            self.logger.info("✅ Aqua Mirror 高度統合アプリケーション初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初期化失敗: {e}")
            return False
    
    def _create_camera_texture(self):
        """カメラテクスチャ作成"""
        try:
            width = self.config.get('camera.width', 1920)
            height = self.config.get('camera.height', 1080)
            
            # 空のテクスチャ作成
            empty_data = np.zeros((height, width, 3), dtype=np.uint8)
            
            self.camera_texture = self.moderngl_app.ctx.texture(
                (width, height), 3, empty_data.tobytes()
            )
            self.camera_texture.filter = (self.moderngl_app.ctx.LINEAR, self.moderngl_app.ctx.LINEAR)
            
            self.logger.info(f"✅ カメラテクスチャ作成: {width}x{height}")
            
        except Exception as e:
            self.logger.error(f"❌ カメラテクスチャ作成失敗: {e}")
    
    def run(self):
        """メインループ実行"""
        try:
            self.running = True
            self.logger.info("🚀 Aqua Mirror Week 2 Advanced 開始")
            
            frame_times = []
            # MediaPipeの処理担当オブジェクトを取得
            mediapipe_processor = self.moderngl_app.mediapipe_processor
            
            while self.running and not self.moderngl_app.should_close():
                frame_start = time.time()
                
                # カメラフレーム取得
                frame = None
                if self.camera_manager:
                    frame = self.camera_manager.get_frame()
                    if frame is not None:
                        self._update_camera_texture(frame)
                
                # AI処理
                emotion_result_dict = None
                gesture_results = []
                
                if frame is not None and mediapipe_processor:
                    # 1. MediaPipeで顔と手をまとめて処理し、結果を辞書で受け取る
                    mp_results = mediapipe_processor.process_frame(frame)
                    
                    # 2. 顔が検出されたかチェックし、ランドマークを取得
                    if mp_results.get('face_detected') and self.emotion_engine:
                        face_landmarks = mp_results['face_landmarks']
                        
                        # 3. 検出したランドマークから詳細な特徴量を抽出
                        features = self.emotion_engine.extract_advanced_features(face_landmarks, frame.shape)
                        
                        # 4. 抽出した特徴量を基に感情を分析
                        emotion_result_dict = self.emotion_engine.analyze_emotion_advanced(features)
                    
                    # 5. 手が検出された場合、ジェスチャー認識システムに渡す
                    if mp_results.get('hands_detected') and self.gesture_recognizer:
                        # process_frameの結果からジェスチャー情報を取得するか、
                        # もしくはGestureRecognizerにmp_results['hands']を渡すなど、
                        # GestureRecognizerの仕様に合わせて調整が必要な場合があります。
                        # ここでは、GestureRecognizerがフレーム全体を必要とすると仮定します。
                        gesture_results = self.gesture_recognizer.recognize(frame)
                
                # インタラクション処理
                interaction_data = {}
                if self.interaction_manager:
                    interaction_data = self.interaction_manager.process_interaction(
                        emotion_result_dict, gesture_results
                    )
                
                # 水面シミュレーション更新
                if self.water_simulation:
                    # ジェスチャーから波源追加
                    effects = interaction_data.get('effects', {})
                    for wave_source in effects.get('wave_sources', []):
                        self.water_simulation.add_wave_source(
                            wave_source['position'],
                            wave_source['intensity']
                        )
                    
                    # 物理更新
                    self.water_simulation.update_physics(0.033)  # 30FPS想定
                
                # 音響更新
                if self.audio_engine:
                    self.audio_engine.update_emotion_audio(emotion_result_dict)
                    if self.water_simulation:
                        self.audio_engine.update_water_audio(self.water_simulation.wave_sources)
                
                # 描画処理
                self._render_advanced_frame(emotion_result_dict, interaction_data)
                
                # パフォーマンス統計更新
                frame_time = time.time() - frame_start
                self._update_performance_stats(frame_time)
                frame_times.append(frame_time)
                
                # 品質動的調整
                if len(frame_times) >= 30:  # 1秒分
                    self._adjust_quality_if_needed(frame_times)
                    frame_times.clear()
                
                # デバッグ情報表示（2秒に1回）
                if self.performance_stats['frame_count'] % 60 == 0:
                    self._display_advanced_debug_info(emotion_result_dict, gesture_results, interaction_data)
                
                # バッファスワップ
                self.moderngl_app.swap_buffers()
                self.moderngl_app.update_fps()
                
                # フレーム制限
                target_frame_time = 1.0 / self.config.get('display.target_fps', 30)
                
                if frame_time < target_frame_time:
                    time.sleep(target_frame_time - frame_time)
            
        except KeyboardInterrupt:
            self.logger.info("🛑 ユーザー中断")
        except Exception as e:
            self.logger.error(f"❌ 実行エラー: {e}")
        finally:
            self.cleanup()
    
    def _update_camera_texture(self, frame):
        """カメラテクスチャ更新"""
        try:
            if self.camera_texture and frame is not None:
                # OpenCV BGR → RGB変換
                rgb_frame = frame[:, :, ::-1]
                self.camera_texture.write(rgb_frame.tobytes())
        except Exception as e:
            self.logger.error(f"❌ カメラテクスチャ更新失敗: {e}")
    
    def _render_advanced_frame(self, emotion_result, interaction_data):
        """高度フレーム描画"""
        try:
            # 画面クリア（深い青）
            self.moderngl_app.ctx.clear(0.02, 0.05, 0.1, 1.0)
            
            # MVP行列生成（簡易版）
            mvp_matrix = np.eye(4, dtype=np.float32)
            
            # 感情色計算
            emotion_color = (0.5, 0.5, 0.5)
            emotion_intensity = 0.0
            
            if emotion_result:
                emotion_colors = {
                    'HAPPY': (1.0, 0.8, 0.0),
                    'SAD': (0.3, 0.5, 0.8),
                    'ANGRY': (0.9, 0.1, 0.1),
                    'SURPRISED': (1.0, 0.1, 0.6),
                    'FEAR': (0.4, 0.2, 0.6),
                    'DISGUST': (0.2, 0.7, 0.2),
                    'NEUTRAL': (0.5, 0.5, 0.5)
                }
                emotion_color = emotion_colors.get(emotion_result.emotion, (0.5, 0.5, 0.5))
                emotion_intensity = emotion_result.confidence
            
            # 高品質水面描画
            if self.water_simulation and self.camera_texture:
                self.water_simulation.render(
                    self.camera_texture, 
                    mvp_matrix,
                    emotion_color,
                    emotion_intensity
                )
            
        except Exception as e:
            self.logger.error(f"❌ 高度フレーム描画失敗: {e}")
    
    def _update_performance_stats(self, frame_time: float):
        """パフォーマンス統計更新"""
        self.performance_stats['frame_count'] += 1
        current_time = time.time()
        
        # FPS計算
        if current_time - self.performance_stats['last_fps_time'] >= 1.0:
            self.performance_stats['current_fps'] = (
                self.performance_stats['frame_count'] / 
                (current_time - self.performance_stats['last_fps_time'])
            )
            self.performance_stats['frame_count'] = 0
            self.performance_stats['last_fps_time'] = current_time
        
        # 平均フレーム時間
        self.performance_stats['avg_frame_time'] = frame_time
    
    def _adjust_quality_if_needed(self, frame_times: list):
        """品質動的調整"""
        try:
            avg_frame_time = np.mean(frame_times)
            target_frame_time = 1.0 / 30.0  # 30FPS目標
            
            current_time = time.time()
            
            # 5秒に1回のみ調整
            if current_time - self.last_quality_adjustment < 5.0:
                return
            
            if avg_frame_time > target_frame_time * 1.2:  # 25FPS未満
                if self.quality_level > 1:
                    self.quality_level -= 1
                    self.last_quality_adjustment = current_time
                    self.logger.info(f"🔽 品質レベル下げ: {self.quality_level}")
                    
            elif avg_frame_time < target_frame_time * 0.8:  # 37.5FPS超過
                if self.quality_level < 5:
                    self.quality_level += 1
                    self.last_quality_adjustment = current_time
                    self.logger.info(f"🔼 品質レベル上げ: {self.quality_level}")
                    
        except Exception as e:
            self.logger.error(f"❌ 品質調整失敗: {e}")
    
    def _display_advanced_debug_info(self, emotion_result, gesture_results, interaction_data):
        """高度デバッグ情報表示"""
        info_lines = []
        info_lines.append(f"🌊 Aqua Mirror Week 2 Advanced - FPS: {self.performance_stats['current_fps']:.1f}")
        info_lines.append(f"⚡ フレーム時間: {self.performance_stats['avg_frame_time']*1000:.1f}ms, 品質: {self.quality_level}/5")
        
        # 感情情報
        if emotion_result:
            info_lines.append(f"😊 感情: {emotion_result.emotion} ({emotion_result.confidence:.2f})")
            if hasattr(emotion_result, 'features'):
                features = emotion_result.features
                info_lines.append(f"   特徴: 口開き{features.get('mouth_open', 0):.2f}, 口角{features.get('mouth_curve', 0):.2f}")
        else:
            info_lines.append("😊 感情: 未検出")
        
        # ジェスチャー情報
        if gesture_results:
            gesture_info = ", ".join([f"{g.type.value}({g.confidence:.2f})" for g in gesture_results])
            info_lines.append(f"👋 ジェスチャー: {gesture_info}")
        else:
            info_lines.append("👋 ジェスチャー: 未検出")
        
        # インタラクション情報
        if interaction_data and 'state' in interaction_data:
            state = interaction_data['state']
            info_lines.append(f"🎭 モード: {state.mode.value}, エネルギー: {state.energy_level:.2f}")
            info_lines.append(f"📊 エンゲージメント: {state.user_engagement:.2f}")
        
        # システム情報
        if self.camera_manager:
            info_lines.append(f"📹 カメラFPS: {self.camera_manager.actual_fps:.1f}")
        
        if self.water_simulation:
            stats = self.water_simulation.get_statistics()
            info_lines.append(f"🌊 水面: {stats['resolution']}, 波源: {stats['active_wave_sources']}")
        
        if self.audio_engine:
            audio_stats = self.audio_engine.get_audio_statistics()
            info_lines.append(f"🔊 音響: {audio_stats['current_emotion']}, 強度: {audio_stats['intensity']:.2f}")
        
        # 統計情報
        if interaction_data and 'statistics' in interaction_data:
            stats = interaction_data['statistics']
            info_lines.append(f"📈 インタラクション: {stats.get('interaction_count', 0)}回")
        
        # 情報出力
        print("\n".join(info_lines))
        print("=" * 100)
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("🧹 Aqua Mirror 高度統合アプリケーション終了処理中...")
        
        try:
            if self.audio_engine:
                self.audio_engine.cleanup()
            
            if self.water_simulation:
                self.water_simulation.cleanup()
            
            if self.interaction_manager:
                self.interaction_manager.cleanup()
            
            if self.camera_manager:
                self.camera_manager.cleanup()
            
            if self.emotion_engine:
                self.emotion_engine.cleanup()
            
            if self.gesture_recognizer:
                self.gesture_recognizer.cleanup()
            
            if self.moderngl_app:
                self.moderngl_app.cleanup()
            
            self.logger.info("✅ Aqua Mirror 高度統合アプリケーション終了完了")
            
        except Exception as e:
            self.logger.error(f"❌ 終了処理エラー: {e}")

def main():
    """メイン関数"""
    print("🌊 Aqua Mirror Week 2 Advanced - Starting...")
    
    # ログディレクトリ作成
    Path("logs").mkdir(exist_ok=True)
    
    # アプリケーション実行
    app = AquaMirrorAdvancedApp()
    
    if app.initialize():
        print("✅ 高度統合初期化完了 - アプリケーション開始")
        app.run()
    else:
        print("❌ アプリケーション初期化失敗")
        sys.exit(1)

if __name__ == "__main__":
    main()

# src/vision.py - 統合完全版（MediaPipe 0.10.x + GPU最適化）
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple
from enum import Enum

class VisionBackend(Enum):
    """Vision処理バックエンドタイプ"""
    MEDIAPIPE_V10 = "mediapipe_v10"        # MediaPipe 0.10.x新プロセッサー
    GPU_OPTIMIZED = "gpu_optimized"        # GPU最適化旧プロセッサー
    CPU_STANDARD = "cpu_standard"          # 標準CPU処理
    FALLBACK = "fallback"                  # 最低限フォールバック

class VisionProcessor:
    """
    統合コンピュータビジョン処理クラス
    
    処理優先順位:
    1. MediaPipe 0.10.x 新プロセッサー（最新・最高性能）
    2. GPU最適化プロセッサー（高性能・安定）
    3. CPU標準プロセッサー（標準・互換）
    4. フォールバックモード（最低限動作保証）
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # バックエンド管理
        self.available_backends = []
        self.current_backend = VisionBackend.FALLBACK
        self.processor = None
        
        # GPU処理管理
        self.gpu_processor = None
        self.use_gpu = False
        
        # フォールバック用
        self.camera = None
        self.face_mesh = None
        self.hands = None
        
        # 品質設定
        self._quality_settings: Dict[str, Any] = {}
        
        # 共通結果フォーマット
        self.last_detection_result = {
            'face_detected': False,
            'hands_detected': False,
            'face_center': None,
            'face_distance': float('inf'),
            'hand_positions': [],
            'frame_shape': (480, 640, 3),
            'processing_backend': 'unknown',
            'processing_time': 0.0
        }
        
        # 統計情報
        self.processing_stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'backend_switches': 0,
            'avg_processing_time': 0.0
        }
        
        # 初期化実行
        self._initialize_backends()
        self._select_optimal_backend()
        self._log_initialization_status()
    
    def _initialize_backends(self):
        """利用可能なバックエンドを段階的に初期化"""
        
        # 1. MediaPipe 0.10.x 新プロセッサー初期化試行
        self._init_mediapipe_v10()
        
        # 2. GPU最適化プロセッサー初期化試行
        self._init_gpu_optimized()
        
        # 3. CPU標準プロセッサー初期化試行
        self._init_cpu_standard()
        
        # 4. フォールバックは常に利用可能
        self.available_backends.append(VisionBackend.FALLBACK)
    
    def _init_mediapipe_v10(self):
        """MediaPipe 0.10.x 新プロセッサー初期化"""
        try:
            from vision.vision_processor_v10 import VisionProcessorV10
            
            # 新プロセッサー初期化テスト
            test_processor = VisionProcessorV10(self.config)
            
            # 基本動作テスト
            if self._test_processor(test_processor):
                self.available_backends.append(VisionBackend.MEDIAPIPE_V10)
                self.logger.info("✅ MediaPipe 0.10.x プロセッサー利用可能")
                # テスト用プロセッサーをクリーンアップ
                if hasattr(test_processor, 'cleanup'):
                    test_processor.cleanup()
            else:
                self.logger.warning("❌ MediaPipe 0.10.x プロセッサーテスト失敗")
                
        except ImportError as e:
            self.logger.warning(f"MediaPipe 0.10.x プロセッサーが見つかりません: {e}")
        except Exception as e:
            self.logger.warning(f"MediaPipe 0.10.x プロセッサー初期化エラー: {e}")
    
    def _init_gpu_optimized(self):
        """GPU最適化プロセッサー初期化"""
        try:
            from core.gpu_processor import GPUProcessor
            
            # GPU プロセッサー初期化
            self.gpu_processor = GPUProcessor()
            self.use_gpu = self.gpu_processor.is_gpu_available()
            
            if self.use_gpu:
                # MediaPipe初期化（GPU最適化版）
                if self._init_mediapipe_components():
                    self.available_backends.append(VisionBackend.GPU_OPTIMIZED)
                    self.logger.info("✅ GPU最適化プロセッサー利用可能")
                else:
                    self.use_gpu = False
                    self.logger.warning("GPU最適化プロセッサー初期化失敗")
            else:
                self.logger.info("GPU処理無効、CPU最適化プロセッサーを準備")
                
        except ImportError as e:
            self.logger.warning(f"GPU プロセッサーが見つかりません: {e}")
        except Exception as e:
            self.logger.warning(f"GPU最適化プロセッサー初期化エラー: {e}")
    
    def _init_cpu_standard(self):
        """CPU標準プロセッサー初期化"""
        try:
            # 標準MediaPipe初期化
            if self._init_mediapipe_components():
                self.available_backends.append(VisionBackend.CPU_STANDARD)
                self.logger.info("✅ CPU標準プロセッサー利用可能")
            else:
                self.logger.warning("CPU標準プロセッサー初期化失敗")
                
        except Exception as e:
            self.logger.warning(f"CPU標準プロセッサー初期化エラー: {e}")
    
    def _init_mediapipe_components(self) -> bool:
        """MediaPipe コンポーネント初期化"""
        try:
            import mediapipe as mp
            
            # MediaPipe初期化
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            # 設定取得
            detection_config = self.config.get('detection', {})
            
            # Face Mesh初期化
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=detection_config.get('max_num_faces', 1),
                refine_landmarks=detection_config.get('refine_landmarks', True),
                min_detection_confidence=detection_config.get('face_detection_confidence', 0.7)
            )
            
            # Hands初期化
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=detection_config.get('max_num_hands', 2),
                min_detection_confidence=detection_config.get('hand_detection_confidence', 0.7)
            )
            
            # カメラ初期化
            self._init_camera()
            
            return True
            
        except Exception as e:
            self.logger.error(f"MediaPipe コンポーネント初期化エラー: {e}")
            return False
    
    def _init_camera(self):
        """カメラ初期化"""
        try:
            camera_config = self.config.get('camera', {})
            device_id = camera_config.get('device_id', 0)
            
            self.camera = cv2.VideoCapture(device_id)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"カメラ {device_id} を開けません")
            
            # カメラ設定
            width = camera_config.get('width', 1920)
            height = camera_config.get('height', 1080)
            fps = camera_config.get('fps', 30)
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            
            self.logger.info(f"カメラ初期化成功: {width}x{height}@{fps}fps")
            
        except Exception as e:
            self.logger.error(f"カメラ初期化エラー: {e}")
            raise
    
    def _test_processor(self, processor) -> bool:
        """プロセッサーの基本動作テスト"""
        try:
            # process_frame メソッド存在確認
            if hasattr(processor, 'process_frame'):
                return True
            else:
                return False
        except Exception as e:
            self.logger.warning(f"プロセッサーテストエラー: {e}")
            return False
    
    def _select_optimal_backend(self):
        """最適なバックエンドを選択"""
        # 優先順位に従ってバックエンド選択
        priority = [
            VisionBackend.MEDIAPIPE_V10,
            VisionBackend.GPU_OPTIMIZED,
            VisionBackend.CPU_STANDARD,
            VisionBackend.FALLBACK
        ]
        
        for backend in priority:
            if backend in self.available_backends:
                self.current_backend = backend
                break
        
        # 選択されたバックエンドに応じてプロセッサー初期化
        self._initialize_current_processor()
    
    def _initialize_current_processor(self):
        """現在のバックエンドに応じたプロセッサー初期化"""
        try:
            if self.current_backend == VisionBackend.MEDIAPIPE_V10:
                from vision.vision_processor_v10 import VisionProcessorV10
                self.processor = VisionProcessorV10(self.config)
                
            elif self.current_backend == VisionBackend.GPU_OPTIMIZED:
                self.processor = None  # 自分自身が処理
                
            elif self.current_backend == VisionBackend.CPU_STANDARD:
                self.processor = None  # 自分自身が処理
                
            else:  # FALLBACK
                self.processor = None
                self._init_fallback()
                
        except Exception as e:
            self.logger.error(f"プロセッサー初期化エラー ({self.current_backend.value}): {e}")
            self._fallback_to_next_backend()
    
    def _init_fallback(self):
        """フォールバック初期化"""
        try:
            if not self.camera:
                camera_config = self.config.get('camera', {})
                device_id = camera_config.get('device_id', 0)
                self.camera = cv2.VideoCapture(device_id)
            
            self.last_detection_result.update({
                'face_detected': False,
                'hands_detected': False,
                'face_center': None,
                'face_distance': float('inf'),
                'hand_positions': [],
                'frame_shape': (480, 640, 3),
                'processing_backend': 'fallback'
            })
            
        except Exception as e:
            self.logger.error(f"フォールバック初期化エラー: {e}")
    
    def _log_initialization_status(self):
        """初期化状況ログ出力"""
        self.logger.info(f"🔧 利用可能バックエンド: {[b.value for b in self.available_backends]}")
        self.logger.info(f"🎯 選択されたバックエンド: {self.current_backend.value}")
        
        if self.use_gpu:
            self.logger.info("⚡ GPU加速: 有効")
        else:
            self.logger.info("🔧 GPU加速: 無効（CPU処理）")
        
        if self.current_backend == VisionBackend.FALLBACK:
            self.logger.info("💡 最適な性能を得るには:")
            self.logger.info("   1. MediaPipe 0.10.x対応プロセッサーのインストール")
            self.logger.info("   2. GPU最適化環境の構築")
    
    def process_frame(self) -> Dict[str, Any]:
        """統合フレーム処理（既存API完全互換）"""
        start_time = time.time()
        
        try:
            # バックエンド別処理
            if self.current_backend == VisionBackend.MEDIAPIPE_V10:
                result = self._process_with_v10()
            elif self.current_backend == VisionBackend.GPU_OPTIMIZED:
                result = self._process_with_gpu_optimization()
            elif self.current_backend == VisionBackend.CPU_STANDARD:
                result = self._process_with_cpu_standard()
            else:  # FALLBACK
                result = self._process_with_fallback()
            
            # 処理時間記録
            processing_time = time.time() - start_time
            result['processing_time'] = processing_time
            result['processing_backend'] = self.current_backend.value
            
            # 統計更新
            self._update_stats(processing_time, True)
            
            self.last_detection_result = result
            return result
            
        except Exception as e:
            self.logger.warning(f"フレーム処理エラー ({self.current_backend.value}): {e}")
            return self._handle_processing_error(e)
    
    def _process_with_v10(self) -> Dict[str, Any]:
        """MediaPipe 0.10.x プロセッサーでの処理"""
        if self.processor:
            return self.processor.process_frame()
        else:
            raise RuntimeError("MediaPipe 0.10.x プロセッサーが初期化されていません")
    
    def _process_with_gpu_optimization(self) -> Dict[str, Any]:
        """GPU最適化処理"""
        if not self.camera or not self.camera.isOpened():
            return self.last_detection_result
        
        ret, frame = self.camera.read()
        if not ret:
            return self.last_detection_result
        
        # GPU最適化前処理
        if self.use_gpu and self.gpu_processor:
            # GPU リサイズ + RGB変換
            processed_frame = self.gpu_processor.process_frame_optimized(
                frame, 
                target_size=(640, 480), 
                convert_to_rgb=True
            )
        else:
            # CPU処理フォールバック
            small_frame = cv2.resize(frame, (640, 480))
            processed_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe処理
        if not self.face_mesh or not self.hands:
            return self._handle_processing_error(
                RuntimeError("MediaPipeコンポーネントが初期化されていません。")
            )
        face_results = self.face_mesh.process(processed_frame)
        hand_results = self.hands.process(processed_frame)
        
        # 結果解析
        return self._analyze_results(face_results, hand_results, processed_frame.shape)
    
    def _process_with_cpu_standard(self) -> Dict[str, Any]:
        """CPU標準処理"""
        if not self.camera or not self.camera.isOpened():
            return self.last_detection_result
        
        ret, frame = self.camera.read()
        if not ret:
            return self.last_detection_result
        
        # CPU前処理
        small_frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe処理
        if not self.face_mesh or not self.hands:
            return self._handle_processing_error(
                RuntimeError("MediaPipeコンポーネントが初期化されていません。")
            )
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # 結果解析
        return self._analyze_results(face_results, hand_results, rgb_frame.shape)
    
    def _process_with_fallback(self) -> Dict[str, Any]:
        """フォールバック処理"""
        result = self.last_detection_result.copy()
        
        # 基本的なカメラフレーム取得のみ
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                result['frame_shape'] = frame.shape
        
        return result
    
    def _analyze_results(self, face_results, hand_results, frame_shape) -> Dict[str, Any]:
        """検出結果の解析（既存コードと互換）"""
        result = {
            'face_detected': False,
            'hands_detected': False,
            'face_center': None,
            'face_distance': float('inf'),
            'hand_positions': [],
            'frame_shape': frame_shape
        }
        
        # 顔の解析
        if face_results.multi_face_landmarks:
            result['face_detected'] = True
            
            for face_landmarks in face_results.multi_face_landmarks:
                # 鼻の先端（ランドマーク1）を中心点とする
                nose_tip = face_landmarks.landmark[1]
                result['face_center'] = (nose_tip.x, nose_tip.y, nose_tip.z)
                
                # Z距離の推定
                result['face_distance'] = abs(nose_tip.z)
                
                break  # 最初の顔のみ処理
        
        # 手の解析
        if hand_results.multi_hand_landmarks:
            result['hands_detected'] = True
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # 手首（ランドマーク0）の座標を取得
                wrist = hand_landmarks.landmark[0]
                result['hand_positions'].append((wrist.x, wrist.y))
        
        return result
    
    def _handle_processing_error(self, error: Exception) -> Dict[str, Any]:
        """処理エラーハンドリング"""
        self.logger.warning(f"処理エラーによりフォールバック: {error}")
        
        # 統計更新
        self._update_stats(0.0, False)
        
        # 次のバックエンドにフォールバック
        self._fallback_to_next_backend()
        
        # エラー時は前回の結果を返す
        return self.last_detection_result
    
    def _fallback_to_next_backend(self):
        """次の利用可能バックエンドにフォールバック"""
        current_index = None
        priority = [
            VisionBackend.MEDIAPIPE_V10,
            VisionBackend.GPU_OPTIMIZED,
            VisionBackend.CPU_STANDARD,
            VisionBackend.FALLBACK
        ]
        
        try:
            current_index = priority.index(self.current_backend)
        except ValueError:
            current_index = len(priority) - 1
        
        # 次の利用可能バックエンドを探す
        for i in range(current_index + 1, len(priority)):
            if priority[i] in self.available_backends:
                old_backend = self.current_backend
                self.current_backend = priority[i]
                self.processing_stats['backend_switches'] += 1
                
                self.logger.warning(f"バックエンド切替: {old_backend.value} → {self.current_backend.value}")
                
                # 新しいプロセッサー初期化
                self._initialize_current_processor()
                break
    
    def _update_stats(self, processing_time: float, success: bool):
        """統計情報更新"""
        self.processing_stats['total_frames'] += 1
        
        if success:
            self.processing_stats['successful_frames'] += 1
            
            # 移動平均で処理時間更新
            alpha = 0.1
            self.processing_stats['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.processing_stats['avg_processing_time']
            )
    
    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報取得（既存API完全互換）"""
        if self.current_backend == VisionBackend.MEDIAPIPE_V10 and self.processor:
            # 新プロセッサーのデバッグ情報
            base_info = self.processor.get_debug_info()
        else:
            # 標準デバッグ情報生成
            result = self.last_detection_result
            base_info = {
                'Face': 'YES' if result.get('face_detected') else 'NO',
                'Hands': 'YES' if result.get('hands_detected') else 'NO',
                'Face Dist': f"{result.get('face_distance', 0):.3f}",
                'Hand Count': len(result.get('hand_positions', []))
            }
        
        # 統合情報追加
        base_info.update({
            'Backend': self.current_backend.value,
            'GPU': 'YES' if self.use_gpu else 'NO',
            'Proc Time': f"{self.last_detection_result.get('processing_time', 0):.3f}s",
            'Success Rate': f"{(self.processing_stats['successful_frames'] / max(1, self.processing_stats['total_frames']) * 100):.1f}%"
        })
        
        return base_info
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報取得"""
        info = {
            'current_backend': self.current_backend.value,
            'available_backends': [b.value for b in self.available_backends],
            'gpu_acceleration': self.use_gpu,
            'processing_stats': self.processing_stats.copy()
        }
        
        # GPU情報
        if self.gpu_processor:
            info['gpu_info'] = self.gpu_processor.get_system_info()
        
        return info
    
    def force_backend(self, backend_name: str) -> bool:
        """バックエンド強制切替"""
        try:
            backend = VisionBackend(backend_name)
            if backend in self.available_backends:
                old_backend = self.current_backend
                self.current_backend = backend
                self._initialize_current_processor()
                
                self.logger.info(f"バックエンド強制切替: {old_backend.value} → {backend.value}")
                return True
            else:
                self.logger.warning(f"バックエンド {backend_name} は利用できません")
                return False
                
        except ValueError:
            self.logger.error(f"不正なバックエンド名: {backend_name}")
            return False
    
    def cleanup(self):
        """リソース解放（既存API完全互換）"""
        try:
            # MediaPipe 0.10.x プロセッサーのクリーンアップ
            if self.processor and hasattr(self.processor, 'cleanup'):
                self.processor.cleanup()
            
            # MediaPipe コンポーネントのクリーンアップ
            if self.face_mesh:
                self.face_mesh.close()
            if self.hands:
                self.hands.close()
            
            # カメラのクリーンアップ
            if self.camera:
                self.camera.release()
            
            # GPU プロセッサーのクリーンアップ
            if self.gpu_processor:
                self.gpu_processor.cleanup()
            
            self.logger.info("VisionProcessor がクリーンアップされました")
            
        except Exception as e:
            self.logger.warning(f"クリーンアップエラー: {e}")
        
        print("VisionProcessor がクリーンアップされました")
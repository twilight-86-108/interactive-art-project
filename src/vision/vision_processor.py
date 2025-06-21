# src/vision/vision_processor.py - 最終統合版
import cv2
import numpy as np
import time
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import gc

class VisionBackend(Enum):
    """Vision処理バックエンドタイプ"""
    MODULAR_GPU = "modular_gpu"          # モジュール分離 + GPU最適化
    MODULAR_CPU = "modular_cpu"          # モジュール分離 + CPU処理
    MEDIAPIPE_DIRECT = "mediapipe_direct" # MediaPipe直接処理
    FALLBACK = "fallback"                # フォールバック

class PerformanceLevel(Enum):
    """パフォーマンスレベル"""
    ULTRA = "ultra"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"

class VisionProcessor:
    """
    最終統合コンピュータビジョン処理クラス
    
    特徴:
    - モジュール分離設計 + 複数バックエンド対応
    - 詳細な結果統合・相互作用分析
    - 適応的品質制御・パフォーマンス監視
    - 堅牢なエラーハンドリング・自動復旧
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
        
        # バックエンド管理
        self.available_backends: List[VisionBackend] = []
        self.current_backend = VisionBackend.FALLBACK
        
        # パフォーマンス管理
        self.target_fps = config.get('performance', {}).get('target_fps', 30)
        self.performance_level = PerformanceLevel.HIGH
        self.adaptive_quality = config.get('adaptive_quality', True)
        
        # 処理統計
        self.processing_times = deque(maxlen=60)
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.frame_skip = 0
        
        # コンポーネント参照
        self.face_detector = None
        self.hand_detector = None
        self.gpu_processor = None
        self.camera = None
        
        # MediaPipe直接処理用
        self.face_mesh = None
        self.hands = None
        self.mp_face_mesh = None
        self.mp_hands = None
        
        # 並列処理
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="Vision")
        
        # 品質制御設定
        self.quality_configs = self._init_quality_configs()
        self._quality_settings: Dict[str, Any] = {}
        
        # 結果管理
        self.last_detection_result = self._create_default_result()
        
        # 統計情報
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'backend_switches': 0,
            'avg_processing_time': 0.0,
            'interaction_detections': 0,
            'gesture_detections': 0
        }
        
        # 初期化実行
        self._initialize_system()
    
    def _setup_logger(self) -> logging.Logger:
        """ロガー設定"""
        logger = logging.getLogger(f"{__name__}.VisionProcessor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_quality_configs(self) -> Dict[PerformanceLevel, Dict[str, Any]]:
        """品質レベル別設定"""
        return {
            PerformanceLevel.ULTRA: {
                'resolution': (1280, 720),
                'model_complexity': 2,
                'refine_landmarks': True,
                'min_detection_confidence': 0.8,
                'min_tracking_confidence': 0.7,
                'max_num_faces': 2,
                'max_num_hands': 2,
                'frame_skip_max': 0,
                'gpu_optimization': True
            },
            PerformanceLevel.HIGH: {
                'resolution': (960, 540),
                'model_complexity': 1,
                'refine_landmarks': True,
                'min_detection_confidence': 0.7,
                'min_tracking_confidence': 0.6,
                'max_num_faces': 1,
                'max_num_hands': 2,
                'frame_skip_max': 1,
                'gpu_optimization': True
            },
            PerformanceLevel.MEDIUM: {
                'resolution': (640, 480),
                'model_complexity': 1,
                'refine_landmarks': False,
                'min_detection_confidence': 0.6,
                'min_tracking_confidence': 0.5,
                'max_num_faces': 1,
                'max_num_hands': 2,
                'frame_skip_max': 2,
                'gpu_optimization': True
            },
            PerformanceLevel.LOW: {
                'resolution': (480, 360),
                'model_complexity': 0,
                'refine_landmarks': False,
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.4,
                'max_num_faces': 1,
                'max_num_hands': 1,
                'frame_skip_max': 3,
                'gpu_optimization': False
            },
            PerformanceLevel.MINIMAL: {
                'resolution': (320, 240),
                'model_complexity': 0,
                'refine_landmarks': False,
                'min_detection_confidence': 0.4,
                'min_tracking_confidence': 0.3,
                'max_num_faces': 1,
                'max_num_hands': 1,
                'frame_skip_max': 4,
                'gpu_optimization': False
            }
        }
    
    def _create_default_result(self) -> Dict[str, Any]:
        """デフォルト結果作成"""
        return {
            # 基本情報
            'timestamp': time.time(),
            'frame_shape': (480, 640, 3),
            'processing_backend': 'unknown',
            'processing_time': 0.0,
            'performance_level': self.performance_level.value,
            
            # 顔検出結果
            'face_detected': False,
            'face_landmarks': None,
            'face_center': None,
            'face_distance': float('inf'),
            'face_rotation': {},
            'face_bbox': (0, 0, 0, 0),
            'face_count': 0,
            
            # 手検出結果
            'hands_detected': False,
            'hand_landmarks': None,
            'hand_positions': [],
            'hand_count': 0,
            'hands_info': [],
            'hand_gestures': [],
            
            # 相互作用情報
            'interaction_data': {},
            'confidence_scores': {},
            
            # パフォーマンス情報
            'fps_estimate': 0.0,
            'gpu_acceleration': False
        }
    
    def _initialize_system(self):
        """システム初期化"""
        self.logger.info("🚀 最終統合VisionProcessor システム初期化開始...")
        
        # バックエンド発見・初期化
        self._discover_and_init_backends()
        
        # 最適バックエンド選択
        self._select_optimal_backend()
        
        # カメラ初期化
        self._init_camera()
        
        # 初期化完了ログ
        self._log_initialization_summary()
    
    def _discover_and_init_backends(self):
        """バックエンド発見・初期化"""
        # モジュール分離GPU最適化バックエンド
        if self._init_modular_gpu_backend():
            self.available_backends.append(VisionBackend.MODULAR_GPU)
        
        # モジュール分離CPU処理バックエンド
        if self._init_modular_cpu_backend():
            self.available_backends.append(VisionBackend.MODULAR_CPU)
        
        # MediaPipe直接処理バックエンド
        if self._init_mediapipe_direct_backend():
            self.available_backends.append(VisionBackend.MEDIAPIPE_DIRECT)
        
        # フォールバック（常に利用可能）
        self.available_backends.append(VisionBackend.FALLBACK)
        
        self.logger.info(f"📋 利用可能バックエンド: {[b.value for b in self.available_backends]}")
    
    def _init_modular_gpu_backend(self) -> bool:
        """モジュール分離GPU最適化バックエンド初期化"""
        try:
            # GPU プロセッサー初期化
            from ..core.gpu_processor import GPUProcessor
            self.gpu_processor = GPUProcessor()
            
            if not self.gpu_processor.get_system_info().get('gpu_available', False):
                return False
            
            # 個別検出器初期化
            from .face_detector import FaceDetector
            from .hand_detector import HandDetector
            
            # 品質設定でコンフィグ更新
            quality_config = self.quality_configs[self.performance_level]
            enhanced_config = self.config.copy()
            enhanced_config.update(quality_config)
            
            self.face_detector = FaceDetector(enhanced_config)
            self.hand_detector = HandDetector(enhanced_config)
            
            self.logger.info("✅ モジュール分離GPU最適化バックエンド初期化完了")
            return True
            
        except (ImportError, Exception) as e:
            self.logger.debug(f"モジュール分離GPU最適化バックエンド初期化失敗: {e}")
            return False
    
    def _init_modular_cpu_backend(self) -> bool:
        """モジュール分離CPU処理バックエンド初期化"""
        try:
            # 個別検出器初期化（CPUモード）
            from .face_detector import FaceDetector
            from .hand_detector import HandDetector
            
            # CPU最適化設定
            quality_config = self.quality_configs[self.performance_level]
            cpu_config = self.config.copy()
            cpu_config.update(quality_config)
            cpu_config['gpu_optimization'] = False
            
            # GPU プロセッサーは使用しない（CPU版では）
            test_face_detector = FaceDetector(cpu_config)
            test_hand_detector = HandDetector(cpu_config)
            
            # 基本機能テスト
            if (hasattr(test_face_detector, 'detect_face') and 
                hasattr(test_hand_detector, 'detect_hands')):
                
                self.logger.info("✅ モジュール分離CPU処理バックエンド利用可能")
                return True
            
            return False
            
        except (ImportError, Exception) as e:
            self.logger.debug(f"モジュール分離CPU処理バックエンド初期化失敗: {e}")
            return False
    
    def _init_mediapipe_direct_backend(self) -> bool:
        """MediaPipe直接処理バックエンド初期化"""
        try:
            import mediapipe as mp
            
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
            
            # テスト用初期化
            test_face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )
            test_hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
            
            # クリーンアップ
            test_face_mesh.close()
            test_hands.close()
            
            self.logger.info("✅ MediaPipe直接処理バックエンド利用可能")
            return True
            
        except Exception as e:
            self.logger.debug(f"MediaPipe直接処理バックエンド初期化失敗: {e}")
            return False
    
    def _select_optimal_backend(self):
        """最適バックエンド選択"""
        # 優先順位
        priority = [
            VisionBackend.MODULAR_GPU,
            VisionBackend.MODULAR_CPU,
            VisionBackend.MEDIAPIPE_DIRECT,
            VisionBackend.FALLBACK
        ]
        
        for backend in priority:
            if backend in self.available_backends:
                self.current_backend = backend
                break
        
        # 選択されたバックエンド初期化
        self._initialize_current_backend()
    
    def _initialize_current_backend(self):
        """現在のバックエンド初期化"""
        try:
            if self.current_backend == VisionBackend.MODULAR_GPU:
                self._init_modular_gpu_components()
            elif self.current_backend == VisionBackend.MODULAR_CPU:
                self._init_modular_cpu_components()
            elif self.current_backend == VisionBackend.MEDIAPIPE_DIRECT:
                self._init_mediapipe_direct_components()
            else:  # FALLBACK
                self._init_fallback_components()
                
        except Exception as e:
            self.logger.error(f"バックエンド初期化失敗 ({self.current_backend.value}): {e}")
            self._fallback_to_next_backend()
    
    def _init_modular_gpu_components(self):
        """モジュール分離GPU最適化コンポーネント初期化"""
        # GPU プロセッサーは既に初期化済み
        # 個別検出器も既に初期化済み
        self.logger.info("⚡ モジュール分離GPU最適化バックエンド アクティブ")
    
    def _init_modular_cpu_components(self):
        """モジュール分離CPU処理コンポーネント初期化"""
        try:
            from .face_detector import FaceDetector
            from .hand_detector import HandDetector
            
            # CPU最適化設定
            quality_config = self.quality_configs[self.performance_level]
            cpu_config = self.config.copy()
            cpu_config.update(quality_config)
            cpu_config['gpu_optimization'] = False
            
            self.face_detector = FaceDetector(cpu_config)
            self.hand_detector = HandDetector(cpu_config)
            
            self.logger.info("🔧 モジュール分離CPU処理バックエンド アクティブ")
            
        except Exception as e:
            self.logger.error(f"モジュール分離CPU処理コンポーネント初期化エラー: {e}")
            raise
    
    def _init_mediapipe_direct_components(self):
        """MediaPipe直接処理コンポーネント初期化"""
        try:
            import mediapipe as mp
            
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
            
            # 現在の品質設定取得
            quality_config = self.quality_configs[self.performance_level]
            
            # Face Mesh初期化
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=quality_config['max_num_faces'],
                refine_landmarks=quality_config['refine_landmarks'],
                min_detection_confidence=quality_config['min_detection_confidence'],
                min_tracking_confidence=quality_config['min_tracking_confidence']
            )
            
            # Hands初期化
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=quality_config['max_num_hands'],
                min_detection_confidence=quality_config['min_detection_confidence'],
                min_tracking_confidence=quality_config['min_tracking_confidence']
            )
            
            self.logger.info("📡 MediaPipe直接処理バックエンド アクティブ")
            
        except Exception as e:
            self.logger.error(f"MediaPipe直接処理コンポーネント初期化エラー: {e}")
            raise
    
    def _init_fallback_components(self):
        """フォールバックコンポーネント初期化"""
        self.logger.info("🛡️ フォールバックバックエンド アクティブ")
    
    def _init_camera(self):
        """カメラ初期化"""
        try:
            camera_config = self.config.get('camera', {})
            device_id = camera_config.get('device_id', 0)
            
            self.camera = cv2.VideoCapture(device_id)
            
            if not self.camera.isOpened():
                self.logger.warning(f"カメラ {device_id} を開けません（デモモードで継続）")
                return
            
            # カメラ設定
            width = camera_config.get('width', 1920)
            height = camera_config.get('height', 1080)
            fps = camera_config.get('fps', 30)
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.logger.info(f"📹 カメラ初期化成功: {width}x{height}@{fps}fps")
            
        except Exception as e:
            self.logger.warning(f"カメラ初期化エラー: {e}（デモモードで継続）")
    
    def _log_initialization_summary(self):
        """初期化サマリーログ"""
        self.logger.info("="*60)
        self.logger.info("🌊 Aqua Mirror 最終統合VisionProcessor 初期化完了")
        self.logger.info(f"🎯 アクティブバックエンド: {self.current_backend.value}")
        self.logger.info(f"⚡ GPU加速: {'有効' if self.gpu_processor else '無効'}")
        self.logger.info(f"🎚️ パフォーマンスレベル: {self.performance_level.value}")
        self.logger.info(f"🎥 目標FPS: {self.target_fps}")
        self.logger.info(f"🔄 適応的品質制御: {'有効' if self.adaptive_quality else '無効'}")
        self.logger.info("="*60)
    
    def process_frame(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """統合フレーム処理（メインAPI）"""
        start_time = time.time()
        
        try:
            # フレーム取得
            if frame is None:
                frame = self._get_camera_frame()
            
            if frame is None:
                return self.last_detection_result
            
            # フレームスキップ制御
            quality_config = self.quality_configs[self.performance_level]
            if hasattr(self, '_frame_skip_counter'):
                if self._frame_skip_counter < quality_config['frame_skip_max']:
                    self._frame_skip_counter += 1
                    return self.last_detection_result
                else:
                    self._frame_skip_counter = 0
            else:
                self._frame_skip_counter = 0
            
            # バックエンド別処理
            if self.current_backend in [VisionBackend.MODULAR_GPU, VisionBackend.MODULAR_CPU]:
                result = self._process_with_modular_backend(frame)
            elif self.current_backend == VisionBackend.MEDIAPIPE_DIRECT:
                result = self._process_with_mediapipe_direct(frame)
            else:  # FALLBACK
                result = self._process_with_fallback(frame)
            
            # 処理時間・統計更新
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, True)
            
            # 結果に処理情報追加
            result.update({
                'processing_time': processing_time,
                'processing_backend': self.current_backend.value,
                'performance_level': self.performance_level.value,
                'gpu_acceleration': self.gpu_processor is not None,
                'timestamp': time.time()
            })
            
            # 適応的品質制御
            if self.adaptive_quality:
                self._adapt_performance()
            
            self.last_detection_result = result
            return result
            
        except Exception as e:
            return self._handle_processing_error(e, start_time)
    
    def _process_with_modular_backend(self, frame: np.ndarray) -> Dict[str, Any]:
        """モジュール分離バックエンドでの処理"""
        try:
            # detectorが初期化されているか確認
            if not self.face_detector or not self.hand_detector:
                self.logger.warning("Detector not initialized for modular backend.")
                return self.last_detection_result

            # GPU前処理（利用可能な場合）
            if (self.current_backend == VisionBackend.MODULAR_GPU and 
                self.gpu_processor):
                processed_frame = self.gpu_processor.process_frame_optimized(frame)
                if processed_frame is None:
                    processed_frame = frame
            else:
                # CPU前処理
                quality_config = self.quality_configs[self.performance_level]
                target_resolution = quality_config['resolution']
                processed_frame = cv2.resize(frame, target_resolution)
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # 並列検出処理
            face_future = self.executor.submit(self.face_detector.detect_face, processed_frame)
            hand_future = self.executor.submit(self.hand_detector.detect_hands, processed_frame)
            
            try:
                face_result = face_future.result(timeout=0.05)
                hand_result = hand_future.result(timeout=0.05)
            except:
                # タイムアウト時は前回結果返却
                return self.last_detection_result
            
            # 結果統合
            detection_results = {'face': face_result, 'hand': hand_result}
            return self._integrate_modular_results(detection_results, frame.shape)
            
        except Exception as e:
            self.logger.error(f"モジュール分離バックエンド処理エラー: {e}")
            return self.last_detection_result
    
    def _process_with_mediapipe_direct(self, frame: np.ndarray) -> Dict[str, Any]:
        """MediaPipe直接処理"""
        try:
            # MediaPipeコンポーネントが初期化されているか確認
            if not self.face_mesh or not self.hands:
                self.logger.warning("MediaPipe components not initialized for direct backend.")
                return self.last_detection_result

            # 前処理
            quality_config = self.quality_configs[self.performance_level]
            target_resolution = quality_config['resolution']
            
            resized_frame = cv2.resize(frame, target_resolution)
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe処理
            face_results = self.face_mesh.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            # 結果解析
            return self._analyze_mediapipe_results(face_results, hand_results, frame.shape)
            
        except Exception as e:
            self.logger.error(f"MediaPipe直接処理エラー: {e}")
            return self.last_detection_result
    
    def _process_with_fallback(self, frame: np.ndarray) -> Dict[str, Any]:
        """フォールバック処理"""
        result = self.last_detection_result.copy()
        result['frame_shape'] = frame.shape
        result['processing_backend'] = 'fallback'
        return result
    
    def _get_camera_frame(self) -> Optional[np.ndarray]:
        """カメラフレーム取得"""
        if not self.camera or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        return frame if ret else None
    
    def _integrate_modular_results(self, detection_results: Dict[str, Any], frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """モジュール分離結果統合"""
        try:
            integrated = self._create_default_result()
            integrated['frame_shape'] = frame_shape
            
            # 顔検出結果統合
            face_data = detection_results.get('face')
            if face_data:
                integrated.update({
                    'face_detected': face_data.get('face_detected', False),
                    'face_landmarks': face_data.get('face_landmarks'),
                    'face_center': face_data.get('face_center'),
                    'face_distance': face_data.get('face_distance', float('inf')),
                    'face_rotation': face_data.get('face_rotation', {}),
                    'face_bbox': face_data.get('face_bbox', (0, 0, 0, 0)),
                    'face_count': face_data.get('face_count', 0)
                })
                
                if face_data.get('face_detected'):
                    integrated['confidence_scores']['face'] = face_data.get('confidence', 0.8)
            
            # 手検出結果統合
            hand_data = detection_results.get('hand')
            if hand_data:
                integrated.update({
                    'hands_detected': hand_data.get('hands_detected', False),
                    'hand_landmarks': hand_data.get('hand_landmarks'),
                    'hand_positions': hand_data.get('hand_positions', []),
                    'hand_count': hand_data.get('hand_count', 0),
                    'hands_info': hand_data.get('hands', [])
                })
                
                # ジェスチャー抽出
                integrated['hand_gestures'] = self._extract_gestures(hand_data)
                
                if hand_data.get('hands_detected'):
                    for i, hand_info in enumerate(hand_data.get('hands', [])):
                        if 'confidence' in hand_info:
                            integrated['confidence_scores'][f'hand_{i}'] = hand_info['confidence']
            
            # 相互作用分析
            integrated['interaction_data'] = self._analyze_interactions(integrated)
            
            # 統計更新
            if integrated.get('hand_gestures'):
                self.stats['gesture_detections'] += len(integrated['hand_gestures'])
            
            if integrated['interaction_data'].get('face_hand_proximity'):
                self.stats['interaction_detections'] += 1
            
            return integrated
            
        except Exception as e:
            self.logger.error(f"モジュール分離結果統合エラー: {e}")
            return self.last_detection_result
    
    def _analyze_mediapipe_results(self, face_results, hand_results, frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """MediaPipe直接処理結果解析"""
        try:
            result = self._create_default_result()
            result['frame_shape'] = frame_shape
            
            # 顔検出解析
            if face_results and face_results.multi_face_landmarks:
                result['face_detected'] = True
                result['face_landmarks'] = face_results
                
                face_landmarks = face_results.multi_face_landmarks[0]
                nose_tip = face_landmarks.landmark[1]
                result['face_center'] = (nose_tip.x, nose_tip.y, nose_tip.z)
                result['face_distance'] = self._calculate_face_distance(face_landmarks)
                result['confidence_scores']['face'] = 0.8
            
            # 手検出解析
            if hand_results and hand_results.multi_hand_landmarks:
                result['hands_detected'] = True
                result['hand_landmarks'] = hand_results.multi_hand_landmarks
                result['hand_count'] = len(hand_results.multi_hand_landmarks)
                
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    wrist = hand_landmarks.landmark[0]
                    result['hand_positions'].append((wrist.x, wrist.y))
                    
                    if i < len(hand_results.multi_handedness):
                        confidence = hand_results.multi_handedness[i].classification[0].score
                        result['confidence_scores'][f'hand_{i}'] = confidence
            
            # 相互作用分析
            result['interaction_data'] = self._analyze_interactions(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"MediaPipe直接処理結果解析エラー: {e}")
            return self.last_detection_result
    
    def _extract_gestures(self, hand_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """ジェスチャー抽出"""
        try:
            gestures = []
            hands_info = hand_data.get('hands', [])
            
            for hand_info in hands_info:
                gesture = hand_info.get('gesture', 'unknown')
                if gesture != 'unknown':
                    gestures.append({
                        'type': gesture,
                        'handedness': hand_info.get('handedness', 'Unknown'),
                        'position': hand_info.get('wrist_position', (0, 0, 0)),
                        'confidence': hand_info.get('confidence', 0.0)
                    })
            
            return gestures
            
        except Exception as e:
            self.logger.error(f"ジェスチャー抽出エラー: {e}")
            return []
    
    def _analyze_interactions(self, integrated_data: Dict[str, Any]) -> Dict[str, Any]:
        """顔と手の相互作用分析"""
        try:
            interaction = {
                'face_hand_proximity': False,
                'hand_near_face': False,
                'pointing_at_face': False,
                'gesture_active': False,
                'proximity_distance': float('inf'),
                'interaction_type': 'none'
            }
            
            # 顔と手の距離分析
            if (integrated_data.get('face_detected') and 
                integrated_data.get('hands_detected')):
                
                face_center = integrated_data.get('face_center')
                hand_positions = integrated_data.get('hand_positions', [])
                
                if face_center and hand_positions:
                    min_distance = float('inf')
                    
                    for hand_pos in hand_positions:
                        # 2D距離計算
                        distance = np.sqrt(
                            (face_center[0] - hand_pos[0])**2 + 
                            (face_center[1] - hand_pos[1])**2
                        )
                        min_distance = min(min_distance, distance)
                        
                        if distance < 0.3:  # 正規化座標での閾値
                            interaction['hand_near_face'] = True
                            interaction['face_hand_proximity'] = True
                            interaction['interaction_type'] = 'proximity'
                    
                    interaction['proximity_distance'] = min_distance
            
            # ジェスチャー活性状態
            gestures = integrated_data.get('hand_gestures', [])
            if gestures:
                interaction['gesture_active'] = True
                
                for gesture in gestures:
                    if gesture['type'] == 'point':
                        interaction['pointing_at_face'] = True
                        interaction['interaction_type'] = 'pointing'
                    elif gesture['type'] in ['wave', 'peace', 'thumbs_up']:
                        interaction['interaction_type'] = 'gesture'
            
            return interaction
            
        except Exception as e:
            self.logger.error(f"相互作用分析エラー: {e}")
            return {}
    
    def _calculate_face_distance(self, face_landmarks) -> float:
        """顔距離計算"""
        try:
            left_face = face_landmarks.landmark[234]
            right_face = face_landmarks.landmark[454]
            
            face_width = abs(right_face.x - left_face.x)
            estimated_distance = 0.14 / (face_width + 1e-6)
            
            return min(estimated_distance, 5.0)
            
        except Exception as e:
            self.logger.error(f"顔距離計算エラー: {e}")
            return float('inf')
    
    def _update_processing_stats(self, processing_time: float, success: bool):
        """処理統計更新"""
        self.processing_times.append(processing_time)
        self.stats['total_frames'] += 1
        
        if success:
            self.stats['successful_frames'] += 1
        else:
            self.stats['failed_frames'] += 1
        
        # 処理時間統計更新
        if self.processing_times:
            self.stats['avg_processing_time'] = sum(self.processing_times) / len(self.processing_times)
        
        # FPS計算
        self._update_fps()
    
    def _update_fps(self):
        """FPS更新"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _adapt_performance(self):
        """適応的パフォーマンス制御"""
        if len(self.processing_times) < 10:
            return
        
        target_frame_time = 1.0 / self.target_fps
        avg_processing_time = self.stats['avg_processing_time']
        
        # パフォーマンスレベル調整
        if avg_processing_time > target_frame_time * 1.5:
            self._decrease_performance_level()
        elif avg_processing_time < target_frame_time * 0.7 and self.current_fps > self.target_fps * 1.1:
            self._increase_performance_level()
    
    def _decrease_performance_level(self):
        """パフォーマンスレベル低下"""
        current_levels = list(PerformanceLevel)
        current_index = current_levels.index(self.performance_level)
        
        if current_index < len(current_levels) - 1:
            old_level = self.performance_level
            self.performance_level = current_levels[current_index + 1]
            
            self.logger.info(f"🔽 パフォーマンスレベル低下: {old_level.value} → {self.performance_level.value}")
            self._reinit_backend_with_quality()
    
    def _increase_performance_level(self):
        """パフォーマンスレベル向上"""
        current_levels = list(PerformanceLevel)
        current_index = current_levels.index(self.performance_level)
        
        if current_index > 0:
            old_level = self.performance_level
            self.performance_level = current_levels[current_index - 1]
            
            self.logger.info(f"🔼 パフォーマンスレベル向上: {old_level.value} → {self.performance_level.value}")
            self._reinit_backend_with_quality()
    
    def _reinit_backend_with_quality(self):
        """品質設定でバックエンド再初期化"""
        try:
            self._initialize_current_backend()
        except Exception as e:
            self.logger.warning(f"バックエンド再初期化エラー: {e}")
    
    def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """処理エラーハンドリング"""
        processing_time = time.time() - start_time
        self._update_processing_stats(processing_time, False)
        
        self.logger.warning(f"フレーム処理エラー ({self.current_backend.value}): {error}")
        
        # バックエンドフォールバック
        self._fallback_to_next_backend()
        
        # エラー時は前回結果返却
        result = self.last_detection_result.copy()
        result.update({
            'processing_time': processing_time,
            'error_occurred': True,
            'error_message': str(error)
        })
        
        return result
    
    def _fallback_to_next_backend(self):
        """次のバックエンドにフォールバック"""
        priority = [
            VisionBackend.MODULAR_GPU,
            VisionBackend.MODULAR_CPU,
            VisionBackend.MEDIAPIPE_DIRECT,
            VisionBackend.FALLBACK
        ]
        
        try:
            current_index = priority.index(self.current_backend)
        except ValueError:
            current_index = -1
        
        # 次の利用可能バックエンドを探索
        for i in range(current_index + 1, len(priority)):
            if priority[i] in self.available_backends:
                old_backend = self.current_backend
                self.current_backend = priority[i]
                self.stats['backend_switches'] += 1
                
                self.logger.warning(f"🔄 バックエンド切替: {old_backend.value} → {self.current_backend.value}")
                
                try:
                    self._initialize_current_backend()
                    break
                except Exception as e:
                    self.logger.error(f"フォールバック初期化失敗: {e}")
                    continue
    
    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報取得（API互換）"""
        result = self.last_detection_result
        
        debug_info = {
            'Face': 'YES' if result.get('face_detected') else 'NO',
            'Hands': 'YES' if result.get('hands_detected') else 'NO',
            'Face Distance': f"{result.get('face_distance', 0):.3f}m",
            'Hand Count': result.get('hand_count', 0),
            'Gestures': len(result.get('hand_gestures', [])),
            'Interaction': result.get('interaction_data', {}).get('interaction_type', 'none'),
            'Backend': self.current_backend.value,
            'GPU': 'ON' if self.gpu_processor else 'OFF',
            'FPS': f"{self.current_fps:.1f}",
            'Performance': self.performance_level.value,
            'Success Rate': f"{(self.stats['successful_frames'] / max(1, self.stats['total_frames']) * 100):.1f}%"
        }
        
        return debug_info
    
    def get_system_info(self) -> Dict[str, Any]:
        """詳細システム情報"""
        return {
            'current_backend': self.current_backend.value,
            'available_backends': [b.value for b in self.available_backends],
            'performance_level': self.performance_level.value,
            'gpu_acceleration': self.gpu_processor is not None,
            'adaptive_quality': self.adaptive_quality,
            'target_fps': self.target_fps,
            'current_fps': self.current_fps,
            'processing_stats': self.stats.copy(),
            'interaction_detections': self.stats.get('interaction_detections', 0),
            'gesture_detections': self.stats.get('gesture_detections', 0)
        }
    
    def force_backend(self, backend_name: str) -> bool:
        """バックエンド強制切替"""
        try:
            backend = VisionBackend(backend_name)
            if backend in self.available_backends:
                old_backend = self.current_backend
                self.current_backend = backend
                self._initialize_current_backend()
                
                self.logger.info(f"🎯 バックエンド強制切替: {old_backend.value} → {backend.value}")
                return True
            else:
                self.logger.warning(f"バックエンド {backend_name} は利用できません")
                return False
                
        except ValueError:
            self.logger.error(f"不正なバックエンド名: {backend_name}")
            return False
    
    def set_performance_level(self, level_name: str) -> bool:
        """パフォーマンスレベル手動設定"""
        try:
            level = PerformanceLevel(level_name)
            old_level = self.performance_level
            self.performance_level = level
            
            self._reinit_backend_with_quality()
            
            self.logger.info(f"🎚️ パフォーマンスレベル設定: {old_level.value} → {level.value}")
            return True
            
        except ValueError:
            self.logger.error(f"不正なパフォーマンスレベル: {level_name}")
            return False
    
    def cleanup(self):
        """リソース解放（API互換）"""
        self.logger.info("🧹 最終統合VisionProcessor クリーンアップ開始...")
        
        try:
            # 個別検出器クリーンアップ
            if self.face_detector and hasattr(self.face_detector, 'cleanup'):
                self.face_detector.cleanup()
            if self.hand_detector and hasattr(self.hand_detector, 'cleanup'):
                self.hand_detector.cleanup()
            
            # MediaPipe直接処理クリーンアップ
            if self.face_mesh:
                self.face_mesh.close()
            if self.hands:
                self.hands.close()
            
            # カメラクリーンアップ
            if self.camera:
                self.camera.release()
            
            # GPU プロセッサークリーンアップ
            if self.gpu_processor and hasattr(self.gpu_processor, 'cleanup'):
                self.gpu_processor.cleanup()
            
            # スレッドプールクリーンアップ
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # メモリクリーンアップ
            gc.collect()
            
            self.logger.info("✅ 最終統合VisionProcessor クリーンアップ完了")
            
        except Exception as e:
            self.logger.warning(f"クリーンアップエラー: {e}")
        
        print("VisionProcessor がクリーンアップされました")
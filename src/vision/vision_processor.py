# src/vision/vision_processor.py
import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple
from collections import deque

# 個別検出器のインポート
from .face_detector import FaceDetector
from .hand_detector import HandDetector
from ..core.gpu_processor import GPUProcessor

class VisionProcessor:
    """統合画像処理エンジン（エラー修正版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # パフォーマンス管理
        self.frame_skip = 0
        self.adaptive_quality = config.get('adaptive_quality', True)
        self.processing_times = deque(maxlen=30)
        self.last_detection_result = {}
        
        try:
            # GPU プロセッサー初期化
            self.gpu_processor = GPUProcessor()
            
            # 個別検出器初期化
            self.face_detector = FaceDetector(config)
            self.hand_detector = HandDetector(config)
            
            # 処理品質設定
            self.quality_level = config.get('quality_level', 'medium')
            self._adjust_quality_settings()
            
            self.logger.info("統合画像処理エンジンが初期化されました")
            
        except Exception as e:
            self.logger.error(f"統合画像処理エンジン初期化エラー: {e}")
            raise
    
    def process_frame(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """メインフレーム処理関数"""
        start_time = time.time()
        
        try:
            # フレーム前処理
            if frame is None:
                return self.last_detection_result
            
            # フレームスキップ制御
            if self.frame_skip > 0:
                self.frame_skip -= 1
                return self.last_detection_result
            
            # GPU前処理
            processed_frame = self.gpu_processor.process_frame_optimized(frame)
            if processed_frame is None:
                processed_frame = frame
            
            # 並列検出処理
            detection_result = self._perform_detections(processed_frame)
            
            # 結果統合
            integrated_result = self._integrate_results(detection_result, tuple(frame.shape))
            
            # パフォーマンス監視
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # 適応的品質制御
            if self.adaptive_quality:
                self._adjust_performance()
            
            self.last_detection_result = integrated_result
            return integrated_result
            
        except Exception as e:
            self.logger.error(f"フレーム処理エラー: {e}")
            return self.last_detection_result
    
    def _perform_detections(self, frame: np.ndarray) -> Dict[str, Any]:
        """検出処理の実行"""
        try:
            results = {}
            
            # 顔検出
            face_result = self.face_detector.detect_face(frame)
            results['face'] = face_result
            
            # 手検出
            hand_result = self.hand_detector.detect_hands(frame)
            results['hand'] = hand_result
            
            return results
            
        except Exception as e:
            self.logger.error(f"検出処理エラー: {e}")
            return {}
    
    def _integrate_results(self, detection_result: Dict[str, Any], frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """検出結果の統合"""
        try:
            integrated = {
                'timestamp': time.time(),
                'frame_shape': frame_shape,
                'processing_quality': self.quality_level
            }
            
            # 顔検出結果の統合
            face_data = detection_result.get('face')
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
            else:
                integrated.update({
                    'face_detected': False,
                    'face_landmarks': None,
                    'face_center': None,
                    'face_distance': float('inf'),
                    'face_rotation': {},
                    'face_bbox': (0, 0, 0, 0),
                    'face_count': 0
                })
            
            # 手検出結果の統合
            hand_data = detection_result.get('hand')
            if hand_data:
                integrated.update({
                    'hands_detected': hand_data.get('hands_detected', False),
                    'hand_landmarks': hand_data.get('hand_landmarks'),
                    'hand_positions': hand_data.get('hand_positions', []),
                    'hand_count': hand_data.get('hand_count', 0),
                    'hands_info': hand_data.get('hands', []),
                    'hand_gestures': self._extract_gestures(hand_data)
                })
            else:
                integrated.update({
                    'hands_detected': False,
                    'hand_landmarks': None,
                    'hand_positions': [],
                    'hand_count': 0,
                    'hands_info': [],
                    'hand_gestures': []
                })
            
            # 相互作用情報
            integrated['interaction_data'] = self._analyze_interactions(integrated)
            
            return integrated
            
        except Exception as e:
            self.logger.error(f"結果統合エラー: {e}")
            return self.last_detection_result
    
    def _extract_gestures(self, hand_data: Dict[str, Any]) -> list:
        """手のジェスチャー情報抽出"""
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
                'gesture_active': False
            }
            
            # 顔と手の距離分析
            if (integrated_data.get('face_detected') and 
                integrated_data.get('hands_detected')):
                
                face_center = integrated_data.get('face_center')
                hand_positions = integrated_data.get('hand_positions', [])
                
                if face_center and hand_positions:
                    for hand_pos in hand_positions:
                        # 2D距離計算
                        distance = np.sqrt(
                            (face_center[0] - hand_pos[0])**2 + 
                            (face_center[1] - hand_pos[1])**2
                        )
                        
                        if distance < 0.3:  # 正規化座標での閾値
                            interaction['hand_near_face'] = True
                            interaction['face_hand_proximity'] = True
            
            # ジェスチャー活性状態
            gestures = integrated_data.get('hand_gestures', [])
            if gestures:
                interaction['gesture_active'] = True
                
                # 指差しジェスチャーの特別処理
                for gesture in gestures:
                    if gesture['type'] == 'point':
                        interaction['pointing_at_face'] = True
            
            return interaction
            
        except Exception as e:
            self.logger.error(f"相互作用分析エラー: {e}")
            return {}
    
    def _adjust_quality_settings(self):
        """品質設定調整"""
        try:
            quality_configs = {
                'low': {
                    'frame_skip_max': 3,
                    'detection_confidence': 0.6,
                    'refine_landmarks': False
                },
                'medium': {
                    'frame_skip_max': 2,
                    'detection_confidence': 0.7,
                    'refine_landmarks': True
                },
                'high': {
                    'frame_skip_max': 1,
                    'detection_confidence': 0.8,
                    'refine_landmarks': True
                }
            }
            
            config = quality_configs.get(self.quality_level, quality_configs['medium'])
            self.frame_skip_max = config['frame_skip_max']
            
        except Exception as e:
            self.logger.error(f"品質設定調整エラー: {e}")
    
    def _adjust_performance(self):
        """適応的パフォーマンス調整"""
        try:
            if len(self.processing_times) >= 10:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                
                # 処理時間が長い場合はフレームスキップ増加
                if avg_time > 0.05:  # 50ms超過
                    self.frame_skip = min(self.frame_skip_max, self.frame_skip + 1)
                elif avg_time < 0.02:  # 20ms未満
                    self.frame_skip = max(0, self.frame_skip - 1)
                    
        except Exception as e:
            self.logger.error(f"パフォーマンス調整エラー: {e}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報取得"""
        try:
            result = self.last_detection_result
            
            debug_info = {
                'Face': 'YES' if result.get('face_detected') else 'NO',
                'Hands': 'YES' if result.get('hands_detected') else 'NO',
                'Face Distance': f"{result.get('face_distance', 0):.3f}",
                'Hand Count': result.get('hand_count', 0),
                'Frame Skip': self.frame_skip,
                'Quality Level': self.quality_level,
                'GPU Available': self.gpu_processor.get_system_info().get('gpu_available', False)
            }
            
            # パフォーマンス情報
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                debug_info['Avg Processing Time'] = f"{avg_time:.3f}s"
                debug_info['Est FPS'] = f"{1/avg_time:.1f}" if avg_time > 0 else "N/A"
            
            return debug_info
            
        except Exception as e:
            self.logger.error(f"デバッグ情報取得エラー: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, float]:
        """パフォーマンス統計取得"""
        try:
            if not self.processing_times:
                return {}
            
            times = list(self.processing_times)
            return {
                'avg_processing_time': sum(times) / len(times),
                'max_processing_time': max(times),
                'min_processing_time': min(times),
                'frame_skip_level': self.frame_skip,
                'fps_estimate': 1 / (sum(times) / len(times)) if times else 0
            }
            
        except Exception as e:
            self.logger.error(f"パフォーマンス統計エラー: {e}")
            return {}
    
    def set_quality_level(self, level: str):
        """品質レベル設定"""
        try:
            if level in ['low', 'medium', 'high']:
                self.quality_level = level
                self._adjust_quality_settings()
                self.logger.info(f"品質レベルを{level}に設定しました")
            else:
                self.logger.warning(f"無効な品質レベル: {level}")
                
        except Exception as e:
            self.logger.error(f"品質レベル設定エラー: {e}")
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.face_detector:
                self.face_detector.cleanup()
            if self.hand_detector:
                self.hand_detector.cleanup()
            if self.gpu_processor:
                self.gpu_processor.cleanup()
            
            self.logger.info("統合画像処理エンジンがクリーンアップされました")
            
        except Exception as e:
            self.logger.error(f"統合画像処理エンジンクリーンアップエラー: {e}")
# src/vision/face_detector.py - 修正版
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh, drawing_utils, drawing_styles
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum

class FaceDetectionQuality(Enum):
    """顔検出品質レベル"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class FaceDetectionResult:
    """顔検出結果データクラス"""
    face_detected: bool
    face_landmarks: Optional[Any] = None
    face_center: Optional[Tuple[float, float, float]] = None
    face_distance: float = float('inf')
    confidence: float = 0.0
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    landmarks_2d: Optional[List[Tuple[float, float]]] = None
    landmarks_3d: Optional[List[Tuple[float, float, float]]] = None

class FaceDetector:
    """MediaPipe顔検出クラス（修正版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe初期化
        self._init_mediapipe()
        
        # 検出履歴（平滑化用）
        self.detection_history = []
        self.max_history = 10
        
        # パフォーマンス監視
        self.processing_times = []
        self.detection_counts = {'success': 0, 'failure': 0}
        
        # 品質制御
        self.current_quality = FaceDetectionQuality.HIGH
        self.adaptive_quality = config.get('adaptive_quality', True)
        
        self.logger.info("顔検出器が初期化されました")
    
    def _init_mediapipe(self):
        """MediaPipe初期化（最新API対応）"""
        try:
            # 明示的インポートで型エラー回避
            self.mp_face_mesh = face_mesh
            self.mp_drawing = drawing_utils
            self.mp_drawing_styles = drawing_styles
            
            # FaceMesh設定取得
            face_config = self.config.get('face_detection', {})
            
            # FaceMeshインスタンス作成
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=face_config.get('max_num_faces', 1),
                refine_landmarks=face_config.get('refine_landmarks', True),
                min_detection_confidence=face_config.get('min_detection_confidence', 0.7),
                min_tracking_confidence=face_config.get('min_tracking_confidence', 0.5)
            )
            
            self.logger.info("MediaPipe FaceMesh初期化成功")
            
        except Exception as e:
            self.logger.error(f"MediaPipe初期化エラー: {e}")
            raise RuntimeError(f"MediaPipe初期化に失敗しました: {e}")
    
    def detect_face(self, frame: np.ndarray) -> FaceDetectionResult:
        """
        顔検出メイン処理
        
        Args:
            frame: 入力画像フレーム (BGR形式)
            
        Returns:
            FaceDetectionResult: 検出結果
        """
        import time
        start_time = time.time()
        
        try:
            # 入力検証
            if frame is None or frame.size == 0:
                self.logger.warning("無効なフレームが入力されました")
                return FaceDetectionResult(face_detected=False)
            
            # フレーム前処理
            processed_frame = self._preprocess_frame(frame)
            
            # MediaPipe処理
            results = self.face_mesh.process(processed_frame)
            
            # 結果処理
            detection_result = self._process_face_results(
                results, 
                tuple(frame.shape)  # 明示的にtupleにキャスト
            )
            
            # 履歴更新
            self._update_detection_history(detection_result)
            
            # パフォーマンス記録
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # 統計更新
            if detection_result.face_detected:
                self.detection_counts['success'] += 1
            else:
                self.detection_counts['failure'] += 1
            
            # 適応的品質制御
            if self.adaptive_quality:
                self._adjust_quality_if_needed(processing_time)
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"顔検出処理エラー: {e}")
            return FaceDetectionResult(face_detected=False)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレーム前処理"""
        try:
            # BGR to RGB変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 品質に応じたリサイズ
            if self.current_quality == FaceDetectionQuality.LOW:
                height, width = frame.shape[:2]
                new_width = min(640, width)
                new_height = int(height * (new_width / width))
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            elif self.current_quality == FaceDetectionQuality.MEDIUM:
                height, width = frame.shape[:2]
                new_width = min(1280, width)
                new_height = int(height * (new_width / width))
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
            # HIGH品質の場合は元のサイズを維持
            
            return rgb_frame
            
        except Exception as e:
            self.logger.warning(f"フレーム前処理エラー: {e}")
            # エラー時は元のフレームを返す（色変換のみ）
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _process_face_results(self, results: Any, frame_shape: Tuple[int, ...]) -> FaceDetectionResult:
        """
        MediaPipe結果処理
        
        Args:
            results: MediaPipe処理結果
            frame_shape: フレーム形状 (height, width, channels)
            
        Returns:
            FaceDetectionResult: 処理された検出結果
        """
        try:
            # フレーム形状の安全な処理
            if len(frame_shape) >= 2:
                height, width = frame_shape[0], frame_shape[1]
                channels = frame_shape[2] if len(frame_shape) > 2 else 3
            else:
                self.logger.warning(f"無効なフレーム形状: {frame_shape}")
                return FaceDetectionResult(face_detected=False)
            
            # 顔が検出されない場合
            if not results.multi_face_landmarks:
                return FaceDetectionResult(face_detected=False)
            
            # 最初の顔のランドマークを処理（複数顔対応は将来的に）
            face_landmarks = results.multi_face_landmarks[0]
            
            # 2D/3Dランドマーク座標取得
            landmarks_2d = []
            landmarks_3d = []
            
            for landmark in face_landmarks.landmark:
                # 正規化座標をピクセル座標に変換
                x_px = int(landmark.x * width)
                y_px = int(landmark.y * height)
                z = landmark.z  # 相対的な深度
                
                landmarks_2d.append((landmark.x, landmark.y))
                landmarks_3d.append((landmark.x, landmark.y, z))
            
            # 顔の中心点計算（鼻先を使用）
            nose_tip_idx = 1  # MediaPipeの鼻先ランドマークインデックス
            if len(face_landmarks.landmark) > nose_tip_idx:
                nose_landmark = face_landmarks.landmark[nose_tip_idx]
                face_center = (nose_landmark.x, nose_landmark.y, nose_landmark.z)
                face_distance = abs(nose_landmark.z)
            else:
                # フォールバック: 全ランドマークの重心
                center_x = sum(lm.x for lm in face_landmarks.landmark) / len(face_landmarks.landmark)
                center_y = sum(lm.y for lm in face_landmarks.landmark) / len(face_landmarks.landmark)
                center_z = sum(lm.z for lm in face_landmarks.landmark) / len(face_landmarks.landmark)
                face_center = (center_x, center_y, center_z)
                face_distance = abs(center_z)
            
            # バウンディングボックス計算
            x_coords = [lm.x * width for lm in face_landmarks.landmark]
            y_coords = [lm.y * height for lm in face_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            bounding_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # 信頼度計算（簡易版）
            confidence = self._calculate_detection_confidence(face_landmarks, frame_shape)
            
            return FaceDetectionResult(
                face_detected=True,
                face_landmarks=face_landmarks,
                face_center=face_center,
                face_distance=face_distance,
                confidence=confidence,
                bounding_box=bounding_box,
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d
            )
            
        except Exception as e:
            self.logger.error(f"顔結果処理エラー: {e}")
            return FaceDetectionResult(face_detected=False)
    
    def _calculate_detection_confidence(self, face_landmarks: Any, frame_shape: Tuple[int, ...]) -> float:
        """検出信頼度計算"""
        try:
            # ランドマーク数による基本信頼度
            landmark_count = len(face_landmarks.landmark)
            base_confidence = min(1.0, landmark_count / 468)  # MediaPipeの標準ランドマーク数
            
            # 顔のサイズによる調整
            if len(frame_shape) >= 2:
                height, width = frame_shape[0], frame_shape[1]
                
                x_coords = [lm.x * width for lm in face_landmarks.landmark]
                y_coords = [lm.y * height for lm in face_landmarks.landmark]
                
                face_width = max(x_coords) - min(x_coords)
                face_height = max(y_coords) - min(y_coords)
                
                # 相対的な顔サイズ
                relative_face_size = (face_width * face_height) / (width * height)
                
                # 適度なサイズの顔に高い信頼度
                if 0.05 <= relative_face_size <= 0.5:
                    size_confidence = 1.0
                elif relative_face_size < 0.05:
                    size_confidence = relative_face_size / 0.05
                else:
                    size_confidence = 0.5 / relative_face_size
                
                return min(1.0, base_confidence * size_confidence)
            
            return base_confidence
            
        except Exception as e:
            self.logger.warning(f"信頼度計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _update_detection_history(self, result: FaceDetectionResult):
        """検出履歴更新"""
        self.detection_history.append(result)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
    
    def _adjust_quality_if_needed(self, processing_time: float):
        """適応的品質調整"""
        target_time = 0.033  # 30FPS相当の処理時間
        
        if processing_time > target_time * 1.5:  # 処理が遅い
            if self.current_quality == FaceDetectionQuality.HIGH:
                self.current_quality = FaceDetectionQuality.MEDIUM
                self.logger.info("顔検出品質をMEDIUMに調整")
            elif self.current_quality == FaceDetectionQuality.MEDIUM:
                self.current_quality = FaceDetectionQuality.LOW
                self.logger.info("顔検出品質をLOWに調整")
        elif processing_time < target_time * 0.5:  # 処理が速い
            if self.current_quality == FaceDetectionQuality.LOW:
                self.current_quality = FaceDetectionQuality.MEDIUM
                self.logger.info("顔検出品質をMEDIUMに調整")
            elif self.current_quality == FaceDetectionQuality.MEDIUM:
                self.current_quality = FaceDetectionQuality.HIGH
                self.logger.info("顔検出品質をHIGHに調整")
    
    def get_smoothed_result(self) -> Optional[FaceDetectionResult]:
        """履歴ベース平滑化結果取得"""
        if not self.detection_history:
            return None
        
        try:
            # 最近の検出成功結果のみを使用
            recent_successful = [
                result for result in self.detection_history[-5:] 
                if result.face_detected
            ]
            
            if not recent_successful:
                return FaceDetectionResult(face_detected=False)
            
            # 平均座標計算
            avg_center_x = sum(r.face_center[0] for r in recent_successful) / len(recent_successful)
            avg_center_y = sum(r.face_center[1] for r in recent_successful) / len(recent_successful)
            avg_center_z = sum(r.face_center[2] for r in recent_successful) / len(recent_successful)
            
            avg_distance = sum(r.face_distance for r in recent_successful) / len(recent_successful)
            avg_confidence = sum(r.confidence for r in recent_successful) / len(recent_successful)
            
            return FaceDetectionResult(
                face_detected=True,
                face_center=(avg_center_x, avg_center_y, avg_center_z),
                face_distance=avg_distance,
                confidence=avg_confidence
            )
            
        except Exception as e:
            self.logger.warning(f"平滑化処理エラー: {e}")
            return self.detection_history[-1] if self.detection_history else None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        if not self.processing_times:
            return {}
        
        total_detections = self.detection_counts['success'] + self.detection_counts['failure']
        
        return {
            'avg_processing_time': sum(self.processing_times) / len(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'min_processing_time': min(self.processing_times),
            'detection_success_rate': self.detection_counts['success'] / max(1, total_detections),
            'total_detections': total_detections,
            'current_quality': self.current_quality.value,
            'fps_estimate': 1.0 / (sum(self.processing_times[-10:]) / min(10, len(self.processing_times)))
        }
    
    def draw_landmarks(self, frame: np.ndarray, result: FaceDetectionResult) -> np.ndarray:
        """ランドマーク描画（デバッグ用）"""
        if not result.face_detected or not result.face_landmarks:
            return frame
        
        try:
            # ランドマーク描画
            annotated_frame = frame.copy()
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                result.face_landmarks,
                list(self.mp_face_mesh.FACEMESH_CONTOURS),
                self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            
            # 中心点描画
            if result.face_center:
                height, width = frame.shape[:2]
                center_x = int(result.face_center[0] * width)
                center_y = int(result.face_center[1] * height)
                cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # バウンディングボックス描画
            if result.bounding_box:
                x, y, w, h = result.bounding_box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            return annotated_frame
            
        except Exception as e:
            self.logger.warning(f"ランドマーク描画エラー: {e}")
            return frame
    
    def cleanup(self):
        """リソース解放"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            self.logger.info("顔検出器がクリーンアップされました")
        except Exception as e:
            self.logger.warning(f"クリーンアップエラー: {e}")


# 使用例とテスト
if __name__ == "__main__":
    import time
    
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # 設定例
    config = {
        'face_detection': {
            'max_num_faces': 1,
            'refine_landmarks': True,
            'min_detection_confidence': 0.7,
            'min_tracking_confidence': 0.5
        },
        'adaptive_quality': True
    }
    
    # 顔検出器初期化
    face_detector = FaceDetector(config)
    
    # カメラからのテスト（カメラが利用可能な場合）
    try:
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("🔴 カメラテスト開始（Qキーで終了）")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 顔検出
                result = face_detector.detect_face(frame)
                
                # 結果表示
                if result.face_detected:
                    frame = face_detector.draw_landmarks(frame, result)
                    
                    # 情報表示
                    info_text = f"Confidence: {result.confidence:.2f}"
                    cv2.putText(frame, info_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Face Detection Test', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # パフォーマンス統計表示
            stats = face_detector.get_performance_stats()
            print("\n📊 パフォーマンス統計:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        else:
            print("❌ カメラが利用できません")
    
    except Exception as e:
        print(f"❌ テストエラー: {e}")
    
    finally:
        face_detector.cleanup()
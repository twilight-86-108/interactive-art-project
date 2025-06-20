# src/vision/face_detector.py
import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any

class FaceDetector:
    """顔検出クラス（エラー修正版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        try:
            # 設定取得（デフォルト値付き）
            face_config = config.get('detection', {})
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=face_config.get('max_num_faces', 1),
                refine_landmarks=face_config.get('refine_landmarks', True),
                min_detection_confidence=face_config.get('face_detection_confidence', 0.7),
                min_tracking_confidence=face_config.get('min_tracking_confidence', 0.5)
            )
            
            self.logger.info("顔検出器が初期化されました")
            
        except Exception as e:
            self.logger.error(f"顔検出器初期化エラー: {e}")
            self.face_mesh = None
    
    def detect_face(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """顔検出メイン関数"""
        if frame is None or self.face_mesh is None:
            return None
        
        try:
            # フレームがBGRの場合、RGBに変換
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # OpenCVはBGR、MediaPipeはRGBを期待
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # 顔検出実行
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                return self._process_face_results(results, frame.shape)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"顔検出エラー: {e}")
            return None
    
    def _process_face_results(self, results, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """顔検出結果の処理"""
        try:
            height, width = frame_shape[:2]
            face_data = {
                'face_detected': True,
                'face_landmarks': results,
                'face_count': len(results.multi_face_landmarks),
                'faces': []
            }
            
            for idx, face_landmarks in enumerate(results.multi_face_landmarks):
                face_info = self._analyze_single_face(face_landmarks, width, height)
                face_data['faces'].append(face_info)
            
            # 最初の顔の情報を主要データとして設定
            if face_data['faces']:
                main_face = face_data['faces'][0]
                face_data.update({
                    'face_center': main_face['center'],
                    'face_distance': main_face['distance'],
                    'face_rotation': main_face['rotation'],
                    'face_bbox': main_face['bbox']
                })
            
            return face_data
            
        except Exception as e:
            self.logger.error(f"顔結果処理エラー: {e}")
            return {'face_detected': False}
    
    def _analyze_single_face(self, landmarks, width: int, height: int) -> Dict[str, Any]:
        """個別顔の分析"""
        try:
            # ランドマークを正規化座標から画素座標に変換
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                landmark_points.append((x, y, z))
            
            # 顔の中心点計算（鼻先を使用）
            nose_tip = landmarks.landmark[1]  # 鼻先のランドマーク
            center = (nose_tip.x, nose_tip.y, nose_tip.z)
            
            # 顔の距離推定（Z座標ベース）
            distance = abs(nose_tip.z)
            
            # 顔の回転角度推定
            rotation = self._estimate_face_rotation(landmarks)
            
            # バウンディングボックス計算
            bbox = self._calculate_face_bbox(landmark_points, width, height)
            
            return {
                'center': center,
                'distance': distance,
                'rotation': rotation,
                'bbox': bbox,
                'landmark_points': landmark_points,
                'confidence': 0.8  # MediaPipeは信頼度を直接提供しないため固定値
            }
            
        except Exception as e:
            self.logger.error(f"顔分析エラー: {e}")
            return {}
    
    def _estimate_face_rotation(self, landmarks) -> Dict[str, float]:
        """顔の回転角度推定"""
        try:
            # 主要ランドマークを使用して回転を推定
            nose_tip = landmarks.landmark[1]
            chin = landmarks.landmark[175] if len(landmarks.landmark) > 175 else nose_tip
            left_eye = landmarks.landmark[33] if len(landmarks.landmark) > 33 else nose_tip
            right_eye = landmarks.landmark[263] if len(landmarks.landmark) > 263 else nose_tip
            
            # ヨー角（左右の回転）
            yaw = (nose_tip.x - 0.5) * 90  # 簡易推定
            
            # ピッチ角（上下の回転）
            pitch = (nose_tip.y - chin.y) * 90  # 簡易推定
            
            # ロール角（傾き）
            eye_diff_y = right_eye.y - left_eye.y
            roll = np.degrees(np.arctan2(eye_diff_y, right_eye.x - left_eye.x))
            
            return {
                'yaw': yaw,
                'pitch': pitch, 
                'roll': roll
            }
            
        except Exception as e:
            self.logger.error(f"回転推定エラー: {e}")
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
    
    def _calculate_face_bbox(self, landmark_points, width: int, height: int) -> Tuple[int, int, int, int]:
        """顔のバウンディングボックス計算"""
        try:
            if not landmark_points:
                return (0, 0, 0, 0)
            
            x_coords = [point[0] for point in landmark_points]
            y_coords = [point[1] for point in landmark_points]
            
            min_x = max(0, min(x_coords))
            max_x = min(width, max(x_coords))
            min_y = max(0, min(y_coords))
            max_y = min(height, max(y_coords))
            
            return (min_x, min_y, max_x - min_x, max_y - min_y)
            
        except Exception as e:
            self.logger.error(f"バウンディングボックス計算エラー: {e}")
            return (0, 0, 0, 0)
    
    def get_face_region(self, landmarks, frame_shape: Tuple[int, int, int], margin: float = 0.2) -> Optional[Tuple[int, int, int, int]]:
        """顔領域の抽出（マージン付き）"""
        try:
            height, width = frame_shape[:2]
            
            if not landmarks or not hasattr(landmarks, 'multi_face_landmarks') or not landmarks.multi_face_landmarks:
                return None
            
            face_landmarks = landmarks.multi_face_landmarks[0]
            
            # ランドマークの最小・最大座標を取得
            x_coords = [landmark.x for landmark in face_landmarks.landmark]
            y_coords = [landmark.y for landmark in face_landmarks.landmark]
            
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            # マージンを追加
            face_width = max_x - min_x
            face_height = max_y - min_y
            
            margin_x = face_width * margin
            margin_y = face_height * margin
            
            # 正規化座標を画素座標に変換
            x1 = int(max(0, (min_x - margin_x) * width))
            y1 = int(max(0, (min_y - margin_y) * height))
            x2 = int(min(width, (max_x + margin_x) * width))
            y2 = int(min(height, (max_y + margin_y) * height))
            
            return (x1, y1, x2 - x1, y2 - y1)
            
        except Exception as e:
            self.logger.error(f"顔領域抽出エラー: {e}")
            return None
    
    def is_face_frontal(self, face_data: Dict[str, Any], threshold: float = 30.0) -> bool:
        """正面顔かどうかの判定"""
        try:
            if not face_data or 'face_rotation' not in face_data:
                return False
            
            rotation = face_data['face_rotation']
            
            # ヨー角とピッチ角が閾値以下であれば正面顔
            if (abs(rotation['yaw']) < threshold and 
                abs(rotation['pitch']) < threshold):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"正面顔判定エラー: {e}")
            return False
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.face_mesh:
                self.face_mesh.close()
            self.logger.info("顔検出器がクリーンアップされました")
        except Exception as e:
            self.logger.error(f"顔検出器クリーンアップエラー: {e}")
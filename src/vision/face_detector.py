import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

class FaceDetector:
    """MediaPipe Face Mesh 統合検出器（Day 3版）"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # MediaPipe Face Mesh 初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Face Mesh 設定
        face_config = config.get('ai_processing', {}).get('vision', {}).get('face_detection', {})
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=face_config.get('max_num_faces', 1),
            refine_landmarks=face_config.get('refine_landmarks', True),
            min_detection_confidence=face_config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=face_config.get('min_tracking_confidence', 0.5)
        )
        
        # 重要ランドマークインデックス定義
        self.FACE_LANDMARKS = {
            'silhouette': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 308, 415, 310, 311, 312, 13, 82, 78, 14, 15, 16, 17, 18],
            'mouth': [61, 84, 17, 314, 405, 320, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305, 292, 308, 415, 310, 311, 312],
            'right_eyebrow': [276, 283, 282, 295, 285, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        }
        
        # 処理統計
        self.detection_times = []
        self.detection_count = 0
        self.last_detection_result = None
        
        print("✅ Face Detector 初期化完了")
    
    def detect_face(self, frame: np.ndarray) -> Dict:
        """顔検出メイン処理"""
        start_time = time.time()
        
        try:
            # RGB変換（MediaPipe用）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe処理
            results = self.face_mesh.process(rgb_frame)
            
            # 結果解析
            detection_result = self._analyze_detection_results(results, frame.shape)
            
            # 処理時間記録
            processing_time = time.time() - start_time
            self.detection_times.append(processing_time)
            if len(self.detection_times) > 100:
                self.detection_times.pop(0)
            
            self.detection_count += 1
            self.last_detection_result = detection_result
            
            return detection_result
            
        except Exception as e:
            print(f"⚠️  顔検出エラー: {e}")
            return self._get_empty_result()
    
    def _analyze_detection_results(self, results, frame_shape) -> Dict:
        """検出結果解析"""
        height, width = frame_shape[:2]
        
        result = {
            'face_detected': False,
            'face_count': 0,
            'landmarks': None,
            'face_center': None,
            'face_size': 0,
            'face_angle': {'yaw': 0, 'pitch': 0, 'roll': 0},
            'face_distance': 0,
            'confidence': 0.0,
            'processing_time': self.detection_times[-1] if self.detection_times else 0
        }
        
        if results.multi_face_landmarks:
            result['face_detected'] = True
            result['face_count'] = len(results.multi_face_landmarks)
            
            # 最初の顔のみ処理（max_num_faces=1の場合）
            face_landmarks = results.multi_face_landmarks[0]
            result['landmarks'] = face_landmarks
            
            # 顔中心計算（鼻先ランドマーク使用）
            nose_tip = face_landmarks.landmark[1]  # 鼻先
            result['face_center'] = (nose_tip.x, nose_tip.y, nose_tip.z)
            
            # 顔サイズ計算（顔の幅）
            left_face = face_landmarks.landmark[234]  # 左顔端
            right_face = face_landmarks.landmark[454]  # 右顔端
            face_width = abs(right_face.x - left_face.x)
            result['face_size'] = face_width
            
            # 顔の向き推定
            result['face_angle'] = self._estimate_face_pose(face_landmarks)
            
            # 距離推定（顔サイズベース）
            result['face_distance'] = self._estimate_distance(face_width)
            
            # 信頼度（簡易版：ランドマークのvisibilityから）
            confidences = [lm.visibility for lm in face_landmarks.landmark if hasattr(lm, 'visibility')]
            result['confidence'] = sum(confidences) / len(confidences) if confidences else 0.8
        
        return result
    
    def _estimate_face_pose(self, landmarks) -> Dict[str, float]:
        """顔の向き推定（簡易版）"""
        # 主要ランドマーク取得
        nose_tip = landmarks.landmark[1]
        chin = landmarks.landmark[152]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]
        
        # ヨー角（左右回転）- 目の位置から推定
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_offset = nose_tip.x - eye_center_x
        yaw = np.degrees(np.arctan(nose_offset * 2))  # 簡易計算
        
        # ピッチ角（上下回転）- 鼻と顎の位置から推定
        nose_chin_distance = abs(nose_tip.y - chin.y)
        pitch = (0.15 - nose_chin_distance) * 300  # 簡易計算
        
        # ロール角（傾き）- 目のライン傾きから推定
        eye_slope = (right_eye.y - left_eye.y) / (right_eye.x - left_eye.x) if right_eye.x != left_eye.x else 0
        roll = np.degrees(np.arctan(eye_slope))
        
        return {
            'yaw': max(-45, min(45, yaw)),      # -45〜45度に制限
            'pitch': max(-30, min(30, pitch)),  # -30〜30度に制限
            'roll': max(-30, min(30, roll))     # -30〜30度に制限
        }
    
    def _estimate_distance(self, face_width: float) -> float:
        """距離推定（顔幅ベース）"""
        # 実際の平均顔幅: 約14cm
        # カメラから1mの距離での画面上の顔幅を基準とした簡易計算
        if face_width > 0:
            # 基準値は実際のカメラ・レンズ特性に応じて調整が必要
            reference_width = 0.15  # 1mでの基準幅
            estimated_distance = reference_width / face_width
            return max(0.3, min(3.0, estimated_distance))  # 0.3m〜3mに制限
        return 1.0
    
    def _get_empty_result(self) -> Dict:
        """空の結果"""
        return {
            'face_detected': False,
            'face_count': 0,
            'landmarks': None,
            'face_center': None,
            'face_size': 0,
            'face_angle': {'yaw': 0, 'pitch': 0, 'roll': 0},
            'face_distance': 0,
            'confidence': 0.0,
            'processing_time': 0
        }
    
    def draw_landmarks(self, frame: np.ndarray, detection_result: Dict, 
                      draw_all: bool = False) -> np.ndarray:
        """ランドマーク描画"""
        if not detection_result['face_detected']:
            return frame
        
        landmarks = detection_result['landmarks']
        if landmarks is None:
            return frame
        
        try:
            if draw_all:
                # 全ランドマーク描画
                self.mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            else:
                # 主要ランドマークのみ描画
                self._draw_key_landmarks(frame, landmarks)
            
            # 顔情報テキスト描画
            self._draw_face_info(frame, detection_result)
            
        except Exception as e:
            print(f"⚠️  ランドマーク描画エラー: {e}")
        
        return frame
    
    def _draw_key_landmarks(self, frame: np.ndarray, landmarks):
        """主要ランドマークのみ描画"""
        height, width = frame.shape[:2]
        
        # 主要部位の色定義
        colors = {
            'nose': (0, 255, 0),      # 緑
            'mouth': (0, 0, 255),     # 赤
            'left_eye': (255, 0, 0),  # 青
            'right_eye': (255, 0, 0), # 青
            'left_eyebrow': (255, 255, 0),  # シアン
            'right_eyebrow': (255, 255, 0)  # シアン
        }
        
        for part_name, indices in self.FACE_LANDMARKS.items():
            if part_name in colors:
                color = colors[part_name]
                for idx in indices:
                    if idx < len(landmarks.landmark):
                        lm = landmarks.landmark[idx]
                        x = int(lm.x * width)
                        y = int(lm.y * height)
                        cv2.circle(frame, (x, y), 2, color, -1)
    
    def _draw_face_info(self, frame: np.ndarray, detection_result: Dict):
        """顔情報テキスト描画"""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # 距離情報
        distance = detection_result['face_distance']
        cv2.putText(frame, f"Distance: {distance:.2f}m", (10, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        
        # 角度情報
        angles = detection_result['face_angle']
        cv2.putText(frame, f"Yaw: {angles['yaw']:.1f}deg", (10, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        
        cv2.putText(frame, f"Pitch: {angles['pitch']:.1f}deg", (10, y_offset), 
                   font, font_scale, color, thickness)
        y_offset += 25
        
        # 信頼度
        confidence = detection_result['confidence']
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_offset), 
                   font, font_scale, color, thickness)
    
    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計取得"""
        if not self.detection_times:
            return {}
        
        return {
            'avg_detection_time': sum(self.detection_times) / len(self.detection_times),
            'max_detection_time': max(self.detection_times),
            'min_detection_time': min(self.detection_times),
            'detection_count': self.detection_count,
            'fps': 1.0 / (sum(self.detection_times) / len(self.detection_times)) if self.detection_times else 0
        }
    
    def cleanup(self):
        """リソース解放"""
        if self.face_mesh:
            self.face_mesh.close()
        print("Face Detector リソース解放完了")

# テスト実行用
if __name__ == "__main__":
    print("🔍 Face Detector テスト開始...")
    
    # テスト設定
    config = {
        'ai_processing': {
            'vision': {
                'face_detection': {
                    'max_num_faces': 1,
                    'refine_landmarks': True,
                    'min_detection_confidence': 0.7,
                    'min_tracking_confidence': 0.5
                }
            }
        }
    }
    
    detector = FaceDetector(config)
    
    # カメラテスト
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラを開けません")
        exit()
    
    print("📹 顔検出テスト開始（ESCで終了）...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 顔検出
        result = detector.detect_face(frame)
        
        # 結果描画
        frame_with_landmarks = detector.draw_landmarks(frame, result, draw_all=False)
        
        # 結果表示
        cv2.imshow('Face Detection Test', frame_with_landmarks)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキー
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 統計表示
    stats = detector.get_performance_stats()
    print(f"📊 性能統計: {stats}")
    
    detector.cleanup()
    print("✅ Face Detector テスト完了")

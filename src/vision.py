# src/vision.py
import cv2
import mediapipe as mp
import numpy as np

class VisionProcessor:
    """コンピュータビジョン処理クラス"""
    
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.last_detection_result = {}
        
        # MediaPipe初期化
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config['detection']['max_num_faces'],
            refine_landmarks=True,
            min_detection_confidence=config['detection']['face_detection_confidence']
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config['detection']['max_num_hands'],
            min_detection_confidence=config['detection']['hand_detection_confidence']
        )
        
        self._init_camera()
    
    def _init_camera(self):
        """カメラ初期化"""
        camera_config = self.config['camera']
        self.camera = cv2.VideoCapture(camera_config['device_id'])
        
        if not self.camera.isOpened():
            raise RuntimeError("カメラにアクセスできません")
        
        # カメラ設定
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
        self.camera.set(cv2.CAP_PROP_FPS, camera_config['fps'])
        
        print("カメラが初期化されました")
    
    def process_frame(self):
        """フレーム処理"""
        if not self.camera or not self.camera.isOpened():
            return self.last_detection_result
        
        ret, frame = self.camera.read()
        if not ret:
            return self.last_detection_result
        
        # BGR to RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 検出実行
        face_results = self.face_mesh.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # 結果を解析
        detection_result = self._analyze_results(face_results, hand_results, frame.shape)
        self.last_detection_result = detection_result
        
        return detection_result
    
    def _analyze_results(self, face_results, hand_results, frame_shape):
        """検出結果の解析"""
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
                
                # Z距離の推定（簡易版）
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
    
    def get_debug_info(self):
        """デバッグ情報取得"""
        result = self.last_detection_result
        return {
            'Face': 'YES' if result.get('face_detected') else 'NO',
            'Hands': 'YES' if result.get('hands_detected') else 'NO',
            'Face Dist': f"{result.get('face_distance', 0):.3f}",
            'Hand Count': len(result.get('hand_positions', []))
        }
    
    def cleanup(self):
        """リソース解放"""
        if self.camera:
            self.camera.release()
        print("VisionProcessor がクリーンアップされました")
# src/vision/vision_processor_v10.py - MediaPipe 0.10.x完全対応版
import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
from typing import Optional, Dict, Any

class VisionProcessorV10:
    """MediaPipe 0.10.x対応版画像処理クラス - 既存コードと完全互換"""
    
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.last_detection_result = {}
        
        # モデルパス
        self.models_dir = "assets/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 検出器初期化
        self._download_and_init_models()
        self._init_camera()
        
        print("✅ MediaPipe 0.10.x VisionProcessor 初期化完了")
    
    def _download_and_init_models(self):
        """モデルダウンロード・初期化"""
        # 顔検出モデル
        self.face_model_path = os.path.join(self.models_dir, "face_landmarker.task")
        if not os.path.exists(self.face_model_path):
            print("📥 顔検出モデルダウンロード中...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                    self.face_model_path
                )
                print("✅ 顔検出モデルダウンロード完了")
            except Exception as e:
                print(f"❌ 顔検出モデルダウンロード失敗: {e}")
                self.face_model_path = None
        
        # 手検出モデル
        self.hand_model_path = os.path.join(self.models_dir, "hand_landmarker.task")
        if not os.path.exists(self.hand_model_path):
            print("📥 手検出モデルダウンロード中...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    self.hand_model_path
                )
                print("✅ 手検出モデルダウンロード完了")
            except Exception as e:
                print(f"❌ 手検出モデルダウンロード失敗: {e}")
                self.hand_model_path = None
        
        # 検出器初期化
        self._init_face_detector()
        self._init_hand_detector()
    
    def _init_face_detector(self):
        """顔検出器初期化"""
        if not self.face_model_path or not os.path.exists(self.face_model_path):
            self.face_landmarker = None
            return
        
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=self.face_model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=self.config.get('detection', {}).get('max_num_faces', 1)
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            print("✅ 顔検出器初期化成功")
            
        except Exception as e:
            print(f"❌ 顔検出器初期化失敗: {e}")
            self.face_landmarker = None
    
    def _init_hand_detector(self):
        """手検出器初期化"""
        if not self.hand_model_path or not os.path.exists(self.hand_model_path):
            self.hand_landmarker = None
            return
        
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            base_options = python.BaseOptions(model_asset_path=self.hand_model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.config.get('detection', {}).get('max_num_hands', 2)
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
            print("✅ 手検出器初期化成功")
            
        except Exception as e:
            print(f"❌ 手検出器初期化失敗: {e}")
            self.hand_landmarker = None
    
    def _init_camera(self):
        """カメラ初期化（既存コードと同じ）"""
        camera_config = self.config.get('camera', {})
        device_id = camera_config.get('device_id', 0)
        
        self.camera = cv2.VideoCapture(device_id)
        
        if not self.camera.isOpened():
            print(f"❌ カメラ {device_id} にアクセスできません")
            return
        
        # カメラ設定
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('width', 1920))
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('height', 1080))
        self.camera.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
        
        print("✅ カメラ初期化完了")
    
    def process_frame(self):
        """フレーム処理 - 既存APIと完全互換"""
        if not self.camera or not self.camera.isOpened():
            return self.last_detection_result
        
        ret, frame = self.camera.read()
        if not ret:
            return self.last_detection_result
        
        # BGR to RGB変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image作成
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 検出実行
        face_results = self._detect_face(mp_image)
        hand_results = self._detect_hands(mp_image)
        
        # 結果を既存形式に変換
        detection_result = self._convert_to_legacy_format(face_results, hand_results, frame.shape)
        self.last_detection_result = detection_result
        
        return detection_result
    
    def _detect_face(self, mp_image):
        """顔検出"""
        if not self.face_landmarker:
            return None
        
        try:
            return self.face_landmarker.detect(mp_image)
        except Exception as e:
            print(f"顔検出エラー: {e}")
            return None
    
    def _detect_hands(self, mp_image):
        """手検出"""
        if not self.hand_landmarker:
            return None
        
        try:
            return self.hand_landmarker.detect(mp_image)
        except Exception as e:
            print(f"手検出エラー: {e}")
            return None
    
    def _convert_to_legacy_format(self, face_results, hand_results, frame_shape):
        """新API結果を既存形式に変換"""
        result = {
            'face_detected': False,
            'hands_detected': False,
            'face_center': None,
            'face_distance': float('inf'),
            'hand_positions': [],
            'frame_shape': frame_shape,
            'face_landmarks': None,  # 感情認識用
            'hand_landmarks': None   # ジェスチャー認識用
        }
        
        # 顔検出結果変換
        if face_results and face_results.face_landmarks:
            result['face_detected'] = True
            result['face_landmarks'] = face_results  # 感情認識で使用
            
            # 最初の顔のランドマーク取得
            if len(face_results.face_landmarks) > 0:
                landmarks = face_results.face_landmarks[0]
                
                # 鼻の先端（ランドマーク1）を中心点とする
                if len(landmarks) > 1:
                    nose_tip = landmarks[1]
                    result['face_center'] = (nose_tip.x, nose_tip.y, nose_tip.z)
                    result['face_distance'] = abs(nose_tip.z)
        
        # 手検出結果変換
        if hand_results and hand_results.hand_landmarks:
            result['hands_detected'] = True
            result['hand_landmarks'] = hand_results  # ジェスチャー認識で使用
            
            for hand_landmarks in hand_results.hand_landmarks:
                if len(hand_landmarks) > 0:
                    # 手首（ランドマーク0）の座標を取得
                    wrist = hand_landmarks[0]
                    result['hand_positions'].append((wrist.x, wrist.y))
        
        return result
    
    def get_debug_info(self):
        """デバッグ情報取得 - 既存APIと互換"""
        result = self.last_detection_result
        return {
            'Face': 'YES' if result.get('face_detected') else 'NO',
            'Hands': 'YES' if result.get('hands_detected') else 'NO',
            'Face Dist': f"{result.get('face_distance', 0):.3f}",
            'Hand Count': len(result.get('hand_positions', [])),
            'API Version': 'MediaPipe 0.10.x',
            'Face Model': 'OK' if self.face_landmarker else 'FAILED',
            'Hand Model': 'OK' if self.hand_landmarker else 'FAILED'
        }
    
    def cleanup(self):
        """リソース解放 - 既存APIと互換"""
        if self.camera:
            self.camera.release()
        print("VisionProcessorV10 がクリーンアップされました")
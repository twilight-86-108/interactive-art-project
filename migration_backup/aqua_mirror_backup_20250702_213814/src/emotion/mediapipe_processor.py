"""
MediaPipe処理基盤
顔・手検出・ランドマーク抽出
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
import time
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

class ProcessingMode(Enum):
    """処理モード"""
    FACE_ONLY = "face_only"
    HANDS_ONLY = "hands_only"
    FACE_AND_HANDS = "face_and_hands"

class MediaPipeProcessor:
    """
    MediaPipe統合処理システム
    顔メッシュ・手検出・ランドマーク抽出
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("MediaPipeProcessor")
        
        # MediaPipeソリューション
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 処理オブジェクト
        self.face_mesh = None
        self.hands = None
        
        # 設定
        self.face_confidence = config.get('ai.mediapipe.face_detection_confidence', 0.7)
        self.face_tracking = config.get('ai.mediapipe.face_tracking_confidence', 0.5)
        self.hand_confidence = config.get('ai.mediapipe.hand_detection_confidence', 0.7)
        self.hand_tracking = config.get('ai.mediapipe.hand_tracking_confidence', 0.5)
        
        # 処理モード
        self.processing_mode = ProcessingMode.FACE_AND_HANDS
        
        # 統計情報
        self.face_detection_count = 0
        self.hand_detection_count = 0
        self.processing_time_total = 0.0
        self.frame_count = 0
        
        # 初期化
        self._initialize_mediapipe()
    
    def _initialize_mediapipe(self):
        """MediaPipe初期化"""
        try:
            # 顔メッシュ初期化
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,  # 1人の顔のみ
                refine_landmarks=True,  # 詳細ランドマーク
                min_detection_confidence=self.face_confidence,
                min_tracking_confidence=self.face_tracking
            )
            
            # 手検出初期化
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.hand_confidence,
                min_tracking_confidence=self.hand_tracking
            )
            
            self.logger.info("✅ MediaPipe初期化完了")
            
        except Exception as e:
            self.logger.error(f"❌ MediaPipe初期化失敗: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        フレーム処理・特徴抽出
        
        Args:
            frame: RGB画像 (H, W, 3)
            
        Returns:
            Dict: 検出結果
        """
        if frame is None:
            return self._empty_result()
        
        processing_start = time.time()
        
        try:
            # MediaPipe用に変換（RGB）
            if frame.shape[2] == 3:
                rgb_frame = frame
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = {}
            
            # 顔検出処理
            if self.processing_mode in [ProcessingMode.FACE_ONLY, ProcessingMode.FACE_AND_HANDS]:
                face_results = self._process_face(rgb_frame)
                results.update(face_results)
            
            # 手検出処理
            if self.processing_mode in [ProcessingMode.HANDS_ONLY, ProcessingMode.FACE_AND_HANDS]:
                hand_results = self._process_hands(rgb_frame)
                results.update(hand_results)
            
            # 統計更新
            processing_time = time.time() - processing_start
            self._update_stats(processing_time, results)
            
            results['processing_time'] = processing_time
            results['frame_count'] = self.frame_count
            
            return results
            
        except Exception as e:
            self.logger.error(f"フレーム処理エラー: {e}")
            return self._empty_result()
    
    def _process_face(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """顔メッシュ処理"""
        try:
            face_results = self.face_mesh.process(rgb_frame)
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # ランドマーク座標抽出
                landmarks = self._extract_face_landmarks(face_landmarks, rgb_frame.shape)
                
                # 重要特徴点抽出
                key_points = self._extract_face_key_points(landmarks)
                
                self.face_detection_count += 1
                
                return {
                    'face_detected': True,
                    'face_landmarks': landmarks,
                    'face_key_points': key_points,
                    'face_confidence': 1.0  # MediaPipeは信頼度を直接提供しない
                }
            else:
                return {
                    'face_detected': False,
                    'face_landmarks': None,
                    'face_key_points': None,
                    'face_confidence': 0.0
                }
                
        except Exception as e:
            self.logger.error(f"顔処理エラー: {e}")
            return {
                'face_detected': False,
                'face_landmarks': None,
                'face_key_points': None,
                'face_confidence': 0.0
            }
    
    def _process_hands(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """手検出処理"""
        try:
            hand_results = self.hands.process(rgb_frame)
            
            detected_hands = []
            
            if hand_results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    # 手の種類取得（左手/右手）
                    handedness = hand_results.multi_handedness[i].classification[0]
                    hand_type = handedness.label  # "Left" or "Right"
                    hand_confidence = handedness.score
                    
                    # ランドマーク座標抽出
                    landmarks = self._extract_hand_landmarks(hand_landmarks, rgb_frame.shape)
                    
                    # ジェスチャー基本分析
                    gesture_info = self._analyze_basic_gesture(landmarks)
                    
                    detected_hands.append({
                        'hand_type': hand_type,
                        'landmarks': landmarks,
                        'confidence': hand_confidence,
                        'gesture_info': gesture_info
                    })
                
                self.hand_detection_count += len(detected_hands)
            
            return {
                'hands_detected': len(detected_hands) > 0,
                'hands_count': len(detected_hands),
                'hands': detected_hands
            }
            
        except Exception as e:
            self.logger.error(f"手処理エラー: {e}")
            return {
                'hands_detected': False,
                'hands_count': 0,
                'hands': []
            }
    
    def _extract_face_landmarks(self, face_landmarks, frame_shape) -> np.ndarray:
        """顔ランドマーク座標抽出"""
        height, width = frame_shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # 相対深度
            landmarks.append([x, y, z])
        
        return np.array(landmarks)
    
    def _extract_hand_landmarks(self, hand_landmarks, frame_shape) -> np.ndarray:
        """手ランドマーク座標抽出"""
        height, width = frame_shape[:2]
        landmarks = []
        
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # 相対深度
            landmarks.append([x, y, z])
        
        return np.array(landmarks)
    
    def _extract_face_key_points(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """顔重要特徴点抽出"""
        if landmarks is None or len(landmarks) == 0:
            return {}
        
        # MediaPipe 468点ランドマークの重要インデックス
        key_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
        
        key_points = {}
        for feature_name, indices in key_indices.items():
            try:
                feature_points = landmarks[indices]
                key_points[feature_name] = feature_points
            except IndexError:
                # インデックスが範囲外の場合はスキップ
                continue
        
        return key_points
    
    def _analyze_basic_gesture(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """基本ジェスチャー分析"""
        if landmarks is None or len(landmarks) < 21:
            return {'gesture': 'unknown', 'confidence': 0.0}
        
        # 指先と指の根元のインデックス
        finger_tips = [4, 8, 12, 16, 20]  # 親指、人差し指、中指、薬指、小指
        finger_bases = [3, 6, 10, 14, 18]
        
        # 指の状態判定
        fingers_up = []
        
        # 親指（X軸での判定）
        if landmarks[4][0] > landmarks[3][0]:  # 右手の場合
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # 他の指（Y軸での判定）
        for i in range(1, 5):
            if landmarks[finger_tips[i]][1] < landmarks[finger_bases[i]][1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # ジェスチャー判定
        total_fingers = sum(fingers_up)
        
        if total_fingers == 0:
            gesture = 'fist'
        elif total_fingers == 1 and fingers_up[1] == 1:
            gesture = 'point'
        elif total_fingers == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:
            gesture = 'peace'
        elif total_fingers == 5:
            gesture = 'open_hand'
        else:
            gesture = 'other'
        
        return {
            'gesture': gesture,
            'fingers_up': fingers_up,
            'total_fingers': total_fingers,
            'confidence': 0.8  # 簡易信頼度
        }
    
    def _update_stats(self, processing_time: float, results: Dict[str, Any]):
        """統計情報更新"""
        self.processing_time_total += processing_time
        self.frame_count += 1
    
    def _empty_result(self) -> Dict[str, Any]:
        """空の結果"""
        return {
            'face_detected': False,
            'face_landmarks': None,
            'face_key_points': None,
            'face_confidence': 0.0,
            'hands_detected': False,
            'hands_count': 0,
            'hands': [],
            'processing_time': 0.0,
            'frame_count': self.frame_count
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        if self.frame_count > 0:
            avg_processing_time = self.processing_time_total / self.frame_count
        else:
            avg_processing_time = 0.0
        
        return {
            'total_frames': self.frame_count,
            'face_detections': self.face_detection_count,
            'hand_detections': self.hand_detection_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'processing_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        }
    
    def set_processing_mode(self, mode: ProcessingMode):
        """処理モード設定"""
        self.processing_mode = mode
        self.logger.info(f"処理モード変更: {mode.value}")
    
    def cleanup(self):
        """リソース解放"""
        if self.face_mesh:
            self.face_mesh.close()
        
        if self.hands:
            self.hands.close()
        
        self.logger.info("✅ MediaPipe リソース解放完了")

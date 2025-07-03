"""
高度ジェスチャー認識システム
複数手・複雑ジェスチャー・GPU連動
"""

import numpy as np
import cv2
import mediapipe as mp
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time

class GestureType(Enum):
    """ジェスチャータイプ列挙"""
    POINT = "point"
    WAVE = "wave"
    PEACE = "peace"
    FIST = "fist"
    OPEN_PALM = "open_palm"
    THUMBS_UP = "thumbs_up"
    OK_SIGN = "ok_sign"
    ROCK = "rock"
    HEART = "heart"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    PINCH = "pinch"
    SPREAD = "spread"

@dataclass
class HandGesture:
    """手ジェスチャーデータ"""
    gesture_type: GestureType
    confidence: float
    position: Tuple[float, float]  # 正規化座標
    hand_side: str  # "Left" or "Right"
    landmarks: List[Tuple[float, float]]
    bounding_box: Tuple[float, float, float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    size: float = 1.0

class GestureRecognizer:
    """
    高度ジェスチャー認識システム
    MediaPipe Hands + カスタム分類器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("GestureRecognizer")
        self.config = config
        
        # MediaPipe Hands設定
        ai_config = config.get('ai', {})
        mediapipe_config = ai_config.get('mediapipe', {})
        
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=mediapipe_config.get('hand_detection_confidence', 0.7),
            min_tracking_confidence=mediapipe_config.get('hand_tracking_confidence', 0.5)
        )
        
        # ジェスチャー履歴（軌跡追跡用）
        self.gesture_history: List[List[HandGesture]] = []
        self.max_history = 30  # 1秒分（30FPS）
        
        # 動的ジェスチャー検出
        self.motion_detector = MotionGestureDetector()
        
        # 指ランドマークインデックス
        self.finger_landmarks = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # ジェスチャー分類器
        self.gesture_classifiers = {
            GestureType.POINT: self._classify_pointing,
            GestureType.WAVE: self._classify_waving,
            GestureType.PEACE: self._classify_peace,
            GestureType.FIST: self._classify_fist,
            GestureType.OPEN_PALM: self._classify_open_palm,
            GestureType.THUMBS_UP: self._classify_thumbs_up,
            GestureType.OK_SIGN: self._classify_ok_sign,
            GestureType.ROCK: self._classify_rock,
            GestureType.HEART: self._classify_heart
        }
        
        self.logger.info("👋 ジェスチャー認識システム初期化完了")
    
    def recognize_gestures(self, frame: np.ndarray) -> List[HandGesture]:
        """フレームからジェスチャー認識"""
        try:
            if frame is None:
                return []
            
            # BGR → RGB変換
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe手検出
            results = self.hands.process(rgb_frame)
            
            gestures = []
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 手の基本情報
                    hand_side = handedness.classification[0].label
                    
                    # ランドマーク座標正規化
                    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    
                    # 手の位置・バウンディングボックス
                    position, bounding_box, size = self._calculate_hand_properties(landmarks)
                    
                    # 静的ジェスチャー認識
                    static_gesture = self._recognize_static_gesture(landmarks, hand_side)
                    
                    if static_gesture:
                        gesture = HandGesture(
                            gesture_type=static_gesture['type'],
                            confidence=static_gesture['confidence'],
                            position=position,
                            hand_side=hand_side,
                            landmarks=landmarks,
                            bounding_box=bounding_box,
                            size=size
                        )
                        
                        # 速度計算（前フレームとの比較）
                        gesture.velocity = self._calculate_velocity(gesture)
                        
                        gestures.append(gesture)
            
            # 動的ジェスチャー検出（波手、スワイプなど）
            dynamic_gestures = self.motion_detector.detect_motion_gestures(gestures, self.gesture_history)
            gestures.extend(dynamic_gestures)
            
            # 履歴更新
            self.gesture_history.append(gestures)
            if len(self.gesture_history) > self.max_history:
                self.gesture_history.pop(0)
            
            return gestures
            
        except Exception as e:
            self.logger.error(f"❌ ジェスチャー認識失敗: {e}")
            return []
    
    def _calculate_hand_properties(self, landmarks: List[Tuple[float, float]]) -> Tuple[Tuple[float, float], Tuple[float, float, float, float], float]:
        """手の位置・サイズ計算"""
        try:
            # 手の中心位置（手首基準）
            wrist = landmarks[0]
            
            # バウンディングボックス
            x_coords = [lm[0] for lm in landmarks]
            y_coords = [lm[1] for lm in landmarks]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            bounding_box = (min_x, min_y, max_x - min_x, max_y - min_y)
            
            # 手のサイズ（対角線距離）
            size = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
            
            return wrist, bounding_box, size
            
        except Exception:
            return (0.5, 0.5), (0.0, 0.0, 0.1, 0.1), 0.1
    
    def _recognize_static_gesture(self, landmarks: List[Tuple[float, float]], hand_side: str) -> Optional[Dict[str, Any]]:
        """静的ジェスチャー認識"""
        try:
            best_gesture = None
            best_confidence = 0.0
            
            # 各ジェスチャー分類器で評価
            for gesture_type, classifier in self.gesture_classifiers.items():
                confidence = classifier(landmarks, hand_side)
                
                if confidence > best_confidence and confidence > 0.6:
                    best_confidence = confidence
                    best_gesture = {
                        'type': gesture_type,
                        'confidence': confidence
                    }
            
            return best_gesture
            
        except Exception as e:
            self.logger.error(f"❌ 静的ジェスチャー認識失敗: {e}")
            return None
    
    def _classify_pointing(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """指差しジェスチャー分類"""
        try:
            # 人差し指が伸びて、他の指が曲がっている
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            index_mcp = landmarks[5]
            
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # 人差し指が伸びているか
            index_extended = index_tip[1] < index_pip[1] < index_mcp[1]
            
            # 他の指が曲がっているか
            middle_folded = middle_tip[1] > middle_pip[1]
            ring_folded = ring_tip[1] > ring_pip[1]
            pinky_folded = pinky_tip[1] > pinky_pip[1]
            
            if index_extended and middle_folded and ring_folded and pinky_folded:
                return 0.9
            elif index_extended and (middle_folded or ring_folded):
                return 0.7
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _classify_open_palm(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """開いた手のひら分類"""
        try:
            extended_fingers = 0
            
            # 各指が伸びているかチェック
            for finger, indices in self.finger_landmarks.items():
                if finger == 'thumb':
                    continue  # 親指は別途処理
                
                tip = landmarks[indices[3]]
                pip = landmarks[indices[2]]
                mcp = landmarks[indices[1]]
                
                # 指が伸びているか（tip < pip < mcp のY座標）
                if tip[1] < pip[1] < mcp[1]:
                    extended_fingers += 1
            
            # 親指チェック（横方向の伸び）
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            thumb_extended = abs(thumb_tip[0] - thumb_mcp[0]) > 0.05
            
            if thumb_extended:
                extended_fingers += 1
            
            # 5本指すべて伸びていれば開いた手のひら
            confidence = extended_fingers / 5.0
            return confidence if confidence > 0.8 else 0.0
            
        except Exception:
            return 0.0
    
    def _classify_fist(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """拳分類"""
        try:
            folded_fingers = 0
            
            # 各指が曲がっているかチェック
            for finger, indices in self.finger_landmarks.items():
                if finger == 'thumb':
                    continue
                
                tip = landmarks[indices[3]]
                pip = landmarks[indices[2]]
                
                # 指が曲がっているか
                if tip[1] > pip[1]:
                    folded_fingers += 1
            
            # 親指も曲がっているかチェック
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            thumb_folded = abs(thumb_tip[0] - thumb_mcp[0]) < 0.03
            
            if thumb_folded:
                folded_fingers += 1
            
            confidence = folded_fingers / 5.0
            return confidence if confidence > 0.8 else 0.0
            
        except Exception:
            return 0.0
    
    def _classify_peace(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """ピースサイン分類"""
        try:
            # 人差し指と中指が伸びて、他が曲がっている
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # 人差し指・中指が伸びているか
            index_extended = index_tip[1] < index_pip[1]
            middle_extended = middle_tip[1] < middle_pip[1]
            
            # 薬指・小指が曲がっているか
            ring_folded = ring_tip[1] > ring_pip[1]
            pinky_folded = pinky_tip[1] > pinky_pip[1]
            
            if index_extended and middle_extended and ring_folded and pinky_folded:
                return 0.9
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _classify_thumbs_up(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """サムズアップ分類"""
        try:
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            
            # 他の指の状態
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            
            # 親指が上向き
            thumb_up = thumb_tip[1] < thumb_mcp[1]
            
            # 他の指が曲がっている
            fingers_folded = all(
                landmarks[self.finger_landmarks[finger][3]][1] > landmarks[self.finger_landmarks[finger][2]][1]
                for finger in ['index', 'middle', 'ring', 'pinky']
            )
            
            if thumb_up and fingers_folded:
                return 0.9
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _classify_ok_sign(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """OKサイン分類"""
        try:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            
            # 親指と人差し指の距離（円を作っているか）
            distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
            
            # 他の指が伸びているか
            other_fingers_extended = all(
                landmarks[self.finger_landmarks[finger][3]][1] < landmarks[self.finger_landmarks[finger][2]][1]
                for finger in ['middle', 'ring', 'pinky']
            )
            
            if distance < 0.05 and other_fingers_extended:
                return 0.9
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _classify_rock(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """ロックサイン分類"""
        try:
            # 人差し指・小指が伸びて、中指・薬指が曲がっている
            index_extended = landmarks[8][1] < landmarks[6][1]
            pinky_extended = landmarks[20][1] < landmarks[18][1]
            
            middle_folded = landmarks[12][1] > landmarks[10][1]
            ring_folded = landmarks[16][1] > landmarks[14][1]
            
            if index_extended and pinky_extended and middle_folded and ring_folded:
                return 0.9
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _classify_heart(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """ハートサイン分類（両手必要）"""
        # 単一手では判定困難、将来の両手ジェスチャーで実装
        return 0.0
    
    def _classify_waving(self, landmarks: List[Tuple[float, float]], hand_side: str) -> float:
        """手振り分類（動的ジェスチャー）"""
        # 動的ジェスチャーは motion_detector で処理
        return 0.0
    
    def _calculate_velocity(self, current_gesture: HandGesture) -> Tuple[float, float]:
        """ジェスチャー速度計算"""
        try:
            if not self.gesture_history:
                return (0.0, 0.0)
            
            # 前フレームの同じ手を探す
            previous_gestures = self.gesture_history[-1] if self.gesture_history else []
            
            for prev_gesture in previous_gestures:
                if prev_gesture.hand_side == current_gesture.hand_side:
                    # 位置の差分から速度計算
                    dx = current_gesture.position[0] - prev_gesture.position[0]
                    dy = current_gesture.position[1] - prev_gesture.position[1]
                    return (dx, dy)
            
            return (0.0, 0.0)
            
        except Exception:
            return (0.0, 0.0)
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("🧹 ジェスチャー認識システムリソース解放中...")
        
        try:
            if hasattr(self.hands, 'close'):
                self.hands.close()
            
            self.logger.info("✅ ジェスチャー認識システムリソース解放完了")
            
        except Exception as e:
            self.logger.error(f"❌ ジェスチャー認識システムリソース解放失敗: {e}")


class MotionGestureDetector:
    """動的ジェスチャー検出器"""
    
    def __init__(self):
        self.wave_detector = WaveDetector()
        self.swipe_detector = SwipeDetector()
    
    def detect_motion_gestures(self, current_gestures: List[HandGesture], 
                             gesture_history: List[List[HandGesture]]) -> List[HandGesture]:
        """動的ジェスチャー検出"""
        motion_gestures = []
        
        try:
            # 手振り検出
            wave_gestures = self.wave_detector.detect(current_gestures, gesture_history)
            motion_gestures.extend(wave_gestures)
            
            # スワイプ検出
            swipe_gestures = self.swipe_detector.detect(current_gestures, gesture_history)
            motion_gestures.extend(swipe_gestures)
            
            return motion_gestures
            
        except Exception as e:
            return []


class WaveDetector:
    """手振り検出器"""
    
    def detect(self, current_gestures: List[HandGesture], 
              gesture_history: List[List[HandGesture]]) -> List[HandGesture]:
        """手振り検出"""
        wave_gestures = []
        
        try:
            if len(gesture_history) < 10:
                return wave_gestures
            
            for current_gesture in current_gestures:
                if current_gesture.gesture_type == GestureType.OPEN_PALM:
                    # 過去のフレームで同じ手の軌跡を追跡
                    positions = self._get_hand_trajectory(current_gesture.hand_side, gesture_history)
                    
                    if self._is_waving_motion(positions):
                        # 手振りジェスチャー作成
                        wave_gesture = HandGesture(
                            gesture_type=GestureType.WAVE,
                            confidence=0.8,
                            position=current_gesture.position,
                            hand_side=current_gesture.hand_side,
                            landmarks=current_gesture.landmarks,
                            bounding_box=current_gesture.bounding_box,
                            size=current_gesture.size
                        )
                        wave_gestures.append(wave_gesture)
            
            return wave_gestures
            
        except Exception:
            return []
    
    def _get_hand_trajectory(self, hand_side: str, gesture_history: List[List[HandGesture]]) -> List[Tuple[float, float]]:
        """手の軌跡取得"""
        positions = []
        
        for gestures in gesture_history[-10:]:  # 最新10フレーム
            for gesture in gestures:
                if gesture.hand_side == hand_side:
                    positions.append(gesture.position)
                    break
        
        return positions
    
    def _is_waving_motion(self, positions: List[Tuple[float, float]]) -> bool:
        """手振り動作判定"""
        if len(positions) < 6:
            return False
        
        # X座標の変化パターンを分析
        x_positions = [pos[0] for pos in positions]
        
        # 左右の往復運動を検出
        direction_changes = 0
        for i in range(1, len(x_positions) - 1):
            if ((x_positions[i] > x_positions[i-1]) != (x_positions[i+1] > x_positions[i])):
                direction_changes += 1
        
        # 2回以上の方向転換があれば手振り
        return direction_changes >= 2


class SwipeDetector:
    """スワイプジェスチャー検出器"""
    
    def detect(self, current_gestures: List[HandGesture], 
              gesture_history: List[List[HandGesture]]) -> List[HandGesture]:
        """スワイプ検出"""
        swipe_gestures = []
        
        try:
            if len(gesture_history) < 5:
                return swipe_gestures
            
            for current_gesture in current_gestures:
                # 速度ベースのスワイプ検出
                if abs(current_gesture.velocity[0]) > 0.1:  # 水平方向の高速移動
                    swipe_type = GestureType.SWIPE_RIGHT if current_gesture.velocity[0] > 0 else GestureType.SWIPE_LEFT
                    
                    swipe_gesture = HandGesture(
                        gesture_type=swipe_type,
                        confidence=min(abs(current_gesture.velocity[0]) * 5, 1.0),
                        position=current_gesture.position,
                        hand_side=current_gesture.hand_side,
                        landmarks=current_gesture.landmarks,
                        bounding_box=current_gesture.bounding_box,
                        velocity=current_gesture.velocity,
                        size=current_gesture.size
                    )
                    swipe_gestures.append(swipe_gesture)
            
            return swipe_gestures
            
        except Exception:
            return []
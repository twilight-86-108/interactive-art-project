import cv2
import numpy as np
import math
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

class GestureType(Enum):
    # 静的ジェスチャー
    POINT = "point"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    FIST = "fist"
    OPEN_HAND = "open_hand"
    
    # 動的ジェスチャー
    WAVE = "wave"
    CIRCLE = "circle"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    
    # 両手ジェスチャー
    CLAP = "clap"
    HEART = "heart"
    UNKNOWN = "unknown"

class GestureEvent:
    """ジェスチャーイベントクラス"""
    def __init__(self, gesture_type: GestureType, confidence: float, 
                 position: Tuple[float, float], hand_label: str = "Unknown",
                 additional_data: Optional[Dict] = None):
        self.gesture_type = gesture_type
        self.confidence = confidence
        self.position = position
        self.hand_label = hand_label
        self.timestamp = time.time()
        self.additional_data = additional_data or {}

class AdvancedGestureRecognizer:
    """高度ジェスチャー認識システム（Day 4版）"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 設定パラメータ
        gesture_config = config.get('ai_processing', {}).get('gesture', {})
        self.recognition_timeout = gesture_config.get('recognition_timeout', 2.0)
        self.min_confidence = gesture_config.get('min_confidence', 0.7)
        
        # 手の軌跡履歴
        self.hand_trails = {
            'Left': deque(maxlen=30),   # 1秒分の履歴（30FPS想定）
            'Right': deque(maxlen=30)
        }
        
        # ジェスチャー状態管理
        self.gesture_states = {}
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # 0.5秒のクールダウン
        
        # 動的ジェスチャー閾値
        self.thresholds = {
            'wave_min_oscillations': 2,
            'wave_amplitude_threshold': 0.05,
            'circle_min_angle': math.pi * 1.5,  # 270度
            'circle_max_deviation': 0.15,
            'swipe_min_distance': 0.2,
            'swipe_max_time': 1.0
        }
        
        print("✅ Advanced Gesture Recognizer 初期化完了")
    
    def recognize_gestures(self, hand_detection_result: Dict) -> List[GestureEvent]:
        """高度ジェスチャー認識メイン処理"""
        current_time = time.time()
        detected_gestures = []
        
        if not hand_detection_result.get('hands_detected'):
            return detected_gestures
        
        # 各手のジェスチャー認識
        for hand_info in hand_detection_result['hands']:
            hand_label = hand_info['label']
            landmarks = hand_info['landmarks']
            
            # 手の軌跡更新
            self._update_hand_trail(hand_label, hand_info['wrist_position'], current_time)
            
            # 静的ジェスチャー認識
            static_gestures = self._recognize_static_gestures(hand_info)
            detected_gestures.extend(static_gestures)
            
            # 動的ジェスチャー認識
            dynamic_gestures = self._recognize_dynamic_gestures(hand_label, current_time)
            detected_gestures.extend(dynamic_gestures)
        
        # 両手ジェスチャー認識
        if len(hand_detection_result['hands']) == 2:
            two_hand_gestures = self._recognize_two_hand_gestures(hand_detection_result['hands'])
            detected_gestures.extend(two_hand_gestures)
        
        # クールダウン適用
        filtered_gestures = self._apply_cooldown(detected_gestures, current_time)
        
        return filtered_gestures
    
    def _update_hand_trail(self, hand_label: str, position: Tuple[float, float], timestamp: float):
        """手の軌跡更新"""
        trail_point = {
            'position': position,
            'timestamp': timestamp
        }
        self.hand_trails[hand_label].append(trail_point)
    
    def _recognize_static_gestures(self, hand_info: Dict) -> List[GestureEvent]:
        """静的ジェスチャー認識"""
        gestures = []
        finger_states = hand_info.get('finger_states', {})
        landmarks = hand_info['landmarks']
        hand_label = hand_info['label']
        wrist_pos = hand_info['wrist_position']
        
        # 基本的な手の形状認識
        extended_fingers = [finger for finger, extended in finger_states.items() if extended]
        extended_count = len(extended_fingers)
        
        gesture_type = None
        confidence = 0.8
        
        if extended_count == 0:
            gesture_type = GestureType.FIST
        elif extended_count == 1:
            if finger_states.get('index', False):
                gesture_type = GestureType.POINT
            elif finger_states.get('thumb', False):
                gesture_type = GestureType.THUMBS_UP
        elif extended_count == 2:
            if finger_states.get('index', False) and finger_states.get('middle', False):
                # ピースサインの詳細検証
                if self._verify_peace_sign(landmarks):
                    gesture_type = GestureType.PEACE
                    confidence = 0.9
        elif extended_count == 5:
            gesture_type = GestureType.OPEN_HAND
        
        if gesture_type:
            gesture_event = GestureEvent(
                gesture_type=gesture_type,
                confidence=confidence,
                position=wrist_pos,
                hand_label=hand_label
            )
            gestures.append(gesture_event)
        
        return gestures
    
    def _verify_peace_sign(self, landmarks) -> bool:
        """ピースサインの詳細検証"""
        try:
            # 人差し指と中指の角度確認
            index_tip = landmarks.landmark[8]
            index_mcp = landmarks.landmark[5]
            middle_tip = landmarks.landmark[12]
            middle_mcp = landmarks.landmark[9]
            
            # 指の方向ベクトル
            index_vector = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])
            middle_vector = np.array([middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y])
            
            # 角度計算
            dot_product = np.dot(index_vector, middle_vector)
            norms = np.linalg.norm(index_vector) * np.linalg.norm(middle_vector)
            
            if norms > 0:
                angle = np.arccos(np.clip(dot_product / norms, -1.0, 1.0))
                # 20-60度の範囲でピースサインと判定
                return math.pi/9 < angle < math.pi/3
            
        except Exception as e:
            print(f"⚠️  ピースサイン検証エラー: {e}")
        
        return False
    
    def _recognize_dynamic_gestures(self, hand_label: str, current_time: float) -> List[GestureEvent]:
        """動的ジェスチャー認識"""
        gestures = []
        trail = self.hand_trails[hand_label]
        
        if len(trail) < 10:  # 最低10フレーム必要
            return gestures
        
        # 手振り検出
        wave_gesture = self._detect_wave_gesture(trail)
        if wave_gesture:
            gestures.append(wave_gesture)
        
        # 円描画検出
        circle_gesture = self._detect_circle_gesture(trail)
        if circle_gesture:
            gestures.append(circle_gesture)
        
        # スワイプ検出
        swipe_gesture = self._detect_swipe_gesture(trail)
        if swipe_gesture:
            gestures.append(swipe_gesture)
        
        return gestures
    
    def _detect_wave_gesture(self, trail: deque) -> Optional[GestureEvent]:
        """手振りジェスチャー検出"""
        if len(trail) < 15:
            return None
        
        # X方向の位置変化を分析
        positions = [point['position'][0] for point in trail]
        
        # 極値（山と谷）検出
        peaks = []
        valleys = []
        
        for i in range(1, len(positions) - 1):
            if positions[i] > positions[i-1] and positions[i] > positions[i+1]:
                peaks.append(i)
            elif positions[i] < positions[i-1] and positions[i] < positions[i+1]:
                valleys.append(i)
        
        # 振動回数確認
        oscillations = min(len(peaks), len(valleys))
        
        if oscillations >= self.thresholds['wave_min_oscillations']:
            # 振幅確認
            if peaks and valleys:
                max_amplitude = max([positions[p] for p in peaks]) - min([positions[v] for v in valleys])
                
                if max_amplitude >= self.thresholds['wave_amplitude_threshold']:
                    center_position = (
                        sum([point['position'][0] for point in trail]) / len(trail),
                        sum([point['position'][1] for point in trail]) / len(trail)
                    )
                    
                    return GestureEvent(
                        gesture_type=GestureType.WAVE,
                        confidence=0.8,
                        position=center_position,
                        additional_data={'oscillations': oscillations, 'amplitude': max_amplitude}
                    )
        
        return None
    
    def _detect_circle_gesture(self, trail: deque) -> Optional[GestureEvent]:
        """円描画ジェスチャー検出"""
        if len(trail) < 20:
            return None
        
        positions = [(point['position'][0], point['position'][1]) for point in trail]
        
        # 重心計算
        center_x = sum(pos[0] for pos in positions) / len(positions)
        center_y = sum(pos[1] for pos in positions) / len(positions)
        center = (center_x, center_y)
        
        # 各点の角度計算
        angles = []
        for pos in positions:
            angle = math.atan2(pos[1] - center_y, pos[0] - center_x)
            angles.append(angle)
        
        # 角度の連続性を保つため調整
        adjusted_angles = [angles[0]]
        for i in range(1, len(angles)):
            angle_diff = angles[i] - adjusted_angles[-1]
            
            # 2πを超える変化を補正
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            adjusted_angles.append(adjusted_angles[-1] + angle_diff)
        
        # 総回転角度計算
        total_rotation = abs(adjusted_angles[-1] - adjusted_angles[0])
        
        if total_rotation >= self.thresholds['circle_min_angle']:
            # 円形度の検証（中心からの距離の分散）
            distances = [math.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2) for pos in positions]
            avg_distance = sum(distances) / len(distances)
            distance_variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
            circle_quality = 1.0 / (1.0 + distance_variance * 100)  # 正規化された品質
            
            if circle_quality > 0.5:  # 品質閾値
                return GestureEvent(
                    gesture_type=GestureType.CIRCLE,
                    confidence=circle_quality,
                    position=center,
                    additional_data={'rotation': total_rotation, 'quality': circle_quality}
                )
        
        return None
    
    def _detect_swipe_gesture(self, trail: deque) -> Optional[GestureEvent]:
        """スワイプジェスチャー検出"""
        if len(trail) < 5:
            return None
        
        start_pos = trail[0]['position']
        end_pos = trail[-1]['position']
        start_time = trail[0]['timestamp']
        end_time = trail[-1]['timestamp']
        
        # 移動距離と時間
        distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        duration = end_time - start_time
        
        if distance >= self.thresholds['swipe_min_distance'] and duration <= self.thresholds['swipe_max_time']:
            # 方向判定
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            
            if abs(dx) > abs(dy):  # 水平方向のスワイプ
                gesture_type = GestureType.SWIPE_RIGHT if dx > 0 else GestureType.SWIPE_LEFT
                center_position = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
                
                return GestureEvent(
                    gesture_type=gesture_type,
                    confidence=0.8,
                    position=center_position,
                    additional_data={'distance': distance, 'duration': duration}
                )
        
        return None
    
    def _recognize_two_hand_gestures(self, hands: List[Dict]) -> List[GestureEvent]:
        """両手ジェスチャー認識"""
        gestures = []
        
        if len(hands) != 2:
            return gestures
        
        left_hand = None
        right_hand = None
        
        for hand in hands:
            if hand['label'] == 'Left':
                left_hand = hand
            else:
                right_hand = hand
        
        if not (left_hand and right_hand):
            return gestures
        
        # 拍手検出
        clap_gesture = self._detect_clap_gesture(left_hand, right_hand)
        if clap_gesture:
            gestures.append(clap_gesture)
        
        # ハート形検出
        heart_gesture = self._detect_heart_gesture(left_hand, right_hand)
        if heart_gesture:
            gestures.append(heart_gesture)
        
        return gestures
    
    def _detect_clap_gesture(self, left_hand: Dict, right_hand: Dict) -> Optional[GestureEvent]:
        """拍手ジェスチャー検出"""
        left_pos = left_hand['wrist_position']
        right_pos = right_hand['wrist_position']
        
        # 手の距離
        distance = math.sqrt((left_pos[0] - right_pos[0])**2 + (left_pos[1] - right_pos[1])**2)
        
        # 両手が開いているかチェック
        left_open = left_hand.get('gesture', '') == 'open_hand'
        right_open = right_hand.get('gesture', '') == 'open_hand'
        
        if distance < 0.15 and left_open and right_open:  # 距離が近く、両手が開いている
            center_position = ((left_pos[0] + right_pos[0]) / 2, (left_pos[1] + right_pos[1]) / 2)
            
            return GestureEvent(
                gesture_type=GestureType.CLAP,
                confidence=0.9,
                position=center_position,
                hand_label="Both",
                additional_data={'distance': distance}
            )
        
        return None
    
    def _detect_heart_gesture(self, left_hand: Dict, right_hand: Dict) -> Optional[GestureEvent]:
        """ハート形ジェスチャー検出（簡易版）"""
        # ここでは簡単な実装として、特定の指の配置をチェック
        left_landmarks = left_hand['landmarks']
        right_landmarks = right_hand['landmarks']
        
        try:
            # 両手の人差し指先端
            left_index_tip = left_landmarks.landmark[8]
            right_index_tip = right_landmarks.landmark[8]
            
            # 両手の親指先端
            left_thumb_tip = left_landmarks.landmark[4]
            right_thumb_tip = right_landmarks.landmark[4]
            
            # ハート形の条件：
            # 1. 人差し指が上で近接
            # 2. 親指が下で近接
            index_distance = math.sqrt((left_index_tip.x - right_index_tip.x)**2 + 
                                     (left_index_tip.y - right_index_tip.y)**2)
            thumb_distance = math.sqrt((left_thumb_tip.x - right_thumb_tip.x)**2 + 
                                     (left_thumb_tip.y - right_thumb_tip.y)**2)
            
            if index_distance < 0.1 and thumb_distance < 0.1:
                # ハート形の中心位置
                center_x = (left_index_tip.x + right_index_tip.x + left_thumb_tip.x + right_thumb_tip.x) / 4
                center_y = (left_index_tip.y + right_index_tip.y + left_thumb_tip.y + right_thumb_tip.y) / 4
                
                return GestureEvent(
                    gesture_type=GestureType.HEART,
                    confidence=0.8,
                    position=(center_x, center_y),
                    hand_label="Both"
                )
        
        except Exception as e:
            print(f"⚠️  ハート検出エラー: {e}")
        
        return None
    
    def _apply_cooldown(self, gestures: List[GestureEvent], current_time: float) -> List[GestureEvent]:
        """クールダウン適用"""
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return []
        
        if gestures:
            self.last_gesture_time = current_time
        
        return gestures
    
    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計取得"""
        return {
            'trail_lengths': {hand: len(trail) for hand, trail in self.hand_trails.items()},
            'last_gesture_time': self.last_gesture_time,
            'gesture_cooldown': self.gesture_cooldown,
            'thresholds': self.thresholds
        }

# テスト実行用
if __name__ == "__main__":
    print("🔍 Advanced Gesture Recognizer テスト開始...")
    
    config = {
        'ai_processing': {
            'gesture': {
                'recognition_timeout': 2.0,
                'min_confidence': 0.7
            }
        }
    }
    
    recognizer = AdvancedGestureRecognizer(config)
    
    # ダミーテスト
    dummy_hand_result = {
        'hands_detected': False,
        'hands': []
    }
    
    gestures = recognizer.recognize_gestures(dummy_hand_result)
    print(f"検出ジェスチャー数: {len(gestures)}")
    
    stats = recognizer.get_performance_stats()
    print(f"性能統計: {stats}")
    
    print("✅ Advanced Gesture Recognizer テスト完了")

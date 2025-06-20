import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

class HandDetector:
    """MediaPipe Hands 統合検出器（Day 3版）"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # MediaPipe Hands 初期化
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hands 設定
        hand_config = config.get('ai_processing', {}).get('vision', {}).get('hand_detection', {})
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=hand_config.get('max_num_hands', 2),
            min_detection_confidence=hand_config.get('min_detection_confidence', 0.7),
            min_tracking_confidence=hand_config.get('min_tracking_confidence', 0.5),
            model_complexity=hand_config.get('model_complexity', 1)
        )
        
        # 手のランドマークインデックス定義
        self.HAND_LANDMARKS = {
            'wrist': 0,
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # 処理統計
        self.detection_times = []
        self.detection_count = 0
        self.last_detection_result = None
        
        print("✅ Hand Detector 初期化完了")
    
    def detect_hands(self, frame: np.ndarray) -> Dict:
        """手検出メイン処理"""
        start_time = time.time()
        
        try:
            # RGB変換（MediaPipe用）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe処理
            results = self.hands.process(rgb_frame)
            
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
            print(f"⚠️  手検出エラー: {e}")
            return self._get_empty_result()
    
    def _analyze_detection_results(self, results, frame_shape) -> Dict:
        """検出結果解析"""
        height, width = frame_shape[:2]
        
        result = {
            'hands_detected': False,
            'hand_count': 0,
            'hands': [],
            'processing_time': self.detection_times[-1] if self.detection_times else 0
        }
        
        if results.multi_hand_landmarks and results.multi_handedness:
            result['hands_detected'] = True
            result['hand_count'] = len(results.multi_hand_landmarks)
            
            # 各手の情報を解析
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_info = self._analyze_single_hand(hand_landmarks, handedness, (width, height))
                result['hands'].append(hand_info)
        
        return result
    
    def _analyze_single_hand(self, landmarks, handedness, frame_size) -> Dict:
        """単一の手の解析"""
        width, height = frame_size
        
        # 手の基本情報
        hand_label = handedness.classification[0].label  # "Left" or "Right"
        hand_score = handedness.classification[0].score
        
        # 手首位置
        wrist = landmarks.landmark[0]
        wrist_pos = (wrist.x, wrist.y)
        
        # 指の状態解析
        finger_states = self._analyze_finger_states(landmarks)
        
        # 手の向き・角度
        hand_angle = self._calculate_hand_angle(landmarks)
        
        # 手のサイズ
        hand_size = self._calculate_hand_size(landmarks)
        
        # 基本ジェスチャー認識
        gesture = self._recognize_basic_gesture(finger_states, landmarks)
        
        return {
            'label': hand_label,
            'confidence': hand_score,
            'landmarks': landmarks,
            'wrist_position': wrist_pos,
            'finger_states': finger_states,
            'hand_angle': hand_angle,
            'hand_size': hand_size,
            'gesture': gesture
        }
    
    def _analyze_finger_states(self, landmarks) -> Dict[str, bool]:
        """指の状態解析（伸展/屈曲）"""
        finger_states = {}
        
        # 親指（水平方向の判定）
        thumb_tip = landmarks.landmark[4]
        thumb_ip = landmarks.landmark[3]
        thumb_mcp = landmarks.landmark[2]
        thumb_extended = abs(thumb_tip.x - thumb_mcp.x) > abs(thumb_ip.x - thumb_mcp.x)
        finger_states['thumb'] = thumb_extended
        
        # 他の指（垂直方向の判定）
        fingers = ['index', 'middle', 'ring', 'pinky']
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for finger, tip_idx, pip_idx in zip(fingers, finger_tips, finger_pips):
            tip = landmarks.landmark[tip_idx]
            pip = landmarks.landmark[pip_idx]
            # 指先がPIP関節より上にある場合は伸展
            extended = tip.y < pip.y
            finger_states[finger] = extended
        
        return finger_states
    
    def _calculate_hand_angle(self, landmarks) -> float:
        """手の角度計算"""
        # 手首から中指MCPへのベクトル
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        
        # 角度計算
        dx = middle_mcp.x - wrist.x
        dy = middle_mcp.y - wrist.y
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _calculate_hand_size(self, landmarks) -> float:
        """手のサイズ計算"""
        # 手首から中指先端までの距離
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        distance = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
        return distance
    
    def _recognize_basic_gesture(self, finger_states: Dict[str, bool], landmarks) -> str:
        """基本ジェスチャー認識"""
        # 指の状態パターンから判定
        extended_fingers = [finger for finger, extended in finger_states.items() if extended]
        extended_count = len(extended_fingers)
        
        # 基本パターン
        if extended_count == 0:
            return "fist"  # 握りこぶし
        elif extended_count == 1:
            if finger_states['index']:
                return "point"  # 指差し
            elif finger_states['thumb']:
                return "thumbs_up"  # サムズアップ
        elif extended_count == 2:
            if finger_states['index'] and finger_states['middle']:
                return "peace"  # ピースサイン
            elif finger_states['thumb'] and finger_states['index']:
                return "gun"  # 銃の形
        elif extended_count == 5:
            return "open_hand"  # 開いた手
        
        return "unknown"  # 不明
    
    def _get_empty_result(self) -> Dict:
        """空の結果"""
        return {
            'hands_detected': False,
            'hand_count': 0,
            'hands': [],
            'processing_time': 0
        }
    
    def draw_landmarks(self, frame: np.ndarray, detection_result: Dict, 
                      draw_connections: bool = True) -> np.ndarray:
        """ランドマーク描画"""
        if not detection_result['hands_detected']:
            return frame
        
        try:
            for hand_info in detection_result['hands']:
                landmarks = hand_info['landmarks']
                
                if draw_connections:
                    # 接続線描画
                    self.mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                else:
                    # ランドマークのみ描画
                    self.mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        None,
                        self.mp_drawing_styles.get_default_hand_landmarks_style()
                    )
                
                # 手の情報テキスト描画
                self._draw_hand_info(frame, hand_info)
        
        except Exception as e:
            print(f"⚠️  ランドマーク描画エラー: {e}")
        
        return frame
    
    def _draw_hand_info(self, frame: np.ndarray, hand_info: Dict):
        """手の情報テキスト描画"""
        height, width = frame.shape[:2]
        
        # 手首位置から情報表示位置計算
        wrist_x, wrist_y = hand_info['wrist_position']
        text_x = int(wrist_x * width)
        text_y = int(wrist_y * height) - 10
        
        # 手の情報
        label = hand_info['label']
        confidence = hand_info['confidence']
        gesture = hand_info['gesture']
        
        # テキスト描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        # 背景矩形
        text = f"{label}: {gesture} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5), 
                     (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        
        # テキスト
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
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
        if self.hands:
            self.hands.close()
        print("Hand Detector リソース解放完了")

# テスト実行用
if __name__ == "__main__":
    print("🔍 Hand Detector テスト開始...")
    
    # テスト設定
    config = {
        'ai_processing': {
            'vision': {
                'hand_detection': {
                    'max_num_hands': 2,
                    'min_detection_confidence': 0.7,
                    'min_tracking_confidence': 0.5,
                    'model_complexity': 1
                }
            }
        }
    }
    
    detector = HandDetector(config)
    
    # カメラテスト
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラを開けません")
        exit()
    
    print("👋 手検出テスト開始（ESCで終了）...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 手検出
        result = detector.detect_hands(frame)
        
        # 結果描画
        frame_with_landmarks = detector.draw_landmarks(frame, result, draw_connections=True)
        
        # 結果表示
        cv2.imshow('Hand Detection Test', frame_with_landmarks)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESCキー
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 統計表示
    stats = detector.get_performance_stats()
    print(f"📊 性能統計: {stats}")
    
    detector.cleanup()
    print("✅ Hand Detector テスト完了")

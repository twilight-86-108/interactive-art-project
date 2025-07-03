# src/vision/hand_detector.py
import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any, List
from mediapipe.python.solutions import hands, drawing_utils

class HandDetector:
    """手検出クラス（エラー修正版）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # MediaPipe初期化
        self.mp_hands = hands
        self.mp_drawing = drawing_utils
        
        try:
            # 設定取得（デフォルト値付き）
            hand_config = config.get('detection', {})
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=hand_config.get('max_num_hands', 2),
                min_detection_confidence=hand_config.get('hand_detection_confidence', 0.7),
                min_tracking_confidence=hand_config.get('min_tracking_confidence', 0.5)
            )
            
            self.logger.info("手検出器が初期化されました")
            
        except Exception as e:
            self.logger.error(f"手検出器初期化エラー: {e}")
            self.hands = None
    
    def detect_hands(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """手検出メイン関数"""
        if frame is None or self.hands is None:
            return None
        
        try:
            # フレームがBGRの場合、RGBに変換
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            # 手検出実行
            results = self.hands.process(rgb_frame)
            
            if hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:  # type: ignore
                return self._process_hand_results(results, frame.shape)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"手検出エラー: {e}")
            return None
    
    def _process_hand_results(self, results, frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """手検出結果の処理"""
        try:
            height, width = frame_shape[:2]
            
            hand_data = {
                'hands_detected': True,
                'hand_landmarks': results,
                'hand_count': len(results.multi_hand_landmarks),
                'hand_positions': [],
                'hands': []
            }
            
            # 各手の情報を処理
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 手の向き情報を取得
                handedness = None
                if results.multi_handedness and idx < len(results.multi_handedness):
                    handedness = results.multi_handedness[idx].classification[0].label
                
                hand_info = self._analyze_single_hand(hand_landmarks, width, height, handedness)
                hand_data['hands'].append(hand_info)
                
                # 手首位置を手の位置として記録
                wrist = hand_landmarks.landmark[0]
                hand_data['hand_positions'].append((wrist.x, wrist.y))
            
            return hand_data
            
        except Exception as e:
            self.logger.error(f"手結果処理エラー: {e}")
            return {'hands_detected': False}
    
    def _analyze_single_hand(self, landmarks, width: int, height: int, handedness: Optional[str]) -> Dict[str, Any]:
        """個別手の分析"""
        try:
            # ランドマークを正規化座標から画素座標に変換
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                landmark_points.append((x, y, z))
            
            # 手首位置（ランドマーク0）
            wrist = landmarks.landmark[0]
            wrist_pos = (wrist.x, wrist.y, wrist.z)
            
            # 指先位置の取得
            fingertips = self._get_fingertip_positions(landmarks)
            
            # 手の向き推定
            orientation = self._estimate_hand_orientation(landmarks)
            
            # バウンディングボックス計算
            bbox = self._calculate_hand_bbox(landmark_points, width, height)
            
            # 基本的なジェスチャー認識
            gesture = self._recognize_basic_gesture(landmarks)
            
            return {
                'handedness': handedness,  # 'Left' or 'Right'
                'wrist_position': wrist_pos,
                'fingertips': fingertips,
                'orientation': orientation,
                'bbox': bbox,
                'gesture': gesture,
                'landmark_points': landmark_points,
                'confidence': 0.8  # MediaPipeは信頼度を直接提供しないため固定値
            }
            
        except Exception as e:
            self.logger.error(f"手分析エラー: {e}")
            return {}
    
    def _get_fingertip_positions(self, landmarks) -> Dict[str, Tuple[float, float, float]]:
        """指先位置の取得"""
        try:
            # MediaPipe手ランドマークの指先インデックス
            fingertip_indices = {
                'thumb': 4,      # 親指
                'index': 8,      # 人差し指
                'middle': 12,    # 中指
                'ring': 16,      # 薬指
                'pinky': 20      # 小指
            }
            
            fingertips = {}
            for finger_name, idx in fingertip_indices.items():
                if idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[idx]
                    fingertips[finger_name] = (landmark.x, landmark.y, landmark.z)
                else:
                    fingertips[finger_name] = (0.0, 0.0, 0.0)
            
            return fingertips
            
        except Exception as e:
            self.logger.error(f"指先位置取得エラー: {e}")
            return {}
    
    def _estimate_hand_orientation(self, landmarks) -> Dict[str, Any]:
        """手の向き推定"""
        try:
            # 手首と中指の基準点を使用
            wrist = landmarks.landmark[0]
            middle_mcp = landmarks.landmark[9]  # 中指の付け根
            
            # 手の向きベクトル
            dx = middle_mcp.x - wrist.x
            dy = middle_mcp.y - wrist.y
            
            # 角度計算（ラジアンから度に変換）
            angle = np.degrees(np.arctan2(dy, dx))
            
            return {
                'angle': angle,
                'vector': (dx, dy)
            }
            
        except Exception as e:
            self.logger.error(f"手向き推定エラー: {e}")
            return {'angle': 0.0, 'vector': (0.0, 0.0)}
    
    def _calculate_hand_bbox(self, landmark_points, width: int, height: int) -> Tuple[int, int, int, int]:
        """手のバウンディングボックス計算"""
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
    
    def _recognize_basic_gesture(self, landmarks) -> str:
        """基本的なジェスチャー認識"""
        try:
            # 指の伸展状態を判定
            finger_states = self._get_finger_states(landmarks)
            
            # パターンマッチング
            extended_count = sum(finger_states.values())
            
            if extended_count == 0:
                return "fist"  # 握りこぶし
            elif extended_count == 1 and finger_states['index']:
                return "point"  # 指差し
            elif extended_count == 2 and finger_states['index'] and finger_states['middle']:
                return "peace"  # ピース
            elif extended_count == 5:
                return "open"  # 開いた手
            else:
                return "unknown"
                
        except Exception as e:
            self.logger.error(f"ジェスチャー認識エラー: {e}")
            return "unknown"
    
    def _get_finger_states(self, landmarks) -> Dict[str, bool]:
        """各指の伸展状態判定"""
        try:
            finger_states = {}
            
            # 親指（特殊処理：横方向の伸展）
            thumb_tip = landmarks.landmark[4]
            thumb_ip = landmarks.landmark[3]
            finger_states['thumb'] = thumb_tip.x > thumb_ip.x  # 簡易判定
            
            # 他の指（縦方向の伸展）
            finger_tips = [8, 12, 16, 20]  # 人差し指、中指、薬指、小指の先端
            finger_pips = [6, 10, 14, 18]  # 対応する関節
            finger_names = ['index', 'middle', 'ring', 'pinky']
            
            for i, (tip_idx, pip_idx, name) in enumerate(zip(finger_tips, finger_pips, finger_names)):
                if tip_idx < len(landmarks.landmark) and pip_idx < len(landmarks.landmark):
                    tip = landmarks.landmark[tip_idx]
                    pip = landmarks.landmark[pip_idx]
                    
                    # 指先が関節より上にあれば伸展
                    finger_states[name] = tip.y < pip.y
                else:
                    finger_states[name] = False
            
            return finger_states
            
        except Exception as e:
            self.logger.error(f"指状態判定エラー: {e}")
            return {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
    
    def get_hand_distance(self, hand_data: Dict[str, Any]) -> float:
        """手とカメラの距離推定"""
        try:
            if not hand_data or 'hands' not in hand_data or not hand_data['hands']:
                return float('inf')
            
            # 最初の手の距離を計算
            hand = hand_data['hands'][0]
            wrist_z = hand['wrist_position'][2]
            
            # Z座標を距離に変換（簡易版）
            distance = abs(wrist_z)
            
            return distance
            
        except Exception as e:
            self.logger.error(f"手距離推定エラー: {e}")
            return float('inf')
    
    def is_pointing_gesture(self, hand_data: Dict[str, Any]) -> bool:
        """指差しジェスチャーの判定"""
        try:
            if not hand_data or 'hands' not in hand_data:
                return False
            
            for hand in hand_data['hands']:
                if hand.get('gesture') == 'point':
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"指差し判定エラー: {e}")
            return False
    
    def get_pointing_direction(self, hand_data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """指差し方向の取得"""
        try:
            if not self.is_pointing_gesture(hand_data):
                return None
            
            for hand in hand_data['hands']:
                if hand.get('gesture') == 'point':
                    orientation = hand.get('orientation', {})
                    vector = orientation.get('vector', (0.0, 0.0))
                    return vector
            
            return None
            
        except Exception as e:
            self.logger.error(f"指差し方向取得エラー: {e}")
            return None
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.hands:
                self.hands.close()
            self.logger.info("手検出器がクリーンアップされました")
        except Exception as e:
            self.logger.error(f"手検出器クリーンアップエラー: {e}")
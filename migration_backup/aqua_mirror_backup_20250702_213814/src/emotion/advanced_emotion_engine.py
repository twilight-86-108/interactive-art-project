"""
高精度感情認識エンジン
個人差対応・時系列分析・信頼度向上システム
"""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import json
import time
from pathlib import Path

@dataclass
class EmotionFeatures:
    """感情特徴量データクラス"""
    # 目の特徴
    left_eye_openness: float = 0.5
    right_eye_openness: float = 0.5
    eye_asymmetry: float = 0.0
    blink_frequency: float = 0.0
    
    # 口の特徴
    mouth_openness: float = 0.0
    mouth_width: float = 0.5
    mouth_corner_left: float = 0.0
    mouth_corner_right: float = 0.0
    lip_tension: float = 0.0
    
    # 眉の特徴
    left_eyebrow_height: float = 0.5
    right_eyebrow_height: float = 0.5
    eyebrow_inner_distance: float = 0.5
    eyebrow_angle: float = 0.0
    
    # 顔全体の特徴
    face_symmetry: float = 1.0
    head_tilt: float = 0.0
    face_tension: float = 0.0
    micro_expressions: float = 0.0

@dataclass
class PersonalCalibration:
    """個人キャリブレーションデータ"""
    neutral_features: EmotionFeatures
    feature_ranges: Dict[str, Tuple[float, float]]
    emotion_thresholds: Dict[str, float]
    calibration_confidence: float = 0.0
    sample_count: int = 0

class AdvancedEmotionEngine:
    """
    高精度感情認識エンジン
    個人差自動学習・時系列分析・微細表情検出
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger("AdvancedEmotionEngine")
        self.config = config
        
        # 感情認識設定
        emotion_config = config.get('ai', {}).get('emotion', {})
        self.smoothing_factor = emotion_config.get('smoothing_factor', 0.8)
        self.confidence_threshold = emotion_config.get('confidence_threshold', 0.6)
        self.calibration_samples = emotion_config.get('calibration_samples', 100)
        
        # 時系列データ管理
        self.feature_history = deque(maxlen=180)  # 6秒分（30FPS）
        self.emotion_history = deque(maxlen=60)   # 2秒分
        self.blink_history = deque(maxlen=30)     # 1秒分
        
        # 個人キャリブレーション
        self.personal_calibration: Optional[PersonalCalibration] = None
        self.calibration_data: List[EmotionFeatures] = []
        self.is_calibrating = False
        
        # 感情分類器（高度なルールベース）
        self.emotion_classifiers = {
            'HAPPY': self._classify_happy,
            'SAD': self._classify_sad,
            'ANGRY': self._classify_angry,
            'SURPRISED': self._classify_surprised,
            'FEAR': self._classify_fear,
            'DISGUST': self._classify_disgust,
            'NEUTRAL': self._classify_neutral
        }
        
        # 微細表情検出
        self.micro_expression_detector = MicroExpressionDetector()
        
        # 顔ランドマーク詳細インデックス
        self._initialize_landmark_indices()
        
        self.logger.info("🧠 高精度感情認識エンジン初期化完了")
    
    def _initialize_landmark_indices(self):
        """詳細顔ランドマークインデックス初期化"""
        # MediaPipe 468点ランドマーク詳細マッピング
        self.landmarks = {
            # 左目詳細
            'left_eye_inner': 133,
            'left_eye_outer': 33,
            'left_eye_top': 159,
            'left_eye_bottom': 145,
            'left_eye_center': 468,  # 仮想中心点
            
            # 右目詳細
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'right_eye_top': 386,
            'right_eye_bottom': 374,
            'right_eye_center': 469,  # 仮想中心点
            
            # 口詳細
            'mouth_left': 61,
            'mouth_right': 291,
            'mouth_top': 13,
            'mouth_bottom': 14,
            'upper_lip_center': 12,
            'lower_lip_center': 15,
            'mouth_corners': [61, 291, 39, 181, 84, 17, 314, 405, 320, 307],
            
            # 眉詳細
            'left_eyebrow_inner': 70,
            'left_eyebrow_outer': 63,
            'left_eyebrow_top': 107,
            'right_eyebrow_inner': 300,
            'right_eyebrow_outer': 293,
            'right_eyebrow_top': 336,
            
            # 鼻
            'nose_tip': 1,
            'nose_bridge': [6, 168, 8, 9, 10],
            
            # 顔輪郭
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        }
    
    def extract_advanced_features(self, landmarks, frame_shape: Tuple[int, int, int]) -> EmotionFeatures:
        """高度特徴量抽出"""
        try:
            height, width = frame_shape[:2]
            
            # ランドマーク座標正規化
            points = np.array([[lm.x * width, lm.y * height] for lm in landmarks.landmark])
            
            features = EmotionFeatures()
            
            # 目の詳細特徴
            features.left_eye_openness = self._calculate_detailed_eye_openness(points, 'left')
            features.right_eye_openness = self._calculate_detailed_eye_openness(points, 'right')
            features.eye_asymmetry = abs(features.left_eye_openness - features.right_eye_openness)
            
            # 口の詳細特徴
            features.mouth_openness = self._calculate_detailed_mouth_openness(points)
            features.mouth_width = self._calculate_mouth_width(points)
            features.mouth_corner_left, features.mouth_corner_right = self._calculate_mouth_corners(points)
            features.lip_tension = self._calculate_lip_tension(points)
            
            # 眉の詳細特徴
            features.left_eyebrow_height = self._calculate_eyebrow_height(points, 'left')
            features.right_eyebrow_height = self._calculate_eyebrow_height(points, 'right')
            features.eyebrow_inner_distance = self._calculate_eyebrow_inner_distance(points)
            features.eyebrow_angle = self._calculate_eyebrow_angle(points)
            
            # 顔全体特徴
            features.face_symmetry = self._calculate_face_symmetry(points)
            features.head_tilt = self._calculate_head_tilt(points)
            features.face_tension = self._calculate_face_tension(points)
            
            # 微細表情
            features.micro_expressions = self.micro_expression_detector.detect(points, self.feature_history)
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ 高度特徴量抽出失敗: {e}")
            return EmotionFeatures()
    
    def _calculate_detailed_eye_openness(self, points: np.ndarray, eye: str) -> float:
        """詳細目開き度計算"""
        try:
            if eye == 'left':
                inner = points[self.landmarks['left_eye_inner']]
                outer = points[self.landmarks['left_eye_outer']]
                top = points[self.landmarks['left_eye_top']]
                bottom = points[self.landmarks['left_eye_bottom']]
            else:
                inner = points[self.landmarks['right_eye_inner']]
                outer = points[self.landmarks['right_eye_outer']]
                top = points[self.landmarks['right_eye_top']]
                bottom = points[self.landmarks['right_eye_bottom']]
            
            # 目の縦横比計算
            vertical_dist = np.linalg.norm(top - bottom)
            horizontal_dist = np.linalg.norm(outer - inner)
            
            # 正規化開き度
            openness = vertical_dist / (horizontal_dist + 1e-6)
            
            # 個人キャリブレーション適用
            if self.personal_calibration:
                neutral_openness = getattr(self.personal_calibration.neutral_features, 
                                         f'{eye}_eye_openness')
                openness = openness / (neutral_openness + 1e-6)
            
            return np.clip(openness, 0.0, 2.0)
            
        except Exception as e:
            self.logger.error(f"❌ 目開き度計算失敗: {e}")
            return 0.5
    
    def _calculate_detailed_mouth_openness(self, points: np.ndarray) -> float:
        """詳細口開き度計算"""
        try:
            upper_lip = points[self.landmarks['upper_lip_center']]
            lower_lip = points[self.landmarks['lower_lip_center']]
            mouth_left = points[self.landmarks['mouth_left']]
            mouth_right = points[self.landmarks['mouth_right']]
            
            # 縦の開き
            vertical_opening = np.linalg.norm(upper_lip - lower_lip)
            
            # 口の幅で正規化
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            normalized_opening = vertical_opening / (mouth_width + 1e-6)
            
            # 個人キャリブレーション適用
            if self.personal_calibration:
                neutral_opening = self.personal_calibration.neutral_features.mouth_openness
                normalized_opening = normalized_opening / (neutral_opening + 1e-6)
            
            return np.clip(normalized_opening, 0.0, 3.0)
            
        except Exception as e:
            self.logger.error(f"❌ 口開き度計算失敗: {e}")
            return 0.0
    
    def _calculate_mouth_corners(self, points: np.ndarray) -> Tuple[float, float]:
        """口角位置計算"""
        try:
            mouth_left = points[self.landmarks['mouth_left']]
            mouth_right = points[self.landmarks['mouth_right']]
            mouth_center = (mouth_left + mouth_right) / 2
            
            # 口角と中心線の相対位置
            left_corner_height = mouth_left[1] - mouth_center[1]
            right_corner_height = mouth_right[1] - mouth_center[1]
            
            # 正規化（上向きが正、下向きが負）
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            left_corner = -left_corner_height / (mouth_width + 1e-6)
            right_corner = -right_corner_height / (mouth_width + 1e-6)
            
            return np.clip(left_corner, -1.0, 1.0), np.clip(right_corner, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ 口角計算失敗: {e}")
            return 0.0, 0.0
    
    def _calculate_lip_tension(self, points: np.ndarray) -> float:
        """唇の緊張度計算"""
        try:
            # 口角点群の曲率から緊張度を推定
            mouth_corners = [points[i] for i in self.landmarks['mouth_corners']]
            
            # 点群の分散から緊張度計算
            mouth_center = np.mean(mouth_corners, axis=0)
            distances = [np.linalg.norm(point - mouth_center) for point in mouth_corners]
            tension = np.std(distances) / (np.mean(distances) + 1e-6)
            
            return np.clip(tension, 0.0, 2.0)
            
        except Exception as e:
            self.logger.error(f"❌ 唇緊張度計算失敗: {e}")
            return 0.0
    
    def _calculate_face_symmetry(self, points: np.ndarray) -> float:
        """顔対称性計算"""
        try:
            # 顔中央線（鼻筋）
            nose_points = [points[i] for i in self.landmarks['nose_bridge']]
            nose_center = np.mean(nose_points, axis=0)
            
            # 左右の対応点の距離比較
            left_eye = points[self.landmarks['left_eye_center']]
            right_eye = points[self.landmarks['right_eye_center']]
            
            # 中央線からの距離
            left_distance = abs(left_eye[0] - nose_center[0])
            right_distance = abs(right_eye[0] - nose_center[0])
            
            # 対称性（1.0が完全対称）
            symmetry = min(left_distance, right_distance) / (max(left_distance, right_distance) + 1e-6)
            
            return np.clip(symmetry, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ 顔対称性計算失敗: {e}")
            return 1.0
    
    def _calculate_head_tilt(self, points: np.ndarray) -> float:
        """頭部傾き計算"""
        try:
            left_eye = points[self.landmarks['left_eye_center']]
            right_eye = points[self.landmarks['right_eye_center']]
            
            # 目線の傾き
            eye_vector = right_eye - left_eye
            tilt_angle = np.arctan2(eye_vector[1], eye_vector[0])
            
            # ラジアンから度数に変換、正規化
            tilt_degrees = np.degrees(tilt_angle)
            normalized_tilt = tilt_degrees / 45.0  # ±45度で正規化
            
            return np.clip(normalized_tilt, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ 頭部傾き計算失敗: {e}")
            return 0.0
    
    def start_personal_calibration(self):
        """個人キャリブレーション開始"""
        self.logger.info("🎯 個人キャリブレーション開始")
        self.is_calibrating = True
        self.calibration_data.clear()
    
    def add_calibration_sample(self, features: EmotionFeatures):
        """キャリブレーションサンプル追加"""
        if self.is_calibrating:
            self.calibration_data.append(features)
            
            if len(self.calibration_data) >= self.calibration_samples:
                self._complete_calibration()
    
    def _complete_calibration(self):
        """キャリブレーション完了処理"""
        try:
            # 中性状態の平均特徴量計算
            neutral_features = self._calculate_average_features(self.calibration_data)
            
            # 特徴量の変動範囲計算
            feature_ranges = self._calculate_feature_ranges(self.calibration_data)
            
            # 感情閾値の個人化
            emotion_thresholds = self._calculate_personal_thresholds(neutral_features, feature_ranges)
            
            # キャリブレーションデータ作成
            self.personal_calibration = PersonalCalibration(
                neutral_features=neutral_features,
                feature_ranges=feature_ranges,
                emotion_thresholds=emotion_thresholds,
                calibration_confidence=self._calculate_calibration_confidence(),
                sample_count=len(self.calibration_data)
            )
            
            self.is_calibrating = False
            self.logger.info(f"✅ 個人キャリブレーション完了 (サンプル数: {len(self.calibration_data)})")
            
            # キャリブレーションデータ保存
            self._save_calibration()
            
        except Exception as e:
            self.logger.error(f"❌ キャリブレーション完了処理失敗: {e}")
            self.is_calibrating = False
    
    def analyze_emotion_advanced(self, features: EmotionFeatures) -> Dict[str, Any]:
        """高精度感情分析"""
        try:
            # 特徴量履歴に追加
            self.feature_history.append(features)
            
            # 各感情の信頼度計算
            emotion_scores = {}
            for emotion, classifier in self.emotion_classifiers.items():
                score = classifier(features)
                emotion_scores[emotion] = score
            
            # 最高スコア感情選択
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[best_emotion]
            
            # 時系列平滑化
            smoothed_result = self._apply_advanced_smoothing(best_emotion, confidence, emotion_scores)
            
            # 微細表情検出結果統合
            micro_emotions = self._detect_micro_emotions(features)
            
            return {
                'emotion': smoothed_result['emotion'],
                'confidence': smoothed_result['confidence'],
                'emotion_scores': emotion_scores,
                'micro_emotions': micro_emotions,
                'features': features,
                'calibration_status': self.personal_calibration is not None,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"❌ 高精度感情分析失敗: {e}")
            return {
                'emotion': 'NEUTRAL',
                'confidence': 0.0,
                'emotion_scores': {},
                'micro_emotions': [],
                'features': features,
                'calibration_status': False,
                'timestamp': time.time()
            }
    
    def _classify_happy(self, features: EmotionFeatures) -> float:
        """幸せ感情分類"""
        score = 0.0
        
        # 口角上がり（重要指標）
        mouth_corners_avg = (features.mouth_corner_left + features.mouth_corner_right) / 2
        if mouth_corners_avg > 0.2:
            score += 0.4 * min(mouth_corners_avg, 1.0)
        
        # 目の細さ（笑顔時の特徴）
        eye_narrowing = 1.0 - (features.left_eye_openness + features.right_eye_openness) / 2
        if eye_narrowing > 0.3:
            score += 0.3 * min(eye_narrowing, 1.0)
        
        # 口の開き（笑い）
        if features.mouth_openness > 0.3:
            score += 0.2 * min(features.mouth_openness, 1.0)
        
        # 微細表情
        score += features.micro_expressions * 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _classify_sad(self, features: EmotionFeatures) -> float:
        """悲しみ感情分類"""
        score = 0.0
        
        # 口角下がり
        mouth_corners_avg = (features.mouth_corner_left + features.mouth_corner_right) / 2
        if mouth_corners_avg < -0.1:
            score += 0.4 * min(abs(mouth_corners_avg), 1.0)
        
        # 眉下がり
        eyebrow_height_avg = (features.left_eyebrow_height + features.right_eyebrow_height) / 2
        if eyebrow_height_avg < 0.4:
            score += 0.3 * (0.4 - eyebrow_height_avg)
        
        # 目の小ささ
        eye_openness_avg = (features.left_eye_openness + features.right_eye_openness) / 2
        if eye_openness_avg < 0.4:
            score += 0.2 * (0.4 - eye_openness_avg)
        
        # 顔の緊張
        score += features.face_tension * 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _classify_angry(self, features: EmotionFeatures) -> float:
        """怒り感情分類"""
        score = 0.0
        
        # 眉を寄せる
        if features.eyebrow_inner_distance < 0.4:
            score += 0.4 * (0.4 - features.eyebrow_inner_distance)
        
        # 眉の角度（八の字）
        if abs(features.eyebrow_angle) > 0.3:
            score += 0.3 * min(abs(features.eyebrow_angle), 1.0)
        
        # 口の緊張
        if features.lip_tension > 0.6:
            score += 0.2 * min(features.lip_tension, 1.0)
        
        # 顔の非対称性
        score += (1.0 - features.face_symmetry) * 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _classify_surprised(self, features: EmotionFeatures) -> float:
        """驚き感情分類"""
        score = 0.0
        
        # 目を大きく見開く
        eye_openness_avg = (features.left_eye_openness + features.right_eye_openness) / 2
        if eye_openness_avg > 0.7:
            score += 0.4 * min(eye_openness_avg, 1.0)
        
        # 眉上がり
        eyebrow_height_avg = (features.left_eyebrow_height + features.right_eyebrow_height) / 2
        if eyebrow_height_avg > 0.6:
            score += 0.3 * min(eyebrow_height_avg, 1.0)
        
        # 口の開き
        if features.mouth_openness > 0.4:
            score += 0.2 * min(features.mouth_openness, 1.0)
        
        # 微細表情（急激な変化）
        score += features.micro_expressions * 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _classify_fear(self, features: EmotionFeatures) -> float:
        """恐怖感情分類"""
        score = 0.0
        
        # 目を見開く（驚きより控えめ）
        eye_openness_avg = (features.left_eye_openness + features.right_eye_openness) / 2
        if eye_openness_avg > 0.6:
            score += 0.3 * min(eye_openness_avg, 1.0)
        
        # 眉上がり + 内側に寄る
        eyebrow_height_avg = (features.left_eyebrow_height + features.right_eyebrow_height) / 2
        if eyebrow_height_avg > 0.5 and features.eyebrow_inner_distance < 0.5:
            score += 0.4
        
        # 口の緊張
        if features.lip_tension > 0.5:
            score += 0.2 * min(features.lip_tension, 1.0)
        
        # 顔の緊張
        score += features.face_tension * 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _classify_disgust(self, features: EmotionFeatures) -> float:
        """嫌悪感情分類"""
        score = 0.0
        
        # 鼻にしわ（口角の微妙な上がり）
        mouth_corners_avg = (features.mouth_corner_left + features.mouth_corner_right) / 2
        if 0.1 < mouth_corners_avg < 0.3:
            score += 0.3
        
        # 上唇の持ち上がり（lip_tensionの特定パターン）
        if features.lip_tension > 0.4:
            score += 0.3 * min(features.lip_tension, 1.0)
        
        # 目の軽い細さ
        eye_openness_avg = (features.left_eye_openness + features.right_eye_openness) / 2
        if 0.3 < eye_openness_avg < 0.6:
            score += 0.2
        
        # 顔の非対称性
        score += (1.0 - features.face_symmetry) * 0.2
        
        return np.clip(score, 0.0, 1.0)
    
    def _classify_neutral(self, features: EmotionFeatures) -> float:
        """中性感情分類"""
        # 他の感情の平均スコアの逆数
        other_emotions = ['HAPPY', 'SAD', 'ANGRY', 'SURPRISED', 'FEAR', 'DISGUST']
        other_scores = [self.emotion_classifiers[emotion](features) for emotion in other_emotions]
        avg_other_score = np.mean(other_scores)
        
        neutral_score = 1.0 - avg_other_score
        return np.clip(neutral_score, 0.0, 1.0)
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("🧹 高精度感情認識エンジンリソース解放完了")


class MicroExpressionDetector:
    """微細表情検出器"""
    
    def __init__(self):
        self.previous_features: Optional[EmotionFeatures] = None
        self.change_threshold = 0.1
    
    def detect(self, points: np.ndarray, feature_history: deque) -> float:
        """微細表情検出"""
        try:
            if len(feature_history) < 2:
                return 0.0
            
            # 短期間での急激な変化を検出
            recent_features = list(feature_history)[-10:]  # 最新10フレーム
            
            if len(recent_features) < 3:
                return 0.0
            
            # 変化量計算
            changes = []
            for i in range(1, len(recent_features)):
                prev = recent_features[i-1]
                curr = recent_features[i]
                
                # 主要特徴の変化量
                eye_change = abs(curr.left_eye_openness - prev.left_eye_openness)
                mouth_change = abs(curr.mouth_openness - prev.mouth_openness)
                eyebrow_change = abs(curr.left_eyebrow_height - prev.left_eyebrow_height)
                
                total_change = eye_change + mouth_change + eyebrow_change
                changes.append(total_change)
            
            # 急激な変化の検出
            max_change = max(changes) if changes else 0.0
            micro_expression_intensity = min(max_change / self.change_threshold, 1.0)
            
            return micro_expression_intensity
            
        except Exception:
            return 0.0
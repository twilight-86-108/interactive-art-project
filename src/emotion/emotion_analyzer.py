# src/emotion/emotion_analyzer.py
import numpy as np
from collections import deque
from enum import Enum
import math
import logging

class Emotion(Enum):
    """感情タイプの定義"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    SURPRISED = "surprised"
    ANGRY = "angry"

class EmotionAnalyzer:
    """感情認識エンジン（エラー修正版）"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 感情履歴（平滑化用）
        self.emotion_history = deque(maxlen=10)
        self.confidence_threshold = 0.6
        
        # 顔の重要ランドマークインデックス（MediaPipe 468点ランドマーク）
        self.mouth_landmarks = [61, 84, 17, 314, 405, 320, 308, 324, 318]
        self.eyebrow_landmarks = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        self.eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # キャリブレーション用基準値
        self.baseline_features = None
        self.calibration_frames = []
        
        self.logger.info("感情認識エンジンが初期化されました")
    
    def analyze_emotion(self, face_landmarks):
        """メイン感情分析関数"""
        try:
            if not face_landmarks or not hasattr(face_landmarks, 'multi_face_landmarks') or not face_landmarks.multi_face_landmarks:
                return Emotion.NEUTRAL, 0.0
            
            landmarks = face_landmarks.multi_face_landmarks[0].landmark
            
            # 特徴量抽出
            features = self._extract_features(landmarks)
            
            # 感情分類
            emotion, confidence = self._classify_emotion(features)
            
            # 履歴ベース平滑化
            self.emotion_history.append((emotion, confidence))
            smoothed_emotion = self._get_smoothed_emotion()
            
            return smoothed_emotion, confidence
            
        except Exception as e:
            self.logger.error(f"感情分析エラー: {e}")
            return Emotion.NEUTRAL, 0.0
    
    def _extract_features(self, landmarks):
        """顔特徴量抽出"""
        try:
            features = {}
            
            # 口の特徴
            features['mouth_curve'] = self._calculate_mouth_curve(landmarks)
            features['mouth_openness'] = self._calculate_mouth_openness(landmarks)
            
            # 眉の特徴
            features['eyebrow_height'] = self._calculate_eyebrow_height(landmarks)
            features['eyebrow_distance'] = self._calculate_eyebrow_distance(landmarks)
            
            # 目の特徴
            features['eye_openness'] = self._calculate_eye_openness(landmarks)
            features['eye_squint'] = self._calculate_eye_squint(landmarks)
            
            return features
            
        except Exception as e:
            self.logger.error(f"特徴量抽出エラー: {e}")
            return {}
    
    def _calculate_mouth_curve(self, landmarks):
        """口角の曲率計算（笑顔検出）"""
        try:
            # 安全なインデックス取得
            if len(landmarks) <= max(61, 291, 13):
                return 0.0
                
            # 左右の口角
            left_corner = landmarks[61]
            right_corner = landmarks[291] if len(landmarks) > 291 else landmarks[61]
            
            # 口の中央点（上唇中央）
            mouth_center = landmarks[13]
            
            # 口角の平均高さと中央の高さの差
            corner_avg_y = (left_corner.y + right_corner.y) / 2
            curve = mouth_center.y - corner_avg_y
            
            return curve
            
        except Exception as e:
            self.logger.error(f"口角曲率計算エラー: {e}")
            return 0.0
    
    def _calculate_mouth_openness(self, landmarks):
        """口の開き具合計算"""
        try:
            if len(landmarks) <= max(13, 14, 61, 291):
                return 0.0
                
            # 上唇中央と下唇中央
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            
            # 口の幅（左右の口角間距離）
            left_corner = landmarks[61]
            right_corner = landmarks[291] if len(landmarks) > 291 else landmarks[61]
            mouth_width = abs(right_corner.x - left_corner.x)
            
            # 正規化された口の開き
            mouth_height = abs(upper_lip.y - lower_lip.y)
            normalized_openness = mouth_height / mouth_width if mouth_width > 0 else 0
            
            return normalized_openness
            
        except Exception as e:
            self.logger.error(f"口開き計算エラー: {e}")
            return 0.0
    
    def _calculate_eyebrow_height(self, landmarks):
        """眉の高さ計算（驚き検出）"""
        try:
            if len(landmarks) <= max(70, 296, 33, 263):
                return 0.0
                
            # 左眉の代表点
            left_eyebrow = landmarks[70]
            left_eye = landmarks[33]
            
            # 右眉の代表点
            right_eyebrow = landmarks[296] if len(landmarks) > 296 else landmarks[70]
            right_eye = landmarks[263] if len(landmarks) > 263 else landmarks[33]
            
            # 眉と目の距離
            left_distance = abs(left_eyebrow.y - left_eye.y)
            right_distance = abs(right_eyebrow.y - right_eye.y)
            
            return (left_distance + right_distance) / 2
            
        except Exception as e:
            self.logger.error(f"眉高さ計算エラー: {e}")
            return 0.0
    
    def _calculate_eyebrow_distance(self, landmarks):
        """眉間の距離計算"""
        try:
            if len(landmarks) <= max(70, 296):
                return 0.0
                
            left_eyebrow = landmarks[70]
            right_eyebrow = landmarks[296] if len(landmarks) > 296 else landmarks[70]
            
            distance = abs(right_eyebrow.x - left_eyebrow.x)
            return distance
            
        except Exception as e:
            self.logger.error(f"眉間距離計算エラー: {e}")
            return 0.0
    
    def _calculate_eye_openness(self, landmarks):
        """目の開き具合計算"""
        try:
            # 安全なインデックス確認
            required_indices = [159, 145, 33, 133, 386, 374, 362, 263]
            if len(landmarks) <= max(required_indices):
                return 0.0
                
            # 左目の開き
            left_eye_top = landmarks[159]
            left_eye_bottom = landmarks[145]
            left_eye_left = landmarks[33]
            left_eye_right = landmarks[133]
            
            left_height = abs(left_eye_top.y - left_eye_bottom.y)
            left_width = abs(left_eye_right.x - left_eye_left.x)
            left_openness = left_height / left_width if left_width > 0 else 0
            
            # 右目の開き
            right_eye_top = landmarks[386] if len(landmarks) > 386 else left_eye_top
            right_eye_bottom = landmarks[374] if len(landmarks) > 374 else left_eye_bottom
            right_eye_left = landmarks[362] if len(landmarks) > 362 else left_eye_left
            right_eye_right = landmarks[263] if len(landmarks) > 263 else left_eye_right
            
            right_height = abs(right_eye_top.y - right_eye_bottom.y)
            right_width = abs(right_eye_right.x - right_eye_left.x)
            right_openness = right_height / right_width if right_width > 0 else 0
            
            return (left_openness + right_openness) / 2
            
        except Exception as e:
            self.logger.error(f"目開き計算エラー: {e}")
            return 0.0
    
    def _calculate_eye_squint(self, landmarks):
        """目を細める度合い計算"""
        try:
            eye_openness = self._calculate_eye_openness(landmarks)
            # 開き具合が小さいほど細めている
            squint = max(0, 0.1 - eye_openness)
            return squint
            
        except Exception as e:
            self.logger.error(f"目細め計算エラー: {e}")
            return 0.0
    
    def _classify_emotion(self, features):
        """特徴量から感情分類"""
        try:
            # 単純なルールベース分類（軽量化）
            confidence = 0.7
            
            # 笑顔検出
            if features.get('mouth_curve', 0) > 0.01:
                return Emotion.HAPPY, min(0.9, confidence + features['mouth_curve'] * 20)
            
            # 驚き検出
            elif features.get('eyebrow_height', 0) > 0.025 and features.get('eye_openness', 0) > 0.15:
                return Emotion.SURPRISED, min(0.9, confidence + features['eyebrow_height'] * 30)
            
            # 悲しみ検出
            elif features.get('mouth_curve', 0) < -0.008:
                return Emotion.SAD, min(0.9, confidence + abs(features['mouth_curve']) * 25)
            
            # 怒り検出（眉を寄せる + 目を細める）
            elif features.get('eyebrow_distance', 1) < 0.02 and features.get('eye_openness', 1) < 0.08:
                return Emotion.ANGRY, min(0.9, confidence + 0.3)
            
            # デフォルト：ニュートラル
            else:
                return Emotion.NEUTRAL, 0.5
                
        except Exception as e:
            self.logger.error(f"感情分類エラー: {e}")
            return Emotion.NEUTRAL, 0.0
    
    def _get_smoothed_emotion(self):
        """履歴ベース平滑化"""
        try:
            if not self.emotion_history:
                return Emotion.NEUTRAL
            
            # 重み付き平均（新しい感情ほど重い）
            emotion_scores = {}
            total_weight = 0
            
            for i, (emotion, confidence) in enumerate(self.emotion_history):
                weight = (i + 1) * confidence  # 新しいほど重い重み
                
                if emotion not in emotion_scores:
                    emotion_scores[emotion] = 0
                emotion_scores[emotion] += weight
                total_weight += weight
            
            if total_weight == 0:
                return Emotion.NEUTRAL
            
            # 最高スコアの感情を返す
            best_emotion = max(emotion_scores, key=lambda k: emotion_scores[k])
            return best_emotion
            
        except Exception as e:
            self.logger.error(f"感情平滑化エラー: {e}")
            return Emotion.NEUTRAL
    
    def calibrate_baseline(self, landmarks_list):
        """ベースライン（ニュートラル状態）キャリブレーション"""
        try:
            if len(landmarks_list) < 30:  # 最低30フレーム必要
                return False
            
            # 平均特徴量計算
            feature_sums = {}
            valid_count = 0
            
            for landmarks in landmarks_list:
                if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                    features = self._extract_features(landmarks.multi_face_landmarks[0].landmark)
                    
                    for key, value in features.items():
                        if key not in feature_sums:
                            feature_sums[key] = 0
                        feature_sums[key] += value
                    
                    valid_count += 1
            
            if valid_count > 0:
                self.baseline_features = {
                    key: value / valid_count for key, value in feature_sums.items()
                }
                self.logger.info("感情認識ベースラインが設定されました")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"キャリブレーションエラー: {e}")
            return False
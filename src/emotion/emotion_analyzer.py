"""
基本感情認識エンジン
顔ランドマーク→感情分析システム
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from collections import deque
import math

class EmotionType(Enum):
    """感情タイプ"""
    HAPPY = 0
    SAD = 1
    ANGRY = 2
    SURPRISED = 3
    NEUTRAL = 4
    FEAR = 5
    DISGUST = 6

class EmotionAnalyzer:
    """
    基本感情認識システム
    顔ランドマーク→7種類感情分析
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("EmotionAnalyzer")
        
        # 設定
        self.smoothing_factor = config.get('ai.emotion.smoothing_factor', 0.8)
        self.confidence_threshold = config.get('ai.emotion.confidence_threshold', 0.6)
        
        # 感情履歴（時系列平滑化用）
        self.emotion_history = deque(maxlen=10)  # 約0.3秒分
        self.current_emotion = EmotionType.NEUTRAL
        self.current_intensity = 0.0
        self.current_confidence = 0.0
        
        # 統計情報
        self.total_analyses = 0
        self.emotion_counts = {emotion: 0 for emotion in EmotionType}
        
        # 感情色彩マッピング
        self.emotion_colors = {
            EmotionType.HAPPY: (1.0, 0.8, 0.0),      # 明るい黄色
            EmotionType.SAD: (0.3, 0.5, 0.8),        # 青
            EmotionType.ANGRY: (0.9, 0.1, 0.1),      # 赤
            EmotionType.SURPRISED: (1.0, 0.1, 0.6),  # ピンク
            EmotionType.NEUTRAL: (0.5, 0.5, 0.5),    # グレー
            EmotionType.FEAR: (0.4, 0.2, 0.6),       # 紫
            EmotionType.DISGUST: (0.2, 0.7, 0.2)     # 緑
        }
        
        self.logger.info("✅ 感情認識エンジン初期化完了")
    
    def analyze_emotion(self, mediapipe_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        感情分析メイン処理
        
        Args:
            mediapipe_results: MediaPipe処理結果
            
        Returns:
            Dict: 感情分析結果
        """
        analysis_start = time.time()
        
        try:
            # 顔が検出されていない場合
            if not mediapipe_results.get('face_detected', False):
                return self._create_neutral_result()
            
            # 顔ランドマーク取得
            face_landmarks = mediapipe_results.get('face_landmarks')
            face_key_points = mediapipe_results.get('face_key_points')
            
            if face_landmarks is None or face_key_points is None:
                return self._create_neutral_result()
            
            # 特徴抽出
            features = self._extract_emotion_features(face_landmarks, face_key_points)
            
            # 感情分析
            emotion_scores = self._analyze_emotion_from_features(features)
            
            # 最も可能性の高い感情決定
            predicted_emotion, confidence = self._determine_primary_emotion(emotion_scores)
            
            # 時系列平滑化
            smoothed_result = self._apply_temporal_smoothing(predicted_emotion, confidence)
            
            # 結果作成
            result = {
                'emotion': smoothed_result['emotion'],
                'intensity': smoothed_result['intensity'],
                'confidence': smoothed_result['confidence'],
                'emotion_scores': emotion_scores,
                'features': features,
                'color': self.emotion_colors[smoothed_result['emotion']],
                'analysis_time': time.time() - analysis_start
            }
            
            # 統計更新
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"感情分析エラー: {e}")
            return self._create_neutral_result()
    
    def _extract_emotion_features(self, landmarks: np.ndarray, key_points: Dict) -> Dict[str, float]:
        """感情特徴抽出"""
        features = {}
        
        try:
            # 口の特徴
            if 'mouth' in key_points:
                mouth_features = self._analyze_mouth_shape(key_points['mouth'])
                features.update(mouth_features)
            
            # 目の特徴
            if 'left_eye' in key_points and 'right_eye' in key_points:
                eye_features = self._analyze_eye_shape(key_points['left_eye'], key_points['right_eye'])
                features.update(eye_features)
            
            # 眉の特徴
            if 'left_eyebrow' in key_points and 'right_eyebrow' in key_points:
                eyebrow_features = self._analyze_eyebrow_position(key_points['left_eyebrow'], key_points['right_eyebrow'])
                features.update(eyebrow_features)
            
            # 顔全体の特徴
            if 'face_oval' in key_points:
                face_features = self._analyze_face_geometry(key_points['face_oval'])
                features.update(face_features)
            
        except Exception as e:
            self.logger.error(f"特徴抽出エラー: {e}")
        
        return features
    
    def _analyze_mouth_shape(self, mouth_points: np.ndarray) -> Dict[str, float]:
        """口の形状分析"""
        if len(mouth_points) < 4:
            return {}
        
        try:
            # 口角の位置関係
            mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
            mouth_height = np.linalg.norm(mouth_points[3] - mouth_points[9])
            
            # 口角の上下位置
            left_corner = mouth_points[0]
            right_corner = mouth_points[6]
            center_top = mouth_points[3]
            center_bottom = mouth_points[9]
            
            # 笑顔指標（口角が上がっているか）
            smile_indicator = ((left_corner[1] + right_corner[1]) / 2 - center_top[1]) / mouth_height
            
            # 口の開き具合
            mouth_openness = mouth_height / mouth_width
            
            return {
                'mouth_smile': max(0, smile_indicator),  # 0-1の範囲
                'mouth_frown': max(0, -smile_indicator),  # 0-1の範囲
                'mouth_openness': mouth_openness,
                'mouth_width_ratio': mouth_width / 100.0  # 正規化
            }
            
        except Exception as e:
            self.logger.error(f"口分析エラー: {e}")
            return {}
    
    def _analyze_eye_shape(self, left_eye: np.ndarray, right_eye: np.ndarray) -> Dict[str, float]:
        """目の形状分析"""
        if len(left_eye) < 6 or len(right_eye) < 6:
            return {}
        
        try:
            # 目の開き具合計算
            def eye_openness(eye_points):
                eye_height = np.linalg.norm(eye_points[1] - eye_points[5])
                eye_width = np.linalg.norm(eye_points[0] - eye_points[3])
                return eye_height / eye_width if eye_width > 0 else 0
            
            left_openness = eye_openness(left_eye)
            right_openness = eye_openness(right_eye)
            avg_eye_openness = (left_openness + right_openness) / 2
            
            # 目の幅（驚き・怒りの判定）
            left_width = np.linalg.norm(left_eye[0] - left_eye[3])
            right_width = np.linalg.norm(right_eye[0] - right_eye[3])
            avg_eye_width = (left_width + right_width) / 2
            
            return {
                'eye_openness': avg_eye_openness,
                'eye_width': avg_eye_width / 50.0,  # 正規化
                'eye_asymmetry': abs(left_openness - right_openness)
            }
            
        except Exception as e:
            self.logger.error(f"目分析エラー: {e}")
            return {}
    
    def _analyze_eyebrow_position(self, left_eyebrow: np.ndarray, right_eyebrow: np.ndarray) -> Dict[str, float]:
        """眉の位置分析"""
        if len(left_eyebrow) < 5 or len(right_eyebrow) < 5:
            return {}
        
        try:
            # 眉の高さ計算
            left_height = np.mean(left_eyebrow[:, 1])
            right_height = np.mean(right_eyebrow[:, 1])
            avg_eyebrow_height = (left_height + right_height) / 2
            
            # 眉の傾き（怒り・悲しみの判定）
            left_slope = (left_eyebrow[-1][1] - left_eyebrow[0][1]) / (left_eyebrow[-1][0] - left_eyebrow[0][0] + 1e-6)
            right_slope = (right_eyebrow[-1][1] - right_eyebrow[0][1]) / (right_eyebrow[-1][0] - right_eyebrow[0][0] + 1e-6)
            
            # 眉間の距離
            eyebrow_distance = np.linalg.norm(left_eyebrow[-1] - right_eyebrow[0])
            
            return {
                'eyebrow_height': avg_eyebrow_height / 100.0,  # 正規化
                'eyebrow_slope': (left_slope + right_slope) / 2,
                'eyebrow_furrow': 1.0 / (eyebrow_distance + 1e-6)  # 眉間のしわ
            }
            
        except Exception as e:
            self.logger.error(f"眉分析エラー: {e}")
            return {}
    
    def _analyze_face_geometry(self, face_oval: np.ndarray) -> Dict[str, float]:
        """顔全体の幾何学的特徴"""
        if len(face_oval) < 10:
            return {}
        
        try:
            # 顔の縦横比
            face_height = np.max(face_oval[:, 1]) - np.min(face_oval[:, 1])
            face_width = np.max(face_oval[:, 0]) - np.min(face_oval[:, 0])
            face_ratio = face_height / face_width if face_width > 0 else 1.0
            
            # 顔の面積
            face_area = face_height * face_width
            
            return {
                'face_ratio': face_ratio,
                'face_area': face_area / 10000.0,  # 正規化
                'face_compactness': 4 * math.pi * face_area / (face_width ** 2) if face_width > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"顔幾何学分析エラー: {e}")
            return {}
    
    def _analyze_emotion_from_features(self, features: Dict[str, float]) -> Dict[EmotionType, float]:
        """特徴から感情スコア計算"""
        scores = {emotion: 0.0 for emotion in EmotionType}
        
        if not features:
            scores[EmotionType.NEUTRAL] = 1.0
            return scores
        
        try:
            # HAPPY（笑顔）
            if 'mouth_smile' in features:
                scores[EmotionType.HAPPY] += features['mouth_smile'] * 0.6
            if 'eye_openness' in features:
                scores[EmotionType.HAPPY] += (1.0 - features['eye_openness']) * 0.3  # 目が細くなる
            
            # SAD（悲しみ）
            if 'mouth_frown' in features:
                scores[EmotionType.SAD] += features['mouth_frown'] * 0.5
            if 'eyebrow_slope' in features:
                scores[EmotionType.SAD] += max(0, -features['eyebrow_slope']) * 0.4
            
            # ANGRY（怒り）
            if 'eyebrow_furrow' in features:
                scores[EmotionType.ANGRY] += features['eyebrow_furrow'] * 0.4
            if 'mouth_frown' in features:
                scores[EmotionType.ANGRY] += features['mouth_frown'] * 0.3
            if 'eye_openness' in features:
                scores[EmotionType.ANGRY] += features['eye_openness'] * 0.2
            
            # SURPRISED（驚き）
            if 'eye_openness' in features:
                scores[EmotionType.SURPRISED] += features['eye_openness'] * 0.5
            if 'mouth_openness' in features:
                scores[EmotionType.SURPRISED] += features['mouth_openness'] * 0.3
            if 'eyebrow_height' in features:
                scores[EmotionType.SURPRISED] += (1.0 - features['eyebrow_height']) * 0.2
            
            # FEAR（恐怖）
            if 'eye_openness' in features:
                scores[EmotionType.FEAR] += features['eye_openness'] * 0.4
            if 'eyebrow_height' in features:
                scores[EmotionType.FEAR] += (1.0 - features['eyebrow_height']) * 0.3
            if 'mouth_openness' in features:
                scores[EmotionType.FEAR] += features['mouth_openness'] * 0.2
            
            # DISGUST（嫌悪）
            if 'mouth_frown' in features:
                scores[EmotionType.DISGUST] += features['mouth_frown'] * 0.4
            if 'eyebrow_furrow' in features:
                scores[EmotionType.DISGUST] += features['eyebrow_furrow'] * 0.3
            
            # NEUTRAL（中性）
            neutral_score = 1.0 - max(scores.values())
            scores[EmotionType.NEUTRAL] = max(0.2, neutral_score)  # 最小値保証
            
            # 正規化
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {emotion: score / total_score for emotion, score in scores.items()}
            
        except Exception as e:
            self.logger.error(f"感情スコア計算エラー: {e}")
            scores = {emotion: 0.0 for emotion in EmotionType}
            scores[EmotionType.NEUTRAL] = 1.0
        
        return scores
    
    def _determine_primary_emotion(self, emotion_scores: Dict[EmotionType, float]) -> Tuple[EmotionType, float]:
        """主要感情決定"""
        if not emotion_scores:
            return EmotionType.NEUTRAL, 0.0
        
        # 最高スコアの感情
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary_emotion]
        
        # 信頼度が閾値以下の場合はNEUTRAL
        if confidence < self.confidence_threshold:
            return EmotionType.NEUTRAL, confidence
        
        return primary_emotion, confidence
    
    def _apply_temporal_smoothing(self, emotion: EmotionType, confidence: float) -> Dict[str, Any]:
        """時系列平滑化"""
        # 感情履歴に追加
        self.emotion_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        if len(self.emotion_history) == 0:
            return {
                'emotion': EmotionType.NEUTRAL,
                'intensity': 0.0,
                'confidence': 0.0
            }
        
        # 重み付き平均計算
        weights = np.exp(np.linspace(-2, 0, len(self.emotion_history)))
        total_weight = np.sum(weights)
        
        # 感情分布計算
        emotion_weights = {emotion: 0.0 for emotion in EmotionType}
        confidence_sum = 0.0
        
        for i, entry in enumerate(self.emotion_history):
            weight = weights[i] / total_weight
            emotion_weights[entry['emotion']] += weight * entry['confidence']
            confidence_sum += weight * entry['confidence']
        
        # 最終感情決定
        if emotion_weights:
            smoothed_emotion = max(emotion_weights, key=emotion_weights.get)
            smoothed_confidence = emotion_weights[smoothed_emotion]
            smoothed_intensity = min(1.0, smoothed_confidence * 1.5)  # 強度調整
        else:
            smoothed_emotion = EmotionType.NEUTRAL
            smoothed_confidence = 0.0
            smoothed_intensity = 0.0
        
        # 現在の状態更新
        self.current_emotion = smoothed_emotion
        self.current_intensity = smoothed_intensity
        self.current_confidence = smoothed_confidence
        
        return {
            'emotion': smoothed_emotion,
            'intensity': smoothed_intensity,
            'confidence': smoothed_confidence
        }
    
    def _create_neutral_result(self) -> Dict[str, Any]:
        """ニュートラル結果作成"""
        return {
            'emotion': EmotionType.NEUTRAL,
            'intensity': 0.0,
            'confidence': 0.0,
            'emotion_scores': {emotion: 0.0 for emotion in EmotionType},
            'features': {},
            'color': self.emotion_colors[EmotionType.NEUTRAL],
            'analysis_time': 0.0
        }
    
    def _update_stats(self, result: Dict[str, Any]):
        """統計更新"""
        self.total_analyses += 1
        emotion = result['emotion']
        if emotion in self.emotion_counts:
            self.emotion_counts[emotion] += 1
    
    def get_current_emotion_info(self) -> Dict[str, Any]:
        """現在の感情情報取得"""
        return {
            'emotion': self.current_emotion,
            'intensity': self.current_intensity,
            'confidence': self.current_confidence,
            'color': self.emotion_colors[self.current_emotion],
            'emotion_name': self.current_emotion.name.lower()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            'total_analyses': self.total_analyses,
            'emotion_distribution': {emotion.name: count for emotion, count in self.emotion_counts.items()},
            'current_emotion': self.current_emotion.name,
            'current_intensity': self.current_intensity,
            'current_confidence': self.current_confidence
        }
    
    def reset_history(self):
        """履歴リセット"""
        self.emotion_history.clear()
        self.current_emotion = EmotionType.NEUTRAL
        self.current_intensity = 0.0
        self.current_confidence = 0.0
        self.logger.info("感情認識履歴リセット")

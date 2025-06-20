import numpy as np
import time
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Tuple

class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"

class EmotionAnalyzer:
    """基本感情認識エンジン（Day 3版）"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 感情履歴（平滑化用）
        emotion_config = config.get('ai_processing', {}).get('emotion', {})
        history_size = emotion_config.get('smoothing_window', 10)
        self.emotion_history = deque(maxlen=history_size)
        
        # 設定パラメータ
        self.confidence_threshold = emotion_config.get('confidence_threshold', 0.6)
        self.smoothing_factor = 0.7  # 履歴重み
        
        # 顔ランドマーク重要インデックス
        self.landmark_indices = {
            'mouth_corners': [61, 291],     # 口角
            'mouth_top': [13],              # 上唇中央
            'mouth_bottom': [14],           # 下唇中央
            'left_eyebrow_inner': [70],     # 左眉内側
            'right_eyebrow_inner': [300],   # 右眉内側
            'left_eyebrow_outer': [46],     # 左眉外側
            'right_eyebrow_outer': [276],   # 右眉外側
            'left_eye_top': [159],          # 左目上
            'left_eye_bottom': [145],       # 左目下
            'right_eye_top': [386],         # 右目上
            'right_eye_bottom': [374],      # 右目下
            'nose_tip': [1],                # 鼻先
            'nose_bridge': [6]              # 鼻根
        }
        
        # ベースライン特徴量（キャリブレーション用）
        self.baseline_features = None
        self.calibration_frames = []
        
        # 処理統計
        self.analysis_times = []
        self.analysis_count = 0
        
        print("✅ Emotion Analyzer 初期化完了")
    
    def analyze_emotion(self, face_detection_result: Dict) -> Tuple[Emotion, float]:
        """感情分析メイン処理"""
        start_time = time.time()
        
        try:
            if not face_detection_result.get('face_detected'):
                return Emotion.NEUTRAL, 0.0
            
            landmarks = face_detection_result.get('landmarks')
            if landmarks is None:
                return Emotion.NEUTRAL, 0.0
            
            # 特徴量抽出
            features = self._extract_facial_features(landmarks)
            
            # 感情分類
            emotion, confidence = self._classify_emotion(features)
            
            # 履歴ベース平滑化
            smoothed_emotion, smoothed_confidence = self._smooth_emotion(emotion, confidence)
            
            # 処理時間記録
            processing_time = time.time() - start_time
            self.analysis_times.append(processing_time)
            if len(self.analysis_times) > 100:
                self.analysis_times.pop(0)
            
            self.analysis_count += 1
            
            return smoothed_emotion, smoothed_confidence
            
        except Exception as e:
            print(f"⚠️  感情分析エラー: {e}")
            return Emotion.NEUTRAL, 0.0
    
    def _extract_facial_features(self, landmarks) -> Dict[str, float]:
        """顔特徴量抽出"""
        features = {}
        
        try:
            # 口の特徴量
            features.update(self._extract_mouth_features(landmarks))
            
            # 眉の特徴量
            features.update(self._extract_eyebrow_features(landmarks))
            
            # 目の特徴量
            features.update(self._extract_eye_features(landmarks))
            
            # 全体的な特徴量
            features.update(self._extract_global_features(landmarks))
            
        except Exception as e:
            print(f"⚠️  特徴量抽出エラー: {e}")
            # エラー時はニュートラルな特徴量を返す
            features = self._get_neutral_features()
        
        return features
    
    def _extract_mouth_features(self, landmarks) -> Dict[str, float]:
        """口の特徴量抽出"""
        features = {}
        
        # 口角の位置
        left_corner = landmarks.landmark[61]   # 左口角
        right_corner = landmarks.landmark[291] # 右口角
        mouth_top = landmarks.landmark[13]     # 上唇中央
        mouth_bottom = landmarks.landmark[14]  # 下唇中央
        
        # 口角の高さ（笑顔検出）
        corner_height = (left_corner.y + right_corner.y) / 2
        mouth_center_height = (mouth_top.y + mouth_bottom.y) / 2
        features['mouth_corner_lift'] = mouth_center_height - corner_height
        
        # 口の開き具合
        mouth_height = abs(mouth_top.y - mouth_bottom.y)
        mouth_width = abs(right_corner.x - left_corner.x)
        features['mouth_openness'] = mouth_height / mouth_width if mouth_width > 0 else 0
        
        # 口の幅（表情の強度）
        features['mouth_width'] = mouth_width
        
        return features
    
    def _extract_eyebrow_features(self, landmarks) -> Dict[str, float]:
        """眉の特徴量抽出"""
        features = {}
        
        # 眉の位置
        left_eyebrow_inner = landmarks.landmark[70]
        right_eyebrow_inner = landmarks.landmark[300]
        left_eyebrow_outer = landmarks.landmark[46]
        right_eyebrow_outer = landmarks.landmark[276]
        
        # 眉の高さ（驚き検出）
        eyebrow_height = (left_eyebrow_inner.y + right_eyebrow_inner.y + 
                         left_eyebrow_outer.y + right_eyebrow_outer.y) / 4
        features['eyebrow_height'] = -eyebrow_height  # Y座標は下向きが正のため反転
        
        # 眉間の距離（怒り検出）
        eyebrow_distance = abs(right_eyebrow_inner.x - left_eyebrow_inner.x)
        features['eyebrow_distance'] = eyebrow_distance
        
        # 眉の傾き
        left_eyebrow_slope = (left_eyebrow_outer.y - left_eyebrow_inner.y) / \
                            (left_eyebrow_outer.x - left_eyebrow_inner.x) if \
                            left_eyebrow_outer.x != left_eyebrow_inner.x else 0
        right_eyebrow_slope = (right_eyebrow_inner.y - right_eyebrow_outer.y) / \
                             (right_eyebrow_inner.x - right_eyebrow_outer.x) if \
                             right_eyebrow_inner.x != right_eyebrow_outer.x else 0
        features['eyebrow_asymmetry'] = abs(left_eyebrow_slope - right_eyebrow_slope)
        
        return features
    
    def _extract_eye_features(self, landmarks) -> Dict[str, float]:
        """目の特徴量抽出"""
        features = {}
        
        # 目の開き具合
        left_eye_top = landmarks.landmark[159]
        left_eye_bottom = landmarks.landmark[145]
        right_eye_top = landmarks.landmark[386]
        right_eye_bottom = landmarks.landmark[374]
        
        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y)
        features['eye_openness'] = (left_eye_openness + right_eye_openness) / 2
        
        # 目の非対称性
        features['eye_asymmetry'] = abs(left_eye_openness - right_eye_openness)
        
        return features
    
    def _extract_global_features(self, landmarks) -> Dict[str, float]:
        """全体的な特徴量抽出"""
        features = {}
        
        # 顔の縦横比
        # 顔の上端（額）
        forehead = landmarks.landmark[10]
        # 顔の下端（顎）
        chin = landmarks.landmark[152]
        # 顔の左端
        left_face = landmarks.landmark[234]
        # 顔の右端
        right_face = landmarks.landmark[454]
        
        face_height = abs(forehead.y - chin.y)
        face_width = abs(right_face.x - left_face.x)
        features['face_aspect_ratio'] = face_height / face_width if face_width > 0 else 1
        
        return features
    
    def _classify_emotion(self, features: Dict[str, float]) -> Tuple[Emotion, float]:
        """特徴量から感情分類"""
        # ベースラインとの差分計算（キャリブレーションされている場合）
        if self.baseline_features:
            features = self._normalize_features(features)
        
        emotion_scores = {}
        
        # 笑顔検出
        happiness_score = 0
        if features.get('mouth_corner_lift', 0) > 0.01:  # 口角が上がっている
            happiness_score += features['mouth_corner_lift'] * 50
        if features.get('eye_openness', 0) > 0.02:  # 目が適度に開いている
            happiness_score += 0.3
        emotion_scores[Emotion.HAPPY] = max(0, min(1, happiness_score))
        
        # 驚き検出
        surprise_score = 0
        if features.get('eyebrow_height', 0) > 0.5:  # 眉が上がっている
            surprise_score += features['eyebrow_height'] - 0.5
        if features.get('eye_openness', 0) > 0.03:  # 目が大きく開いている
            surprise_score += (features['eye_openness'] - 0.03) * 10
        if features.get('mouth_openness', 0) > 0.02:  # 口が開いている
            surprise_score += features['mouth_openness'] * 5
        emotion_scores[Emotion.SURPRISED] = max(0, min(1, surprise_score))
        
        # 悲しみ検出
        sadness_score = 0
        if features.get('mouth_corner_lift', 0) < -0.005:  # 口角が下がっている
            sadness_score += abs(features['mouth_corner_lift']) * 30
        if features.get('eyebrow_height', 0) < 0.4:  # 眉が下がっている
            sadness_score += (0.4 - features['eyebrow_height']) * 2
        emotion_scores[Emotion.SAD] = max(0, min(1, sadness_score))
        
        # 怒り検出
        anger_score = 0
        if features.get('eyebrow_distance', 0) < 0.08:  # 眉間が狭い
            anger_score += (0.08 - features['eyebrow_distance']) * 5
        if features.get('eyebrow_asymmetry', 0) > 0.02:  # 眉が非対称
            anger_score += features['eyebrow_asymmetry'] * 10
        if features.get('mouth_corner_lift', 0) < -0.002:  # 口角が下がっている
            anger_score += abs(features['mouth_corner_lift']) * 20
        emotion_scores[Emotion.ANGRY] = max(0, min(1, anger_score))
        
        # ニュートラル（他の感情が低い場合）
        max_other_score = max([score for emotion, score in emotion_scores.items()])
        emotion_scores[Emotion.NEUTRAL] = max(0, 1 - max_other_score * 1.5)
        
        # 最高スコアの感情を選択
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[best_emotion]
        
        # 信頼度が閾値以下の場合はニュートラル
        if confidence < self.confidence_threshold:
            return Emotion.NEUTRAL, confidence
        
        return best_emotion, confidence
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """ベースライン基準で特徴量正規化"""
        normalized = {}
        for key, value in features.items():
            baseline_value = self.baseline_features.get(key, value)
            normalized[key] = value - baseline_value
        return normalized
    
    def _smooth_emotion(self, emotion: Emotion, confidence: float) -> Tuple[Emotion, float]:
        """履歴ベース感情平滑化"""
        # 現在の結果を履歴に追加
        self.emotion_history.append((emotion, confidence))
        
        if len(self.emotion_history) < 3:
            return emotion, confidence
        
        # 重み付き平均計算
        emotion_weights = {}
        total_weight = 0
        
        for i, (hist_emotion, hist_confidence) in enumerate(self.emotion_history):
            # 新しい履歴ほど重い重み
            weight = (i + 1) * hist_confidence
            
            if hist_emotion not in emotion_weights:
                emotion_weights[hist_emotion] = 0
            emotion_weights[hist_emotion] += weight
            total_weight += weight
        
        if total_weight == 0:
            return Emotion.NEUTRAL, 0.0
        
        # 最も重い感情を選択
        best_emotion = max(emotion_weights, key=emotion_weights.get)
        avg_confidence = emotion_weights[best_emotion] / total_weight
        
        return best_emotion, min(1.0, avg_confidence)
    
    def _get_neutral_features(self) -> Dict[str, float]:
        """ニュートラル特徴量"""
        return {
            'mouth_corner_lift': 0.0,
            'mouth_openness': 0.01,
            'mouth_width': 0.1,
            'eyebrow_height': 0.5,
            'eyebrow_distance': 0.1,
            'eyebrow_asymmetry': 0.0,
            'eye_openness': 0.02,
            'eye_asymmetry': 0.0,
            'face_aspect_ratio': 1.3
        }
    
    def calibrate_baseline(self, face_detection_results: List[Dict]) -> bool:
        """ベースライン キャリブレーション"""
        if len(face_detection_results) < 10:
            return False
        
        print("🎯 感情認識ベースライン キャリブレーション中...")
        
        feature_sums = {}
        valid_count = 0
        
        for result in face_detection_results:
            if result.get('face_detected') and result.get('landmarks'):
                features = self._extract_facial_features(result['landmarks'])
                
                for key, value in features.items():
                    if key not in feature_sums:
                        feature_sums[key] = 0
                    feature_sums[key] += value
                
                valid_count += 1
        
        if valid_count > 0:
            self.baseline_features = {
                key: value / valid_count for key, value in feature_sums.items()
            }
            print(f"✅ ベースラインキャリブレーション完了（{valid_count}フレーム）")
            return True
        
        return False
    
    def get_performance_stats(self) -> Dict:
        """パフォーマンス統計取得"""
        if not self.analysis_times:
            return {}
        
        return {
            'avg_analysis_time': sum(self.analysis_times) / len(self.analysis_times),
            'max_analysis_time': max(self.analysis_times),
            'min_analysis_time': min(self.analysis_times),
            'analysis_count': self.analysis_count,
            'fps': 1.0 / (sum(self.analysis_times) / len(self.analysis_times)) if self.analysis_times else 0,
            'emotion_history_size': len(self.emotion_history),
            'baseline_calibrated': self.baseline_features is not None
        }

# テスト実行用
if __name__ == "__main__":
    print("🔍 Emotion Analyzer テスト開始...")
    
    # テスト設定
    config = {
        'ai_processing': {
            'emotion': {
                'smoothing_window': 10,
                'confidence_threshold': 0.6
            }
        }
    }
    
    analyzer = EmotionAnalyzer(config)
    
    # 基本機能テスト
    print("📊 基本機能テスト...")
    
    # ダミーの検出結果でテスト
    dummy_result = {
        'face_detected': False,
        'landmarks': None
    }
    
    emotion, confidence = analyzer.analyze_emotion(dummy_result)
    print(f"顔なし: {emotion.value}, 信頼度: {confidence:.3f}")
    
    # 統計表示
    stats = analyzer.get_performance_stats()
    print(f"📊 性能統計: {stats}")
    
    print("✅ Emotion Analyzer テスト完了")

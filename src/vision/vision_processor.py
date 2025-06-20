import cv2
import time
from typing import Dict, Optional
from .face_detector import FaceDetector
from .hand_detector import HandDetector
from ..emotion.emotion_analyzer import EmotionAnalyzer, Emotion

class VisionProcessor:
    """統合画像処理・AI システム（Day 3版）"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # AI コンポーネント初期化
        self.face_detector = FaceDetector(config)
        self.hand_detector = HandDetector(config)
        self.emotion_analyzer = EmotionAnalyzer(config)
        
        # 処理状態
        self.processing_enabled = True
        self.debug_mode = config.get('debug_mode', False)
        
        # パフォーマンス統計
        self.total_processing_times = []
        self.frame_count = 0
        
        print("✅ Vision Processor 統合初期化完了")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """統合フレーム処理"""
        if not self.processing_enabled or frame is None:
            return self._get_empty_result()
        
        start_time = time.time()
        
        try:
            # 1. 顔検出
            face_result = self.face_detector.detect_face(frame)
            
            # 2. 手検出
            hand_result = self.hand_detector.detect_hands(frame)
            
            # 3. 感情分析（顔が検出された場合）
            emotion = Emotion.NEUTRAL
            emotion_confidence = 0.0
            if face_result['face_detected']:
                emotion, emotion_confidence = self.emotion_analyzer.analyze_emotion(face_result)
            
            # 統合結果作成
            integrated_result = {
                'timestamp': time.time(),
                'frame_processed': True,
                
                # 顔検出結果
                'face': face_result,
                
                # 手検出結果
                'hands': hand_result,
                
                # 感情認識結果
                'emotion': {
                    'emotion': emotion,
                    'confidence': emotion_confidence,
                    'emotion_name': emotion.value
                },
                
                # 統合処理時間
                'processing_time': 0  # 後で設定
            }
            
            # 処理時間記録
            total_time = time.time() - start_time
            integrated_result['processing_time'] = total_time
            
            self.total_processing_times.append(total_time)
            if len(self.total_processing_times) > 100:
                self.total_processing_times.pop(0)
            
            self.frame_count += 1
            
            return integrated_result
            
        except Exception as e:
            print(f"⚠️  統合処理エラー: {e}")
            return self._get_empty_result()
    
    def _get_empty_result(self) -> Dict:
        """空の結果"""
        return {
            'timestamp': time.time(),
            'frame_processed': False,
            'face': {'face_detected': False},
            'hands': {'hands_detected': False},
            'emotion': {
                'emotion': Emotion.NEUTRAL,
                'confidence': 0.0,
                'emotion_name': 'neutral'
            },
            'processing_time': 0
        }
    
    def draw_all_results(self, frame: np.ndarray, result: Dict, 
                        show_face: bool = True, show_hands: bool = True, 
                        show_emotion: bool = True) -> np.ndarray:
        """全検出結果描画"""
        if not result['frame_processed']:
            return frame
        
        output_frame = frame.copy()
        
        try:
            # 顔検出結果描画
            if show_face and result['face']['face_detected']:
                output_frame = self.face_detector.draw_landmarks(
                    output_frame, result['face'], draw_all=False
                )
            
            # 手検出結果描画
            if show_hands and result['hands']['hands_detected']:
                output_frame = self.hand_detector.draw_landmarks(
                    output_frame, result['hands'], draw_connections=True
                )
            
            # 感情情報描画
            if show_emotion:
                output_frame = self._draw_emotion_info(output_frame, result['emotion'])
            
            # 統合情報描画（デバッグモード時）
            if self.debug_mode:
                output_frame = self._draw_debug_info(output_frame, result)
        
        except Exception as e:
            print(f"⚠️  結果描画エラー: {e}")
        
        return output_frame
    
    def _draw_emotion_info(self, frame: np.ndarray, emotion_result: Dict) -> np.ndarray:
        """感情情報描画"""
        height, width = frame.shape[:2]
        
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        emotion_name = emotion_result['emotion_name']
        
        # 感情に応じた色設定
        emotion_colors = {
            'happy': (0, 255, 255),      # 黄色
            'sad': (255, 0, 0),          # 青
            'angry': (0, 0, 255),        # 赤
            'surprised': (255, 0, 255),  # マゼンタ
            'neutral': (255, 255, 255)   # 白
        }
        
        color = emotion_colors.get(emotion_name, (255, 255, 255))
        
        # 感情情報テキスト
        text = f"Emotion: {emotion_name.upper()}"
        confidence_text = f"Confidence: {confidence:.2f}"
        
        # 背景矩形
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, font, 0.8, thickness)
        
        max_width = max(text_width, conf_width)
        total_height = text_height + conf_height + 20
        
        # 右上角に配置
        rect_x = width - max_width - 20
        rect_y = 10
        
        cv2.rectangle(frame, (rect_x - 10, rect_y), 
                     (rect_x + max_width + 10, rect_y + total_height + 10), 
                     (0, 0, 0), -1)
        
        # テキスト描画
        cv2.putText(frame, text, (rect_x, rect_y + text_height + 5), 
                   font, font_scale, color, thickness)
        cv2.putText(frame, confidence_text, (rect_x, rect_y + text_height + conf_height + 15), 
                   font, 0.8, color, thickness)
        
        return frame
    
    def _draw_debug_info(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """デバッグ情報描画"""
        height, width = frame.shape[:2]
        
        debug_info = [
            f"Frame: {self.frame_count}",
            f"Processing: {result['processing_time']:.3f}s",
            f"Face: {'YES' if result['face']['face_detected'] else 'NO'}",
            f"Hands: {result['hands']['hand_count']}",
            f"FPS: {self._get_current_fps():.1f}"
        ]
        
        # 左下に描画
        y_start = height - len(debug_info) * 25 - 10
        
        for i, info in enumerate(debug_info):
            y_pos = y_start + i * 25
            cv2.putText(frame, info, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _get_current_fps(self) -> float:
        """現在のFPS取得"""
        if len(self.total_processing_times) < 10:
            return 0.0
        
        recent_times = self.total_processing_times[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def calibrate_emotion_baseline(self, frame_count: int = 30) -> bool:
        """感情認識ベースライン キャリブレーション"""
        print(f"🎯 感情認識キャリブレーション開始（{frame_count}フレーム）...")
        
        calibration_results = []
        
        # カメラからフレーム取得してキャリブレーション
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ キャリブレーション用カメラアクセス失敗")
            return False
        
        print("😐 ニュートラルな表情を保ってください...")
        
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 顔検出のみ実行
            face_result = self.face_detector.detect_face(frame)
            if face_result['face_detected']:
                calibration_results.append(face_result)
            
            # 進捗表示
            if i % 10 == 0:
                print(f"キャリブレーション進捗: {i}/{frame_count}")
        
        cap.release()
        
        # ベースライン設定
        success = self.emotion_analyzer.calibrate_baseline(calibration_results)
        
        if success:
            print("✅ 感情認識キャリブレーション完了")
        else:
            print("❌ 感情認識キャリブレーション失敗")
        
        return success
    
    def get_performance_stats(self) -> Dict:
        """総合パフォーマンス統計"""
        face_stats = self.face_detector.get_performance_stats()
        hand_stats = self.hand_detector.get_performance_stats()
        emotion_stats = self.emotion_analyzer.get_performance_stats()
        
        return {
            'total_frames_processed': self.frame_count,
            'avg_total_time': sum(self.total_processing_times) / len(self.total_processing_times) if self.total_processing_times else 0,
            'current_fps': self._get_current_fps(),
            'face_detection': face_stats,
            'hand_detection': hand_stats,
            'emotion_analysis': emotion_stats
        }
    
    def enable_processing(self, enabled: bool):
        """処理有効/無効切替"""
        self.processing_enabled = enabled
        print(f"Vision Processing: {'ENABLED' if enabled else 'DISABLED'}")
    
    def cleanup(self):
        """リソース解放"""
        self.face_detector.cleanup()
        self.hand_detector.cleanup()
        print("Vision Processor リソース解放完了")

# テスト実行用
if __name__ == "__main__":
    print("🔍 Vision Processor 統合テスト開始...")
    
    # テスト設定
    config = {
        'ai_processing': {
            'vision': {
                'face_detection': {
                    'max_num_faces': 1,
                    'refine_landmarks': True,
                    'min_detection_confidence': 0.7
                },
                'hand_detection': {
                    'max_num_hands': 2,
                    'min_detection_confidence': 0.7
                }
            },
            'emotion': {
                'smoothing_window': 10,
                'confidence_threshold': 0.6
            }
        },
        'debug_mode': True
    }
    
    processor = VisionProcessor(config)
    
    # カメラテスト
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ カメラを開けません")
        exit()
    
    print("🎬 統合テスト開始（C: キャリブレーション, ESC: 終了）...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 統合処理
        result = processor.process_frame(frame)
        
        # 結果描画
        output_frame = processor.draw_all_results(
            frame, result, 
            show_face=True, show_hands=True, show_emotion=True
        )
        
        # 表示
        cv2.imshow('Vision Processor Test', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):  # キャリブレーション
            processor.calibrate_emotion_baseline()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 統計表示
    stats = processor.get_performance_stats()
    print(f"📊 総合性能統計:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    processor.cleanup()
    print("✅ Vision Processor 統合テスト完了")

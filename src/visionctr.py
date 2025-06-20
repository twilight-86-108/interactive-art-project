# src/vision.py - MediaPipe 0.10.x対応版（既存インターフェース維持）
import cv2
import numpy as np

# === MediaPipe 0.10.x対応 ===
try:
    from src.vision.vision_processor_v10 import VisionProcessorV10
    USE_NEW_PROCESSOR = True
    print("✅ MediaPipe 0.10.x プロセッサー使用")
except ImportError:
    print("❌ 新プロセッサーが見つかりません")
    USE_NEW_PROCESSOR = False

class VisionProcessor:
    """コンピュータビジョン処理クラス（MediaPipe 0.10.x対応・既存API互換）"""
    
    def __init__(self, config):
        self.config = config
        
        if USE_NEW_PROCESSOR:
            # 新しいプロセッサー使用
            self.processor = VisionProcessorV10(config)
            print("✅ MediaPipe 0.10.x プロセッサー初期化完了")
        else:
            # フォールバック：基本機能のみ
            print("⚠️ 基本機能モードで動作")
            self.processor = None
            self._init_fallback()
    
    def _init_fallback(self):
        """フォールバック初期化"""
        self.camera = cv2.VideoCapture(0)
        self.last_detection_result = {
            'face_detected': False,
            'hands_detected': False,
            'face_center': None,
            'face_distance': float('inf'),
            'hand_positions': [],
            'frame_shape': (480, 640, 3)
        }
    
    def process_frame(self):
        """フレーム処理 - 既存APIと完全互換"""
        if self.processor:
            return self.processor.process_frame()
        else:
            # フォールバック：カメラフレームのみ
            ret, frame = self.camera.read()
            if ret:
                self.last_detection_result['frame_shape'] = frame.shape
            return self.last_detection_result
    
    def get_debug_info(self):
        """デバッグ情報取得 - 既存APIと互換"""
        if self.processor:
            return self.processor.get_debug_info()
        else:
            return {
                'Face': 'NO',
                'Hands': 'NO', 
                'Face Dist': '0.000',
                'Hand Count': 0,
                'API Version': 'Fallback Mode'
            }
    
    def cleanup(self):
        """リソース解放 - 既存APIと互換"""
        if self.processor:
            self.processor.cleanup()
        elif hasattr(self, 'camera'):
            self.camera.release()
        print("VisionProcessor がクリーンアップされました")
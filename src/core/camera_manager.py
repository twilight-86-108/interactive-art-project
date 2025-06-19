import cv2
import numpy as np
from typing import Optional
import threading
import queue
import time

class CameraManager:
    """基本カメラマネージャー - Day 1 版"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.camera = None
        self.running = False
        print(f"カメラマネージャー初期化: デバイス {device_id}")
    
    def initialize(self) -> bool:
        """カメラ初期化"""
        try:
            print("カメラ初期化中...")
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                print("❌ カメラを開けませんでした")
                return False
            
            # 基本設定
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # テスト撮影
            ret, frame = self.camera.read()
            if not ret:
                print("❌ フレーム取得に失敗")
                return False
            
            print(f"✅ カメラ初期化成功: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"❌ カメラ初期化エラー: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """フレーム取得"""
        if not self.camera or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        return frame if ret else None
    
    def cleanup(self):
        """リソース解放"""
        if self.camera:
            self.camera.release()
        print("カメラリソース解放完了")

# テスト実行用
if __name__ == "__main__":
    import sys
    
    print("🔍 カメラテスト開始...")
    
    manager = CameraManager()
    if manager.initialize():
        print("📸 5秒間フレーム取得テスト...")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            frame = manager.get_frame()
            if frame is not None:
                frame_count += 1
            time.sleep(0.1)
        
        print(f"✅ {frame_count}フレーム取得成功")
        manager.cleanup()
    else:
        print("❌ カメラテスト失敗")
        sys.exit(1)

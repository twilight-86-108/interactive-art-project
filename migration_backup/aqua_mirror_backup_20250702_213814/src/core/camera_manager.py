import cv2
import numpy as np
from typing import Optional, Tuple
import threading
import queue
import time
import logging 

class CameraManager:
    """カメラ管理・最適化クラス"""
    
    def __init__(self, device_index: int = 0, resolution: Tuple[int, int] = (1920, 1080)):
        self.logger = logging.getLogger("CameraManager")
        self.device_index = device_index
        self.resolution = resolution
        
        # OpenCV VideoCapture
        self.cap: Optional[cv2.VideoCapture] = None
        
        # フレーム管理
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0.0
        
        # エラー回復
        self.consecutive_failures = 0
        self.max_failures = 5
    
    def initialize(self) -> bool:
        """カメラ初期化"""
        try:
            self.cap = cv2.VideoCapture(self.device_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"❌ カメラ{self.device_index}を開けません")
                return False
            
            # Logicool C922N最適化設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            # 実際の設定値確認
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"✅ カメラ初期化完了: {actual_width}x{actual_height} @{actual_fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ カメラ初期化失敗: {e}")
            return False
    
    def get_frame(self) -> Optional[np.ndarray]:
        """フレーム取得"""
        try:
            if not self.cap or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            
            if ret:
                self.consecutive_failures = 0
                self._update_fps()
                return frame
            else:
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_failures:
                    self.logger.warning("⚠️ カメラエラー: 再初期化を試行")
                    self._attempt_recovery()
                return None
                
        except Exception as e:
            self.logger.error(f"❌ フレーム取得失敗: {e}")
            return None
    
    def _update_fps(self):
        """FPS計算"""
        current_time = time.time()
        if current_time - self.last_frame_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_frame_time)
            self.frame_count = 0
            self.last_frame_time = current_time
        self.frame_count += 1
    
    def _attempt_recovery(self):
        """カメラ復旧試行"""
        try:
            if self.cap:
                self.cap.release()
            time.sleep(1)
            self.initialize()
        except Exception as e:
            self.logger.error(f"❌ カメラ復旧失敗: {e}")
    
    def get_fps(self) -> float:
        """現在のFPS取得"""
        return self.fps
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.cap:
                self.cap.release()
            self.logger.info("✅ カメラリソース解放完了")
        except Exception as e:
            self.logger.error(f"❌ カメラリソース解放失敗: {e}")

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

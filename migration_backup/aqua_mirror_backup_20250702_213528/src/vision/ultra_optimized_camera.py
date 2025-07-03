"""
超最適化カメラマネージャー
WSL2 + USB カメラ特化最適化
"""

import cv2
import numpy as np
import logging
import threading
import time
from queue import Queue, Empty
from typing import Optional, Dict, Any

class UltraOptimizedCamera:
    """
    WSL2 + USB カメラ特化超最適化マネージャー
    最大FPS重視設計
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("UltraOptimizedCamera")
        
        # 最小リソース設定
        self.device_id = config.get('camera.device_id', 0)
        self.width = 320   # さらに小さく
        self.height = 240  # さらに小さく
        self.target_fps = 30
        
        # カメラオブジェクト
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_available = False
        self.is_streaming = False
        
        # 単一フレームバッファ（最小メモリ）
        self.current_frame = None
        self.frame_lock = threading.RLock()
        
        # 高速処理スレッド
        self.capture_thread = None
        self.stop_event = threading.Event()
        
        # FPS統計
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0.0
        self.last_fps_update = 0
        
    def initialize(self) -> bool:
        """超最適化初期化"""
        try:
            self.logger.info("🚀 超最適化カメラ初期化")
            
            # カメラオープン
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                # フォールバック
                self.cap = cv2.VideoCapture(self.device_id)
                
            if not self.cap.isOpened():
                self.logger.error("❌ カメラオープン失敗")
                return False
            
            # 激的最適化設定
            optimizations = [
                # 解像度最小化
                (cv2.CAP_PROP_FRAME_WIDTH, self.width),
                (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
    
                # FPS最大化
                (cv2.CAP_PROP_FPS, self.target_fps),
    
                # バッファ最小化（重要！）
                (cv2.CAP_PROP_BUFFERSIZE, 1),
    
                # フォーマット強制設定
                (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')),  # MJPEG強制
                (cv2.CAP_PROP_CONVERT_RGB, 1),  # RGB変換有効化
    
                # 自動機能無効化（処理高速化）
                (cv2.CAP_PROP_AUTOFOCUS, 0),
                (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
            ]
            
            # 設定適用
            for prop, value in optimizations:
                try:
                    self.cap.set(prop, value)
                except:
                    pass  # エラーは無視して継続

            # バッファサイズ強制設定（重要なので個別処理）
            for attempt in range(3):  # 複数回設定試行
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
                    if buffer_size <= 1.0:
                        break  # 成功
                except:
                    pass
                time.sleep(0.01)  # 短い待機
            
            # 実際の設定確認
            actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            buffer_size = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            
            self.logger.info(f"📐 実際設定: {actual_w}x{actual_h}@{actual_fps}fps")
            self.logger.info(f"📦 バッファサイズ: {buffer_size}")
            
            # 初期フレーム確認
            success_count = 0
            for attempt in range(10):
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    success_count += 1
                    if success_count >= 3:  # 3回成功したら安定
                        self.current_frame = self._minimal_preprocess(frame)
                        self.is_available = True
                        self.logger.info("✅ 超最適化カメラ初期化完了")
                        return True
                time.sleep(0.01)
            
            self.logger.error("❌ 初期フレーム取得失敗")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ 初期化エラー: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """超高速ストリーミング開始"""
        if self.is_streaming or not self.is_available:
            return False
        
        try:
            self.stop_event.clear()
            self.capture_thread = threading.Thread(
                target=self._ultra_fast_capture_loop, 
                daemon=True
            )
            self.capture_thread.start()
            
            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0
            
            self.logger.info("🚀 超高速ストリーミング開始")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ストリーミング開始エラー: {e}")
            return False
    
    def _ultra_fast_capture_loop(self):
        """超高速フレーム取得ループ"""
        consecutive_failures = 0
        max_failures = 10
        
        while not self.stop_event.is_set():
            try:
                # ノンブロッキング読み取り
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # 最小限前処理
                    processed = self._minimal_preprocess(frame)
                    
                    # 高速フレーム更新
                    with self.frame_lock:
                        self.current_frame = processed
                    
                    self.frame_count += 1
                    consecutive_failures = 0
                    
                    # FPS統計更新（軽量化）
                    if self.frame_count % 30 == 0:
                        self._update_fps_lightweight()
                        
                else:
                    consecutive_failures += 1
                    if consecutive_failures > max_failures:
                        self.logger.warning("⚠️ 連続フレーム取得失敗")
                        break
                    
                    # 短い待機
                    time.sleep(0.001)
                    
            except Exception as e:
                self.logger.error(f"❌ フレーム取得エラー: {e}")
                consecutive_failures += 1
                if consecutive_failures > max_failures:
                    break
                time.sleep(0.01)
        
        self.logger.info("📹 キャプチャループ終了")
    
    def _minimal_preprocess(self, frame: np.ndarray) -> np.ndarray:
        """最小限前処理（高速化）"""
        # フレーム検証
        if frame is None or frame.size == 0:
            return None
    
        # 形状確認・修正
        if len(frame.shape) == 2:
            # グレースケールの場合、3チャンネル化
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # RGBA→BGR変換
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
        # 水平反転
        return cv2.flip(frame, 1)
    
    def _update_fps_lightweight(self):
        """軽量FPS統計更新"""
        if self.start_time:
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.actual_fps = self.frame_count / elapsed
    
    def get_frame(self) -> Optional[np.ndarray]:
        """最新フレーム高速取得"""
        with self.frame_lock:
            if self.current_frame is not None:
                # フレーム形状チェック追加
                if len(self.current_frame.shape) == 3 and self.current_frame.shape[2] == 3:
                    return cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                elif len(self.current_frame.shape) == 2:
                    # グレースケール→RGB変換
                    return cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2RGB)
                else:
                    # 異常形状の場合は生データ返却
                    return self.current_frame
            return None
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """生フレーム取得（変換なし）"""
        with self.frame_lock:
            return self.current_frame
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            'is_available': self.is_available,
            'is_streaming': self.is_streaming,
            'actual_fps': self.actual_fps,
            'frame_count': self.frame_count,
            'resolution': f"{self.width}x{self.height}",
            'optimization_level': 'ULTRA'
        }
    
    def force_frame_grab(self) -> bool:
        """強制フレーム取得（デバッグ用）"""
        if not self.cap:
            return False
        
        try:
            # バッファクリア
            for _ in range(5):
                self.cap.grab()
            
            ret, frame = self.cap.retrieve()
            if ret:
                with self.frame_lock:
                    self.current_frame = self._minimal_preprocess(frame)
                return True
        except:
            pass
        
        return False
    
    def stop_streaming(self):
        """ストリーミング停止"""
        if self.is_streaming:
            self.stop_event.set()
            if self.capture_thread:
                self.capture_thread.join(timeout=2.0)
            self.is_streaming = False
            self.logger.info("📹 ストリーミング停止")
    
    def cleanup(self):
        """リソース解放"""
        self.stop_streaming()
        
        if self.cap:
            self.cap.release()
        
        self.is_available = False
        self.current_frame = None
        
        self.logger.info("✅ 超最適化カメラリソース解放完了")

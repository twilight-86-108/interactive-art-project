"""
Logicool C922N最適化カメラマネージャー
GPU テクスチャストリーミング対応
"""

import cv2
import numpy as np
import logging
import threading
import time
from queue import Queue, Empty
from typing import Optional, Tuple, Dict, Any

class CameraManager:
    """
    Logicool C922N専用最適化カメラマネージャー
    GPU統合・リアルタイムストリーミング対応
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("CameraManager")
        
        # カメラ設定
        self.device_id = config.get('camera.device_id', 0)
        self.width = config.get('camera.width', 1920)
        self.height = config.get('camera.height', 1080)
        self.fps = config.get('camera.fps', 30)
        
        # カメラオブジェクト
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_available = False
        self.is_streaming = False
        
        # フレームバッファ
        self.frame_queue = Queue(maxsize=3)  # 最新3フレーム保持
        self.latest_frame = None
        
        # ストリーミングスレッド
        self.capture_thread = None
        self.stop_event = threading.Event()
        
        # 統計情報
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
        
    def initialize(self) -> bool:
        """カメラ初期化・C922N最適化設定"""
        try:
            self.logger.info(f"Logicool C922N初期化開始（デバイス{self.device_id}）")
            
            # OpenCVカメラオブジェクト作成
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                self.logger.error("❌ カメラオープン失敗")
                return False
            
            # C922N最適化設定
            self._configure_c922n_settings()
            
            # 解像度・FPS設定確認
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"✅ カメラ設定完了: {actual_width}x{actual_height}@{actual_fps}fps")
            
            # 初期フレーム取得テスト
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.latest_frame = frame
                self.is_available = True
                self.logger.info("✅ カメラ初期化完了")
                return True
            else:
                self.logger.error("❌ 初期フレーム取得失敗")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ カメラ初期化エラー: {e}")
            return False
    
    def _configure_c922n_settings(self):
        """Logicool C922N専用最適化設定"""
        try:
            # 解像度設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # FPS設定
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # C922N最適化設定
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # バッファサイズ最小化
            
            # 画質設定
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手動露出
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # オートフォーカス無効
            
            # 明度・コントラスト最適化
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)
            
            self.logger.info("✅ C922N最適化設定完了")
            
        except Exception as e:
            self.logger.warning(f"⚠️ C922N設定エラー（継続）: {e}")
    
    def start_streaming(self) -> bool:
        """非同期ストリーミング開始"""
        if self.is_streaming:
            self.logger.warning("カメラストリーミングは既に開始済み")
            return True
        
        if not self.is_available:
            self.logger.error("カメラが利用できません")
            return False
        
        try:
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.is_streaming = True
            self.logger.info("✅ カメラストリーミング開始")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ストリーミング開始エラー: {e}")
            return False
    
    def _capture_loop(self):
        """フレーム取得ループ（別スレッド）"""
        frame_interval = 1.0 / self.fps
        next_capture = time.time()
        
        while not self.stop_event.is_set():
            current_time = time.time()
            
            # FPS制御
            if current_time < next_capture:
                time.sleep(0.001)  # 1ms待機
                continue
            
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # フレーム前処理
                    processed_frame = self._preprocess_frame(frame)
                    
                    # 最新フレーム更新
                    self.latest_frame = processed_frame
                    
                    # キューに追加（古いフレーム破棄）
                    try:
                        self.frame_queue.put_nowait(processed_frame)
                    except:
                        # キューが満杯の場合、古いフレームを破棄
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(processed_frame)
                            self.dropped_frames += 1
                        except:
                            pass
                    
                    # 統計更新
                    self.frame_count += 1
                    self._update_fps_stats(current_time)
                    
                    next_capture = current_time + frame_interval
                else:
                    self.logger.warning("フレーム取得失敗")
                    time.sleep(0.01)
                    
            except Exception as e:
                self.logger.error(f"フレーム取得エラー: {e}")
                time.sleep(0.01)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """フレーム前処理・最適化"""
        # 水平反転（鏡効果）
        frame = cv2.flip(frame, 1)
        
        # 色空間変換（BGR → RGB）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def _update_fps_stats(self, current_time: float):
        """FPS統計更新"""
        if current_time - self.last_fps_time >= 1.0:
            elapsed = current_time - self.last_fps_time
            self.actual_fps = self.frame_count / elapsed
            
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_frame(self) -> Optional[np.ndarray]:
        """最新フレーム取得（GPU転送用）"""
        return self.latest_frame
    
    def get_frame_from_queue(self) -> Optional[np.ndarray]:
        """キューからフレーム取得"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def get_camera_stats(self) -> Dict[str, Any]:
        """カメラ統計情報取得"""
        return {
            'is_available': self.is_available,
            'is_streaming': self.is_streaming,
            'actual_fps': self.actual_fps,
            'dropped_frames': self.dropped_frames,
            'queue_size': self.frame_queue.qsize(),
            'resolution': f"{self.width}x{self.height}"
        }
    
    def stop_streaming(self):
        """ストリーミング停止"""
        if self.is_streaming:
            self.stop_event.set()
            
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            
            self.is_streaming = False
            self.logger.info("✅ カメラストリーミング停止")
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("カメラリソース解放中...")
        
        self.stop_streaming()
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # キュークリア
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        self.is_available = False
        self.logger.info("✅ カメラリソース解放完了")

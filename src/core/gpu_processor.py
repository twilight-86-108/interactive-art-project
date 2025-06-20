# src/core/gpu_processor.py
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

class GPUProcessor:
    """GPU加速処理クラス（エラー修正版）"""
    
    def __init__(self):
        self.gpu_available = False
        self.logger = logging.getLogger(__name__)
        
        try:
            # CUDA利用可能性確認
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_devices > 0:
                self.gpu_available = True
                self.logger.info(f"CUDA GPU利用可能: {cuda_devices} devices")
            else:
                self.logger.warning("CUDA GPU利用不可、CPU処理で継続")
        except Exception as e:
            self.logger.warning(f"GPU初期化エラー: {e}、CPU処理で継続")
            self.gpu_available = False
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """フレーム前処理（GPU最適化）"""
        if frame is None:
            return None
            
        try:
            if self.gpu_available:
                return self._gpu_preprocess(frame, target_size)
            else:
                return self._cpu_preprocess(frame, target_size)
        except Exception as e:
            self.logger.error(f"フレーム前処理エラー: {e}")
            return self._cpu_preprocess(frame, target_size)
    
    def _gpu_preprocess(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """GPU加速前処理"""
        try:
            # GPU メモリにアップロード
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # GPU でリサイズ
            gpu_resized = cv2.cuda.resize(gpu_frame, target_size)
            
            # RGB変換
            gpu_rgb = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2RGB)
            
            # CPU メモリにダウンロード
            result = gpu_rgb.download()
            
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU処理失敗、CPUにフォールバック: {e}")
            return self._cpu_preprocess(frame, target_size)
    
    def _cpu_preprocess(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """CPU前処理（フォールバック）"""
        try:
            # リサイズ
            resized = cv2.resize(frame, target_size)
            
            # RGB変換
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            return rgb_frame
            
        except Exception as e:
            self.logger.error(f"CPU前処理エラー: {e}")
            return frame
    
    def is_gpu_available(self) -> bool:
        """GPU利用可能性取得"""
        return self.gpu_available
    
    def cleanup(self):
        """リソース解放"""
        # GPU リソースのクリーンアップは自動的に行われる
        pass
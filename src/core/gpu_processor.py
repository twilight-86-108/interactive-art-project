import cv2
import numpy as np
import time
from typing import Optional, Tuple

class GPUProcessor:
    """GPU加速処理クラス（Day 2版）"""
    
    def __init__(self):
        self.gpu_available = False
        self.device_count = 0
        self.processing_times = []
        
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """GPU利用可能性確認"""
        try:
            # CUDA デバイス数確認
            self.device_count = cv2.cuda.getCudaEnabledDeviceCount()
            
            if self.device_count > 0:
                self.gpu_available = True
                print(f"✅ GPU利用可能: {self.device_count} devices")
                
                # GPU情報表示
                for i in range(self.device_count):
                    device_info = cv2.cuda.DeviceInfo(i)
                    print(f"   Device {i}: {device_info.name()}")
            else:
                print("⚠️  CUDA GPU が見つかりません（CPU処理で継続）")
                
        except Exception as e:
            print(f"⚠️  GPU確認エラー: {e}（CPU処理で継続）")
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int], 
                    interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        """GPU加速フレームリサイズ"""
        start_time = time.time()
        
        try:
            if self.gpu_available:
                result = self._gpu_resize(frame, target_size, interpolation)
            else:
                result = self._cpu_resize(frame, target_size, interpolation)
                
            # 処理時間記録
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return result
            
        except Exception as e:
            print(f"⚠️  リサイズエラー: {e}")
            # エラー時はCPU処理にフォールバック
            return self._cpu_resize(frame, target_size, interpolation)
    
    def _gpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int], 
                   interpolation) -> np.ndarray:
        """GPU処理実装"""
        # GPU メモリにアップロード
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # GPU上でリサイズ
        gpu_resized = cv2.cuda.resize(gpu_frame, target_size, interpolation=interpolation)
        
        # CPU メモリにダウンロード
        result = gpu_resized.download()
        
        return result
    
    def _cpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int], 
                   interpolation) -> np.ndarray:
        """CPU処理実装"""
        return cv2.resize(frame, target_size, interpolation=interpolation)
    
    def convert_color(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """GPU加速色空間変換"""
        try:
            if self.gpu_available:
                return self._gpu_convert_color(frame, conversion_code)
            else:
                return self._cpu_convert_color(frame, conversion_code)
                
        except Exception as e:
            print(f"⚠️  色変換エラー: {e}")
            return self._cpu_convert_color(frame, conversion_code)
    
    def _gpu_convert_color(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """GPU色変換"""
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        gpu_converted = cv2.cuda.cvtColor(gpu_frame, conversion_code)
        
        return gpu_converted.download()
    
    def _cpu_convert_color(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """CPU色変換"""
        return cv2.cvtColor(frame, conversion_code)
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        if not self.processing_times:
            return {}
        
        times = self.processing_times
        return {
            'gpu_available': self.gpu_available,
            'device_count': self.device_count,
            'avg_processing_time': sum(times) / len(times),
            'max_processing_time': max(times),
            'min_processing_time': min(times),
            'sample_count': len(times)
        }
    
    def cleanup(self):
        """リソース解放"""
        # GPU処理のクリーンアップ（必要に応じて）
        pass

# テスト実行用
if __name__ == "__main__":
    print("🔍 GPU処理テスト開始...")
    
    processor = GPUProcessor()
    
    # テスト画像作成
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    print(f"テスト画像: {test_image.shape}")
    
    # リサイズテスト
    start_time = time.time()
    resized = processor.resize_frame(test_image, (640, 480))
    resize_time = time.time() - start_time
    
    print(f"リサイズ結果: {resized.shape}")
    print(f"処理時間: {resize_time:.4f}秒")
    
    # 色変換テスト
    start_time = time.time()
    gray = processor.convert_color(test_image, cv2.COLOR_BGR2GRAY)
    convert_time = time.time() - start_time
    
    print(f"色変換結果: {gray.shape}")
    print(f"処理時間: {convert_time:.4f}秒")
    
    # パフォーマンス統計
    stats = processor.get_performance_stats()
    print(f"パフォーマンス統計: {stats}")
    
    processor.cleanup()
    print("✅ GPU処理テスト完了")

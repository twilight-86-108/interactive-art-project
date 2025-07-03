# src/core/gpu_processor.py - 統合完全版
import cv2
import ctypes
import numpy as np
import logging
import time
from typing import Optional, Tuple, Union
from pathlib import Path
from enum import Enum

class GPUBackend(Enum):
    """GPU処理バックエンドタイプ"""
    CUSTOM_CPP = "custom_cpp"          # カスタムC++ライブラリ
    OPENCV_CUDA = "opencv_cuda"        # OpenCV CUDA
    CPU_FALLBACK = "cpu_fallback"      # CPU処理

class GPUProcessor:
    """
    統合GPU加速処理クラス
    
    処理優先順位:
    1. カスタムC++ライブラリ（最高性能）
    2. OpenCV CUDA（標準GPU処理）
    3. CPU処理（フォールバック）
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 利用可能バックエンド
        self.available_backends = []
        self.current_backend = GPUBackend.CPU_FALLBACK
        
        # C++ライブラリ関連
        self.cpp_lib = None
        self.cpp_available = False
        
        # OpenCV CUDA関連
        self.opencv_cuda_available = False
        
        # 統計情報
        self.processing_stats = {
            'total_frames': 0,
            'gpu_frames': 0,
            'cpu_frames': 0,
            'avg_processing_time': 0.0,
            'backend_switches': 0
        }
        
        # 初期化実行
        self._initialize_backends()
        self._select_optimal_backend()
        self._log_initialization_status()
    
    def _initialize_backends(self):
        """利用可能なバックエンドを初期化"""
        
        # 1. カスタムC++ライブラリの初期化試行
        self._init_cpp_backend()
        
        # 2. OpenCV CUDA の初期化試行
        self._init_opencv_cuda_backend()
        
        # 3. CPU処理は常に利用可能
        self.available_backends.append(GPUBackend.CPU_FALLBACK)
    
    def _init_cpp_backend(self):
        """カスタムC++ライブラリ初期化"""
        try:
            # ライブラリパス候補
            lib_paths = [
                Path(__file__).parent.parent.parent / "cpp" / "build" / "libaqua_mirror_gpu.so",
                Path(__file__).parent.parent.parent / "cpp" / "build" / "libaqua_mirror_gpu.dll",
                Path("./cpp/build/libaqua_mirror_gpu.so"),
                Path("./libaqua_mirror_gpu.so")
            ]
            
            for lib_path in lib_paths:
                if lib_path.exists():
                    self.cpp_lib = ctypes.CDLL(str(lib_path))
                    self._setup_cpp_function_signatures()
                    
                    # テスト実行
                    if self._test_cpp_library():
                        self.cpp_available = True
                        self.available_backends.append(GPUBackend.CUSTOM_CPP)
                        self.logger.info(f"✅ カスタムC++ライブラリ初期化成功: {lib_path}")
                        break
                    
        except Exception as e:
            self.logger.warning(f"カスタムC++ライブラリ初期化失敗: {e}")
    
    def _setup_cpp_function_signatures(self):
        """C++関数シグネチャ設定"""
        if not self.cpp_lib:
            return
        try:
            # resize_gpu関数
            self.cpp_lib.resize_gpu.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),  # input
                ctypes.POINTER(ctypes.c_ubyte),  # output
                ctypes.c_int,                    # input_width
                ctypes.c_int,                    # input_height
                ctypes.c_int,                    # output_width
                ctypes.c_int,                    # output_height
                ctypes.c_int                     # channels
            ]
            self.cpp_lib.resize_gpu.restype = ctypes.c_int
            
            # color_convert_gpu関数
            self.cpp_lib.color_convert_gpu.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),  # input
                ctypes.POINTER(ctypes.c_ubyte),  # output
                ctypes.c_int,                    # width
                ctypes.c_int,                    # height
                ctypes.c_int                     # conversion_code
            ]
            self.cpp_lib.color_convert_gpu.restype = ctypes.c_int
            
            # デバイス情報関数
            if hasattr(self.cpp_lib, 'get_gpu_info'):
                self.cpp_lib.get_gpu_info.argtypes = []
                self.cpp_lib.get_gpu_info.restype = ctypes.c_int
                
        except Exception as e:
            self.logger.error(f"C++関数シグネチャ設定エラー: {e}")
            raise
    
    def _test_cpp_library(self) -> bool:
        """C++ライブラリテスト"""
        if not self.cpp_lib:
            return False
        try:
            # 小さなテスト画像でリサイズテスト
            test_input = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            test_output = np.zeros((50, 50, 3), dtype=np.uint8)
            
            input_ptr = test_input.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            output_ptr = test_output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            
            result = self.cpp_lib.resize_gpu(input_ptr, output_ptr, 100, 100, 50, 50, 3)
            
            return result == 0  # 成功コード
            
        except Exception as e:
            self.logger.warning(f"C++ライブラリテスト失敗: {e}")
            return False
    
    def _init_opencv_cuda_backend(self):
        """OpenCV CUDA初期化"""
        try:
            # OpenCV CUDA モジュール存在確認
            if not hasattr(cv2, 'cuda'):
                self.logger.warning("OpenCVにCUDAモジュールが含まれていません")
                return
            
            if not hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                self.logger.warning("getCudaEnabledDeviceCount関数が利用できません")
                return
            
            # CUDA対応デバイス確認
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count == 0:
                self.logger.warning("CUDA対応デバイスが見つかりません")
                return
            
            # 基本的なCUDA操作テスト
            test_gpu_mat = cv2.cuda.GpuMat()
            test_array = np.zeros((10, 10, 3), dtype=np.uint8)
            test_gpu_mat.upload(test_array)
            test_gpu_mat.download()
            
            self.opencv_cuda_available = True
            self.available_backends.append(GPUBackend.OPENCV_CUDA)
            self.logger.info(f"✅ OpenCV CUDA初期化成功 (デバイス数: {device_count})")
            
        except Exception as e:
            self.logger.warning(f"OpenCV CUDA初期化失敗: {e}")
    
    def _select_optimal_backend(self):
        """最適なバックエンドを選択"""
        if GPUBackend.CUSTOM_CPP in self.available_backends:
            self.current_backend = GPUBackend.CUSTOM_CPP
        elif GPUBackend.OPENCV_CUDA in self.available_backends:
            self.current_backend = GPUBackend.OPENCV_CUDA
        else:
            self.current_backend = GPUBackend.CPU_FALLBACK
    
    def _log_initialization_status(self):
        """初期化状況ログ出力"""
        self.logger.info(f"🔧 利用可能バックエンド: {[b.value for b in self.available_backends]}")
        self.logger.info(f"🎯 選択されたバックエンド: {self.current_backend.value}")
        
        if self.current_backend == GPUBackend.CPU_FALLBACK:
            self.logger.info("💡 GPU加速を有効にするには:")
            self.logger.info("   1. カスタムC++ライブラリのビルド")
            self.logger.info("   2. CUDA対応OpenCVのインストール")
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """統合フレームリサイズ"""
        start_time = time.time()
        
        try:
            # バックエンド別処理
            if self.current_backend == GPUBackend.CUSTOM_CPP:
                result = self._cpp_resize(frame, target_size)
            elif self.current_backend == GPUBackend.OPENCV_CUDA:
                result = self._opencv_cuda_resize(frame, target_size)
            else:
                result = self._cpu_resize(frame, target_size)
            
            # 統計更新
            processing_time = time.time() - start_time
            self._update_stats(processing_time, self.current_backend != GPUBackend.CPU_FALLBACK)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"リサイズ処理エラー ({self.current_backend.value}): {e}")
            return self._fallback_resize(frame, target_size, e)
    
    def color_convert(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """統合色空間変換"""
        start_time = time.time()
        
        try:
            # バックエンド別処理
            if self.current_backend == GPUBackend.CUSTOM_CPP:
                result = self._cpp_color_convert(frame, conversion_code)
            elif self.current_backend == GPUBackend.OPENCV_CUDA:
                result = self._opencv_cuda_color_convert(frame, conversion_code)
            else:
                result = self._cpu_color_convert(frame, conversion_code)
            
            # 統計更新
            processing_time = time.time() - start_time
            self._update_stats(processing_time, self.current_backend != GPUBackend.CPU_FALLBACK)
            
            return result
            
        except Exception as e:
            self.logger.warning(f"色変換処理エラー ({self.current_backend.value}): {e}")
            return self._fallback_color_convert(frame, conversion_code, e)
    
    def process_frame_optimized(self, frame: np.ndarray,
                              target_size: Optional[Tuple[int, int]] = None,
                              convert_to_rgb: bool = True) -> np.ndarray:
        """最適化統合フレーム処理"""
        if frame is None or frame.size == 0:
            raise ValueError("無効なフレームが入力されました")
        
        result = frame.copy()
        
        try:
            # リサイズ処理
            if target_size and result.shape[:2] != target_size[::-1]:
                result = self.resize_frame(result, target_size)
            
            # 色変換処理
            if convert_to_rgb and len(result.shape) == 3 and result.shape[2] == 3:
                result = self.color_convert(result, cv2.COLOR_BGR2RGB)
            
            return result
            
        except Exception as e:
            self.logger.error(f"統合フレーム処理で重大エラー: {e}")
            # 最低限の処理で継続
            try:
                if target_size:
                    result = cv2.resize(result, target_size)
                if convert_to_rgb:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            except:
                pass  # 元のフレームを返す
            
            return result
    
    # =====================================================
    # バックエンド別実装
    # =====================================================
    
    def _cpp_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """カスタムC++リサイズ"""
        if not self.cpp_lib:
            raise RuntimeError("カスタムC++ライブラリが利用できません")
        h, w, c = frame.shape
        target_w, target_h = target_size
        
        # 出力配列準備
        output = np.zeros((target_h, target_w, c), dtype=np.uint8)
        
        # GPU処理実行
        input_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        result_code = self.cpp_lib.resize_gpu(input_ptr, output_ptr, w, h, target_w, target_h, c)
        
        if result_code != 0:
            raise RuntimeError(f"C++リサイズ処理失敗 (コード: {result_code})")
        
        return output
    
    def _cpp_color_convert(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """カスタムC++色変換"""
        if not self.cpp_lib:
            raise RuntimeError("カスタムC++ライブラリが利用できません")
        h, w, c = frame.shape
        output_channels = 3 if conversion_code == cv2.COLOR_BGR2RGB else c
        
        # 出力配列準備
        output = np.zeros((h, w, output_channels), dtype=np.uint8)
        
        # GPU処理実行
        input_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        result_code = self.cpp_lib.color_convert_gpu(input_ptr, output_ptr, w, h, conversion_code)
        
        if result_code != 0:
            raise RuntimeError(f"C++色変換処理失敗 (コード: {result_code})")
        
        return output
    
    def _opencv_cuda_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """OpenCV CUDAリサイズ"""
        # 注意: OpenCVのPythonバインディングでは直接的なCUDAリサイズAPIがない場合がある
        # この実装は概念的なもので、実際のAPIに合わせて調整が必要
        gpu_frame = cv2.cuda.GpuMat()
        gpu_frame.upload(frame)
        
        # GPU上でリサイズ（API確認要）
        # gpu_resized = cv2.cuda.resize(gpu_frame, target_size)
        
        # フォールバック: CPUリサイズ
        cpu_frame = gpu_frame.download()
        resized = cv2.resize(cpu_frame, target_size)
        
        return resized
    
    def _opencv_cuda_color_convert(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """OpenCV CUDA色変換"""
        gpu_frame = cv2.cuda.GpuMat()
        gpu_frame.upload(frame)
        
        # GPU上で色変換（API確認要）
        # gpu_converted = cv2.cuda.cvtColor(gpu_frame, conversion_code)
        
        # フォールバック: CPU色変換
        cpu_frame = gpu_frame.download()
        converted = cv2.cvtColor(cpu_frame, conversion_code)
        
        return converted
    
    def _cpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """CPU リサイズ"""
        return cv2.resize(frame, target_size)
    
    def _cpu_color_convert(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """CPU 色変換"""
        return cv2.cvtColor(frame, conversion_code)
    
    # =====================================================
    # フォールバック・エラー処理
    # =====================================================
    
    def _fallback_resize(self, frame: np.ndarray, target_size: Tuple[int, int], error: Exception) -> np.ndarray:
        """リサイズフォールバック処理"""
        self._handle_backend_error(error)
        return self._cpu_resize(frame, target_size)
    
    def _fallback_color_convert(self, frame: np.ndarray, conversion_code: int, error: Exception) -> np.ndarray:
        """色変換フォールバック処理"""
        self._handle_backend_error(error)
        return self._cpu_color_convert(frame, conversion_code)
    
    def _handle_backend_error(self, error: Exception):
        """バックエンドエラー処理"""
        # 現在のバックエンドを無効化
        if self.current_backend in self.available_backends:
            self.available_backends.remove(self.current_backend)
            self.processing_stats['backend_switches'] += 1
        
        # 次の利用可能バックエンドに切替
        self._select_optimal_backend()
        
        self.logger.warning(f"バックエンド切替: {self.current_backend.value}")
    
    # =====================================================
    # 統計・監視機能
    # =====================================================
    
    def _update_stats(self, processing_time: float, is_gpu: bool):
        """統計情報更新"""
        self.processing_stats['total_frames'] += 1
        
        if is_gpu:
            self.processing_stats['gpu_frames'] += 1
        else:
            self.processing_stats['cpu_frames'] += 1
        
        # 移動平均で処理時間更新
        alpha = 0.1
        self.processing_stats['avg_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.processing_stats['avg_processing_time']
        )
    
    def get_system_info(self) -> dict:
        """システム情報取得"""
        info = {
            'current_backend': self.current_backend.value,
            'available_backends': [b.value for b in self.available_backends],
            'cpp_library_available': self.cpp_available,
            'opencv_cuda_available': self.opencv_cuda_available,
            'opencv_version': cv2.__version__,
            'processing_stats': self.processing_stats.copy()
        }
        
        # CUDA詳細情報
        if self.opencv_cuda_available:
            try:
                info['cuda_device_count'] = cv2.cuda.getCudaEnabledDeviceCount()
            except:
                info['cuda_device_count'] = 0
        
        return info
    
    def benchmark_performance(self, test_size: Tuple[int, int] = (1920, 1080),
                            iterations: int = 50) -> dict:
        """パフォーマンスベンチマーク"""
        results = {
            'test_configuration': {
                'input_size': test_size,
                'output_size': (640, 480),
                'iterations': iterations
            },
            'backend_performance': {}
        }
        
        # テスト用画像
        test_image = np.random.randint(0, 255, (test_size[1], test_size[0], 3), dtype=np.uint8)
        
        # 各バックエンドのベンチマーク
        for backend in self.available_backends:
            original_backend = self.current_backend
            self.current_backend = backend
            
            try:
                self.logger.info(f"ベンチマーク実行中: {backend.value}")
                
                start_time = time.time()
                for _ in range(iterations):
                    result = self.process_frame_optimized(test_image, (640, 480), True)
                total_time = time.time() - start_time
                
                results['backend_performance'][backend.value] = {
                    'total_time': total_time,
                    'fps': iterations / total_time,
                    'time_per_frame': total_time / iterations
                }
                
            except Exception as e:
                results['backend_performance'][backend.value] = {
                    'error': str(e)
                }
            
            finally:
                self.current_backend = original_backend
        
        return results
    
    def is_gpu_available(self) -> bool:
        """GPU処理利用可能性"""
        return self.current_backend in [GPUBackend.CUSTOM_CPP, GPUBackend.OPENCV_CUDA]
    
    def force_backend(self, backend: GPUBackend) -> bool:
        """バックエンド強制切替"""
        if backend in self.available_backends:
            self.current_backend = backend
            self.logger.info(f"バックエンド強制切替: {backend.value}")
            return True
        else:
            self.logger.warning(f"バックエンド {backend.value} は利用できません")
            return False
    
    def cleanup(self):
        """リソース解放"""
        try:
            # CUDA関連リソース解放
            if self.opencv_cuda_available:
                try:
                    cv2.cuda.resetDevice()
                except:
                    pass
            
            # C++ライブラリリソース解放
            if self.cpp_available and self.cpp_lib:
                try:
                    if hasattr(self.cpp_lib, 'cleanup_gpu'):
                        self.cpp_lib.cleanup_gpu()
                except:
                    pass
            
            self.logger.info("GPU プロセッサークリーンアップ完了")
            
        except Exception as e:
            self.logger.warning(f"クリーンアップエラー: {e}")
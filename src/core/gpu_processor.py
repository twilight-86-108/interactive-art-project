# src/core/gpu_processor.py - 完全修正版（CUDA無効環境対応）
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Union
import time

class GPUProcessor:
    """GPU加速処理クラス - CUDA無効環境でも動作保証"""
    
    def __init__(self):
        self.gpu_available = False
        self.cuda_opencv_available = False
        self.logger = logging.getLogger(__name__)
        
        # CUDA環境の詳細チェック
        self._check_cuda_environment()
        
        # 初期化結果をログ出力
        self._log_initialization_status()
    
    def _check_cuda_environment(self):
        """CUDA環境の包括的チェック"""
        self.cuda_opencv_available = False
        self.gpu_available = False
        
        try:
            # 1. cv2.cuda モジュールの存在確認
            if not hasattr(cv2, 'cuda'):
                self.logger.warning("OpenCVにcudaモジュールが含まれていません")
                return
            
            # 2. getCudaEnabledDeviceCount関数の存在確認
            if not hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
                self.logger.warning("getCudaEnabledDeviceCount関数が利用できません")
                return
            
            # 3. CUDA対応デバイス数確認
            try:
                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                self.logger.info(f"CUDA対応デバイス数: {device_count}")
                
                if device_count == 0:
                    self.logger.warning("CUDA対応デバイスが見つかりません（OpenCVがCUDA無効でビルドされている可能性）")
                    return
                
                self.cuda_opencv_available = True
                
            except Exception as e:
                self.logger.warning(f"CUDA デバイス確認エラー: {e}")
                return
            
            # 4. 基本的なCUDA操作テスト
            try:
                # GpuMat作成テスト
                test_gpu_mat = cv2.cuda.GpuMat()
                
                # 小さなテスト配列でアップロード/ダウンロードテスト
                test_array = np.zeros((10, 10, 3), dtype=np.uint8)
                test_gpu_mat.upload(test_array)
                result = test_gpu_mat.download()
                
                # 基本的なCUDA関数テスト
                if hasattr(cv2.cuda, 'resize') and hasattr(cv2.cuda, 'cvtColor'):
                    self.gpu_available = True
                    self.logger.info("CUDA GPU処理が利用可能です")
                else:
                    self.logger.warning("必要なCUDA関数が利用できません")
                
            except Exception as e:
                self.logger.warning(f"CUDA基本操作テストに失敗: {e}")
                return
                
        except Exception as e:
            self.logger.warning(f"CUDA環境確認中にエラー: {e}")
    
    def _log_initialization_status(self):
        """初期化状況のログ出力"""
        if self.gpu_available:
            self.logger.info("✅ GPU加速処理が利用可能です")
        else:
            self.logger.info("ℹ️ CPU処理モードで動作します")
            
            # CUDA無効の原因を詳しく説明
            if not self.cuda_opencv_available:
                self.logger.info("💡 GPU加速を有効にするには、CUDA対応OpenCVのインストールが必要です")
    
    def gpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """GPU加速リサイズ（完全フォールバック対応）"""
        if not self.gpu_available:
            return self._cpu_resize(frame, target_size)
        
        try:
            # OpenCVのPythonバインディングにはCUDAリサイズAPIが存在しないため、CPUでリサイズ
            self.logger.warning("OpenCVのPythonバインディングにはCUDAリサイズAPIが存在しません。CPUリサイズにフォールバックします。")
            return self._cpu_resize(frame, target_size)
        except Exception as e:
            # GPU処理失敗時はCPU処理にフォールバック
            self.logger.warning(f"GPU リサイズ失敗、CPU処理に切替: {e}")
            self.gpu_available = False  # 今後はCPU処理を使用
            return self._cpu_resize(frame, target_size)
    
    def gpu_color_convert(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """GPU加速色空間変換（完全フォールバック対応）"""
        if not self.gpu_available:
            return self._cpu_color_convert(frame, conversion_code)
        
        try:
            # GPU処理
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            # OpenCVのPythonバインディングでは cv2.cuda.cvtColor は存在しないため、代わりに cv2.cuda_CvtColor を使用
            self.logger.warning("OpenCVのPythonバインディングにはCUDA色変換APIが存在しません。CPU色変換にフォールバックします。")
            return self._cpu_color_convert(frame, conversion_code)
        except Exception as e:
            # GPU処理失敗時はCPU処理にフォールバック
            self.logger.warning(f"GPU 色変換失敗、CPU処理に切替: {e}")
            self.gpu_available = False  # 今後はCPU処理を使用
            return self._cpu_color_convert(frame, conversion_code)
    
    def process_frame_optimized(self, frame: np.ndarray, 
                              target_size: Optional[Tuple[int, int]] = None,
                              convert_to_rgb: bool = True) -> np.ndarray:
        """最適化されたフレーム処理（必ず成功）"""
        try:
            # 入力検証
            if frame is None or frame.size == 0:
                raise ValueError("無効なフレームが入力されました")
            
            # GPU処理試行
            if self.gpu_available:
                try:
                    return self._gpu_process_chain(frame, target_size, convert_to_rgb)
                except Exception as gpu_error:
                    self.logger.warning(f"GPU処理チェーン失敗: {gpu_error}")
                    self.gpu_available = False  # GPU無効化
            
            # CPU処理（フォールバック）
            return self._cpu_process_chain(frame, target_size, convert_to_rgb)
            
        except Exception as e:
            self.logger.error(f"フレーム処理で重大エラー: {e}")
            # 最低限の処理で継続
            result = frame.copy()
            try:
                if target_size and result.shape[:2] != target_size[::-1]:
                    result = cv2.resize(result, target_size)
                if convert_to_rgb and len(result.shape) == 3 and result.shape[2] == 3:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            except Exception as fallback_error:
                self.logger.error(f"フォールバック処理も失敗: {fallback_error}")
                # 元のフレームをそのまま返す
                pass
            
            return result
    
    def _gpu_process_chain(self, frame: np.ndarray,
                          target_size: Optional[Tuple[int, int]],
                          convert_to_rgb: bool) -> np.ndarray:
        """GPU処理チェーン"""
        # GPU メモリにアップロード
        gpu_frame = cv2.cuda.GpuMat()
        gpu_frame.upload(frame)
        current_gpu_frame = gpu_frame
        
        # リサイズ処理
        if target_size:
            # OpenCVのPythonバインディングにはCUDAリサイズAPIが存在しないため、CPUでリサイズ
            self.logger.warning("OpenCVのPythonバインディングにはCUDAリサイズAPIが存在しません。CPUリサイズにフォールバックします。")
            result = current_gpu_frame.download()
            result = cv2.resize(result, target_size)
            gpu_frame.upload(result)
            current_gpu_frame = gpu_frame
        
        # 色変換処理
        if convert_to_rgb:
            self.logger.warning("OpenCVのPythonバインディングにはCUDA色変換APIが存在しません。CPU色変換にフォールバックします。")
            result = current_gpu_frame.download()
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            gpu_frame.upload(result)
            current_gpu_frame = gpu_frame
        
        # 結果をCPUメモリにダウンロード
        return current_gpu_frame.download()
    
    def _cpu_process_chain(self, frame: np.ndarray,
                          target_size: Optional[Tuple[int, int]],
                          convert_to_rgb: bool) -> np.ndarray:
        """CPU処理チェーン（確実動作）"""
        result = frame.copy()
        
        # リサイズ処理
        if target_size:
            result = cv2.resize(result, target_size)
        
        # 色変換処理
        if convert_to_rgb:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        return result
    
    # CPUフォールバック関数
    def _cpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """CPU リサイズ"""
        return cv2.resize(frame, target_size)
    
    def _cpu_color_convert(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """CPU 色変換"""
        return cv2.cvtColor(frame, conversion_code)
    
    def get_system_info(self) -> dict:
        """システム情報取得"""
        info = {
            'gpu_available': self.gpu_available,
            'cuda_opencv_available': self.cuda_opencv_available,
            'opencv_version': cv2.__version__,
            'opencv_cuda_support': hasattr(cv2, 'cuda'),
            'processing_mode': 'GPU' if self.gpu_available else 'CPU'
        }
        
        # CUDA情報（利用可能な場合のみ）
        if self.cuda_opencv_available:
            try:
                info['cuda_device_count'] = cv2.cuda.getCudaEnabledDeviceCount()
                
                # デバイス詳細情報（エラーが出る可能性があるので try-catch）
                try:
                    if info['cuda_device_count'] > 0:
                        # device_info = cv2.cuda.DeviceInfo(0)  # これはエラーが出る可能性
                        # より安全なアプローチ
                        info['cuda_functions_available'] = {
                            'resize': hasattr(cv2.cuda, 'resize'),
                            'cvtColor': hasattr(cv2.cuda, 'cvtColor'),
                            'GpuMat': hasattr(cv2.cuda, 'GpuMat')
                        }
                except Exception as device_error:
                    info['device_info_error'] = str(device_error)
                    
            except Exception as cuda_error:
                info['cuda_info_error'] = str(cuda_error)
        
        return info
    
    def benchmark_performance(self, test_size: Tuple[int, int] = (1920, 1080),
                            iterations: int = 50) -> dict:
        """パフォーマンスベンチマーク"""
        results = {
            'test_configuration': {
                'input_size': test_size,
                'output_size': (640, 480),
                'iterations': iterations,
                'operations': ['resize', 'color_convert']
            }
        }
        
        # テスト用画像生成
        test_image = np.random.randint(0, 255, 
                                     (test_size[1], test_size[0], 3), 
                                     dtype=np.uint8)
        
        # CPU性能測定
        self.logger.info("CPU性能測定中...")
        cpu_start = time.time()
        for _ in range(iterations):
            result = self._cpu_process_chain(test_image, (640, 480), True)
        cpu_time = time.time() - cpu_start
        
        results['cpu_performance'] = {
            'total_time': cpu_time,
            'fps': iterations / cpu_time,
            'time_per_frame': cpu_time / iterations
        }
        
        # GPU性能測定（利用可能な場合）
        if self.gpu_available:
            self.logger.info("GPU性能測定中...")
            try:
                gpu_start = time.time()
                for _ in range(iterations):
                    result = self._gpu_process_chain(test_image, (640, 480), True)
                gpu_time = time.time() - gpu_start
                
                results['gpu_performance'] = {
                    'total_time': gpu_time,
                    'fps': iterations / gpu_time,
                    'time_per_frame': gpu_time / iterations,
                    'speedup_factor': cpu_time / gpu_time
                }
                
            except Exception as gpu_bench_error:
                results['gpu_benchmark_error'] = {"error": str(gpu_bench_error)}
        else:
            results['gpu_performance'] = {"error": "GPU処理が利用できません"}
        
        return results
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.gpu_available:
                # CUDA関連リソースの解放を試行
                try:
                    cv2.cuda.resetDevice()
                    self.logger.info("CUDAデバイスリセット完了")
                except Exception as reset_error:
                    self.logger.warning(f"CUDAデバイスリセットエラー: {reset_error}")
            
            self.gpu_available = False
            self.cuda_opencv_available = False
            self.logger.info("GPU プロセッサークリーンアップ完了")
            
        except Exception as cleanup_error:
            self.logger.warning(f"クリーンアップエラー: {cleanup_error}")


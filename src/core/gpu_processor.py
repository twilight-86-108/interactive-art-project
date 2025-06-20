# src/core/gpu_processor.py を完全修正
import cv2
import numpy as np
import time
from typing import Optional, Tuple, List, Union

class GPUProcessor:
    """GPU加速処理クラス（CUDA API完全対応版）"""
    
    def __init__(self):
        self.gpu_available = False
        self.device_count = 0
        self.device_info: List[str] = []
        self.processing_times: List[float] = []
        self.cuda_context_created = False
        self.cuda_functions_available = {
            'resize': False,
            'cvtColor': False,
            'GaussianBlur': False,
            'GpuMat': False
        }
        
        self._check_gpu_availability()
    
    def _check_gpu_availability(self):
        """GPU利用可能性確認（完全版）"""
        try:
            # Step 1: OpenCVのCUDAサポート確認
            if not hasattr(cv2, 'cuda'):
                print("⚠️  OpenCVにCUDAモジュールがありません（CPU処理で継続）")
                return
            
            # Step 2: CUDA デバイス数確認
            try:
                self.device_count = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception as e:
                print(f"⚠️  CUDA デバイス数取得エラー: {e}（CPU処理で継続）")
                return
            
            if self.device_count <= 0:
                print("⚠️  CUDA対応デバイスが見つかりません（CPU処理で継続）")
                return
            
            # Step 3: CUDA機能確認
            self._check_cuda_functions()
            
            # Step 4: 基本的なCUDA機能があれば有効化
            if self.cuda_functions_available['GpuMat']:
                self.gpu_available = True
                print(f"✅ GPU利用可能: {self.device_count} devices")
                
                # GPU情報取得（簡素化版）
                self._get_device_info_safe()
                
                # CUDA コンテキスト初期化テスト
                self._test_cuda_context()
            else:
                print("⚠️  必要なCUDA機能が利用できません（CPU処理で継続）")
                
        except Exception as e:
            print(f"⚠️  GPU確認エラー: {e}（CPU処理で継続）")
            self.gpu_available = False
    
    def _check_cuda_functions(self):
        """CUDA機能の利用可能性確認"""
        # GpuMat 確認
        try:
            if hasattr(cv2.cuda, 'GpuMat'):
                test_mat = cv2.cuda.GpuMat()
                self.cuda_functions_available['GpuMat'] = True
        except:
            pass
        
        # resize 確認
        try:
            if hasattr(cv2.cuda, 'resize'):
                self.cuda_functions_available['resize'] = True
        except:
            pass
        
        # cvtColor 確認
        try:
            if hasattr(cv2.cuda, 'cvtColor'):
                self.cuda_functions_available['cvtColor'] = True
        except:
            pass
        
        # GaussianBlur 確認
        try:
            if hasattr(cv2.cuda, 'GaussianBlur'):
                self.cuda_functions_available['GaussianBlur'] = True
        except:
            pass
        
        print(f"CUDA機能確認: {self.cuda_functions_available}")
    
    def _get_device_info_safe(self):
        """GPU デバイス情報取得（安全版）"""
        try:
            for i in range(self.device_count):
                try:
                    # デバイス情報取得を試行（複数の方法）
                    device_name = self._get_single_device_info(i)
                    self.device_info.append(device_name)
                    print(f"   Device {i}: {device_name}")
                    
                except Exception as device_error:
                    fallback_name = f"CUDA Device {i} (Details unavailable)"
                    self.device_info.append(fallback_name)
                    print(f"   Device {i}: 情報取得エラー ({device_error})")
                    
        except Exception as e:
            print(f"⚠️  デバイス情報取得エラー: {e}")
            # フォールバック：デバイス数分の基本情報
            for i in range(self.device_count):
                self.device_info.append(f"CUDA Device {i}")
    
    def _get_single_device_info(self, device_id: int) -> str:
        """単一デバイス情報取得"""
        try:
            # DeviceInfo 作成を試行
            if hasattr(cv2.cuda, 'DeviceInfo'):
                device_info = cv2.cuda.DeviceInfo(device_id)
                
                # 様々な方法でデバイス名取得を試行
                methods = [
                    lambda: device_info.name() if hasattr(device_info, 'name') and callable(device_info.name) else None,
                    lambda: str(device_info.name) if hasattr(device_info, 'name') else None,
                    lambda: device_info.getName() if hasattr(device_info, 'getName') and callable(device_info.getName) else None,
                    lambda: str(device_info) if hasattr(device_info, '__str__') else None
                ]
                
                for method in methods:
                    try:
                        result = method()
                        if result and isinstance(result, str) and result.strip():
                            return result.strip()
                    except:
                        continue
            
            # すべての方法が失敗した場合のフォールバック
            return f"CUDA Device {device_id}"
            
        except Exception as e:
            return f"CUDA Device {device_id} (Error: {e})"
    
    def _test_cuda_context(self):
        """CUDA コンテキストテスト"""
        if not self.cuda_functions_available['GpuMat']:
            return
        
        try:
            # 簡単なCUDA操作でコンテキストをテスト
            test_array = np.ones((10, 10), dtype=np.uint8)
            gpu_mat = cv2.cuda.GpuMat()
            gpu_mat.upload(test_array)
            result = gpu_mat.download()
            
            if result is not None and result.shape == (10, 10):
                self.cuda_context_created = True
                print("✅ CUDA コンテキスト初期化成功")
            else:
                raise RuntimeError("CUDA テスト結果が無効")
                
        except Exception as e:
            print(f"⚠️  CUDA コンテキストエラー: {e}")
            self.gpu_available = False
            self.cuda_context_created = False
    
    def resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """GPU加速フレームリサイズ（完全修正版）"""
        if frame is None:
            raise ValueError("入力フレームがNoneです")
        
        start_time = time.time()
        
        try:
            if (self.gpu_available and 
                self.cuda_context_created and 
                self.cuda_functions_available['resize'] and
                self.cuda_functions_available['GpuMat']):
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
                   interpolation: int) -> np.ndarray:
        """GPU処理実装（存在確認版）"""
        try:
            # GPU メモリにアップロード
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            
            # GPU上でリサイズ（実際にresizeが利用可能な場合のみ）
            if hasattr(cv2.cuda, 'resize'):
                try:
                    # OpenCV 4.x 形式
                    gpu_resized = cv2.cuda.resize(gpu_frame, target_size)
                    return gpu_resized.download()
                except Exception as api_error:
                    print(f"⚠️  cv2.cuda.resize API エラー: {api_error}")
                    raise api_error
            else:
                raise AttributeError("cv2.cuda.resize not available")
            
        except Exception as e:
            print(f"⚠️  GPU リサイズ詳細エラー: {e}")
            raise e
    
    def _cpu_resize(self, frame: np.ndarray, target_size: Tuple[int, int], 
                   interpolation: int) -> np.ndarray:
        """CPU処理実装"""
        return cv2.resize(frame, target_size, interpolation=interpolation)
    
    def convert_color(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """GPU加速色空間変換（完全修正版）"""
        if frame is None:
            raise ValueError("入力フレームがNoneです")
        
        try:
            if (self.gpu_available and 
                self.cuda_context_created and 
                self.cuda_functions_available['cvtColor'] and
                self.cuda_functions_available['GpuMat']):
                return self._gpu_convert_color(frame, conversion_code)
            else:
                return self._cpu_convert_color(frame, conversion_code)
                
        except Exception as e:
            print(f"⚠️  色変換エラー: {e}")
            return self._cpu_convert_color(frame, conversion_code)
    
    def _gpu_convert_color(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """GPU色変換（存在確認版）"""
        try:
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            
            if hasattr(cv2.cuda, 'cvtColor'):
                try:
                    gpu_converted = cv2.cuda.cvtColor(gpu_frame, conversion_code)
                    return gpu_converted.download()
                except Exception as api_error:
                    print(f"⚠️  cv2.cuda.cvtColor API エラー: {api_error}")
                    raise api_error
            else:
                raise AttributeError("cv2.cuda.cvtColor not available")
            
        except Exception as e:
            print(f"⚠️  GPU色変換詳細エラー: {e}")
            raise e
    
    def _cpu_convert_color(self, frame: np.ndarray, conversion_code: int) -> np.ndarray:
        """CPU色変換"""
        return cv2.cvtColor(frame, conversion_code)
    
    def blur_frame(self, frame: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """ブラー処理（GPU対応）"""
        if frame is None:
            raise ValueError("入力フレームがNoneです")
        
        try:
            if (self.gpu_available and 
                self.cuda_context_created and 
                self.cuda_functions_available['GaussianBlur'] and
                self.cuda_functions_available['GpuMat']):
                return self._gpu_blur(frame, kernel_size)
            else:
                return self._cpu_blur(frame, kernel_size)
        except Exception as e:
            print(f"⚠️  ブラーエラー: {e}")
            return self._cpu_blur(frame, kernel_size)
    
    def _gpu_blur(self, frame: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """GPU ブラー処理（存在確認版）"""
        try:
            gpu_frame = cv2.cuda.GpuMat()
            gpu_frame.upload(frame)
            
            if hasattr(cv2.cuda, 'GaussianBlur'):
                try:
                    gpu_blurred = cv2.cuda.GaussianBlur(gpu_frame, kernel_size, 0)
                    return gpu_blurred.download()
                except Exception as api_error:
                    print(f"⚠️  cv2.cuda.GaussianBlur API エラー: {api_error}")
                    raise api_error
            else:
                raise AttributeError("cv2.cuda.GaussianBlur not available")
            
        except Exception as e:
            print(f"⚠️  GPU ブラー詳細エラー: {e}")
            raise e
    
    def _cpu_blur(self, frame: np.ndarray, kernel_size: Tuple[int, int]) -> np.ndarray:
        """CPU ブラー処理"""
        return cv2.GaussianBlur(frame, kernel_size, 0)
    
    def get_performance_stats(self) -> dict:
        """パフォーマンス統計取得"""
        base_stats = {
            'gpu_available': self.gpu_available,
            'device_count': self.device_count,
            'device_info': self.device_info,
            'cuda_context_created': self.cuda_context_created,
            'cuda_functions_available': self.cuda_functions_available,
            'processing_samples': len(self.processing_times)
        }
        
        if self.processing_times:
            times = self.processing_times
            base_stats.update({
                'avg_processing_time': sum(times) / len(times),
                'max_processing_time': max(times),
                'min_processing_time': min(times)
            })
        
        return base_stats
    
    def get_cuda_info(self) -> dict:
        """CUDA詳細情報取得"""
        cuda_info = {
            'opencv_cuda_support': hasattr(cv2, 'cuda'),
            'gpu_available': self.gpu_available,
            'device_count': self.device_count,
            'cuda_context_created': self.cuda_context_created,
            'cuda_functions_available': self.cuda_functions_available
        }
        
        if self.gpu_available:
            cuda_info['device_info'] = self.device_info
        
        # OpenCV ビルド情報（可能な場合）
        try:
            build_info = cv2.getBuildInformation()
            cuda_info['opencv_has_cuda_in_build'] = 'CUDA:' in build_info and 'YES' in build_info
        except:
            cuda_info['opencv_has_cuda_in_build'] = 'Unknown'
        
        return cuda_info
    
    def is_gpu_processing_available(self) -> bool:
        """GPU処理が実際に利用可能かチェック"""
        return (self.gpu_available and 
                self.cuda_context_created and 
                self.cuda_functions_available['GpuMat'])
    
    def cleanup(self):
        """リソース解放"""
        try:
            # CUDA コンテキストのクリーンアップ
            # OpenCVでは通常自動的に行われる
            pass
        except Exception as e:
            print(f"⚠️  GPU クリーンアップエラー: {e}")

# テスト実行用
if __name__ == "__main__":
    print("🔍 GPU処理テスト開始...")
    
    processor = GPUProcessor()
    
    # CUDA情報表示
    cuda_info = processor.get_cuda_info()
    print("CUDA情報:")
    for key, value in cuda_info.items():
        print(f"  {key}: {value}")
    
    print(f"\nGPU処理利用可能: {processor.is_gpu_processing_available()}")
    
    # テスト画像作成
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print(f"テスト画像: {test_image.shape}")
    
    # リサイズテスト
    try:
        start_time = time.time()
        resized = processor.resize_frame(test_image, (320, 240))
        resize_time = time.time() - start_time
        
        print(f"リサイズ結果: {resized.shape}")
        print(f"処理時間: {resize_time:.4f}秒")
        print(f"処理方法: {'GPU' if processor.is_gpu_processing_available() else 'CPU'}")
    except Exception as e:
        print(f"❌ リサイズテストエラー: {e}")
    
    # 色変換テスト
    try:
        start_time = time.time()
        gray = processor.convert_color(test_image, cv2.COLOR_BGR2GRAY)
        convert_time = time.time() - start_time
        
        print(f"色変換結果: {gray.shape}")
        print(f"処理時間: {convert_time:.4f}秒")
    except Exception as e:
        print(f"❌ 色変換テストエラー: {e}")
    
    # パフォーマンス統計
    stats = processor.get_performance_stats()
    print(f"\nパフォーマンス統計:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    processor.cleanup()
    print("✅ GPU処理テスト完了")
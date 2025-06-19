# src/core/gpu_troubleshoot.py
import subprocess
import sys


class GPUTroubleshooter:
    """GPU問題診断・修復"""
    
    @staticmethod
    def check_cuda_environment():
        """CUDA環境診断"""
        import subprocess
        import sys
        
        checks = {}
        
        # NVIDIA ドライバー確認
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            checks['nvidia_driver'] = 'NVIDIA' in result.stdout
        except:
            checks['nvidia_driver'] = False
        
        # CUDA インストール確認
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            checks['cuda_toolkit'] = 'release' in result.stdout
        except:
            checks['cuda_toolkit'] = False
        
        # Python CUDA バインディング
        try:
            import cupy
            checks['cupy'] = True
        except ImportError:
            checks['cupy'] = False
        
        # OpenCV CUDA サポート
        try:
            import cv2
            checks['opencv_cuda'] = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            checks['opencv_cuda'] = False
        
        return checks
    
    @staticmethod
    def fix_cuda_issues():
        """CUDA問題修復"""
        fixes = []
        
        # 環境変数設定
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        fixes.append("CUDA環境変数設定")
        
        # メモリ成長設定（TensorFlow使用時）
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                fixes.append("GPU メモリ成長設定")
        except:
            pass
        
        return fixes
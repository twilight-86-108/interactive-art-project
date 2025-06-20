# test_gpu_setup.py
import sys

def test_gpu_libraries():
    """GPU ライブラリ動作確認"""
    
    print("🔍 GPU環境テスト開始...")
    print("-" * 50)
    
    # 1. 基本情報
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError as e:
        print(f"❌ PyTorch エラー: {e}")
    
    print("-" * 50)
    
    # 2. CuPy テスト
    try:
        import cupy as cp
        print(f"✅ CuPy: {cp.__version__}")
        
        # 簡単な計算テスト
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([2, 3, 4, 5, 6])
        c = a + b
        print(f"   計算テスト: {c.get()} (GPU上で計算)")
        
        # メモリ情報
        mempool = cp.get_default_memory_pool()
        print(f"   GPU Memory Pool: {mempool.used_bytes() / 1024**2:.1f} MB used")
        
    except ImportError as e:
        print(f"❌ CuPy インポートエラー: {e}")
        print("   対処法: pip install cupy-cuda12x")
    except Exception as e:
        print(f"❌ CuPy 動作エラー: {e}")
    
    print("-" * 50)
    
    # 3. TensorFlow テスト
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        
        # GPU認識確認
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   認識GPU数: {len(gpus)}")
        
        if gpus:
            print(f"   GPU Details: {gpus[0]}")
            
            # メモリ成長設定
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("   ✅ GPU メモリ成長設定完了")
            
            # 簡単な計算テスト
            with tf.device('/GPU:0'):  # type: ignore
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[2.0, 1.0], [4.0, 3.0]])
                c = tf.matmul(a, b)
                print(f"   計算テスト結果:\n{c.numpy()}")
                
        else:
            print("   ⚠️ GPU認識されていません（CPU使用）")
            
    except ImportError as e:
        print(f"❌ TensorFlow インポートエラー: {e}")
        print("   対処法: pip install tensorflow[and-cuda]")
    except Exception as e:
        print(f"❌ TensorFlow 動作エラー: {e}")
    
    print("-" * 50)
    
    # 4. OpenCV CUDA テスト
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        print(f"   CUDA対応デバイス数: {cuda_devices}")
        
        if cuda_devices > 0:
            print("   ✅ OpenCV CUDA サポート有効")
        else:
            print("   ⚠️ OpenCV CUDA サポート無効")
            
    except Exception as e:
        print(f"❌ OpenCV エラー: {e}")
    
    print("-" * 50)
    print("🎯 テスト完了")

if __name__ == "__main__":
    test_gpu_libraries()
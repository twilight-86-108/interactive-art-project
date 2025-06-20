# test_aqua_mirror_basic.py
"""
Aqua Mirror プロジェクト基本動作テスト
GPU環境確認後の統合テスト
"""

import sys
import os
import time

def test_basic_imports():
    """基本ライブラリインポートテスト"""
    print("📦 基本ライブラリテスト...")
    
    tests = [
        ("OpenCV", "cv2"),
        ("MediaPipe", "mediapipe"),
        ("Pygame", "pygame"),
        ("NumPy", "numpy"),
        ("CuPy", "cupy"),
        ("TensorFlow", "tensorflow")
    ]
    
    results = {}
    
    for name, module in tests:
        try:
            exec(f"import {module}")
            print(f"✅ {name}: OK")
            results[name] = True
        except ImportError as e:
            print(f"❌ {name}: {e}")
            results[name] = False
    
    return results

def test_mediapipe_setup():
    """MediaPipe セットアップテスト"""
    print("\n🤖 MediaPipe セットアップテスト...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Face Mesh初期化テスト
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        )
        print("✅ Face Mesh 初期化成功")
        
        # Hands初期化テスト
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        print("✅ Hands 初期化成功")
        
        # ダミー画像でテスト
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_image_rgb = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB)
        
        # 処理テスト
        face_results = face_mesh.process(dummy_image_rgb)
        hand_results = hands.process(dummy_image_rgb)
        
        print("✅ MediaPipe 処理テスト成功")
        
        # クリーンアップ
        face_mesh.close()
        hands.close()
        
        return True
        
    except Exception as e:
        print(f"❌ MediaPipe テストエラー: {e}")
        return False

def test_pygame_setup():
    """Pygame セットアップテスト"""
    print("\n🎮 Pygame セットアップテスト...")
    
    try:
        import pygame
        
        # Pygame初期化
        pygame.init()
        
        # ダミーサーフェス作成テスト
        screen = pygame.Surface((640, 480))
        screen.fill((0, 100, 200))  # 青色
        
        print("✅ Pygame 初期化成功")
        print(f"   バージョン: {pygame.version.ver}")
        
        # フォント機能テスト
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        text = font.render("Test", True, (255, 255, 255))
        print("✅ Pygame フォント機能OK")
        
        pygame.quit()
        return True
        
    except Exception as e:
        print(f"❌ Pygame テストエラー: {e}")
        return False

def test_gpu_optimization():
    """GPU最適化テスト"""
    print("\n⚡ GPU最適化テスト...")
    
    try:
        import cupy as cp
        import numpy as np
        import time
        
        # テストデータ生成
        size = 1000000
        cpu_array = np.random.random(size).astype(np.float32)
        
        # CPU処理時間測定
        start_time = time.time()
        cpu_result = np.sqrt(cpu_array)
        cpu_time = time.time() - start_time
        
        # GPU処理時間測定
        gpu_array = cp.asarray(cpu_array)
        start_time = time.time()
        gpu_result = cp.sqrt(gpu_array)
        cp.cuda.Stream.null.synchronize()  # 同期待ち
        gpu_time = time.time() - start_time
        
        speedup = cpu_time / gpu_time
        
        print(f"✅ CPU処理時間: {cpu_time:.4f}秒")
        print(f"✅ GPU処理時間: {gpu_time:.4f}秒")
        print(f"✅ 高速化倍率: {speedup:.2f}x")
        
        return speedup > 1.0
        
    except Exception as e:
        print(f"❌ GPU最適化テストエラー: {e}")
        return False

def test_memory_management():
    """メモリ管理テスト"""
    print("\n💾 メモリ管理テスト...")
    
    try:
        import cupy as cp
        import tensorflow as tf
        
        # CuPyメモリ制限設定
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=6 * 1024**3)  # 6GB制限
        print("✅ CuPy メモリ制限設定: 6GB")
        
        # TensorFlowメモリ成長設定確認
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            memory_growth = tf.config.experimental.get_memory_growth(gpus[0])
            print(f"✅ TensorFlow メモリ成長設定: {memory_growth}")
        
        # メモリ使用量確認
        print(f"   CuPy使用中メモリ: {mempool.used_bytes() / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ メモリ管理テストエラー: {e}")
        return False

def create_minimal_config():
    """最小設定ファイル作成"""
    print("\n⚙️ 最小設定ファイル作成...")
    
    config = {
        "system": {
            "name": "Aqua Mirror",
            "version": "1.0.0",
            "debug_mode": True,
            "demo_mode": False
        },
        "hardware": {
            "camera": {
                "device_id": 0,
                "resolution": {"width": 1280, "height": 720},
                "fps": 30
            },
            "display": {
                "width": 1280,
                "height": 720,
                "fullscreen": False
            },
            "gpu": {
                "enabled": True,
                "memory_limit_gb": 6,
                "optimization_level": "medium"
            }
        },
        "ai_processing": {
            "vision": {
                "face_detection": {
                    "model_complexity": 0,  # 軽量設定
                    "min_detection_confidence": 0.6,
                    "max_num_faces": 1
                },
                "hand_detection": {
                    "model_complexity": 0,  # 軽量設定
                    "min_detection_confidence": 0.6,
                    "max_num_hands": 2
                }
            }
        },
        "performance": {
            "target_fps": 30,
            "adaptive_quality": True
        }
    }
    
    import json
    
    # configディレクトリ作成
    if not os.path.exists("config"):
        os.makedirs("config")
    
    # 設定ファイル保存
    with open("config/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ config/config.json 作成完了")
    return True

def main():
    """メインテスト実行"""
    print("🌊 Aqua Mirror 基本動作テスト開始")
    print("=" * 60)
    
    test_results = {}
    
    # 各テスト実行
    tests = [
        ("基本ライブラリ", test_basic_imports),
        ("MediaPipe", test_mediapipe_setup),
        ("Pygame", test_pygame_setup),
        ("GPU最適化", test_gpu_optimization),
        ("メモリ管理", test_memory_management),
        ("設定ファイル", create_minimal_config)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} テスト {'='*20}")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} テスト中にエラー: {e}")
            test_results[test_name] = False
    
    # 結果サマリー
    print("\n" + "="*60)
    print("📊 テスト結果サマリー")
    print("-"*60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    print("-"*60)
    
    if all_passed:
        print("🎉 全テストパス！Aqua Mirrorを起動できます")
        print("\n次のステップ:")
        print("1. python main.py --debug  # デバッグモードで起動")
        print("2. または python main.py --demo  # デモモード")
    else:
        print("⚠️ 一部テストに失敗しました")
        print("問題があるコンポーネントを修正してください")
    
    return all_passed

if __name__ == "__main__":
    main()
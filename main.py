#!/usr/bin/env python3
"""
Aqua Mirror - Interactive Art Project
Day 1 基本版
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_camera():
    """カメラテスト"""
    print("🔍 カメラテスト実行...")
    
    try:
        from core.camera_manager import CameraManager
        
        manager = CameraManager()
        if manager.initialize():
            print("✅ カメラテスト成功")
            manager.cleanup()
            return True
        else:
            print("❌ カメラテスト失敗")
            return False
            
    except Exception as e:
        print(f"❌ カメラテストエラー: {e}")
        return False

def test_opencv():
    """OpenCVテスト"""
    print("🔍 OpenCVテスト実行...")
    
    try:
        import cv2
        print(f"OpenCV バージョン: {cv2.__version__}")
        
        # CUDA確認
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"CUDA デバイス数: {cuda_devices}")
        except:
            print("CUDA: 利用不可")
        
        print("✅ OpenCVテスト成功")
        return True
        
    except Exception as e:
        print(f"❌ OpenCVテストエラー: {e}")
        return False

def test_mediapipe():
    """MediaPipeテスト"""
    print("🔍 MediaPipeテスト実行...")
    
    try:
        import mediapipe as mp
        print(f"MediaPipe バージョン: {mp.__version__}")
        
        # 基本初期化テスト
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1
        )
        
        print("✅ MediaPipeテスト成功")
        return True
        
    except Exception as e:
        print(f"❌ MediaPipeテストエラー: {e}")
        return False

def test_pygame():
    """Pygameテスト"""
    print("🔍 Pygameテスト実行...")
    
    try:
        import pygame
        pygame.init()
        
        print(f"Pygame バージョン: {pygame.version.ver}")
        
        # 基本画面作成テスト
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Aqua Mirror - Test")
        
        # 2秒間表示
        clock = pygame.time.Clock()
        for _ in range(60):  # 30FPS * 2秒
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            screen.fill((0, 100, 150))  # 青色背景
            pygame.display.flip()
            clock.tick(30)
        
        pygame.quit()
        print("✅ Pygameテスト成功")
        return True
        
    except Exception as e:
        print(f"❌ Pygameテストエラー: {e}")
        return False

def main():
    """Day 1 メイン関数"""
    print("🌊 Aqua Mirror Day 1 セットアップテスト")
    print("=" * 50)
    
    tests = [
        ("OpenCV", test_opencv),
        ("MediaPipe", test_mediapipe),
        ("Pygame", test_pygame),
        ("Camera", test_camera)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} テスト開始...")
        results[test_name] = test_func()
        print("-" * 30)
    
    # 結果サマリー
    print("\n📊 テスト結果サマリー:")
    print("=" * 50)
    
    success_count = 0
    for test_name, result in results.items():
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name:12}: {status}")
        if result:
            success_count += 1
    
    print(f"\n🎯 成功率: {success_count}/{len(tests)} ({success_count/len(tests)*100:.1f}%)")
    
    if success_count == len(tests):
        print("\n🎉 おめでとうございます！Day 1 セットアップ完了です！")
        print("次のステップ: Day 2 の開発に進むことができます。")
    else:
        print("\n⚠️  いくつかのテストが失敗しました。")
        print("失敗したコンポーネントの確認と修正が必要です。")
    
    return success_count == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

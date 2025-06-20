#!/usr/bin/env python3
# test_new_vision.py - 新VisionProcessor動作確認テスト

import sys
import os
import json
import time

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def test_vision_processor():
    """VisionProcessor動作確認"""
    print("🧪 VisionProcessor動作確認テスト開始\n")
    
    # 1. 設定ファイル読み込み
    print("📋 設定ファイル読み込み...")
    try:
        with open('config/config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ config.json読み込み成功")
    except Exception as e:
        print(f"❌ config.json読み込み失敗: {e}")
        return False
    
    # 2. VisionProcessor初期化
    print("\n🤖 VisionProcessor初期化...")
    try:
        from src.visionctr import VisionProcessor
        vision_processor = VisionProcessor(config)
        print("✅ VisionProcessor初期化成功")
    except Exception as e:
        print(f"❌ VisionProcessor初期化失敗: {e}")
        return False
    
    # 3. デバッグ情報確認
    print("\n📊 デバッグ情報確認...")
    try:
        debug_info = vision_processor.get_debug_info()
        print("✅ デバッグ情報取得成功:")
        for key, value in debug_info.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"❌ デバッグ情報取得失敗: {e}")
    
    # 4. フレーム処理テスト
    print("\n🎥 フレーム処理テスト...")
    try:
        for i in range(5):
            print(f"フレーム {i+1}/5 処理中...")
            result = vision_processor.process_frame()
            
            if result:
                print(f"   顔検出: {result.get('face_detected', 'Unknown')}")
                print(f"   手検出: {result.get('hands_detected', 'Unknown')}")
                print(f"   フレーム形状: {result.get('frame_shape', 'Unknown')}")
            else:
                print("   結果なし")
            
            time.sleep(0.5)
        
        print("✅ フレーム処理テスト完了")
    except Exception as e:
        print(f"❌ フレーム処理テスト失敗: {e}")
    
    # 5. クリーンアップ
    print("\n🧹 クリーンアップ...")
    try:
        vision_processor.cleanup()
        print("✅ クリーンアップ完了")
    except Exception as e:
        print(f"❌ クリーンアップ失敗: {e}")
    
    print("\n🎉 VisionProcessor動作確認テスト完了！")
    return True

def test_main_app():
    """メインアプリケーション動作確認"""
    print("\n🚀 メインアプリケーション動作確認...")
    
    try:
        # main.pyの動作確認（5秒間）
        import subprocess
        import signal
        
        print("main.py を5秒間実行...")
        process = subprocess.Popen([sys.executable, "main.py", "--debug"], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # 5秒後に終了
        time.sleep(5)
        process.terminate()
        
        stdout, stderr = process.communicate(timeout=2)
        
        if process.returncode is None or process.returncode == 0:
            print("✅ メインアプリケーション起動成功")
            return True
        else:
            print(f"❌ メインアプリケーション起動失敗:")
            if stderr:
                print(f"エラー: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ メインアプリケーション確認失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    print("=" * 60)
    print("🌊 Aqua Mirror - MediaPipe 0.10.x 対応確認テスト")
    print("=" * 60)
    
    # VisionProcessor単体テスト
    vision_test_result = test_vision_processor()
    
    # メインアプリケーションテスト
    if vision_test_result:
        app_test_result = test_main_app()
    else:
        app_test_result = False
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー:")
    print("=" * 60)
    print(f"VisionProcessor: {'✅ 成功' if vision_test_result else '❌ 失敗'}")
    print(f"メインアプリ:    {'✅ 成功' if app_test_result else '❌ 失敗'}")
    
    if vision_test_result and app_test_result:
        print("\n🎉 すべてのテストが成功！")
        print("🚀 Aqua Mirror の開発を継続できます！")
        return True
    elif vision_test_result:
        print("\n⚠️ VisionProcessorは動作しています")
        print("🔧 メインアプリケーションの微調整が必要な可能性があります")
        return True
    else:
        print("\n❌ 重大な問題があります")
        print("🔍 手動での確認・修正が必要です")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
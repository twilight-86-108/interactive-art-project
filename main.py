#!/usr/bin/env python3
"""
Aqua Mirror - Interactive Art Project
Day 2 版 - GPU加速・エラーハンドリング統合
"""

import sys
import os
import json
from pathlib import Path

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def load_config():
    """設定ファイル読み込み"""
    config_path = PROJECT_ROOT / "config" / "config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 設定読み込み完了: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 設定読み込みエラー: {e}")
        # デフォルト設定使用
        return {
            'hardware': {
                'camera': {'device_id': 0},
                'display': {'width': 1280, 'height': 720, 'fullscreen': False}
            },
            'performance': {'target_fps': 30},
            'debug_mode': True
        }

def run_component_tests():
    """コンポーネント単体テスト実行"""
    print("🧪 コンポーネントテスト実行...")
    
    tests = []
    
    # GPU処理テスト
    try:
        from src.core.gpu_processor import GPUProcessor
        gpu = GPUProcessor()
        print(f"✅ GPU処理: {gpu.device_count} devices")
        tests.append(True)
    except Exception as e:
        print(f"❌ GPU処理テストエラー: {e}")
        tests.append(False)
    
    # エラーマネージャーテスト
    try:
        from src.core.error_manager import ErrorManager
        error_mgr = ErrorManager()
        print("✅ エラーマネージャー: OK")
        tests.append(True)
    except Exception as e:
        print(f"❌ エラーマネージャーテストエラー: {e}")
        tests.append(False)
    
    # パフォーマンスモニターテスト
    try:
        from src.core.performance_monitor import PerformanceMonitor
        perf_mon = PerformanceMonitor()
        print("✅ パフォーマンスモニター: OK")
        tests.append(True)
    except Exception as e:
        print(f"❌ パフォーマンスモニターテストエラー: {e}")
        tests.append(False)
    
    success_rate = sum(tests) / len(tests) * 100
    print(f"📊 コンポーネントテスト成功率: {success_rate:.1f}%")
    
    return all(tests)

def main():
    """Day 2 メイン関数"""
    print("🌊 Aqua Mirror Day 2 - GPU加速・エラーハンドリング版")
    print("=" * 60)
    
    # 設定読み込み
    config = load_config()
    
    # コンポーネントテスト
    if not run_component_tests():
        print("⚠️  コンポーネントテストに失敗がありますが、継続します...")
    
    # メインアプリケーション実行
    try:
        print("\n🚀 メインアプリケーション起動...")
        from src.core.app import AquaMirrorApp
        
        app = AquaMirrorApp(config)
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  ユーザーによって停止されました")
    except Exception as e:
        print(f"\n❌ アプリケーションエラー: {e}")
        return False
    
    print("\n🎉 Day 2 実行完了！")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

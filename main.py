#!/usr/bin/env python3.11
"""
Aqua Mirror ModernGL版 - エントリーポイント
感情を映す水鏡：GPU加速インタラクティブアート作品

起動方法:
  python main.py
  python main.py --config config/development_config.json
  python main.py --debug
"""

import sys
import argparse
from pathlib import Path

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent))

from src.core.moderngl_app import ModernGLApp


def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(
        description="Aqua Mirror ModernGL版 - GPU加速感情認識アート",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.json",
        help="設定ファイルパス (デフォルト: config/config.json)"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="デバッグモード有効化"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Aqua Mirror ModernGL 1.0.0"
    )
    
    return parser.parse_args()


def main():
    """メイン関数"""
    print("🌊 Aqua Mirror ModernGL版 起動中...")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # 引数解析
    args = parse_arguments()
    
    # 設定ファイル確認
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {config_path}")
        print("基本設定ファイルを作成してください。")
        return 1
    
    try:
        # アプリケーション初期化
        app = ModernGLApp(config_path=str(config_path))
        
        # デバッグモード設定
        if args.debug:
            app.config.set('debug.enable_debug', True)
            app.config.set('debug.show_fps', True)
            app.config.set('debug.show_gpu_stats', True)
            print("🐛 デバッグモード有効")
        
        # 初期化・実行
        if app.initialize():
            print("🚀 アプリケーション開始")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("操作方法:")
            print("  ESC: 終了")
            print("  P: パフォーマンス統計表示")
            print("  F11: フルスクリーン切り替え")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            app.run()
            return 0
        else:
            print("❌ アプリケーション初期化失敗")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによる終了")
        return 0
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
インタラクティブアートプロジェクト - メインエントリーポイント
"""

import sys
import os

# srcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from app import App
from config_loader import ConfigLoader

def main():
    """メイン関数"""
    try:
        # 設定ファイル読み込み
        config_loader = ConfigLoader('config.json')
        config = config_loader.load()
        
        # アプリケーション初期化・実行
        app = App(config)
        app.run()
        
    except KeyboardInterrupt:
        print("アプリケーションを終了します...")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
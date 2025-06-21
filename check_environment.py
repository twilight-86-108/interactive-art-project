# check_environment.py（設定確認用）
#!/usr/bin/env python3
import sys
import os
from pathlib import Path

print("🔍 Python環境診断")
print("=" * 50)

print(f"Python実行ファイル: {sys.executable}")
print(f"Pythonバージョン: {sys.version}")
print(f"現在の作業ディレクトリ: {os.getcwd()}")
print(f"仮想環境: {os.environ.get('VIRTUAL_ENV', 'なし')}")

print("\n📦 重要パッケージ確認:")
packages = ['cv2', 'mediapipe', 'pygame', 'numpy']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'バージョン不明')
        print(f"✅ {pkg}: {version}")
    except ImportError:
        print(f"❌ {pkg}: インストールされていません")

print("\n📁 プロジェクト構造確認:")
project_files = ['main.py', 'src/', 'config/', 'requirements.txt']
for file in project_files:
    if Path(file).exists():
        print(f"✅ {file}")
    else:
        print(f"❌ {file}: 見つかりません")
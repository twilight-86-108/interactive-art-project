@echo off
echo 🔧 Aqua Mirror トラブルシューティング
echo ==================================

echo 1. GPU ドライバー確認:
nvidia-smi
echo.

echo 2. Python 環境確認:
python --version
python -c "import sys; print(f'実行ファイル: {sys.executable}')"
echo.

echo 3. 主要パッケージ確認:
python -c "
import sys
packages = ['pygame', 'moderngl', 'numpy']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', '不明')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: 未インストール')
"
echo.

echo 4. GPU 環境変数:
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo NVIDIA_VISIBLE_DEVICES=%NVIDIA_VISIBLE_DEVICES%
echo.

echo 5. Windows グラフィック設定確認:
echo Windows設定 > システム > ディスプレイ > グラフィックの設定
echo Python.exe を高パフォーマンス GPU に設定してください
echo.

pause

@echo off
echo 🌊 Aqua Mirror - Pygame + ModernGL版 起動
echo ==========================================

REM GPU環境変数設定
call set_gpu_env.bat

REM 仮想環境有効化
if exist "venv\Scripts\activate.bat" (
    echo 🔧 仮想環境有効化中...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ 仮想環境が見つかりません
    echo venv\Scripts\activate.bat を確認してください
    pause
    exit /b 1
)

REM GPU確認
echo 🔍 GPU確認中...
nvidia-smi --query-gpu=name --format=csv,noheader

REM アプリケーション起動
echo 🚀 Aqua Mirror 起動中...
python pygame_moderngl_app_complete.py

pause

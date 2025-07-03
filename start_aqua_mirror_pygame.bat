@echo off
echo 🌊 Aqua Mirror - Pygame版起動
echo =============================

REM 環境変数設定
set CUDA_VISIBLE_DEVICES=0
set NVIDIA_VISIBLE_DEVICES=all
set __GL_SYNC_TO_VBLANK=1

REM 仮想環境有効化
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM アプリケーション起動
echo 🚀 起動中...
python main.py

pause

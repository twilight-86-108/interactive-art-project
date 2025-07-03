@echo off
echo 🔍 GPU設定確認
echo ================

echo 環境変数確認:
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo NVIDIA_VISIBLE_DEVICES=%NVIDIA_VISIBLE_DEVICES%
echo.

echo GPU情報:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo.

echo Python環境:
python -c "import sys; print(f'Python: {sys.version}'); print(f'実行ファイル: {sys.executable}')"
echo.

pause

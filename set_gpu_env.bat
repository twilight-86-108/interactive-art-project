@echo off
REM Aqua Mirror GPU環境変数設定
echo 🔧 GPU環境変数設定中...

set CUDA_VISIBLE_DEVICES=0
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set NVIDIA_VISIBLE_DEVICES=all
set NVIDIA_DRIVER_CAPABILITIES=all
set __GL_SYNC_TO_VBLANK=1
set __GL_SYNC_DISPLAY_DEVICE=DP-0
set LIBGL_ALWAYS_INDIRECT=0
set LIBGL_ALWAYS_SOFTWARE=0
set __GLX_VENDOR_LIBRARY_NAME=nvidia

echo ✅ GPU環境変数設定完了
echo 現在のプロセスでのみ有効です

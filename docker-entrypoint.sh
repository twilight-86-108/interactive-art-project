#!/bin/bash
set -e

echo "🔍 GPU・OpenGL環境確認中..."

# NVIDIA GPU確認
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU検出:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️ NVIDIA GPU未検出（CPU描画モード）"
fi

# OpenGL確認
if command -v glxinfo &> /dev/null; then
    echo "✅ OpenGL情報:"
    glxinfo | grep "OpenGL version" || echo "OpenGL version情報取得不可"
    glxinfo | grep "OpenGL renderer" || echo "OpenGL renderer情報取得不可"
else
    echo "⚠️ glxinfo未インストール"
fi

# ModernGL環境確認
echo "🔍 ModernGL環境確認中..."
python3.11 -c "
try:
    import moderngl
    import glfw
    print(f'ModernGL: {moderngl.__version__}')
    print(f'GLFW: {glfw.__version__}')
    print('✅ ModernGL環境確認完了')
except ImportError as e:
    print(f'❌ ModernGL環境確認失敗: {e}')
    exit(1)
" || {
    echo "❌ ModernGL環境確認失敗"
    echo "CPU描画モードで継続します"
}

# カメラデバイス確認
echo "📹 カメラデバイス確認中..."
ls /dev/video* 2>/dev/null || echo "⚠️ カメラデバイス未検出"

echo "🚀 Aqua Mirror ModernGL版を起動中..."
exec "$@"
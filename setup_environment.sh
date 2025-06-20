#!/bin/bash
# setup_environment.sh - 環境設定最適化スクリプト

echo "🔧 Aqua Mirror 環境設定最適化"

# 1. TensorFlow警告抑制
echo "📊 TensorFlow警告抑制設定..."
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# 2. CUDA警告抑制
echo "🖥️ CUDA警告抑制設定..."
export CUDA_VISIBLE_DEVICES=0

# 3. SDL/Pygame設定
echo "🎮 SDL/Pygame設定..."
export SDL_VIDEODRIVER=x11
export SDL_AUDIODRIVER=pulse

# 4. MediaPipe設定
echo "🤖 MediaPipe設定..."
export MEDIAPIPE_DISABLE_GPU=0

# 5. 環境変数をファイルに保存
echo "💾 環境変数をファイルに保存..."
cat > .env << EOF
# Aqua Mirror 環境設定
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=x11
export SDL_AUDIODRIVER=pulse
export MEDIAPIPE_DISABLE_GPU=0
EOF

echo "✅ 環境設定完了"
echo ""
echo "📋 使用方法:"
echo "   source .env       # 環境変数読み込み"
echo "   python main.py    # アプリケーション実行"
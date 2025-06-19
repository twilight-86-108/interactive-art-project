#!/bin/bash
# Aqua Mirror - 開発環境セットアップスクリプト

echo "🌊 Aqua Mirror セットアップを開始します..."

# 1. システム更新
echo "📦 システム更新中..."
sudo apt update && sudo apt upgrade -y

# 2. 基本開発ツール
echo "🛠️ 基本ツールインストール..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    pkg-config

# 3. OpenCV依存関係
echo "📹 OpenCV依存関係インストール..."
sudo apt install -y \
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0

# 4. 音響関係
echo "🔊 音響ライブラリインストール..."
sudo apt install -y \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    pulseaudio \
    pavucontrol

# 5. GPU関係（NVIDIA）
echo "🖥️ GPU環境セットアップ..."
# NVIDIA GPG キー追加
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# CUDA インストール
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# 6. Docker セットアップ
echo "🐳 Docker セットアップ..."
# Docker公式リポジトリ追加
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Docker権限設定
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# 7. プロジェクト環境構築
echo "🏗️ プロジェクト環境構築..."
cd /mnt/c/projects  # Windows側のプロジェクトフォルダ
git clone https://github.com/your-username/aqua-mirror.git  # あなたのリポジトリ
cd aqua-mirror

# Python仮想環境
python3.11 -m venv venv
source venv/bin/activate

# Python依存関係インストール
pip install --upgrade pip
pip install -r requirements.txt

# 8. カメラ権限設定
echo "📹 カメラ権限設定..."
sudo usermod -aG video $USER

# 9. 設定ファイル初期化
echo "⚙️ 設定ファイル初期化..."
cp config/config.template.json config/config.json

# 10. 初期テスト実行
echo "🧪 初期テスト実行..."
python tests/test_camera.py
python tests/test_gpu.py

echo "✅ セットアップ完了！"
echo ""
echo "🔄 再起動またはログアウト・ログインしてください（Docker権限適用のため）"
echo ""
echo "次のステップ:"
echo "1. 再起動後、'source venv/bin/activate' で仮想環境を有効化"
echo "2. 'python main.py' でアプリケーション起動"
echo "3. カメラが認識されない場合は 'tools/troubleshoot.py' を実行"
echo ""
echo "🌊 Aqua Mirror開発を始めましょう！"
# setup_aqua_mirror_gpu.py
"""
Aqua Mirror プロジェクト用 GPU環境自動セットアップ
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """コマンド実行ヘルパー"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 完了")
            return True
        else:
            print(f"⚠️ {description} 警告: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} エラー: {e}")
        return False

def install_cuda_toolkit():
    """CUDA Toolkit インストール"""
    commands = [
        ("sudo apt update", "システム更新"),
        ("sudo apt install -y nvidia-cuda-toolkit", "CUDA Toolkit インストール")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    # 環境変数追加
    bashrc_lines = [
        'export PATH=/usr/local/cuda/bin:$PATH',
        'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
    ]
    
    with open(os.path.expanduser("~/.bashrc"), "a") as f:
        f.write("\n# CUDA環境変数 (Aqua Mirror)\n")
        for line in bashrc_lines:
            f.write(f"{line}\n")
    
    print("✅ 環境変数を~/.bashrcに追加しました")
    return True

def install_python_packages():
    """Python GPU パッケージインストール"""
    packages = [
        "cupy-cuda12x",  # CUDA 12.x用CuPy
        "tensorflow[and-cuda]",  # TensorFlow GPU版
    ]
    
    for package in packages:
        cmd = f"pip install {package}"
        if not run_command(cmd, f"{package} インストール"):
            print(f"⚠️ {package} インストール失敗 - 手動インストールが必要です")

def update_requirements():
    """requirements.txt 更新"""
    requirements_path = "requirements.txt"
    
    gpu_requirements = [
        "# GPU関連ライブラリ",
        "cupy-cuda12x>=12.0.0",
        "tensorflow[and-cuda]>=2.13.0",
        "",
        "# 既存ライブラリ",
        "opencv-python==4.9.0",
        "mediapipe==0.10.11",
        "pygame==2.5.2",
        "numpy==1.26.4"
    ]
    
    try:
        with open(requirements_path, "w") as f:
            f.write("\n".join(gpu_requirements))
        print(f"✅ {requirements_path} を更新しました")
    except Exception as e:
        print(f"⚠️ {requirements_path} 更新失敗: {e}")

def create_gpu_config():
    """GPU設定ファイル作成"""
    gpu_config = {
        "gpu_optimization": {
            "enabled": True,
            "memory_limit_gb": 6,
            "memory_growth": True,
            "device_id": 0,
            "fallback_to_cpu": True
        },
        "rtx4060_settings": {
            "thermal_management": True,
            "power_limit": 75,
            "memory_optimization": True
        }
    }
    
    import json
    config_dir = "config"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    config_path = os.path.join(config_dir, "gpu_config.json")
    
    try:
        with open(config_path, "w") as f:
            json.dump(gpu_config, f, indent=2)
        print(f"✅ GPU設定ファイル作成: {config_path}")
    except Exception as e:
        print(f"⚠️ GPU設定ファイル作成失敗: {e}")

def main():
    """メイン実行"""
    print("🌊 Aqua Mirror GPU環境セットアップ開始")
    print("=" * 60)
    
    # 前提条件確認
    if not run_command("nvidia-smi", "NVIDIA GPU確認"):
        print("❌ NVIDIA GPUが認識されていません")
        return False
    
    # 仮想環境確認
    if "VIRTUAL_ENV" not in os.environ:
        print("⚠️ 仮想環境がアクティブでありません")
        print("   以下を実行してから再試行してください:")
        print("   source venv/bin/activate")
        return False
    
    # ステップ実行
    steps = [
        ("CUDA Toolkit インストール", install_cuda_toolkit),
        ("Python GPU パッケージインストール", install_python_packages),
        ("requirements.txt 更新", update_requirements),
        ("GPU設定ファイル作成", create_gpu_config),
    ]
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 40)
        if not step_func():
            print(f"⚠️ {step_name} で問題が発生しました")
        
    print("\n" + "=" * 60)
    print("🎯 セットアップ完了!")
    print("\n次のステップ:")
    print("1. 新しいターミナルを開く（環境変数読み込みのため）")
    print("2. source venv/bin/activate")
    print("3. python test_gpu_setup.py  # 動作確認")
    print("4. python main.py  # Aqua Mirror 起動")
    
    return True

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Windows用 NVIDIA GPU完全有効化スクリプト
GeForce RTX 4060 Laptop GPU を ModernGL で使用するための包括的設定
"""

import os
import sys
import subprocess
import json
import logging
import winreg
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class WindowsNVIDIAGPUEnabler:
    """Windows環境でのNVIDIA GPU完全有効化管理"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.gpu_info = {}
        self.system_info = {}
        
    def _setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'gpu_setup_windows.log', encoding='utf-8')
            ]
        )
        return logging.getLogger("WindowsGPUEnabler")
    
    def check_windows_environment(self) -> bool:
        """Windows環境確認"""
        self.logger.info("🖥️ Windows環境確認中...")
        
        try:
            # Windows バージョン確認
            import platform
            system_info = {
                'os': platform.system(),
                'version': platform.version(),
                'architecture': platform.architecture()[0],
                'processor': platform.processor()
            }
            
            self.system_info = system_info
            
            self.logger.info(f"OS: {system_info['os']} {system_info['version']}")
            self.logger.info(f"アーキテクチャ: {system_info['architecture']}")
            
            if system_info['os'] != 'Windows':
                self.logger.error("❌ Windows環境ではありません")
                return False
            
            # Python環境確認
            python_info = {
                'version': sys.version,
                'executable': sys.executable,
                'platform': sys.platform
            }
            
            self.logger.info(f"Python: {python_info['version'].split()[0]}")
            self.logger.info(f"実行ファイル: {python_info['executable']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Windows環境確認失敗: {e}")
            return False
    
    def check_nvidia_gpu_detailed(self) -> bool:
        """詳細なNVIDIA GPU確認"""
        self.logger.info("🔍 詳細GPU確認中...")
        
        try:
            # nvidia-smi コマンド実行
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader'],
                capture_output=True, text=True, check=True
            )
            
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 4:
                    gpu_info = {
                        'index': i,
                        'name': parts[0],
                        'driver_version': parts[1],
                        'memory_total': parts[2],
                        'compute_capability': parts[3]
                    }
                    
                    self.gpu_info[f'gpu_{i}'] = gpu_info
                    
                    self.logger.info(f"GPU {i}: {gpu_info['name']}")
                    self.logger.info(f"  ドライバー: {gpu_info['driver_version']}")
                    self.logger.info(f"  メモリ: {gpu_info['memory_total']}")
                    self.logger.info(f"  計算能力: {gpu_info['compute_capability']}")
            
            # GeForce RTX 4060 確認
            found_4060 = False
            for gpu_key, gpu_data in self.gpu_info.items():
                if 'RTX 4060' in gpu_data['name'] or '4060' in gpu_data['name']:
                    found_4060 = True
                    self.logger.info("🎯 GeForce RTX 4060 Laptop GPU 検出！")
                    break
            
            if not found_4060:
                self.logger.warning("⚠️ GeForce RTX 4060が見つかりません")
                self.logger.info("利用可能なGPU:")
                for gpu_key, gpu_data in self.gpu_info.items():
                    self.logger.info(f"  - {gpu_data['name']}")
            
            return len(self.gpu_info) > 0
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ nvidia-smi実行エラー: {e}")
            return False
        except FileNotFoundError:
            self.logger.error("❌ nvidia-smi が見つかりません")
            self.logger.error("NVIDIAドライバーがインストールされているか確認してください")
            return False
    
    def check_cuda_installation(self) -> bool:
        """CUDA インストール確認"""
        self.logger.info("🔧 CUDA インストール確認中...")
        
        try:
            # nvcc確認
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True)
            cuda_version = "不明"
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.strip()
                    break
            
            self.logger.info(f"✅ CUDA インストール済み: {cuda_version}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("⚠️ CUDA が見つかりません")
            self.logger.info("CUDA Toolkit のインストールを推奨します")
            return False
    
    def set_windows_gpu_environment_variables(self) -> bool:
        """Windows GPU環境変数設定"""
        self.logger.info("⚙️ Windows GPU環境変数設定中...")
        
        # 現在のプロセス用環境変数
        gpu_env_vars = {
            'CUDA_VISIBLE_DEVICES': '0',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'NVIDIA_VISIBLE_DEVICES': 'all',
            'NVIDIA_DRIVER_CAPABILITIES': 'all',
            '__GL_SYNC_TO_VBLANK': '1',
            '__GL_SYNC_DISPLAY_DEVICE': 'DP-0',
            'LIBGL_ALWAYS_INDIRECT': '0',
            'LIBGL_ALWAYS_SOFTWARE': '0',
            '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'
        }
        
        # 現在のプロセスに設定
        for key, value in gpu_env_vars.items():
            os.environ[key] = value
            self.logger.info(f"  {key} = {value}")
        
        # Windows レジストリに永続化
        success_count = self._set_windows_registry_variables(gpu_env_vars)
        
        # バッチファイル作成
        self._create_environment_batch_file(gpu_env_vars)
        
        self.logger.info(f"✅ 環境変数設定完了: {success_count}/{len(gpu_env_vars)} 永続化")
        return True
    
    def _set_windows_registry_variables(self, env_vars: Dict[str, str]) -> int:
        """Windows レジストリに環境変数設定"""
        success_count = 0
        
        try:
            # ユーザー環境変数にsetxで設定
            for key, value in env_vars.items():
                try:
                    subprocess.run(['setx', key, value], 
                                 capture_output=True, check=True, timeout=30)
                    success_count += 1
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"⚠️ {key} 設定失敗: {e}")
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"⚠️ {key} 設定タイムアウト")
        
        except Exception as e:
            self.logger.warning(f"⚠️ レジストリ設定エラー: {e}")
        
        return success_count
    
    def _create_environment_batch_file(self, env_vars: Dict[str, str]):
        """環境変数設定バッチファイル作成"""
        batch_content = "@echo off\n"
        batch_content += "REM Aqua Mirror GPU環境変数設定\n"
        batch_content += "echo 🔧 GPU環境変数設定中...\n\n"
        
        for key, value in env_vars.items():
            batch_content += f"set {key}={value}\n"
        
        batch_content += "\necho ✅ GPU環境変数設定完了\n"
        batch_content += "echo 現在のプロセスでのみ有効です\n"
        
        batch_path = Path("set_gpu_env.bat")
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        self.logger.info(f"✅ 環境変数バッチファイル作成: {batch_path}")
    
    def configure_windows_graphics_preference(self) -> bool:
        """Windows グラフィック優先設定"""
        self.logger.info("🎮 Windows グラフィック設定確認中...")
        
        python_exe = Path(sys.executable)
        
        # レジストリを使った自動設定を試行
        try:
            self._set_graphics_preference_registry(python_exe)
        except Exception as e:
            self.logger.warning(f"⚠️ 自動グラフィック設定失敗: {e}")
        
        # 手動設定の案内
        self.logger.info("💡 手動でのグラフィック設定手順:")
        self.logger.info("1. Windows設定 (Win + I) を開く")
        self.logger.info("2. システム > ディスプレイ を選択")
        self.logger.info("3. 'グラフィックの設定' をクリック")
        self.logger.info("4. '参照' をクリックして以下を追加:")
        self.logger.info(f"   {python_exe}")
        self.logger.info("5. 追加後、'オプション' → '高パフォーマンス' を選択")
        self.logger.info("6. '保存' をクリック")
        
        return True
    
    def _set_graphics_preference_registry(self, python_exe: Path):
        """レジストリを使ったグラフィック優先設定"""
        try:
            # Windows 10/11 のグラフィック設定レジストリキー
            key_path = r"SOFTWARE\Microsoft\DirectX\UserGpuPreferences"
            
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                # 高パフォーマンス GPU を優先
                winreg.SetValueEx(
                    key, 
                    str(python_exe), 
                    0, 
                    winreg.REG_SZ, 
                    "GpuPreference=2;"  # 2 = 高パフォーマンス
                )
            
            self.logger.info("✅ レジストリでグラフィック優先設定完了")
            
        except Exception as e:
            self.logger.warning(f"⚠️ レジストリグラフィック設定失敗: {e}")
    
    def test_pygame_moderngl_gpu(self) -> bool:
        """Pygame + ModernGL GPU テスト"""
        self.logger.info("🧪 Pygame + ModernGL GPU テスト実行中...")
        
        try:
            import pygame
            import moderngl
            
            # Pygame初期化
            pygame.init()
            
            # OpenGL設定
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, 
                                           pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            
            # 非表示ウィンドウ作成
            screen = pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
            
            # ModernGLコンテキスト作成
            ctx = moderngl.create_context()
            
            # GPU情報取得
            gpu_info = {
                'renderer': ctx.info.get('GL_RENDERER', 'Unknown'),
                'version': ctx.info.get('GL_VERSION', 'Unknown'),
                'vendor': ctx.info.get('GL_VENDOR', 'Unknown'),
                'shading_language': ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown')
            }
            
            self.logger.info("🖥️ OpenGL GPU情報:")
            for key, value in gpu_info.items():
                self.logger.info(f"  {key}: {value}")
            
            # NVIDIA GPU確認
            nvidia_detected = 'NVIDIA' in gpu_info['renderer']
            rtx4060_detected = '4060' in gpu_info['renderer']
            
            if nvidia_detected:
                if rtx4060_detected:
                    self.logger.info("🎯 GeForce RTX 4060 Laptop GPU 有効化成功！")
                else:
                    self.logger.info("✅ NVIDIA GPU 有効化成功")
            else:
                self.logger.warning("⚠️ NVIDIA GPU が使用されていません")
                self.logger.warning("統合グラフィックスが使用されている可能性があります")
            
            # 簡単なGPU演算テスト
            vertex_shader = """
            #version 330 core
            in vec2 position;
            void main() { gl_Position = vec4(position, 0.0, 1.0); }
            """
            
            fragment_shader = """
            #version 330 core
            out vec4 color;
            void main() { color = vec4(1.0, 0.0, 0.0, 1.0); }
            """
            
            program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
            
            # テストデータ
            vertices = np.array([-1, -1, 1, -1, 0, 1], dtype=np.float32)
            vbo = ctx.buffer(vertices.tobytes())
            vao = ctx.vertex_array(program, [(vbo, '2f', 'position')])
            
            # レンダリングテスト
            vao.render()
            
            pygame.quit()
            
            self.logger.info("✅ GPU演算テスト成功")
            return nvidia_detected
            
        except ImportError as e:
            self.logger.error(f"❌ 必要なライブラリが不足: {e}")
            self.logger.info("pip install pygame moderngl で追加してください")
            return False
        except Exception as e:
            self.logger.error(f"❌ GPU テスト失敗: {e}")
            return False
        finally:
            try:
                pygame.quit()
            except:
                pass
    
    def create_gpu_config_file(self) -> Path:
        """GPU設定ファイル作成"""
        self.logger.info("📝 GPU設定ファイル作成中...")
        
        # 検出されたGPU情報を基に設定作成
        primary_gpu = None
        if self.gpu_info:
            primary_gpu = list(self.gpu_info.values())[0]
        
        config = {
            "gpu": {
                "enable_nvidia": True,
                "device_id": 0,
                "memory_limit": 0.8,
                "opengl_version": [4, 3],
                "opengl_profile": "core",
                "enable_vsync": True,
                "enable_multisample": True,
                "samples": 4
            },
            "rendering": {
                "target_fps": 60,
                "enable_gpu_compute": True,
                "texture_streaming": True,
                "gpu_memory_management": "auto"
            },
            "windows": {
                "force_nvidia": True,
                "prefer_dedicated_gpu": True,
                "disable_integrated_graphics": False,
                "graphics_preference": "high_performance"
            },
            "detected_hardware": {
                "system_info": self.system_info,
                "gpu_info": self.gpu_info,
                "detection_timestamp": time.time()
            },
            "debug": {
                "enable_debug_context": False,
                "gpu_profiling": False,
                "log_gpu_stats": True
            }
        }
        
        # 設定ファイル保存
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / "gpu_config_windows.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"✅ GPU設定ファイル作成: {config_path}")
        return config_path
    
    def create_startup_scripts(self) -> List[Path]:
        """起動スクリプト作成"""
        self.logger.info("📄 起動スクリプト作成中...")
        
        scripts = []
        
        # 1. GPU設定確認スクリプト
        gpu_check_script = '''@echo off
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
'''
        
        # 2. Aqua Mirror起動スクリプト
        aqua_mirror_script = '''@echo off
echo 🌊 Aqua Mirror - Pygame + ModernGL版 起動
echo ==========================================

REM GPU環境変数設定
call set_gpu_env.bat

REM 仮想環境有効化
if exist "venv\\Scripts\\activate.bat" (
    echo 🔧 仮想環境有効化中...
    call venv\\Scripts\\activate.bat
) else (
    echo ⚠️ 仮想環境が見つかりません
    echo venv\\Scripts\\activate.bat を確認してください
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
'''
        
        # 3. 依存関係インストールスクリプト
        install_deps_script = '''@echo off
echo 📦 依存関係インストール
echo =======================

REM 仮想環境有効化
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
) else (
    echo 仮想環境を作成しています...
    python -m venv venv
    call venv\\Scripts\\activate.bat
)

REM 基本パッケージ更新
echo 📦 基本パッケージ更新中...
python -m pip install --upgrade pip

REM 必要なパッケージインストール
echo 📦 必要なパッケージインストール中...
pip install pygame==2.5.2
pip install moderngl==5.8.2
pip install numpy==1.26.4
pip install rich==13.7.1

REM requirements.txt があればインストール
if exist "requirements.txt" (
    echo 📦 requirements.txt からインストール中...
    pip install -r requirements.txt
)

echo ✅ 依存関係インストール完了
pause
'''
        
        # スクリプトファイル作成
        script_files = [
            ("check_gpu.bat", gpu_check_script),
            ("start_aqua_mirror.bat", aqua_mirror_script),
            ("install_dependencies.bat", install_deps_script)
        ]
        
        for filename, content in script_files:
            script_path = Path(filename)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            scripts.append(script_path)
            self.logger.info(f"  ✅ {filename}")
        
        return scripts
    
    def run_complete_setup(self) -> bool:
        """完全セットアップ実行"""
        self.logger.info("🚀 Windows NVIDIA GPU 完全セットアップ開始")
        
        setup_steps = [
            ("Windows環境確認", self.check_windows_environment),
            ("NVIDIA GPU詳細確認", self.check_nvidia_gpu_detailed),
            ("CUDA インストール確認", self.check_cuda_installation),
            ("GPU環境変数設定", self.set_windows_gpu_environment_variables),
            ("Windowsグラフィック設定", self.configure_windows_graphics_preference),
            ("Pygame + ModernGL テスト", self.test_pygame_moderngl_gpu),
            ("GPU設定ファイル作成", lambda: bool(self.create_gpu_config_file())),
            ("起動スクリプト作成", lambda: bool(self.create_startup_scripts()))
        ]
        
        success_count = 0
        results = {}
        
        for step_name, step_func in setup_steps:
            self.logger.info(f"\n📋 {step_name}...")
            try:
                result = step_func()
                results[step_name] = result
                if result:
                    success_count += 1
                    self.logger.info(f"✅ {step_name} 完了")
                else:
                    self.logger.warning(f"⚠️ {step_name} で問題発生")
            except Exception as e:
                self.logger.error(f"❌ {step_name} 失敗: {e}")
                results[step_name] = False
        
        # 結果レポート
        self._print_setup_report(success_count, len(setup_steps), results)
        
        return success_count >= 6  # 最低6個のステップが成功していれば OK
    
    def _print_setup_report(self, success_count: int, total_steps: int, results: Dict[str, bool]):
        """セットアップレポート表示"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 Windows GPU セットアップ完了レポート")
        self.logger.info("=" * 60)
        
        for step_name, result in results.items():
            status = "✅" if result else "❌"
            self.logger.info(f"{status} {step_name}")
        
        self.logger.info(f"\n🎯 成功率: {success_count}/{total_steps} ({success_count/total_steps*100:.1f}%)")
        
        if success_count >= 6:
            self.logger.info("🎉 セットアップ完了！")
            self.logger.info("\n次の手順:")
            self.logger.info("1. install_dependencies.bat を実行（初回のみ）")
            self.logger.info("2. 新しいコマンドプロンプトを開く")
            self.logger.info("3. start_aqua_mirror.bat を実行")
            self.logger.info("または check_gpu.bat で設定確認")
        else:
            self.logger.warning("⚠️ セットアップに問題があります")
            self.logger.info("エラーを確認して再実行してください")

def main():
    """メイン実行"""
    print("🌊 Windows NVIDIA GPU 完全有効化ツール")
    print("=" * 50)
    print("対象: GeForce RTX 4060 Laptop GPU")
    print("用途: Aqua Mirror Pygame + ModernGL版")
    print("=" * 50)
    
    enabler = WindowsNVIDIAGPUEnabler()
    success = enabler.run_complete_setup()
    
    if success:
        print(f"\n🎉 GPU有効化セットアップ完了！")
        print("\n作成されたファイル:")
        print("- check_gpu.bat (GPU設定確認)")
        print("- install_dependencies.bat (依存関係インストール)")
        print("- start_aqua_mirror.bat (アプリケーション起動)")
        print("- set_gpu_env.bat (環境変数設定)")
        print("- config/gpu_config_windows.json (GPU設定)")
    else:
        print(f"\n❌ セットアップに問題があります")
        print("ログを確認して問題を解決してください")
    
    input("\nEnterキーを押して終了...")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
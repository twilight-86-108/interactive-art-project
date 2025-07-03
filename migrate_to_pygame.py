#!/usr/bin/env python3
"""
Aqua Mirror 完全移行スクリプト - Windows版
GLFW → Pygame + ModernGL + NVIDIA GPU 有効化
"""

import os
import sys
import shutil
import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class AquaMirrorCompleteMigration:
    """Aqua Mirror完全移行管理 - Windows版"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = Path.cwd()
        self.backup_dir = Path("migration_backup")
        self.migration_log = []
        self.migration_config = {}
        
    def _setup_logging(self):
        """ログ設定"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / 'complete_migration.log', encoding='utf-8')
            ]
        )
        return logging.getLogger("CompleteMigration")
    
    def analyze_current_project(self) -> Dict:
        """現在のプロジェクト分析"""
        self.logger.info("🔍 プロジェクト構造分析中...")
        
        analysis = {
            'glfw_files': [],
            'moderngl_files': [],
            'config_files': [],
            'shader_files': [],
            'component_files': [],
            'main_files': []
        }
        
        # プロジェクト内のファイル分析
        for py_file in self.project_root.rglob("*.py"):
            if py_file.is_file() and ".git" not in str(py_file):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # GLFW使用ファイル
                    if 'import glfw' in content or 'glfw.' in content:
                        analysis['glfw_files'].append(py_file)
                    
                    # ModernGL使用ファイル
                    if 'import moderngl' in content or 'moderngl.' in content:
                        analysis['moderngl_files'].append(py_file)
                    
                    # コンポーネントファイル
                    if any(comp in str(py_file) for comp in ['camera', 'texture', 'emotion', 'effect']):
                        analysis['component_files'].append(py_file)
                    
                    # メインファイル
                    if py_file.name in ['main.py', 'app.py'] or 'main' in py_file.name:
                        analysis['main_files'].append(py_file)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ ファイル読み込みエラー: {py_file} - {e}")
        
        # 設定ファイル
        config_patterns = ['*.json', '*.yaml', '*.yml', '*.toml']
        for pattern in config_patterns:
            # analysis['config_files'].extend(self.project_root.rglob(pattern))
            # venvディレクトリのパスを定義
            venv_path = self.project_root / 'venv-311-8'

            # venvディレクトリを除外しながらファイルを検索
            for file_path in self.project_root.rglob(pattern):
            # ファイルのパスがvenvディレクトリの中に含まれていないかチェック
                try:
                    # Python 3.9以降で利用可能な is_relative_to を使用
                    if not file_path.is_relative_to(venv_path):
                        analysis['config_files'].append(file_path)
                except AttributeError:
                    # is_relative_to が使えない古いPythonバージョンのためのフォールバック
                    if not str(file_path).startswith(str(venv_path)):
                        analysis['config_files'].append(file_path)

        # シェーダーファイル
        shader_patterns = ['*.glsl', '*.vert', '*.frag', '*.comp']
        for pattern in shader_patterns:
            analysis['shader_files'].extend(self.project_root.rglob(pattern))
        
        # 分析結果ログ
        self.logger.info("📊 プロジェクト分析結果:")
        for key, files in analysis.items():
            self.logger.info(f"  {key}: {len(files)}ファイル")
            for file in files[:3]:  # 最初の3ファイルのみ表示
                self.logger.info(f"    - {file.relative_to(self.project_root)}")
            if len(files) > 3:
                self.logger.info(f"    ... 他{len(files)-3}ファイル")
        
        return analysis
    
    def create_comprehensive_backup(self) -> bool:
        """包括的バックアップ作成"""
        self.logger.info("📦 包括的バックアップ作成中...")
        
        try:
            # バックアップディレクトリ作成
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"aqua_mirror_backup_{timestamp}"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # 重要ファイル・ディレクトリのバックアップ
            backup_targets = [
                'src/',
                'config/',
                'assets/',
                'shaders/',
                'main.py',
                'requirements.txt',
                'README.md'
            ]
            
            backup_count = 0
            for target in backup_targets:
                source = self.project_root / target
                if source.exists():
                    dest = backup_path / target
                    if source.is_dir():
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)
                    backup_count += 1
                    self.logger.info(f"  ✅ {target}")
            
            # バックアップ情報保存
            backup_info = {
                'timestamp': timestamp,
                'backup_path': str(backup_path),
                'backed_up_items': backup_count,
                'migration_version': '1.0',
                'original_project_path': str(self.project_root)
            }
            
            with open(backup_path / 'backup_info.json', 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            self.migration_log.append(f"バックアップ作成: {backup_path}")
            self.logger.info(f"✅ バックアップ完了: {backup_count}項目 → {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ バックアップ失敗: {e}")
            return False
    
    def check_dependencies(self) -> Dict[str, bool]:
        """依存関係確認"""
        self.logger.info("📦 依存関係確認中...")
        
        dependencies = {
            'pygame': False,
            'moderngl': False,
            'numpy': False,
            'opencv-python': False,
            'mediapipe': False,
            'rich': False
        }
        
        for package in dependencies.keys():
            try:
                __import__(package.replace('-', '_'))
                dependencies[package] = True
                self.logger.info(f"  ✅ {package}")
            except ImportError:
                self.logger.info(f"  ❌ {package}")
        
        missing_packages = [pkg for pkg, installed in dependencies.items() if not installed]
        
        if missing_packages:
            self.logger.warning(f"⚠️ 不足パッケージ: {', '.join(missing_packages)}")
            self.logger.info("pip install で追加する必要があります")
        else:
            self.logger.info("✅ すべての依存関係が満たされています")
        
        return dependencies
    
    def install_missing_dependencies(self, missing_deps: List[str]) -> bool:
        """不足依存関係インストール"""
        if not missing_deps:
            return True
        
        self.logger.info(f"📦 不足パッケージインストール中: {', '.join(missing_deps)}")
        
        try:
            # pip install コマンド実行
            cmd = [sys.executable, '-m', 'pip', 'install'] + missing_deps
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            self.logger.info("✅ パッケージインストール完了")
            self.migration_log.append(f"インストール: {', '.join(missing_deps)}")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ パッケージインストール失敗: {e}")
            self.logger.error(f"stdout: {e.stdout}")
            self.logger.error(f"stderr: {e.stderr}")
            return False
    
    def deploy_pygame_application(self) -> bool:
        """Pygame版アプリケーション配置"""
        self.logger.info("🚀 Pygame版アプリケーション配置中...")
        
        try:
            # pygame_moderngl_app.py が存在するか確認
            pygame_app_file = self.project_root / 'pygame_moderngl_app.py'
            
            if not pygame_app_file.exists():
                self.logger.error("❌ pygame_moderngl_app.py が見つかりません")
                self.logger.info("先にファイルを作成してください")
                return False
            
            # 新しいメインファイル作成
            new_main_content = f'''#!/usr/bin/env python3
"""
Aqua Mirror - Pygame + ModernGL版メインエントリーポイント
移行版: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """メイン実行"""
    print("🌊 Aqua Mirror - Pygame + ModernGL版")
    print("=" * 50)
    
    try:
        from pygame_moderngl_app import PygameModernGLApp
        
        app = PygameModernGLApp()
        
        if app.initialize():
            print("✅ 初期化完了")
            app.run()
            return 0
        else:
            print("❌ 初期化失敗")
            return 1
            
    except ImportError as e:
        print(f"❌ インポートエラー: {{e}}")
        return 1
    except Exception as e:
        print(f"❌ 実行エラー: {{e}}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
            
            # 既存のmain.pyをバックアップ
            main_py = self.project_root / 'main.py'
            if main_py.exists():
                backup_main = self.project_root / 'main_glfw_backup.py'
                shutil.copy2(main_py, backup_main)
                self.logger.info(f"  既存main.pyバックアップ: {backup_main}")
            
            # 新しいmain.py作成
            with open(main_py, 'w', encoding='utf-8') as f:
                f.write(new_main_content)
            
            self.logger.info("✅ Pygame版アプリケーション配置完了")
            self.migration_log.append("Pygame版アプリケーション配置")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ アプリケーション配置失敗: {e}")
            return False
    
    def create_migration_config(self) -> Dict:
        """移行設定作成"""
        self.logger.info("⚙️ 移行設定作成中...")
        
        config = {
            "migration": {
                "version": "1.0",
                "timestamp": time.time(),
                "from_framework": "glfw",
                "to_framework": "pygame",
                "target_gpu": "nvidia_rtx_4060",
                "platform": "windows"
            },
            "pygame": {
                "display": {
                    "width": 1920,
                    "height": 1080,
                    "fullscreen": False,
                    "title": "Aqua Mirror - Pygame + ModernGL (Windows)"
                },
                "opengl": {
                    "major_version": 4,
                    "minor_version": 3,
                    "profile": "core",
                    "double_buffer": True,
                    "depth_size": 24,
                    "multisample": True
                }
            },
            "gpu": {
                "enable_nvidia": True,
                "force_dedicated_gpu": True,
                "environment_variables": {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "NVIDIA_VISIBLE_DEVICES": "all",
                    "__GL_SYNC_TO_VBLANK": "1"
                }
            },
            "compatibility": {
                "preserve_existing_components": True,
                "glfw_callback_mapping": {
                    "key_callback": "pygame.KEYDOWN/KEYUP",
                    "window_size_callback": "pygame.VIDEORESIZE",
                    "mouse_callback": "pygame.MOUSEBUTTONDOWN/UP"
                }
            }
        }
        
        # 設定ファイル保存
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / "migration_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.migration_config = config
        self.logger.info(f"✅ 移行設定作成: {config_path}")
        return config
    
    def create_windows_scripts(self) -> List[Path]:
        """Windows用スクリプト作成"""
        self.logger.info("📄 Windows用スクリプト作成中...")
        
        scripts = []
        
        # 1. 完全テストスクリプト
        test_script = '''@echo off
echo 🧪 Aqua Mirror 移行完全テスト
echo ===============================

echo 📦 1. 依存関係確認...
python -c "
packages = ['pygame', 'moderngl', 'numpy', 'opencv-python', 'mediapipe']
missing = []
for pkg in packages:
    try:
        __import__(pkg.replace('-', '_'))
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'\\n⚠️ 不足: {missing}')
    print('install_dependencies.bat を実行してください')
else:
    print('\\n✅ すべての依存関係OK')
"

echo.
echo 🔍 2. GPU確認...
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo.
echo 🎮 3. Pygame + ModernGL テスト...
python -c "
try:
    import pygame
    import moderngl
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)
    screen = pygame.display.set_mode((1, 1), pygame.OPENGL | pygame.HIDDEN)
    ctx = moderngl.create_context()
    gpu = ctx.info.get('GL_RENDERER', 'Unknown')
    print(f'GPU: {gpu}')
    if 'NVIDIA' in gpu:
        print('✅ NVIDIA GPU 検出')
    else:
        print('⚠️ NVIDIA GPU 未検出')
    pygame.quit()
except Exception as e:
    print(f'❌ テスト失敗: {e}')
"

echo.
echo ===============================
pause
'''
        
        # 2. アプリケーション起動スクリプト
        app_script = '''@echo off
echo 🌊 Aqua Mirror - Pygame版起動
echo =============================

REM 環境変数設定
set CUDA_VISIBLE_DEVICES=0
set NVIDIA_VISIBLE_DEVICES=all
set __GL_SYNC_TO_VBLANK=1

REM 仮想環境有効化
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
)

REM アプリケーション起動
echo 🚀 起動中...
python main.py

pause
'''
        
        # 3. 依存関係インストールスクリプト
        deps_script = '''@echo off
echo 📦 依存関係インストール
echo =====================

REM 仮想環境確認・作成
if not exist "venv\\Scripts\\activate.bat" (
    echo 仮想環境作成中...
    python -m venv venv
)

call venv\\Scripts\\activate.bat

REM パッケージインストール
echo 基本パッケージ更新...
python -m pip install --upgrade pip

echo 必要パッケージインストール...
pip install pygame==2.5.2
pip install moderngl==5.8.2
pip install numpy==1.26.4
pip install opencv-python==4.9.0.80
pip install mediapipe==0.10.14
pip install rich==13.7.1

REM requirements.txt があれば追加インストール
if exist "requirements.txt" (
    echo requirements.txt インストール...
    pip install -r requirements.txt
)

echo ✅ インストール完了
pause
'''
        
        # 4. 問題解決スクリプト
        troubleshoot_script = '''@echo off
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
'''
        
        # スクリプトファイル作成
        script_files = [
            ("test_migration.bat", test_script),
            ("start_aqua_mirror_pygame.bat", app_script),
            ("install_dependencies.bat", deps_script),
            ("troubleshoot.bat", troubleshoot_script)
        ]
        
        for filename, content in script_files:
            script_path = Path(filename)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(content)
            scripts.append(script_path)
            self.logger.info(f"  ✅ {filename}")
        
        return scripts
    
    def run_complete_migration(self) -> bool:
        """完全移行実行"""
        self.logger.info("🚀 Aqua Mirror 完全移行開始 (Windows)")
        
        migration_steps = [
            ("プロジェクト分析", self.analyze_current_project),
            ("包括的バックアップ", self.create_comprehensive_backup),
            ("依存関係確認", self.check_dependencies),
            ("移行設定作成", self.create_migration_config),
            ("Pygame版配置", self.deploy_pygame_application),
            ("Windowsスクリプト作成", self.create_windows_scripts)
        ]
        
        success_count = 0
        results = {}
        
        for step_name, step_func in migration_steps:
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
        
        # 依存関係インストール（必要な場合）
        dep_results = results.get("依存関係確認", {})
        if isinstance(dep_results, dict):
            missing_deps = [pkg for pkg, installed in dep_results.items() if not installed]
            if missing_deps:
                self.logger.info(f"\n📦 不足依存関係インストール...")
                if self.install_missing_dependencies(missing_deps):
                    success_count += 0.5  # 部分的な追加ポイント
        
        # 移行完了レポート
        self._print_migration_report(success_count, len(migration_steps), results)
        
        return success_count >= len(migration_steps) - 1  # 1つ失敗しても許容
    
    def _print_migration_report(self, success_count: float, total_steps: int, results: Dict):
        """移行完了レポート"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 Aqua Mirror 完全移行レポート (Windows)")
        self.logger.info("=" * 60)
        
        for step_name, result in results.items():
            if isinstance(result, bool):
                status = "✅" if result else "❌"
                self.logger.info(f"{status} {step_name}")
            else:
                self.logger.info(f"ℹ️ {step_name}: 完了")
        
        for log_entry in self.migration_log:
            self.logger.info(f"  📝 {log_entry}")
        
        self.logger.info(f"\n🎯 成功率: {success_count:.1f}/{total_steps}")
        
        if success_count >= total_steps - 1:
            self.logger.info("🎉 移行完了！")
            self.logger.info("\n📋 次の手順:")
            self.logger.info("1. install_dependencies.bat を実行")
            self.logger.info("2. python gpu_enabler_windows_complete.py を実行")
            self.logger.info("3. test_migration.bat で動作確認")
            self.logger.info("4. start_aqua_mirror_pygame.bat でアプリ起動")
            
            self.logger.info("\n📁 作成されたファイル:")
            self.logger.info("- main.py (新しいPygame版エントリーポイント)")
            self.logger.info("- config/migration_config.json (移行設定)")
            self.logger.info("- *.bat ファイル (Windows用スクリプト)")
            
        else:
            self.logger.warning("⚠️ 移行に問題があります")
            self.logger.info("troubleshoot.bat を実行して問題を確認してください")

def main():
    """メイン実行"""
    print("🌊 Aqua Mirror 完全移行ツール - Windows版")
    print("=" * 50)
    print("GLFW → Pygame + ModernGL + NVIDIA GPU")
    print("=" * 50)
    
    migrator = AquaMirrorCompleteMigration()
    success = migrator.run_complete_migration()
    
    if success:
        print("\n✅ 移行準備完了！")
        print("\n🚀 次のステップ:")
        print("1. install_dependencies.bat")
        print("2. python gpu_enabler_windows_complete.py")
        print("3. start_aqua_mirror_pygame.bat")
    else:
        print("\n❌ 移行で問題が発生しました")
        print("ログを確認して問題を解決してください")
    
    input("\nEnterキーを押して終了...")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
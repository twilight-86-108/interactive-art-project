#!/usr/bin/env python3
"""
Aqua Mirror - Interactive Art Project
メインエントリーポイント (修正版)

実行方法:
    python main.py [--config CONFIG_FILE] [--debug] [--presentation]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# プロジェクトルートディレクトリの設定
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"

# Pythonパスにsrcディレクトリを追加（相対インポートエラー解決）
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 環境設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow警告を抑制

def setup_logging(debug_mode: bool = False):
    """ログ設定"""
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # ログディレクトリ作成
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # ログ設定
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / "app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def check_dependencies():
    """依存関係チェック"""
    required_packages = [
        'cv2',           # opencv-python
        'mediapipe',     # mediapipe
        'pygame',        # pygame
        'numpy',         # numpy
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 不足している依存関係: {', '.join(missing_packages)}")
        print("以下のコマンドでインストールしてください:")
        print("pip install opencv-python mediapipe pygame numpy")
        return False
    
    return True

def create_default_config():
    """デフォルト設定作成"""
    return {
        "system": {
            "name": "Aqua Mirror",
            "version": "1.0.0",
            "debug_mode": False,
            "presentation_mode": False,
            "demo_mode": False
        },
        "camera": {
            "device_id": 0,
            "width": 1920,
            "height": 1080,
            "fps": 30
        },
        "display": {
            "width": 1920,
            "height": 1080,
            "fullscreen": False
        },
        "detection": {
            "face_detection_confidence": 0.7,
            "hand_detection_confidence": 0.7,
            "max_num_faces": 1,
            "max_num_hands": 2,
            "face_detection": {
                "model_complexity": 1,
                "min_detection_confidence": 0.7,
                "min_tracking_confidence": 0.5,
                "max_num_faces": 1,
                "refine_landmarks": True
            }
        },
        "performance": {
            "target_fps": 30,
            "adaptive_quality": True
        },
        "assets": {
            "background_image": "assets/images/underwater_scene.jpg"
        }
    }

def load_config(config_path: str, logger):
    """設定読み込み（エラー耐性付き）"""
    config_file = Path(config_path)
    
    # 必要ディレクトリの作成
    required_dirs = [
        PROJECT_ROOT / "assets" / "images",
        PROJECT_ROOT / "assets" / "audio", 
        PROJECT_ROOT / "logs"
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ディレクトリ確認/作成: {dir_path}")
    
    # 設定ファイル読み込み
    if config_file.exists():
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"設定ファイル読み込み成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"設定ファイル読み込みエラー: {e}")
            logger.info("デフォルト設定を使用します")
    else:
        logger.warning(f"設定ファイルが見つかりません: {config_path}")
        logger.info("デフォルト設定で動作します")
    
    return create_default_config()

def safe_import_modules():
    """安全なモジュールインポート"""
    try:
        # 基本的なモジュールのテストインポート
        import numpy as np
        import cv2
        import pygame
        
        print("✅ 基本依存関係 OK")
        
        # MediaPipeのテストインポート
        try:
            import mediapipe as mp
            print(f"✅ MediaPipe OK (v{mp.__version__})")
        except Exception as e:
            print(f"⚠️ MediaPipe警告: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ モジュールインポートエラー: {e}")
        print("必要な依存関係がインストールされていない可能性があります")
        print("pip install -r requirements.txt を実行してください")
        return False

def try_import_app_modules():
    """アプリケーションモジュールの段階的インポート"""
    try:
        # コアモジュールのインポート試行
        from src.core.config_loader import ConfigLoader
        print("✅ ConfigLoader インポート成功")
        return ConfigLoader
        
    except ImportError as core_error:
        print(f"⚠️ コアモジュールインポートエラー: {core_error}")
        
        # フォールバック: 簡易設定ローダー
        class SimpleConfigLoader:
            def __init__(self, config_path):
                self.config_path = config_path
            
            def load(self):
                return create_default_config()
        
        return SimpleConfigLoader

def main():
    """メイン実行関数（エラー耐性強化版）"""
    # コマンドライン引数解析
    parser = argparse.ArgumentParser(description='Aqua Mirror Interactive Art')
    parser.add_argument('--config', default='config.json', help='設定ファイルパス')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--presentation', action='store_true', help='プレゼンテーションモード')
    parser.add_argument('--demo', action='store_true', help='デモモード（カメラなし）')
    parser.add_argument('--safe-mode', action='store_true', help='セーフモード（最小機能）')
    
    args = parser.parse_args()
    
    # ログ設定
    logger = setup_logging(args.debug)
    logger.info("🌊 Aqua Mirror を起動します...")
    
    try:
        # 依存関係チェック
        if not check_dependencies():
            return 1
        
        # 基本モジュールインポート
        if not safe_import_modules():
            return 1
        
        # 設定読み込み（ファイルまたはデフォルト）
        config = load_config(args.config, logger)
        
        # コマンドライン引数で設定を上書き
        if args.debug:
            config['system']['debug_mode'] = True
        if args.presentation:
            config['system']['presentation_mode'] = True
        if args.demo:
            config['system']['demo_mode'] = True
        
        # アプリケーションモジュールのインポート試行
        ConfigLoader = try_import_app_modules()
        
        try:
            # 完全版アプリケーションの実行試行
            config_loader = ConfigLoader(args.config)
            
            # 実際のアプリケーション設定読み込み
            try:
                config = config_loader.load()
            except Exception as config_error:
                logger.warning(f"設定に問題がありますが、継続します: {config_error}")
            
            # メインアプリケーションインポート・実行
            try:
                from src.core.app import AquaMirrorApp
                app = AquaMirrorApp(config)
                logger.info("🚀 メインアプリケーション開始")
                app.run()
                
            except ImportError as app_import_error:
                logger.error(f"❌ モジュールインポートエラー: {app_import_error}")
                
                # セーフモード実行
                logger.info("🔧 セーフモードで実行を試行します...")
                run_safe_mode(config, logger)
            
        except Exception as app_error:
            logger.error(f"❌ アプリケーション実行エラー: {app_error}")
            
            # セーフモード実行
            logger.info("🔧 セーフモードで実行を試行します...")
            run_safe_mode(config, logger)
        
        logger.info("✅ 正常終了")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️ ユーザーによって停止されました")
        return 0
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        return 1
    finally:
        logger.info("🌊 Aqua Mirror を終了します")

def run_safe_mode(config, logger):
    """セーフモード実行"""
    logger.info("🛡️ セーフモード開始")
    
    try:
        import pygame
        import numpy as np
        import time
        
        # 最小限のPygame初期化
        pygame.init()
        
        display_config = config.get('display', {})
        width = display_config.get('width', 800)
        height = display_config.get('height', 600)
        
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Aqua Mirror - Safe Mode")
        clock = pygame.time.Clock()
        
        logger.info("✅ セーフモード画面初期化完了")
        
        # 基本的なアニメーション
        running = True
        frame_count = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # 簡単なアニメーション
            screen.fill((0, 50, 100))  # 深い青
            
            # 波のような効果
            for i in range(5):
                y = height // 2 + np.sin(frame_count * 0.1 + i) * 50
                color = (100 + i * 30, 150 + i * 20, 200)
                pygame.draw.circle(screen, color, (width // 2, int(y)), 20 + i * 5)
            
            # 情報表示
            font = pygame.font.Font(None, 36)
            text = font.render("Aqua Mirror - Safe Mode", True, (255, 255, 255))
            screen.blit(text, (width // 2 - text.get_width() // 2, 50))
            
            info_text = font.render("ESC to exit", True, (200, 200, 200))
            screen.blit(info_text, (width // 2 - info_text.get_width() // 2, height - 50))
            
            pygame.display.flip()
            clock.tick(30)
            frame_count += 1
        
        pygame.quit()
        logger.info("✅ セーフモード正常終了")
        
    except Exception as safe_error:
        logger.error(f"❌ セーフモードエラー: {safe_error}")

if __name__ == "__main__":
    sys.exit(main())
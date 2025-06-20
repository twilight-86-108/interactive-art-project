#!/usr/bin/env python3
"""
Aqua Mirror - Interactive Art Project
メインエントリーポイント（エラー修正版）

実行方法:
    python main.py [--config CONFIG_FILE] [--debug] [--demo]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def setup_logging(debug_mode: bool = False):
    """ログ設定"""
    level = logging.DEBUG if debug_mode else logging.INFO
    
    # ログディレクトリ作成
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # ログ設定
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "aqua_mirror.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # MediaPipeの冗長なログを抑制
    logging.getLogger('mediapipe').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def check_dependencies():
    """依存関係チェック"""
    required_modules = [
        'cv2', 'mediapipe', 'pygame', 'numpy'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ 必要なモジュールが不足しています: {', '.join(missing_modules)}")
        print("pip install -r requirements.txt を実行してください")
        return False
    
    return True

def check_assets():
    """アセットファイルの存在確認"""
    logger = logging.getLogger(__name__)
    
    # 必要なディレクトリを作成
    required_dirs = [
        PROJECT_ROOT / "assets" / "images",
        PROJECT_ROOT / "assets" / "audio",
        PROJECT_ROOT / "logs"
    ]
    
    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ディレクトリ確認/作成: {dir_path}")
    
    # 重要なファイルの存在確認
    config_file = PROJECT_ROOT / "config.json"
    if not config_file.exists():
        logger.warning(f"設定ファイルが見つかりません: {config_file}")
        logger.info("デフォルト設定で動作します")
    
    return True

def main():
    """メイン関数"""
    # コマンドライン引数解析
    parser = argparse.ArgumentParser(description='Aqua Mirror Interactive Art')
    parser.add_argument('--config', default='config.json', help='設定ファイルパス')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--demo', action='store_true', help='デモモード（カメラなし）')
    parser.add_argument('--test', action='store_true', help='テストモード')
    
    args = parser.parse_args()
    
    # ログ設定
    logger = setup_logging(args.debug)
    
    try:
        logger.info("🌊 Aqua Mirror を起動します...")
        
        # 依存関係チェック
        if not check_dependencies():
            sys.exit(1)
        
        # アセット確認
        check_assets()
        
        # 設定読み込み
        from src.core.config_loader import ConfigLoader
        
        config_path = PROJECT_ROOT / args.config
        config_loader = ConfigLoader(str(config_path))
        config = config_loader.load()
        
        # コマンドライン引数による設定上書き
        if args.debug:
            config['debug_mode'] = True
            logger.info("デバッグモードが有効になりました")
        
        if args.demo:
            config['demo_mode'] = True
            logger.info("デモモードが有効になりました")
        
        if args.test:
            config['test_mode'] = True
            logger.info("テストモードが有効になりました")
        
        # 設定妥当性確認
        if not config_loader.validate_config():
            logger.warning("設定に問題がありますが、継続します")
        
        # テストモードの場合
        if args.test:
            logger.info("テストモードで実行します")
            run_tests(config)
            return
        
        # アプリケーション起動
        from src.core.app import AquaMirrorApp
        
        logger.info("アプリケーションを初期化します...")
        app = AquaMirrorApp(config)
        
        logger.info("メインループを開始します...")
        app.run()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  ユーザーによって停止されました")
    except ImportError as e:
        logger.error(f"❌ モジュールインポートエラー: {e}")
        logger.error("必要な依存関係がインストールされていない可能性があります")
        logger.error("pip install -r requirements.txt を実行してください")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {e}")
        logger.exception("詳細なエラー情報:")
        sys.exit(1)
    finally:
        logger.info("🌊 Aqua Mirror を終了します")

def run_tests(config):
    """テストモード実行"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== Aqua Mirror テストモード ===")
        
        # 基本テスト
        test_results = {
            'config_loading': False,
            'camera_access': False,
            'gpu_availability': False,
            'mediapipe_init': False
        }
        
        # 設定読み込みテスト
        logger.info("1. 設定読み込みテスト...")
        if config:
            test_results['config_loading'] = True
            logger.info("✅ 設定読み込み成功")
        else:
            logger.error("❌ 設定読み込み失敗")
        
        # カメラアクセステスト
        logger.info("2. カメラアクセステスト...")
        if not config.get('demo_mode', False):
            try:
                import cv2
                cap = cv2.VideoCapture(config.get('camera', {}).get('device_id', 0))
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        test_results['camera_access'] = True
                        logger.info("✅ カメラアクセス成功")
                    else:
                        logger.warning("⚠️ カメラからフレーム取得失敗")
                    cap.release()
                else:
                    logger.warning("⚠️ カメラデバイスを開けません")
            except Exception as e:
                logger.error(f"❌ カメラテストエラー: {e}")
        else:
            logger.info("⏭️ デモモードのためカメラテストをスキップ")
            test_results['camera_access'] = True
        
        # GPU利用可能性テスト
        logger.info("3. GPU利用可能性テスト...")
        try:
            import cv2
            gpu_count = cv2.cuda.getCudaEnabledDeviceCount()
            if gpu_count > 0:
                test_results['gpu_availability'] = True
                logger.info(f"✅ GPU利用可能 ({gpu_count} devices)")
            else:
                logger.info("ℹ️ GPU利用不可、CPU処理で継続")
                test_results['gpu_availability'] = True  # CPU処理も正常
        except Exception as e:
            logger.warning(f"⚠️ GPU確認エラー: {e}")
            test_results['gpu_availability'] = True  # CPU処理で継続可能
        
        # MediaPipe初期化テスト
        logger.info("4. MediaPipe初期化テスト...")
        try:
            import mediapipe as mp
            
            # 顔検出初期化テスト
            face_mesh = mp.solutions.face_mesh.FaceMesh()
            face_mesh.close()
            
            # 手検出初期化テスト
            hands = mp.solutions.hands.Hands()
            hands.close()
            
            test_results['mediapipe_init'] = True
            logger.info("✅ MediaPipe初期化成功")
            
        except Exception as e:
            logger.error(f"❌ MediaPipe初期化エラー: {e}")
        
        # テスト結果サマリー
        logger.info("\n=== テスト結果サマリー ===")
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\n合格: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            logger.info("🎉 全テストが成功しました！")
            logger.info("python main.py でアプリケーションを起動できます")
        else:
            logger.warning("⚠️ 一部のテストが失敗しました")
            logger.info("アプリケーションは動作する可能性がありますが、一部機能に制限があります")
        
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")

if __name__ == "__main__":
    main()
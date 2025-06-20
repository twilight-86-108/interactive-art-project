#!/usr/bin/env python3
"""
Aqua Mirror - Interactive Art Project
MediaPipe 0.10.x対応版（エラーハンドリング強化）
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

# プロジェクトルートをPythonパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def parse_arguments():
    """コマンドライン引数解析"""
    parser = argparse.ArgumentParser(description='Aqua Mirror - Interactive Art')
    parser.add_argument('--config', default='config.json', help='設定ファイルパス')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    parser.add_argument('--demo', action='store_true', help='デモモード（5秒で自動終了）')
    parser.add_argument('--test', action='store_true', help='テストモード（コンポーネント確認のみ）')
    parser.add_argument('--no-camera', action='store_true', help='カメラなしモード')
    return parser.parse_args()

def load_config(config_path):
    """設定ファイル読み込み"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 設定ファイル読み込み成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ 設定ファイルが見つかりません: {config_path}")
        print("📋 デフォルト設定を使用します")
        return get_default_config()
    except Exception as e:
        print(f"❌ 設定ファイル読み込みエラー: {e}")
        print("📋 デフォルト設定を使用します")
        return get_default_config()

def get_default_config():
    """デフォルト設定"""
    return {
        "camera": {
            "device_id": 0,
            "width": 1280,
            "height": 720,
            "fps": 30
        },
        "display": {
            "width": 1280,
            "height": 720,
            "fullscreen": False
        },
        "detection": {
            "face_detection_confidence": 0.7,
            "hand_detection_confidence": 0.7,
            "max_num_faces": 1,
            "max_num_hands": 2
        }
    }

def test_components(config):
    """コンポーネントテスト"""
    print("🧪 コンポーネントテスト実行...")
    
    results = {}
    
    # VisionProcessor テスト
    print("📹 VisionProcessor テスト...")
    try:
        from src.visionctr import VisionProcessor
        vision_processor = VisionProcessor(config)
        
        # 簡単な動作確認
        debug_info = vision_processor.get_debug_info()
        print(f"   APIバージョン: {debug_info.get('API Version', 'Unknown')}")
        print(f"   顔モデル: {debug_info.get('Face Model', 'Unknown')}")
        print(f"   手モデル: {debug_info.get('Hand Model', 'Unknown')}")
        
        vision_processor.cleanup()
        results['VisionProcessor'] = True
        print("   ✅ VisionProcessor テスト成功")
        
    except Exception as e:
        print(f"   ❌ VisionProcessor テストエラー: {e}")
        results['VisionProcessor'] = False
    
    # Pygame テスト
    print("🎮 Pygame テスト...")
    try:
        import pygame
        
        # 環境変数設定（ヘッドレス対応）
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Aqua Mirror Test")
        
        # 簡単な描画テスト
        screen.fill((0, 100, 150))
        pygame.display.flip()
        
        pygame.quit()
        results['Pygame'] = True
        print("   ✅ Pygame テスト成功")
        
    except Exception as e:
        print(f"   ❌ Pygame テストエラー: {e}")
        results['Pygame'] = False
    
    return results

def run_demo_mode(config):
    """デモモード実行（5秒間）"""
    print("🚀 デモモード開始（5秒間実行）...")
    
    try:
        # 環境変数設定
        os.environ['SDL_VIDEODRIVER'] = 'x11'  # Linux用
        
        import pygame
        from src.visionctr import VisionProcessor
        
        # 初期化
        pygame.init()
        
        display_config = config.get('display', {})
        screen = pygame.display.set_mode((
            display_config.get('width', 1280),
            display_config.get('height', 720)
        ))
        pygame.display.set_caption("Aqua Mirror - Demo Mode")
        
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        
        # VisionProcessor 初期化
        vision_processor = VisionProcessor(config)
        
        print("✅ デモモード初期化完了")
        
        # メインループ（5秒間）
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            # フレーム処理
            try:
                detection_result = vision_processor.process_frame()
                
                # 画面描画
                screen.fill((0, 50, 100))
                
                # 情報表示
                fps = clock.get_fps()
                fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
                
                # 検出結果表示
                if detection_result:
                    face_status = "顔: 検出" if detection_result.get('face_detected') else "顔: 未検出"
                    hand_status = f"手: {len(detection_result.get('hand_positions', []))}個"
                    
                    face_text = font.render(face_status, True, (255, 255, 255))
                    hand_text = font.render(hand_status, True, (255, 255, 255))
                    
                    screen.blit(face_text, (10, 50))
                    screen.blit(hand_text, (10, 90))
                
                # デモ時間表示
                remaining = 5.0 - (time.time() - start_time)
                time_text = font.render(f"残り時間: {remaining:.1f}秒", True, (255, 255, 0))
                screen.blit(time_text, (10, 130))
                
                pygame.display.flip()
                frame_count += 1
                
            except Exception as e:
                print(f"⚠️ フレーム処理エラー: {e}")
            
            clock.tick(30)
        
        # クリーンアップ
        vision_processor.cleanup()
        pygame.quit()
        
        print(f"✅ デモモード完了 - {frame_count}フレーム処理")
        return True
        
    except Exception as e:
        print(f"❌ デモモード実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_application(config, debug_mode=False):
    """フルアプリケーション実行"""
    print("🚀 Aqua Mirror フルアプリケーション開始...")
    
    try:
        # 必要なモジュールインポート
        import pygame
        from src.visionctr import VisionProcessor
        
        # 初期化
        pygame.init()
        
        display_config = config.get('display', {})
        if display_config.get('fullscreen', False):
            screen = pygame.display.set_mode((
                display_config.get('width', 1920),
                display_config.get('height', 1080)
            ), pygame.FULLSCREEN)
        else:
            screen = pygame.display.set_mode((
                display_config.get('width', 1280),
                display_config.get('height', 720)
            ))
        
        pygame.display.set_caption("Aqua Mirror - Interactive Art")
        
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        
        # VisionProcessor 初期化
        vision_processor = VisionProcessor(config)
        
        print("✅ フルアプリケーション初期化完了")
        print("🎮 操作方法: ESCキー終了, F1キーデバッグ切替")
        
        # メインループ
        running = True
        show_debug = debug_mode
        frame_count = 0
        
        while running:
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_F1:
                        show_debug = not show_debug
                        print(f"🔧 デバッグモード: {'ON' if show_debug else 'OFF'}")
            
            # フレーム処理
            try:
                detection_result = vision_processor.process_frame()
                
                # 画面描画
                screen.fill((0, 30, 80))  # 深い青色
                
                # 簡単な視覚エフェクト
                current_time = time.time()
                for i in range(10):
                    x = (current_time * 50 + i * 100) % screen.get_width()
                    y = 200 + i * 30
                    radius = int(20 + 10 * abs(math.sin(current_time + i)))
                    color = (100, 150 + i * 10, 200)
                    pygame.draw.circle(screen, color, (int(x), int(y)), radius, 2)
                
                # デバッグ情報表示
                if show_debug:
                    fps = clock.get_fps()
                    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
                    screen.blit(fps_text, (10, 10))
                    
                    if detection_result:
                        face_status = "顔: 検出" if detection_result.get('face_detected') else "顔: 未検出"
                        hand_status = f"手: {len(detection_result.get('hand_positions', []))}個"
                        
                        face_text = font.render(face_status, True, (255, 255, 255))
                        hand_text = font.render(hand_status, True, (255, 255, 255))
                        
                        screen.blit(face_text, (10, 50))
                        screen.blit(hand_text, (10, 90))
                
                pygame.display.flip()
                frame_count += 1
                
                if frame_count % 300 == 0:  # 10秒ごと
                    print(f"📊 {frame_count}フレーム処理完了")
                
            except Exception as e:
                print(f"⚠️ フレーム処理エラー: {e}")
                if debug_mode:
                    import traceback
                    traceback.print_exc()
            
            clock.tick(30)
        
        # クリーンアップ
        vision_processor.cleanup()
        pygame.quit()
        
        print(f"✅ アプリケーション正常終了 - {frame_count}フレーム処理")
        return True
        
    except Exception as e:
        print(f"❌ アプリケーション実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン関数"""
    # 警告メッセージ抑制
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("🌊 Aqua Mirror - Interactive Art Project")
    print("=" * 60)
    
    # 引数解析
    args = parse_arguments()
    
    # 設定読み込み
    config = load_config(args.config)
    
    # カメラなしモード
    if args.no_camera:
        config['camera']['device_id'] = -1  # カメラ無効
        print("📷 カメラなしモードで実行")
    
    # モード別実行
    success = False
    
    if args.test:
        # テストモード
        print("\n🧪 テストモード実行...")
        results = test_components(config)
        success = all(results.values())
        
        print(f"\n📊 テスト結果:")
        for component, result in results.items():
            status = "✅" if result else "❌"
            print(f"   {component}: {status}")
        
    elif args.demo:
        # デモモード
        print("\n🎮 デモモード実行...")
        success = run_demo_mode(config)
        
    else:
        # フルアプリケーション
        print("\n🚀 フルアプリケーション実行...")
        success = run_full_application(config, args.debug)
    
    # 結果表示
    if success:
        print("\n🎉 実行完了！")
    else:
        print("\n❌ 実行中にエラーが発生しました")
        print("💡 --test オプションでコンポーネント確認を行ってください")
    
    return success

if __name__ == "__main__":
    # 必要なモジュール追加インポート
    import math
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ ユーザーによって中断されました")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
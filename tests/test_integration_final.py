# ファイル: tests/test_integration_final.py (新規作成)
# 時間: 1-2時間 | 優先度: 🟡 高

import unittest
import time
import threading
import sys
import os

# パス追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.app import AquaMirrorApp
from src.core.config_loader import ConfigLoader
from src.core.error_manager import ErrorManager, ErrorSeverity

class TestFinalIntegration(unittest.TestCase):
    """最終統合テスト"""
    
    def setUp(self):
        """テスト準備"""
        self.config_loader = ConfigLoader('config.json')
        self.config = self.config_loader.load()
        
        # テスト用設定調整
        self.config['debug_mode'] = True
        self.config['display']['fullscreen'] = False
        self.config['display']['width'] = 640
        self.config['display']['height'] = 480
    
    def test_app_initialization_and_cleanup(self):
        """アプリケーション初期化・終了テスト"""
        app = AquaMirrorApp(self.config)
        
        # 初期化確認
        self.assertIsNotNone(app.vision_processor)
        self.assertIsNotNone(app.scene_manager)
        self.assertIsNotNone(app.error_manager)
        self.assertIsNotNone(app.performance_monitor)
        self.assertIsNotNone(app.quality_manager)
        
        # クリーンアップ
        app._cleanup()
    
    def test_error_recovery_system(self):
        """エラー復旧システムテスト"""
        error_manager = ErrorManager(self.config)
        
        # 各種エラーのテスト
        test_errors = [
            (RuntimeError("テストエラー"), ErrorSeverity.ERROR),
            (MemoryError("メモリ不足"), ErrorSeverity.CRITICAL),
            (Exception("カメラエラー"), ErrorSeverity.WARNING)
        ]
        
        for error, severity in test_errors:
            result = error_manager.handle_error(error, severity)
            self.assertIsInstance(result, bool)
        
        # エラー統計確認
        stats = error_manager.get_error_statistics()
        self.assertGreater(stats['total_errors'], 0)
    
    def test_performance_monitoring(self):
        """パフォーマンス監視テスト"""
        from src.core.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        # フレーム時間記録
        for i in range(30):
            monitor.record_frame_time(0.033)  # 30FPS相当
            time.sleep(0.01)
        
        # メトリクス取得
        metrics = monitor.get_current_metrics()
        self.assertIsNotNone(metrics)
        if metrics:
            self.assertGreater(metrics.fps, 0)
        
        monitor.cleanup()
    
    def test_gpu_processor_fallback(self):
        """GPU処理フォールバックテスト"""
        from src.core.gpu_processor import GPUProcessor
        
        gpu_processor = GPUProcessor()
        
        # テスト用画像データ
        import numpy as np
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # リサイズテスト（GPU/CPU自動切替）
        resized = gpu_processor.resize_frame(test_frame, (320, 240))
        
        self.assertEqual(resized.shape[:2], (240, 320))
        self.assertEqual(resized.dtype, np.uint8)
    
    def test_demo_mode_functionality(self):
        """デモモード機能テスト"""
        error_manager = ErrorManager(self.config)
        
        # デモモード移行
        error_manager._enter_demo_mode(None)
        
        self.assertTrue(error_manager.is_demo_mode())
        
        # デモデータ取得
        demo_data = error_manager.get_demo_detection_result()
        self.assertIn('face_detected', demo_data)
        self.assertIn('hands_detected', demo_data)
    
    def test_short_run_stability(self):
        """短時間動作安定性テスト"""
        self.config['demo_mode'] = True  # カメラなしテスト
        
        app = AquaMirrorApp(self.config)
        
        # 別スレッドで5秒間実行
        def run_app():
            start_time = time.time()
            while time.time() - start_time < 5.0 and app.running:
                try:
                    app._safe_handle_events()
                    app._safe_update()
                    # 描画はスキップ（ヘッドレス環境）
                    time.sleep(0.033)  # 30FPS相当
                except Exception as e:
                    print(f"実行中エラー: {e}")
                    break
            
            app.running = False
        
        test_thread = threading.Thread(target=run_app)
        test_thread.start()
        test_thread.join(timeout=10.0)  # 最大10秒待機
        
        # クリーンアップ
        app._cleanup()
        
        # エラー統計確認
        self.assertIsNotNone(app.error_manager)
        if app.error_manager:
            error_stats = app.error_manager.get_error_statistics()
            print(f"テスト結果 - 総エラー数: {error_stats['total_errors']}")

if __name__ == '__main__':
    # ログディレクトリ作成
    os.makedirs('logs', exist_ok=True)
    
    # テスト実行
    unittest.main()
# src/core/error_manager.py - 統合版
# Aqua Mirror - 統合エラーハンドリングシステム

import logging
import traceback
import time
import os
import gc
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
from collections import defaultdict, deque
import unittest

class ErrorSeverity(Enum):
    """エラー重要度レベル"""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class RecoveryResult(Enum):
    """復旧結果ステータス"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    DEMO_MODE = "demo_mode"

class ErrorManager:
    """統合エラーハンドリングシステム
    
    機能:
    - 統一エラーハンドリング・ログ管理
    - 自動復旧戦略実行
    - デモモード自動切替
    - エラー統計・監視
    - クールダウン機能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, log_file: str = "logs/error.log"):
        self.config = config or {}
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies: Dict[str, Callable[[Exception, Any], bool]] = {}
        self.last_error_time = {}
        
        # 設定パラメータ
        self.cooldown_period = self.config.get('error_cooldown', 5.0)  # 5秒
        self.max_retry_count = self.config.get('max_retry_count', 3)
        self.auto_demo_mode = self.config.get('auto_demo_mode', True)
        
        # デモモード管理
        self.demo_mode = False
        self.demo_data = self._create_demo_data()
        
        # ログシステム設定
        self._setup_logging(log_file)
        
        # 復旧戦略登録
        self._register_recovery_strategies()
        
        self.logger.info("✅ エラーマネージャー初期化完了")
    
    def _setup_logging(self, log_file: str):
        """ログシステム設定"""
        # ログディレクトリ作成
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # ログフォーマット設定
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # ログハンドラー設定
        handlers = [
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=handlers,
            force=True  # 既存設定を上書き
        )
        
        self.logger = logging.getLogger('AquaMirror.ErrorManager')
        self.logger.info("エラーマネージャー: ログシステム初期化完了")
    
    def _register_recovery_strategies(self):
        """復旧戦略登録"""
        self.recovery_strategies = {
            'CameraError': self._recover_camera_error,
            'RuntimeError': self._recover_runtime_error,
            'cv2.error': self._recover_opencv_error,
            'GPUError': self._recover_gpu_error,
            'MemoryError': self._recover_memory_error,
            'PerformanceError': self._recover_performance_error,
            'FileNotFoundError': self._recover_file_error,
            'ImportError': self._recover_import_error,
            'MediaPipeError': self._recover_mediapipe_error,
            'OSError': self._recover_os_error
        }
        
        self.logger.info(f"復旧戦略登録完了: {len(self.recovery_strategies)}種類")
    
    def handle_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                    context: Any = None, auto_recover: bool = True) -> RecoveryResult:
        """統一エラーハンドリング
        
        Args:
            error: 発生したエラー
            severity: エラー重要度
            context: エラー発生コンテキスト
            auto_recover: 自動復旧試行フラグ
            
        Returns:
            復旧結果ステータス
        """
        error_type = type(error).__name__
        current_time = time.time()
        
        # エラー記録作成
        error_record = {
            'timestamp': current_time,
            'error_type': error_type,
            'message': str(error),
            'severity': severity.name,
            'context': self._extract_context_info(context),
            'traceback': traceback.format_exc(),
            'retry_count': self.error_counts[error_type]
        }
        
        # エラー履歴更新
        self.error_history.append(error_record)
        self.error_counts[error_type] += 1
        
        # ログ出力
        self._log_error(error_record)
        
        # 重要エラーの即座対応
        if severity == ErrorSeverity.CRITICAL:
            return self._handle_critical_error(error, context)
        
        # クールダウン確認
        if not self._check_cooldown(error_type, current_time):
            self.logger.warning(f"⏱️ エラー復旧クールダウン中: {error_type}")
            return RecoveryResult.FAILED
        
        # 自動復旧試行
        if auto_recover:
            return self._attempt_recovery(error_type, error, context)
        
        return RecoveryResult.FAILED
    
    def _extract_context_info(self, context: Any) -> str:
        """コンテキスト情報抽出"""
        if context is None:
            return "No context"
        
        context_info = str(context.__class__.__name__) if hasattr(context, '__class__') else str(context)
        
        # 追加の有用な情報
        additional_info = []
        if hasattr(context, 'current_state'):
            additional_info.append(f"state={context.current_state}")
        if hasattr(context, 'frame_count'):
            additional_info.append(f"frame={context.frame_count}")
        
        if additional_info:
            context_info += f" ({', '.join(additional_info)})"
        
        return context_info
    
    def _log_error(self, error_record: Dict[str, Any]):
        """エラーログ出力"""
        severity_name = error_record['severity']
        message = f"[{error_record['error_type']}] {error_record['message']}"
        
        if error_record['context'] != "No context":
            message += f" | Context: {error_record['context']}"
        
        if error_record['retry_count'] > 1:
            message += f" | Retry: {error_record['retry_count']}"
        
        # 重要度別ログ出力
        if severity_name == 'CRITICAL':
            self.logger.critical(f"🚨 {message}")
        elif severity_name == 'ERROR':
            self.logger.error(f"❌ {message}")
        elif severity_name == 'WARNING':
            self.logger.warning(f"⚠️ {message}")
        else:
            self.logger.info(f"ℹ️ {message}")
    
    def _check_cooldown(self, error_type: str, current_time: float) -> bool:
        """クールダウン確認"""
        if error_type in self.last_error_time:
            time_since_last = current_time - self.last_error_time[error_type]
            if time_since_last < self.cooldown_period:
                return False
        
        self.last_error_time[error_type] = current_time
        return True
    
    def _handle_critical_error(self, error: Exception, context: Any) -> RecoveryResult:
        """重要エラーの即座処理"""
        self.logger.critical(f"🚨 重要エラー即座対応: {type(error).__name__}")
        
        # システム状態保存
        self._save_system_state(context)
        
        # 即座にデモモードに移行
        if self.auto_demo_mode:
            self._enter_demo_mode(context)
            return RecoveryResult.DEMO_MODE
        
        return RecoveryResult.FAILED
    
    def _attempt_recovery(self, error_type: str, error: Exception, context: Any) -> RecoveryResult:
        """復旧試行"""
        if error_type not in self.recovery_strategies:
            self.logger.warning(f"⚠️ 未対応エラータイプ: {error_type}")
            return RecoveryResult.FAILED
        
        # 再試行回数確認
        if self.error_counts[error_type] > self.max_retry_count:
            self.logger.warning(f"⚠️ 最大再試行回数超過: {error_type}")
            if self.auto_demo_mode:
                self._enter_demo_mode(context)
                return RecoveryResult.DEMO_MODE
            return RecoveryResult.FAILED
        
        try:
            self.logger.info(f"🔧 エラー復旧試行: {error_type}")
            recovery_result = self.recovery_strategies[error_type](error, context)
            
            if recovery_result:
                self.logger.info(f"✅ エラー復旧成功: {error_type}")
                # 成功時はエラーカウントリセット
                self.error_counts[error_type] = 0
                return RecoveryResult.SUCCESS
            else:
                self.logger.warning(f"⚠️ エラー復旧失敗: {error_type}")
                return RecoveryResult.FAILED
                
        except Exception as recovery_error:
            self.logger.error(f"❌ 復旧処理中エラー: {recovery_error}")
            return RecoveryResult.FAILED
    
    # =============================================================================
    # 個別復旧戦略
    # =============================================================================
    
    def _recover_camera_error(self, error: Exception, context: Any) -> bool:
        """カメラエラー復旧"""
        self.logger.info("📹 カメラエラー復旧試行...")
        
        error_count = self.error_counts['CameraError']
        
        if error_count <= 2:
            # 軽度復旧: カメラ再初期化
            if hasattr(context, 'camera_manager'):
                try:
                    self.logger.info("カメラ再初期化中...")
                    context.camera_manager.cleanup()
                    time.sleep(1)
                    return context.camera_manager.initialize()
                except Exception as e:
                    self.logger.error(f"カメラ再初期化失敗: {e}")
        
        elif error_count <= 4:
            # 中度復旧: 他のデバイス試行
            if hasattr(context, 'camera_manager'):
                for device_id in range(3):
                    try:
                        self.logger.info(f"カメラデバイス{device_id}試行中...")
                        context.camera_manager.device_id = device_id
                        if context.camera_manager.initialize():
                            self.logger.info(f"カメラデバイス{device_id}で復旧成功")
                            return True
                    except:
                        continue
        
        # 重度復旧: デモモード移行
        self.logger.warning("カメラ復旧失敗、デモモードに移行")
        self._enter_demo_mode(context)
        return True
    
    def _recover_gpu_error(self, error: Exception, context: Any) -> bool:
        """GPU エラー復旧"""
        self.logger.info("🖥️ GPU エラー復旧: CPU処理に切替")
        
        # GPU処理を無効化
        if hasattr(context, 'vision_processor'):
            if hasattr(context.vision_processor, 'use_gpu'):
                context.vision_processor.use_gpu = False
            if hasattr(context.vision_processor, 'gpu_processor'):
                context.vision_processor.gpu_processor.gpu_available = False
        
        # 品質設定を下げる
        if hasattr(context, 'quality_manager'):
            context.quality_manager.reduce_quality()
        
        self.logger.info("CPU処理モードに切替完了")
        return True
    
    def _recover_memory_error(self, error: Exception, context: Any) -> bool:
        """メモリエラー復旧"""
        self.logger.info("🧠 メモリエラー復旧試行...")
        
        # 強制ガベージコレクション
        collected = gc.collect()
        self.logger.info(f"ガベージコレクション: {collected}オブジェクト解放")
        
        # エフェクト削減
        if hasattr(context, 'visual_engine'):
            if hasattr(context.visual_engine, 'reduce_effects'):
                context.visual_engine.reduce_effects()
        
        # 品質レベル強制削減
        if hasattr(context, 'quality_manager'):
            context.quality_manager.force_low_quality()
        
        self.logger.info("メモリ使用量削減完了")
        return True
    
    def _recover_performance_error(self, error: Exception, context: Any) -> bool:
        """パフォーマンスエラー復旧"""
        self.logger.info("⚡ パフォーマンス復旧: 品質調整")
        
        if hasattr(context, 'performance_monitor'):
            context.performance_monitor.trigger_optimization()
        
        if hasattr(context, 'frame_pipeline'):
            context.frame_pipeline.increase_frame_skip()
        
        return True
    
    def _recover_opencv_error(self, error: Exception, context: Any) -> bool:
        """OpenCV エラー復旧"""
        self.logger.info("📷 OpenCV エラー復旧...")
        
        # OpenCV リソース解放
        import cv2
        cv2.destroyAllWindows()
        
        # カメラ再初期化
        if hasattr(context, 'camera_manager'):
            try:
                context.camera_manager.cleanup()
                time.sleep(0.5)
                return context.camera_manager.initialize()
            except:
                pass
        
        return False
    
    def _recover_mediapipe_error(self, error: Exception, context: Any) -> bool:
        """MediaPipe エラー復旧"""
        self.logger.info("🤖 MediaPipe エラー復旧...")
        
        if hasattr(context, 'vision_processor'):
            try:
                # MediaPipe インスタンス再作成
                context.vision_processor._init_mediapipe()
                return True
            except:
                pass
        
        return False
    
    def _recover_runtime_error(self, error: Exception, context: Any) -> bool:
        """ランタイムエラー復旧"""
        self.logger.info("🔧 ランタイムエラー復旧...")
        
        # 一般的な復旧処理
        gc.collect()
        
        if hasattr(context, 'reset_state'):
            context.reset_state()
            return True
        
        return False
    
    def _recover_file_error(self, error: Exception, context: Any) -> bool:
        """ファイルエラー復旧"""
        self.logger.info("📁 ファイルエラー復旧: デフォルト設定使用")
        
        if hasattr(context, 'load_default_config'):
            context.load_default_config()
            return True
        
        return False
    
    def _recover_import_error(self, error: Exception, context: Any) -> bool:
        """インポートエラー復旧"""
        self.logger.warning("📦 インポートエラー: 機能制限モードに移行")
        
        if hasattr(context, 'enable_limited_mode'):
            context.enable_limited_mode()
            return True
        
        return False
    
    def _recover_os_error(self, error: Exception, context: Any) -> bool:
        """OS エラー復旧"""
        self.logger.info("💻 OS エラー復旧...")
        
        # リソース解放
        if hasattr(context, 'cleanup_resources'):
            context.cleanup_resources()
        
        time.sleep(1)  # システム安定化待機
        return False
    
    # =============================================================================
    # デモモード機能
    # =============================================================================
    
    def _enter_demo_mode(self, context: Any):
        """デモモード移行"""
        self.demo_mode = True
        self.logger.info("🎭 デモモードに移行しました")
        
        if hasattr(context, 'enable_demo_mode'):
            context.enable_demo_mode()
        elif hasattr(context, 'demo_mode'):
            context.demo_mode = True
    
    def _create_demo_data(self) -> Dict[str, Any]:
        """デモ用データ生成"""
        return {
            'face_detected': True,
            'face_center': (0.5, 0.4, 0.5),
            'face_distance': 0.7,
            'hands_detected': True,
            'hand_positions': [(0.3, 0.6), (0.7, 0.6)],
            'frame_shape': (480, 640, 3),
            'emotion': 'happy',
            'confidence': 0.8
        }
    
    def get_demo_detection_result(self) -> Dict[str, Any]:
        """デモ用検出結果取得"""
        return self.demo_data.copy()
    
    def is_demo_mode(self) -> bool:
        """デモモード確認"""
        return self.demo_mode
    
    # =============================================================================
    # ユーティリティ・管理機能
    # =============================================================================
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable[[Exception, Any], bool]):
        """カスタム復旧戦略登録"""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"復旧戦略登録: {error_type}")
    
    def _save_system_state(self, context: Any):
        """システム状態保存"""
        try:
            state_data = {
                'timestamp': time.time(),
                'error_counts': dict(self.error_counts),
                'context_type': str(type(context).__name__) if context else None
            }
            
            import json
            with open('logs/crash_state.json', 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info("システム状態保存完了")
        except Exception as e:
            self.logger.error(f"システム状態保存失敗: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """詳細エラー統計取得"""
        recent_errors = list(self.error_history)[-10:] if self.error_history else []
        
        return {
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': recent_errors,
            'error_types': len(self.error_counts),
            'recovery_strategies': len(self.recovery_strategies),
            'demo_mode': self.demo_mode,
            'cooldown_period': self.cooldown_period,
            'max_retry_count': self.max_retry_count
        }
    
    def clear_error_history(self):
        """エラー履歴クリア"""
        self.error_history.clear()
        self.error_counts.clear()
        self.last_error_time.clear()
        self.demo_mode = False
        self.logger.info("エラー履歴をクリアしました")
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """最近のエラー取得"""
        return list(self.error_history)[-count:] if self.error_history else []

# =============================================================================
# テスト・デバッグ機能
# =============================================================================

class TestErrorManager(unittest.TestCase):
    def setUp(self):
        """テスト準備"""
        self.config = {
            'error_cooldown': 0.1,
            'max_retry_count': 2,
            'auto_demo_mode': True,
        }

    def test_error_manager_basic(self):
        """ErrorManager基本テスト（修正版）"""
        print("\n🧪 ErrorManager基本テスト開始...")
        
        try:
            error_manager = ErrorManager(self.config)
            self.assertIsNotNone(error_manager)
            print("✅ ErrorManager インスタンス作成成功")
            
            # テストエラーの処理（返り値型を確認）
            test_error = RuntimeError("テストエラー")
            result = error_manager.handle_error(test_error, ErrorSeverity.ERROR)
            
            # 返り値の型を確認して適切にテスト
            self.assertIsInstance(result, RecoveryResult)
            print(f"✅ エラーハンドリング動作確認: {result}")
            
            # エラー統計確認（メソッドが存在する場合）
            stats = error_manager.get_error_statistics()
            print(f"✅ エラー統計取得: {stats}")
            self.assertIsInstance(stats, dict)
            self.assertIn('total_errors', stats)
            self.assertGreater(stats['total_errors'], 0)
            
        except Exception as e:
            print(f"❌ ErrorManagerテストエラー: {e}")
            self.fail(f"ErrorManagerテスト中に例外が発生: {e}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
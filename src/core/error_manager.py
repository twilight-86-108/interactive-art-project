import logging
import traceback
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional
from collections import defaultdict, deque

class ErrorSeverity(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ErrorManager:
    """統合エラーハンドリングシステム（Day 2版）"""
    
    def __init__(self, log_file: str = "logs/app.log"):
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.recovery_strategies = {}
        self.last_error_time = {}
        self.cooldown_period = 5.0  # 5秒のクールダウン
        
        # ログシステムセットアップ
        self._setup_logging(log_file)
        
        # 基本復旧戦略登録
        self._register_basic_strategies()
        
        print("✅ エラーマネージャー初期化完了")
    
    def _setup_logging(self, log_file: str):
        """ログシステム設定"""
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('AquaMirror')
        self.logger.info("エラーマネージャー: ログシステム初期化完了")
    
    def _register_basic_strategies(self):
        """基本復旧戦略登録"""
        self.recovery_strategies = {
            'CameraError': self._recover_camera_error,
            'GPUError': self._recover_gpu_error,
            'MemoryError': self._recover_memory_error,
            'PerformanceError': self._recover_performance_error,
            'FileNotFoundError': self._recover_file_error,
            'ImportError': self._recover_import_error
        }
    
    def handle_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.ERROR, 
                    context: Any = None, auto_recover: bool = True) -> bool:
        """統合エラーハンドリング"""
        error_type = type(error).__name__
        current_time = time.time()
        
        # エラー記録
        error_record = {
            'timestamp': current_time,
            'error_type': error_type,
            'message': str(error),
            'severity': severity,
            'context': str(context) if context else None,
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_record)
        self.error_counts[error_type] += 1
        
        # ログ出力
        self._log_error(error_record)
        
        # クールダウン確認
        if error_type in self.last_error_time:
            time_since_last = current_time - self.last_error_time[error_type]
            if time_since_last < self.cooldown_period:
                self.logger.warning(f"エラー復旧クールダウン中: {error_type}")
                return False
        
        self.last_error_time[error_type] = current_time
        
        # 自動復旧試行
        if auto_recover and error_type in self.recovery_strategies:
            try:
                self.logger.info(f"エラー復旧試行: {error_type}")
                recovery_result = self.recovery_strategies[error_type](error, context)
                
                if recovery_result:
                    self.logger.info(f"エラー復旧成功: {error_type}")
                else:
                    self.logger.warning(f"エラー復旧失敗: {error_type}")
                
                return recovery_result
                
            except Exception as recovery_error:
                self.logger.error(f"復旧処理中エラー: {recovery_error}")
                return False
        
        return False
    
    def _log_error(self, error_record: Dict[str, Any]):
        """エラーログ出力"""
        severity = error_record['severity']
        message = f"{error_record['error_type']}: {error_record['message']}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
        elif severity == ErrorSeverity.ERROR:
            self.logger.error(message)
        elif severity == ErrorSeverity.WARNING:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _recover_camera_error(self, error: Exception, context: Any) -> bool:
        """カメラエラー復旧"""
        self.logger.info("カメラエラー復旧試行...")
        
        error_count = self.error_counts['CameraError']
        
        if error_count <= 3:
            # 軽度復旧: カメラ再初期化
            if hasattr(context, 'camera_manager'):
                try:
                    context.camera_manager.cleanup()
                    time.sleep(1)
                    return context.camera_manager.initialize()
                except:
                    pass
        
        elif error_count <= 5:
            # 中度復旧: 他のカメラデバイス試行
            if hasattr(context, 'camera_manager'):
                for device_id in range(3):
                    try:
                        context.camera_manager.device_id = device_id
                        if context.camera_manager.initialize():
                            self.logger.info(f"カメラデバイス{device_id}で復旧成功")
                            return True
                    except:
                        continue
        
        # 重度復旧: デモモード有効化
        self.logger.warning("カメラ復旧失敗、デモモードに移行")
        if hasattr(context, 'enable_demo_mode'):
            context.enable_demo_mode()
            return True
        
        return False
    
    def _recover_gpu_error(self, error: Exception, context: Any) -> bool:
        """GPU エラー復旧"""
        self.logger.info("GPU エラー復旧: CPU処理に切替")
        
        if hasattr(context, 'gpu_processor'):
            context.gpu_processor.gpu_available = False
            return True
        
        return False
    
    def _recover_memory_error(self, error: Exception, context: Any) -> bool:
        """メモリエラー復旧"""
        self.logger.info("メモリエラー復旧試行...")
        
        import gc
        
        # 強制ガベージコレクション
        collected = gc.collect()
        self.logger.info(f"ガベージコレクション: {collected}オブジェクト解放")
        
        # コンテキストの品質設定下げ
        if hasattr(context, 'reduce_quality'):
            context.reduce_quality()
            self.logger.info("処理品質を下げました")
        
        return True
    
    def _recover_performance_error(self, error: Exception, context: Any) -> bool:
        """パフォーマンスエラー復旧"""
        self.logger.info("パフォーマンス復旧: 品質調整")
        
        if hasattr(context, 'adjust_performance'):
            context.adjust_performance()
            return True
        
        return False
    
    def _recover_file_error(self, error: Exception, context: Any) -> bool:
        """ファイルエラー復旧"""
        self.logger.info("ファイルエラー復旧: デフォルトファイル使用")
        # デフォルトファイル作成など
        return False
    
    def _recover_import_error(self, error: Exception, context: Any) -> bool:
        """インポートエラー復旧"""
        self.logger.warning("インポートエラー: 代替機能に切替")
        # 代替実装への切替など
        return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """エラー統計取得"""
        return {
            'error_counts': dict(self.error_counts),
            'total_errors': sum(self.error_counts.values()),
            'recent_errors': list(self.error_history)[-10:] if self.error_history else [],
            'error_types': len(self.error_counts),
            'recovery_strategies': len(self.recovery_strategies)
        }
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """カスタム復旧戦略登録"""
        self.recovery_strategies[error_type] = strategy
        self.logger.info(f"復旧戦略登録: {error_type}")
    
    def clear_error_history(self):
        """エラー履歴クリア"""
        self.error_history.clear()
        self.error_counts.clear()
        self.last_error_time.clear()
        self.logger.info("エラー履歴をクリアしました")

# テスト実行用
if __name__ == "__main__":
    print("🔍 エラーマネージャーテスト開始...")
    
    manager = ErrorManager()
    
    # テストエラー発生
    try:
        raise ValueError("テストエラー")
    except Exception as e:
        result = manager.handle_error(e, ErrorSeverity.ERROR)
        print(f"エラーハンドリング結果: {result}")
    
    # 統計確認
    stats = manager.get_error_stats()
    print(f"エラー統計: {stats}")
    
    print("✅ エラーマネージャーテスト完了")

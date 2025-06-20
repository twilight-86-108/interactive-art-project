# src/core/performance_monitor.py を修正
import time
import threading
import platform
import os
from collections import deque
from typing import Dict, Any, Optional, Union

# psutil のオプション依存
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("✅ psutil利用可能")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil未インストール（基本機能で継続）")

class PerformanceMonitor:
    """パフォーマンス監視システム（型エラー修正版）"""
    
    def __init__(self, history_size: int = 300):
        self.history_size = history_size
        self.frame_times = deque(maxlen=history_size)
        self.cpu_usage = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.gpu_usage = deque(maxlen=history_size)
        
        # 監視設定
        self.target_fps = 30
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 警告閾値（float型で統一）
        self.thresholds: Dict[str, float] = {
            'fps_warning': 25.0,
            'cpu_warning': 80.0,
            'memory_warning': 85.0,
            'gpu_warning': 90.0
        }
        
        # システム情報
        self.system_info = self._get_system_info()
        self.psutil_available = PSUTIL_AVAILABLE
        
        print("✅ パフォーマンスモニター初期化完了")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得（フォールバック付き）"""
        try:
            if PSUTIL_AVAILABLE:
                return {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total,
                    'python_process': psutil.Process().pid,
                    'platform': platform.system(),
                    'psutil_version': psutil.__version__
                }
            else:
                # psutil なしでの基本情報
                return {
                    'cpu_count': os.cpu_count() or 1,
                    'memory_total': 'Unknown',
                    'python_process': os.getpid(),
                    'platform': platform.system(),
                    'psutil_version': 'Not installed'
                }
        except Exception as e:
            print(f"⚠️  システム情報取得エラー: {e}")
            return {
                'cpu_count': 1,
                'memory_total': 'Unknown',
                'python_process': os.getpid(),
                'platform': platform.system(),
                'error': str(e)
            }
    
    def start_monitoring(self, interval: float = 1.0):
        """バックグラウンド監視開始"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        if PSUTIL_AVAILABLE:
            print("📊 詳細パフォーマンス監視開始（psutil使用）")
        else:
            print("📊 基本パフォーマンス監視開始（psutil非使用）")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  パフォーマンス監視停止")
    
    def _monitoring_loop(self, interval: float):
        """監視ループ（フォールバック対応）"""
        while self.monitoring_active:
            try:
                if PSUTIL_AVAILABLE:
                    # psutil を使用した詳細監視
                    self._monitor_with_psutil()
                else:
                    # psutil なしの基本監視
                    self._monitor_basic()
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  監視ループエラー: {e}")
                time.sleep(interval)
    
    def _monitor_with_psutil(self):
        """psutil を使用した詳細監視"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(float(cpu_percent))
            
            # メモリ使用率
            memory = psutil.virtual_memory()
            self.memory_usage.append(float(memory.percent))
            
            # GPU使用率（ダミー値）
            self.gpu_usage.append(0.0)
            
        except Exception as e:
            print(f"⚠️  psutil監視エラー: {e}")
            # エラー時は基本監視にフォールバック
            self._monitor_basic()
    
    def _monitor_basic(self):
        """基本監視（psutil非依存）"""
        try:
            # CPU使用率（概算）
            start_time = time.time()
            # 簡易CPU負荷測定（精度は低い）
            for _ in range(1000):
                pass
            cpu_time = time.time() - start_time
            cpu_estimate = min(100.0, cpu_time * 10000)  # 概算値
            self.cpu_usage.append(cpu_estimate)
            
            # メモリ使用率（概算値）
            # 実際の使用量は取得できないのでダミー値
            memory_estimate = 50.0  # デフォルト50%として扱う
            self.memory_usage.append(memory_estimate)
            
            # GPU使用率（ダミー値）
            self.gpu_usage.append(0.0)
            
        except Exception as e:
            print(f"⚠️  基本監視エラー: {e}")
    
    def record_frame_time(self, frame_time: float):
        """フレーム時間記録"""
        self.frame_times.append(float(frame_time))
    
    def get_current_fps(self) -> float:
        """現在のFPS取得"""
        if len(self.frame_times) < 10:
            return 0.0
        
        recent_times = list(self.frame_times)[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        current_fps = self.get_current_fps()
        
        stats = {
            'fps': {
                'current': current_fps,
                'target': float(self.target_fps),
                'warning': current_fps < self.thresholds['fps_warning']
            },
            'frame_times': {
                'count': len(self.frame_times),
                'avg': sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0.0,
                'max': max(self.frame_times) if self.frame_times else 0.0,
                'min': min(self.frame_times) if self.frame_times else 0.0
            },
            'monitoring': {
                'psutil_available': PSUTIL_AVAILABLE,
                'monitoring_active': self.monitoring_active
            }
        }
        
        # CPU統計（利用可能な場合のみ詳細情報）
        if self.cpu_usage:
            current_cpu = float(self.cpu_usage[-1])
            stats['cpu'] = {
                'current': current_cpu,
                'avg': sum(self.cpu_usage) / len(self.cpu_usage),
                'warning': current_cpu > self.thresholds['cpu_warning'],
                'note': 'Accurate' if PSUTIL_AVAILABLE else 'Estimated'
            }
        
        # メモリ統計
        if self.memory_usage:
            current_memory = float(self.memory_usage[-1])
            stats['memory'] = {
                'current': current_memory,
                'avg': sum(self.memory_usage) / len(self.memory_usage),
                'warning': current_memory > self.thresholds['memory_warning'],
                'note': 'Accurate' if PSUTIL_AVAILABLE else 'Estimated'
            }
        
        return stats
    
    def check_performance_warnings(self) -> list:
        """パフォーマンス警告確認"""
        warnings = []
        stats = self.get_performance_stats()
        
        if stats.get('fps', {}).get('warning', False):
            warnings.append(f"FPS低下: {stats['fps']['current']:.1f}")
        
        if not PSUTIL_AVAILABLE:
            warnings.append("psutil未インストール: 概算値使用")
        
        if PSUTIL_AVAILABLE:
            if stats.get('cpu', {}).get('warning', False):
                warnings.append(f"CPU使用率高: {stats['cpu']['current']:.1f}%")
            
            if stats.get('memory', {}).get('warning', False):
                warnings.append(f"メモリ使用率高: {stats['memory']['current']:.1f}%")
        
        return warnings
    
    def adjust_performance_targets(self, new_fps_target: Union[int, float]):
        """パフォーマンス目標調整（型安全版）"""
        self.target_fps = int(new_fps_target)
        # float型で計算して閾値設定
        self.thresholds['fps_warning'] = float(new_fps_target * 0.8)
        print(f"📊 FPS目標を{self.target_fps}に調整（警告閾値: {self.thresholds['fps_warning']:.1f}）")
    
    def set_threshold(self, metric: str, value: Union[int, float]):
        """閾値設定（型安全版）"""
        if metric in self.thresholds:
            self.thresholds[metric] = float(value)
            print(f"📊 {metric} 閾値を {value} に設定")
        else:
            print(f"⚠️  未知のメトリクス: {metric}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """デバッグ情報取得"""
        stats = self.get_performance_stats()
        warnings = self.check_performance_warnings()
        
        return {
            'performance_stats': stats,
            'warnings': warnings,
            'system_info': self.system_info,
            'monitoring_active': self.monitoring_active,
            'psutil_available': PSUTIL_AVAILABLE,
            'thresholds': self.thresholds
        }

# テスト実行用
if __name__ == "__main__":
    print("🔍 パフォーマンスモニターテスト開始...")
    
    monitor = PerformanceMonitor()
    
    # 閾値調整テスト
    print("🔧 閾値調整テスト...")
    monitor.adjust_performance_targets(25)  # int
    monitor.adjust_performance_targets(27.5)  # float
    monitor.set_threshold('cpu_warning', 75.0)
    
    monitor.start_monitoring(interval=0.5)
    
    # 5秒間のテスト
    print("📊 5秒間の監視テスト...")
    for i in range(50):
        # フレーム時間をシミュレート
        frame_time = 0.033 + (i % 10) * 0.001  # 30FPS前後
        monitor.record_frame_time(frame_time)
        time.sleep(0.1)
    
    # 統計表示
    stats = monitor.get_performance_stats()
    print(f"FPS: {stats['fps']['current']:.1f} (目標: {stats['fps']['target']:.1f})")
    print(f"CPU: {stats.get('cpu', {}).get('current', 'N/A')} ({stats.get('cpu', {}).get('note', 'N/A')})")
    print(f"Memory: {stats.get('memory', {}).get('current', 'N/A')} ({stats.get('memory', {}).get('note', 'N/A')})")
    
    # 閾値確認
    print(f"閾値: {monitor.thresholds}")
    
    # 警告確認
    warnings = monitor.check_performance_warnings()
    if warnings:
        print(f"⚠️  警告: {', '.join(warnings)}")
    else:
        print("✅ パフォーマンス正常")
    
    monitor.stop_monitoring()
    print("✅ パフォーマンスモニターテスト完了")

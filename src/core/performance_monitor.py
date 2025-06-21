# src/core/performance_monitor.py - 統合版
# 時間: 2-3時間 | 優先度: 🟡 高

import time
import threading
import platform
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List

# psutil のオプション依存
try:
    import psutil
    PSUTIL_AVAILABLE = True
    print("✅ psutil利用可能 - 詳細監視機能有効")
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil未インストール（基本機能で継続）")

@dataclass
class PerformanceMetrics:
    """パフォーマンス指標データクラス"""
    fps: float
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: float
    processing_time: float
    frame_time: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, float]:
        """辞書形式に変換"""
        return {
            'fps': self.fps,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_memory_usage': self.gpu_memory_usage,
            'processing_time': self.processing_time,
            'frame_time': self.frame_time,
            'timestamp': self.timestamp
        }

class PerformanceMonitor:
    """
    統合パフォーマンス監視システム
    
    特徴:
    - psutil オプション依存（なくても基本機能で動作）
    - GPU メモリ監視（nvidia-smi経由）
    - 型安全性
    - 堅牢なエラーハンドリング
    - 詳細な統計情報
    """
    
    def __init__(self, history_size: int = 300):
        self.history_size = history_size
        
        # メトリクス履歴
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=history_size)
        self.frame_times: deque[float] = deque(maxlen=60)  # 2秒分（30FPS）
        self.cpu_usage: deque[float] = deque(maxlen=history_size)
        self.memory_usage: deque[float] = deque(maxlen=history_size)
        self.gpu_usage: deque[float] = deque(maxlen=history_size)
        
        # 処理時間記録
        self.processing_time: float = 0.0
        self.last_gpu_check: float = 0.0
        self.gpu_check_interval: float = 1.0  # GPU確認間隔（秒）
        
        # 監視設定
        self.target_fps = 30
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # 警告閾値（float型で統一）
        self.thresholds: Dict[str, float] = {
            'fps_warning': 25.0,
            'fps_critical': 20.0,
            'cpu_warning': 80.0,
            'cpu_critical': 90.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'gpu_warning': 90.0,
            'gpu_critical': 98.0,
            'processing_time_warning': 0.03,  # 30ms
            'processing_time_critical': 0.05   # 50ms
        }
        
        # システム情報・機能確認
        self.system_info = self._get_system_info()
        self.psutil_available = PSUTIL_AVAILABLE
        self.gpu_available = self._check_gpu_availability()
        
        print("✅ 統合パフォーマンスモニター初期化完了")
        self._print_capabilities()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """システム情報取得（フォールバック付き）"""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return {
                    'cpu_count': psutil.cpu_count(),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'memory_total': memory.total,
                    'memory_total_gb': round(memory.total / (1024**3), 2),
                    'python_process': psutil.Process().pid,
                    'platform': platform.system(),
                    'platform_release': platform.release(),
                    'psutil_version': psutil.__version__
                }
            else:
                return {
                    'cpu_count': os.cpu_count() or 1,
                    'cpu_count_logical': os.cpu_count() or 1,
                    'memory_total': 'Unknown',
                    'memory_total_gb': 'Unknown',
                    'python_process': os.getpid(),
                    'platform': platform.system(),
                    'platform_release': platform.release(),
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
    
    def _check_gpu_availability(self) -> bool:
        """GPU利用可能性確認"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
            ], capture_output=True, text=True, timeout=3)
            
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                print(f"🖥️  GPU検出: {gpu_name}")
                return True
            else:
                return False
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _print_capabilities(self):
        """利用可能機能の表示"""
        capabilities = []
        
        if self.psutil_available:
            capabilities.append("詳細CPU/メモリ監視")
        else:
            capabilities.append("基本監視")
            
        if self.gpu_available:
            capabilities.append("GPU監視")
        else:
            capabilities.append("GPU監視不可")
        
        print(f"📊 機能: {' | '.join(capabilities)}")
    
    def start_monitoring(self, interval: float = 1.0):
        """バックグラウンド監視開始"""
        if self.monitoring_active:
            print("⚠️  既に監視中です")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        print(f"📊 パフォーマンス監視開始（間隔: {interval:.1f}s）")
    
    def stop_monitoring(self):
        """監視停止"""
        if not self.monitoring_active:
            return
            
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        print("⏹️  パフォーマンス監視停止")
    
    def _monitoring_loop(self, interval: float):
        """監視ループ（統合版）"""
        while self.monitoring_active:
            try:
                metrics = self._collect_comprehensive_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self._check_performance_warnings(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"⚠️  監視ループエラー: {e}")
                time.sleep(interval)
    
    def _collect_comprehensive_metrics(self) -> Optional[PerformanceMetrics]:
        """包括的メトリクス収集"""
        try:
            current_time = time.time()
            
            # FPS計算
            fps = self._calculate_current_fps()
            
            # CPU・メモリ使用率
            cpu_usage, memory_usage = self._get_system_metrics()
            
            # GPU メモリ使用率
            gpu_memory = self._get_gpu_memory_usage()
            
            # フレーム時間
            frame_time = self.frame_times[-1] if self.frame_times else 0.0
            
            return PerformanceMetrics(
                fps=fps,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_memory_usage=gpu_memory,
                processing_time=self.processing_time,
                frame_time=frame_time,
                timestamp=current_time
            )
            
        except Exception as e:
            print(f"⚠️  メトリクス収集エラー: {e}")
            return None
    
    def _calculate_current_fps(self) -> float:
        """現在のFPS計算"""
        if len(self.frame_times) < 10:
            return 0.0
        
        recent_times = list(self.frame_times)[-10:]
        avg_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def _get_system_metrics(self) -> tuple[float, float]:
        """システムメトリクス取得"""
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # 履歴に追加
                self.cpu_usage.append(float(cpu_percent))
                self.memory_usage.append(float(memory.percent))
                
                return float(cpu_percent), float(memory.percent)
                
            except Exception as e:
                print(f"⚠️  psutil取得エラー: {e}")
                return self._get_basic_system_metrics()
        else:
            return self._get_basic_system_metrics()
    
    def _get_basic_system_metrics(self) -> tuple[float, float]:
        """基本システムメトリクス（psutil非依存）"""
        try:
            # CPU使用率概算
            start_time = time.time()
            for _ in range(1000):
                pass
            cpu_time = time.time() - start_time
            cpu_estimate = min(100.0, cpu_time * 10000)
            
            # メモリ使用率（概算値）
            memory_estimate = 50.0
            
            self.cpu_usage.append(cpu_estimate)
            self.memory_usage.append(memory_estimate)
            
            return cpu_estimate, memory_estimate
            
        except Exception as e:
            print(f"⚠️  基本監視エラー: {e}")
            return 0.0, 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """GPU メモリ使用量取得"""
        current_time = time.time()
        
        # GPU確認間隔制御
        if current_time - self.last_gpu_check < self.gpu_check_interval:
            return self.gpu_usage[-1] if self.gpu_usage else 0.0
        
        self.last_gpu_check = current_time
        
        if not self.gpu_available:
            return 0.0
        
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                used, total = map(int, result.stdout.strip().split(', '))
                gpu_percent = (used / total) * 100 if total > 0 else 0.0
                self.gpu_usage.append(gpu_percent)
                return gpu_percent
            
        except Exception as e:
            if len(str(e)) < 100:  # 短いエラーのみ表示
                print(f"⚠️  GPU監視エラー: {e}")
        
        return 0.0
    
    def _check_performance_warnings(self, metrics: PerformanceMetrics):
        """パフォーマンス警告チェック"""
        warnings = []
        critical = []
        
        # FPS警告
        if metrics.fps < self.thresholds['fps_critical']:
            critical.append(f"FPS危険: {metrics.fps:.1f}")
        elif metrics.fps < self.thresholds['fps_warning']:
            warnings.append(f"FPS低下: {metrics.fps:.1f}")
        
        # CPU警告（psutil利用可能時のみ詳細）
        if PSUTIL_AVAILABLE:
            if metrics.cpu_usage > self.thresholds['cpu_critical']:
                critical.append(f"CPU危険: {metrics.cpu_usage:.1f}%")
            elif metrics.cpu_usage > self.thresholds['cpu_warning']:
                warnings.append(f"CPU高負荷: {metrics.cpu_usage:.1f}%")
        
        # メモリ警告
        if PSUTIL_AVAILABLE:
            if metrics.memory_usage > self.thresholds['memory_critical']:
                critical.append(f"メモリ危険: {metrics.memory_usage:.1f}%")
            elif metrics.memory_usage > self.thresholds['memory_warning']:
                warnings.append(f"メモリ高使用: {metrics.memory_usage:.1f}%")
        
        # GPU警告
        if self.gpu_available and metrics.gpu_memory_usage > 0:
            if metrics.gpu_memory_usage > self.thresholds['gpu_critical']:
                critical.append(f"GPU危険: {metrics.gpu_memory_usage:.1f}%")
            elif metrics.gpu_memory_usage > self.thresholds['gpu_warning']:
                warnings.append(f"GPU高使用: {metrics.gpu_memory_usage:.1f}%")
        
        # 処理時間警告
        if metrics.processing_time > self.thresholds['processing_time_critical']:
            critical.append(f"処理時間危険: {metrics.processing_time*1000:.1f}ms")
        elif metrics.processing_time > self.thresholds['processing_time_warning']:
            warnings.append(f"処理時間超過: {metrics.processing_time*1000:.1f}ms")
        
        # 警告出力
        if critical:
            print(f"🚨 危険: {' | '.join(critical)}")
        elif warnings:
            print(f"⚠️  警告: {' | '.join(warnings)}")
    
    def record_frame_time(self, frame_time: float):
        """フレーム時間記録"""
        self.frame_times.append(float(frame_time))
    
    def record_processing_time(self, processing_time: float):
        """処理時間記録"""
        self.processing_time = float(processing_time)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """現在のメトリクス取得"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, duration_seconds: int = 5) -> Optional[PerformanceMetrics]:
        """平均メトリクス取得"""
        if not self.metrics_history:
            return None
        
        samples = min(duration_seconds, len(self.metrics_history))
        recent_metrics = list(self.metrics_history)[-samples:]
        
        if not recent_metrics:
            return None
        
        return PerformanceMetrics(
            fps=sum(m.fps for m in recent_metrics) / samples,
            cpu_usage=sum(m.cpu_usage for m in recent_metrics) / samples,
            memory_usage=sum(m.memory_usage for m in recent_metrics) / samples,
            gpu_memory_usage=sum(m.gpu_memory_usage for m in recent_metrics) / samples,
            processing_time=sum(m.processing_time for m in recent_metrics) / samples,
            frame_time=sum(m.frame_time for m in recent_metrics) / samples,
            timestamp=time.time()
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """詳細パフォーマンス統計取得"""
        current_metrics = self.get_current_metrics()
        avg_metrics = self.get_average_metrics(5)
        
        stats = {
            'current': current_metrics.to_dict() if current_metrics else None,
            'average_5s': avg_metrics.to_dict() if avg_metrics else None,
            'capabilities': {
                'psutil_available': PSUTIL_AVAILABLE,
                'gpu_available': self.gpu_available,
                'monitoring_active': self.monitoring_active
            },
            'thresholds': self.thresholds.copy(),
            'history_size': len(self.metrics_history)
        }
        
        # FPS統計
        if self.frame_times:
            stats['fps_stats'] = {
                'count': len(self.frame_times),
                'avg': sum(self.frame_times) / len(self.frame_times),
                'max': max(self.frame_times),
                'min': min(self.frame_times),
                'current_fps': self._calculate_current_fps()
            }
        
        return stats
    
    def check_performance_warnings(self) -> List[str]:
        """現在の警告一覧取得"""
        warnings = []
        current_metrics = self.get_current_metrics()
        
        if not current_metrics:
            return ["メトリクスデータなし"]
        
        # FPS警告
        if current_metrics.fps < self.thresholds['fps_warning']:
            warnings.append(f"FPS低下: {current_metrics.fps:.1f}")
        
        # システム監視警告
        if not PSUTIL_AVAILABLE:
            warnings.append("psutil未インストール: 概算値使用")
        
        if not self.gpu_available:
            warnings.append("GPU監視不可")
        
        # 各種閾値警告
        if PSUTIL_AVAILABLE:
            if current_metrics.cpu_usage > self.thresholds['cpu_warning']:
                warnings.append(f"CPU使用率高: {current_metrics.cpu_usage:.1f}%")
            
            if current_metrics.memory_usage > self.thresholds['memory_warning']:
                warnings.append(f"メモリ使用率高: {current_metrics.memory_usage:.1f}%")
        
        if self.gpu_available and current_metrics.gpu_memory_usage > self.thresholds['gpu_warning']:
            warnings.append(f"GPU使用率高: {current_metrics.gpu_memory_usage:.1f}%")
        
        return warnings
    
    def adjust_performance_targets(self, new_fps_target: Union[int, float]):
        """パフォーマンス目標調整"""
        self.target_fps = int(new_fps_target)
        self.thresholds['fps_warning'] = float(new_fps_target * 0.8)
        self.thresholds['fps_critical'] = float(new_fps_target * 0.67)
        print(f"📊 FPS目標を{self.target_fps}に調整（警告: {self.thresholds['fps_warning']:.1f}）")
    
    def set_threshold(self, metric: str, value: Union[int, float]):
        """閾値設定"""
        if metric in self.thresholds:
            self.thresholds[metric] = float(value)
            print(f"📊 {metric} 閾値を {value} に設定")
        else:
            available_metrics = ', '.join(self.thresholds.keys())
            print(f"⚠️  未知のメトリクス: {metric}")
            print(f"利用可能: {available_metrics}")
    
    def get_debug_info(self) -> Dict[str, Any]:
        """詳細デバッグ情報取得"""
        return {
            'performance_stats': self.get_performance_stats(),
            'warnings': self.check_performance_warnings(),
            'system_info': self.system_info,
            'capabilities': {
                'psutil_available': PSUTIL_AVAILABLE,
                'gpu_available': self.gpu_available,
                'monitoring_active': self.monitoring_active
            },
            'configuration': {
                'target_fps': self.target_fps,
                'history_size': self.history_size,
                'gpu_check_interval': self.gpu_check_interval,
                'thresholds': self.thresholds
            }
        }
    
    def cleanup(self):
        """クリーンアップ"""
        self.stop_monitoring()
        print("🧹 パフォーマンスモニターをクリーンアップしました")

# テスト実行
if __name__ == "__main__":
    print("🔍 統合パフォーマンスモニターテスト開始...")
    
    monitor = PerformanceMonitor()
    
    # 設定テスト
    print("\n🔧 設定テスト...")
    monitor.adjust_performance_targets(25)
    monitor.set_threshold('cpu_warning', 75.0)
    monitor.set_threshold('invalid_metric', 100.0)  # エラーテスト
    
    # 監視開始
    monitor.start_monitoring(interval=0.5)
    
    # シミュレーション
    print("\n📊 5秒間の監視シミュレーション...")
    for i in range(50):
        # フレーム時間シミュレート
        frame_time = 0.033 + (i % 10) * 0.001  # 30FPS前後
        monitor.record_frame_time(frame_time)
        
        # 処理時間シミュレート
        processing_time = 0.02 + (i % 5) * 0.005
        monitor.record_processing_time(processing_time)
        
        time.sleep(0.1)
    
    # 結果表示
    print("\n📈 結果:")
    stats = monitor.get_performance_stats()
    
    if stats['current']:
        current = stats['current']
        print(f"FPS: {current['fps']:.1f}")
        print(f"CPU: {current['cpu_usage']:.1f}%")
        print(f"Memory: {current['memory_usage']:.1f}%")
        print(f"GPU: {current['gpu_memory_usage']:.1f}%")
        print(f"処理時間: {current['processing_time']*1000:.1f}ms")
    
    # 警告確認
    warnings = monitor.check_performance_warnings()
    if warnings:
        print(f"\n⚠️  警告: {' | '.join(warnings)}")
    else:
        print("\n✅ パフォーマンス正常")
    
    # デバッグ情報
    debug_info = monitor.get_debug_info()
    print(f"\n🔍 機能: psutil={debug_info['capabilities']['psutil_available']}, "
          f"GPU={debug_info['capabilities']['gpu_available']}")
    
    monitor.cleanup()
    print("\n✅ 統合パフォーマンスモニターテスト完了")
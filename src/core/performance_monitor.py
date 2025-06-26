"""
パフォーマンス監視システム
"""

import time
import psutil
import logging
from typing import Dict, List, Optional
from collections import deque
import threading

class PerformanceMonitor:
    """
    アプリケーション・GPU・システムパフォーマンス監視
    """
    
    def __init__(self, config, history_size: int = 100):
        self.config = config
        self.logger = logging.getLogger("PerformanceMonitor")
        self.history_size = history_size
        
        # パフォーマンス履歴
        self.frame_times = deque(maxlen=history_size)
        self.cpu_usage = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.gpu_memory = deque(maxlen=history_size)
        
        # 統計情報
        self.stats = {
            'total_frames': 0,
            'dropped_frames': 0,
            'average_fps': 0.0,
            'min_fps': float('inf'),
            'max_fps': 0.0,
            'cpu_average': 0.0,
            'memory_peak': 0.0
        }
        
        # 監視フラグ
        self.monitoring = False
        self.target_fps = config.get('rendering.target_fps', 30)
        self.max_frame_time = 1.0 / self.target_fps
        
        # GPUモニタリング
        self.gpu_available = self._check_gpu_availability()
        
    def _check_gpu_availability(self) -> bool:
        """GPU監視可能性確認"""
        try:
            import nvidia_ml_py3 as nvml
            nvml.nvmlInit()
            return True
        except ImportError:
            self.logger.warning("nvidia-ml-py3が利用できません（CPU監視のみ）")
            return False
        except Exception as e:
            self.logger.warning(f"GPU監視初期化失敗: {e}")
            return False
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.logger.info("パフォーマンス監視開始")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        self.logger.info("パフォーマンス監視停止")
    
    def update_frame_stats(self, frame_time: float):
        """フレーム統計更新"""
        if not self.monitoring:
            return
        
        self.frame_times.append(frame_time)
        self.stats['total_frames'] += 1
        
        # フレームドロップ検出
        if frame_time > self.max_frame_time * 1.5:
            self.stats['dropped_frames'] += 1
        
        # FPS計算
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            self.stats['min_fps'] = min(self.stats['min_fps'], current_fps)
            self.stats['max_fps'] = max(self.stats['max_fps'], current_fps)
        
        # 平均FPS更新
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.stats['average_fps'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # システム統計更新
        self._update_system_stats()
    
    def _update_system_stats(self):
        """システム統計更新"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu_percent)
            
            # メモリ使用量
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            self.memory_usage.append(memory_mb)
            
            # 統計更新
            if len(self.cpu_usage) > 0:
                self.stats['cpu_average'] = sum(self.cpu_usage) / len(self.cpu_usage)
            
            self.stats['memory_peak'] = max(self.memory_usage) if self.memory_usage else 0
            
            # GPU メモリ監視
            if self.gpu_available:
                self._update_gpu_stats()
                
        except Exception as e:
            self.logger.error(f"システム統計更新エラー: {e}")
    
    def _update_gpu_stats(self):
        """GPU統計更新"""
        try:
            import nvidia_ml_py3 as nvml
            
            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_memory_mb = info.used / (1024 * 1024)
            self.gpu_memory.append(gpu_memory_mb)
            
        except Exception as e:
            self.logger.error(f"GPU統計更新エラー: {e}")
    
    def get_current_stats(self) -> Dict:
        """現在の統計情報取得"""
        current_stats = self.stats.copy()
        
        # リアルタイム情報追加
        if len(self.frame_times) > 0:
            current_stats['current_frame_time'] = self.frame_times[-1]
        
        if len(self.cpu_usage) > 0:
            current_stats['current_cpu'] = self.cpu_usage[-1]
        
        if len(self.memory_usage) > 0:
            current_stats['current_memory'] = self.memory_usage[-1]
        
        if len(self.gpu_memory) > 0:
            current_stats['current_gpu_memory'] = self.gpu_memory[-1]
        
        return current_stats
    
    def get_performance_summary(self) -> str:
        """パフォーマンス概要文字列取得"""
        stats = self.get_current_stats()
        
        summary = f"""
📊 パフォーマンス概要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎮 フレーム統計:
   - 平均FPS: {stats['average_fps']:.1f}
   - 最小FPS: {stats['min_fps']:.1f}
   - 最大FPS: {stats['max_fps']:.1f}
   - 総フレーム数: {stats['total_frames']}
   - ドロップフレーム: {stats['dropped_frames']}

💻 システム統計:
   - CPU使用率: {stats.get('current_cpu', 0):.1f}% (平均: {stats['cpu_average']:.1f}%)
   - メモリ使用量: {stats.get('current_memory', 0):.1f}MB (ピーク: {stats['memory_peak']:.1f}MB)
"""
        
        if 'current_gpu_memory' in stats:
            summary += f"   - GPU メモリ: {stats['current_gpu_memory']:.1f}MB\n"
        
        summary += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        return summary
    
    def should_adjust_quality(self) -> Optional[str]:
        """品質調整提案"""
        if len(self.frame_times) < 10:
            return None
        
        avg_fps = self.stats['average_fps']
        target_fps = self.target_fps
        
        if avg_fps < target_fps * 0.8:
            return "quality_down"
        elif avg_fps > target_fps * 1.2:
            return "quality_up"
        
        return None

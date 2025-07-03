"""
GPU パフォーマンス最適化システム
動的品質調整・負荷分散制御
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from collections import deque

class QualityLevel(Enum):
    """品質レベル"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    ULTRA = 4

class PerformanceOptimizer:
    """
    動的パフォーマンス最適化システム
    GPU負荷監視・品質自動調整
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("PerformanceOptimizer")
        
        # 目標性能
        self.target_fps = config.get('rendering.target_fps', 30)
        self.max_frame_time = 1.0 / self.target_fps
        self.gpu_usage_limit = 85.0  # 85%以下
        
        # 現在の品質設定
        self.current_quality = QualityLevel.HIGH
        self.auto_adjustment = True
        
        # パフォーマンス履歴
        self.frame_times = deque(maxlen=60)  # 2秒分
        self.gpu_usage_history = deque(maxlen=30)  # 1秒分
        self.adjustment_cooldown = 0.0
        self.last_adjustment_time = 0.0
        
        # 品質設定パラメータ
        self.quality_params = {
            QualityLevel.LOW: {
                'camera_resolution_scale': 0.5,
                'ai_processing_interval': 3,  # フレーム間隔
                'effect_intensity_scale': 0.6,
                'texture_quality': 0.5,
                'enable_glow': False,
                'enable_ripples': True,
                'enable_color_blend': True
            },
            QualityLevel.MEDIUM: {
                'camera_resolution_scale': 0.75,
                'ai_processing_interval': 2,
                'effect_intensity_scale': 0.8,
                'texture_quality': 0.75,
                'enable_glow': True,
                'enable_ripples': True,
                'enable_color_blend': True
            },
            QualityLevel.HIGH: {
                'camera_resolution_scale': 1.0,
                'ai_processing_interval': 1,
                'effect_intensity_scale': 1.0,
                'texture_quality': 1.0,
                'enable_glow': True,
                'enable_ripples': True,
                'enable_color_blend': True
            },
            QualityLevel.ULTRA: {
                'camera_resolution_scale': 1.0,
                'ai_processing_interval': 1,
                'effect_intensity_scale': 1.2,
                'texture_quality': 1.0,
                'enable_glow': True,
                'enable_ripples': True,
                'enable_color_blend': True
            }
        }
        
        # 統計情報
        self.total_adjustments = 0
        self.performance_score = 100.0
        
        self.logger.info("✅ パフォーマンス最適化システム初期化完了")
    
    def update_performance_data(self, frame_time: float, gpu_usage: Optional[float] = None):
        """パフォーマンスデータ更新"""
        self.frame_times.append(frame_time)
        
        if gpu_usage is not None:
            self.gpu_usage_history.append(gpu_usage)
        
        # 自動調整判定
        if self.auto_adjustment:
            current_time = time.time()
            
            # クールダウン期間チェック（5秒間隔）
            if current_time - self.last_adjustment_time > 5.0:
                self._evaluate_performance_adjustment()
    
    def _evaluate_performance_adjustment(self):
        """パフォーマンス調整評価"""
        if len(self.frame_times) < 30:  # 1秒分のデータが必要
            return
        
        # 平均フレーム時間計算
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        # GPU使用率平均
        avg_gpu_usage = 0.0
        if self.gpu_usage_history:
            avg_gpu_usage = sum(self.gpu_usage_history) / len(self.gpu_usage_history)
        
        # パフォーマンス評価
        fps_ratio = current_fps / self.target_fps
        gpu_load_factor = avg_gpu_usage / 100.0
        
        # 調整判定
        should_decrease_quality = (
            fps_ratio < 0.9 or  # 目標FPSの90%以下
            avg_gpu_usage > self.gpu_usage_limit
        )
        
        should_increase_quality = (
            fps_ratio > 1.1 and  # 目標FPSの110%以上
            avg_gpu_usage < 70.0 and  # GPU使用率70%以下
            self.current_quality != QualityLevel.ULTRA
        )
        
        # 品質調整実行
        if should_decrease_quality and self.current_quality != QualityLevel.LOW:
            self._decrease_quality()
            self.logger.info(f"パフォーマンス低下により品質下げ: FPS={current_fps:.1f}, GPU={avg_gpu_usage:.1f}%")
        elif should_increase_quality:
            self._increase_quality()
            self.logger.info(f"パフォーマンス余裕により品質上げ: FPS={current_fps:.1f}, GPU={avg_gpu_usage:.1f}%")
        
        # パフォーマンススコア更新
        self._update_performance_score(fps_ratio, gpu_load_factor)
    
    def _decrease_quality(self):
        """品質レベル下げ"""
        quality_levels = list(QualityLevel)
        current_index = quality_levels.index(self.current_quality)
        
        if current_index > 0:
            self.current_quality = quality_levels[current_index - 1]
            self.total_adjustments += 1
            self.last_adjustment_time = time.time()
    
    def _increase_quality(self):
        """品質レベル上げ"""
        quality_levels = list(QualityLevel)
        current_index = quality_levels.index(self.current_quality)
        
        if current_index < len(quality_levels) - 1:
            self.current_quality = quality_levels[current_index + 1]
            self.total_adjustments += 1
            self.last_adjustment_time = time.time()
    
    def _update_performance_score(self, fps_ratio: float, gpu_load_factor: float):
        """パフォーマンススコア更新"""
        # FPS スコア（0-50点）
        fps_score = min(50.0, fps_ratio * 50.0)
        
        # GPU効率スコア（0-50点）
        if gpu_load_factor < 0.7:
            gpu_score = 50.0  # 最高効率
        elif gpu_load_factor < 0.85:
            gpu_score = 40.0
        elif gpu_load_factor < 0.95:
            gpu_score = 25.0
        else:
            gpu_score = 10.0  # 過負荷
        
        self.performance_score = fps_score + gpu_score
    
    def get_current_quality_params(self) -> Dict[str, Any]:
        """現在の品質パラメータ取得"""
        return self.quality_params[self.current_quality].copy()
    
    def set_quality_level(self, quality: QualityLevel):
        """品質レベル手動設定"""
        self.current_quality = quality
        self.logger.info(f"品質レベル手動設定: {quality.name}")
    
    def toggle_auto_adjustment(self):
        """自動調整オンオフ切り替え"""
        self.auto_adjustment = not self.auto_adjustment
        self.logger.info(f"自動品質調整: {'ON' if self.auto_adjustment else 'OFF'}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """パフォーマンス統計取得"""
        current_fps = 0.0
        avg_gpu_usage = 0.0
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        if self.gpu_usage_history:
            avg_gpu_usage = sum(self.gpu_usage_history) / len(self.gpu_usage_history)
        
        return {
            'current_quality': self.current_quality.name,
            'current_fps': current_fps,
            'target_fps': self.target_fps,
            'avg_gpu_usage': avg_gpu_usage,
            'performance_score': self.performance_score,
            'total_adjustments': self.total_adjustments,
            'auto_adjustment': self.auto_adjustment
        }
    
    def force_low_quality_mode(self):
        """緊急低品質モード"""
        self.current_quality = QualityLevel.LOW
        self.auto_adjustment = False
        self.logger.warning("🚨 緊急低品質モード有効化")
    
    def reset_optimization(self):
        """最適化リセット"""
        self.current_quality = QualityLevel.HIGH
        self.auto_adjustment = True
        self.frame_times.clear()
        self.gpu_usage_history.clear()
        self.total_adjustments = 0
        self.performance_score = 100.0
        self.logger.info("パフォーマンス最適化リセット")

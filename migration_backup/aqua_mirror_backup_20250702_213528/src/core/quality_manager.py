# ファイル: src/core/quality_manager.py (新規作成)
# 時間: 2-3時間 | 優先度: 🟡 高
import time
from enum import Enum
from typing import Dict, Any

class QualityLevel(Enum):
    ULTRA = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1

class AdaptiveQualityManager:
    """適応的品質制御"""
    
    def __init__(self, performance_monitor):
        self.performance_monitor = performance_monitor
        self.current_level = QualityLevel.HIGH
        self.adjustment_cooldown = 3.0  # 3秒間隔で調整
        self.last_adjustment = 0
        
        # 品質レベル別設定
        self.quality_settings = {
            QualityLevel.ULTRA: {
                'mediapipe_model_complexity': 2,
                'frame_skip': 0,
                'particle_count': 2000,
                'effect_quality': 'ultra',
                'render_scale': 1.0
            },
            QualityLevel.HIGH: {
                'mediapipe_model_complexity': 1,
                'frame_skip': 0,
                'particle_count': 1000,
                'effect_quality': 'high',
                'render_scale': 1.0
            },
            QualityLevel.MEDIUM: {
                'mediapipe_model_complexity': 1,
                'frame_skip': 1,
                'particle_count': 500,
                'effect_quality': 'medium',
                'render_scale': 0.9
            },
            QualityLevel.LOW: {
                'mediapipe_model_complexity': 0,
                'frame_skip': 2,
                'particle_count': 200,
                'effect_quality': 'low',
                'render_scale': 0.8
            },
            QualityLevel.MINIMAL: {
                'mediapipe_model_complexity': 0,
                'frame_skip': 3,
                'particle_count': 50,
                'effect_quality': 'minimal',
                'render_scale': 0.7
            }
        }
    
    def update(self) -> bool:
        """品質レベル更新（必要時）"""
        current_time = time.time()
        
        # クールダウン中は調整しない
        if current_time - self.last_adjustment < self.adjustment_cooldown:
            return False
        
        # 現在のパフォーマンス取得
        metrics = self.performance_monitor.get_average_metrics(duration_seconds=2)
        if not metrics:
            return False
        
        # 品質調整判定
        new_level = self._determine_quality_level(metrics)
        
        if new_level != self.current_level:
            self.current_level = new_level
            self.last_adjustment = current_time
            print(f"🎯 品質レベル調整: {new_level.name}")
            return True
        
        return False
    
    def _determine_quality_level(self, metrics) -> QualityLevel:
        """パフォーマンスに基づく品質レベル決定"""
        # FPSベース判定
        if metrics.fps < 20:
            return QualityLevel.MINIMAL
        elif metrics.fps < 25:
            return QualityLevel.LOW
        elif metrics.fps < 28:
            return QualityLevel.MEDIUM
        elif metrics.fps >= 30:
            # CPU・メモリ使用率も確認
            if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
                return QualityLevel.MEDIUM
            else:
                return QualityLevel.HIGH
        else:
            return QualityLevel.MEDIUM
    
    def get_current_settings(self) -> Dict[str, Any]:
        """現在の品質設定取得"""
        return self.quality_settings[self.current_level].copy()
    
    def force_quality_level(self, level: QualityLevel):
        """品質レベル強制設定"""
        self.current_level = level
        print(f"🎯 品質レベル強制設定: {level.name}")

    def cycle_quality_level(self):
        """品質レベルを順番に切り替え"""
        levels = list(QualityLevel)
        try:
            current_index = levels.index(self.current_level)
            # HIGH -> MEDIUM -> LOW -> HIGH... のように循環させたいので、
            # MINIMALとULTRAは手動設定のみとする
            cycle_levels = [QualityLevel.HIGH, QualityLevel.MEDIUM, QualityLevel.LOW]
            if self.current_level in cycle_levels:
                current_cycle_index = cycle_levels.index(self.current_level)
                next_level = cycle_levels[(current_cycle_index + 1) % len(cycle_levels)]
            else:
                next_level = QualityLevel.MEDIUM
            
            self.force_quality_level(next_level)
        except ValueError:
            # 現在のレベルがリストにない場合、デフォルトに設定
            self.force_quality_level(QualityLevel.MEDIUM)
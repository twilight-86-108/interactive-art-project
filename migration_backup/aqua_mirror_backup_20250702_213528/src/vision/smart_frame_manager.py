"""
スマートフレーム管理システム
AI処理とカメラFPSの最適バランス
"""

import time
import threading
from typing import Optional, Dict, Any
from collections import deque

class SmartFrameManager:
    """
    インテリジェントフレーム管理
    カメラFPS < AI処理能力の場合の最適化
    """
    
    def __init__(self, camera_manager, ai_processor):
        self.camera = camera_manager
        self.ai_processor = ai_processor
        
        # フレーム管理設定
        self.ai_processing_interval = 3  # 3フレームに1回AI処理
        self.frame_counter = 0
        
        # 最新結果キャッシュ
        self.latest_ai_result = None
        self.last_ai_time = 0
        
        # 統計
        self.total_frames = 0
        self.ai_processed_frames = 0
        self.skipped_frames = 0
        
        # 動的調整
        self.auto_adjust = True
        self.target_ai_fps = 10  # AI処理目標FPS
        
    def process_frame_smart(self) -> Optional[Dict[str, Any]]:
        """スマートフレーム処理"""
        frame = self.camera.get_frame()
        if frame is None:
            return self.latest_ai_result
        
        self.total_frames += 1
        self.frame_counter += 1
        
        # AI処理判定
        should_process_ai = (
            self.frame_counter >= self.ai_processing_interval or
            self.latest_ai_result is None or
            time.time() - self.last_ai_time > 1.0  # 1秒以上更新がない場合
        )
        
        if should_process_ai:
            # AI処理実行
            ai_start = time.time()
            
            if hasattr(self.ai_processor, 'process_frame'):
                # MediaPipe処理
                mp_results = self.ai_processor.process_frame(frame)
                
                # 感情認識（別のオブジェクトの場合）
                if hasattr(self, 'emotion_analyzer'):
                    ai_result = self.emotion_analyzer.analyze_emotion(mp_results)
                else:
                    ai_result = mp_results
            else:
                # 統合AI処理オブジェクトの場合
                ai_result = self.ai_processor.analyze_frame(frame)
            
            self.latest_ai_result = ai_result
            self.last_ai_time = time.time()
            self.ai_processed_frames += 1
            self.frame_counter = 0
            
            # 動的間隔調整
            if self.auto_adjust:
                self._adjust_processing_interval(time.time() - ai_start)
            
        else:
            # フレームスキップ
            self.skipped_frames += 1
        
        return self.latest_ai_result
    
    def _adjust_processing_interval(self, ai_processing_time: float):
        """AI処理間隔動的調整"""
        current_ai_fps = 1.0 / ai_processing_time if ai_processing_time > 0 else 0
        
        if current_ai_fps > self.target_ai_fps * 1.5:
            # 余裕があるので間隔短縮
            self.ai_processing_interval = max(1, self.ai_processing_interval - 1)
        elif current_ai_fps < self.target_ai_fps * 0.7:
            # 負荷が高いので間隔延長
            self.ai_processing_interval = min(10, self.ai_processing_interval + 1)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """処理統計取得"""
        ai_fps = 0
        efficiency = 0
        
        if self.total_frames > 0:
            efficiency = (self.ai_processed_frames / self.total_frames) * 100
        
        if self.ai_processed_frames > 0 and self.last_ai_time > 0:
            # 概算AI FPS
            total_time = time.time() - (self.last_ai_time - self.ai_processed_frames * 0.1)
            ai_fps = self.ai_processed_frames / total_time if total_time > 0 else 0
        
        return {
            'total_frames': self.total_frames,
            'ai_processed_frames': self.ai_processed_frames,
            'skipped_frames': self.skipped_frames,
            'processing_interval': self.ai_processing_interval,
            'ai_fps': ai_fps,
            'efficiency_percent': efficiency
        }
    
    def set_ai_target_fps(self, target_fps: int):
        """AI目標FPS設定"""
        self.target_ai_fps = max(1, min(30, target_fps))
    
    def force_ai_processing(self):
        """強制AI処理"""
        self.frame_counter = self.ai_processing_interval
    
    def reset_stats(self):
        """統計リセット"""
        self.total_frames = 0
        self.ai_processed_frames = 0
        self.skipped_frames = 0
        self.frame_counter = 0

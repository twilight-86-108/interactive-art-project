"""
デバッグUI システム
リアルタイム統計表示・パフォーマンス監視
"""

import moderngl
import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

class DebugUIManager:
    """
    デバッグ情報表示システム
    GPU統計・AI性能・エフェクト状態可視化
    """
    
    def __init__(self, ctx: moderngl.Context, config):
        self.ctx = ctx
        self.config = config
        self.logger = logging.getLogger("DebugUI")
        
        # デバッグ表示設定
        self.show_fps = config.get('debug.show_fps', True)
        self.show_gpu_stats = config.get('debug.show_gpu_stats', True)
        self.show_ai_stats = True
        self.show_effect_params = True
        self.show_camera_info = True
        
        # UI レンダリング用リソース
        self.ui_program = None
        self.text_texture = None
        self.ui_vao = None
        
        # 統計データ
        self.stats_data = {
            'fps': 0.0,
            'frame_time': 0.0,
            'gpu_usage': 0.0,
            'memory_usage': 0.0,
            'camera_fps': 0.0,
            'ai_fps': 0.0,
            'emotion': 'NEUTRAL',
            'emotion_intensity': 0.0,
            'quality_level': 'HIGH',
            'effect_status': {}
        }
        
        # 表示位置・サイズ
        self.ui_position = (10, 10)  # 左上
        self.line_height = 20
        self.text_size = 14
        
        self._initialize_ui_rendering()
    
    def _initialize_ui_rendering(self):
        """UI レンダリング初期化"""
        try:
            # シンプルなテキスト表示用シェーダー
            vertex_shader = """
            #version 410 core
            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 texcoord;
            
            uniform vec2 u_screen_size;
            uniform vec2 u_position;
            
            out vec2 v_texcoord;
            
            void main() {
                vec2 screen_pos = (position * 300.0 + u_position) / u_screen_size * 2.0 - 1.0;
                screen_pos.y *= -1.0;  // Y軸反転
                gl_Position = vec4(screen_pos, 0.0, 1.0);
                v_texcoord = texcoord;
            }
            """
            
            fragment_shader = """
            #version 410 core
            in vec2 v_texcoord;
            out vec4 fragColor;
            
            uniform float u_alpha;
            
            void main() {
                // シンプルな半透明背景
                vec3 bg_color = vec3(0.0, 0.0, 0.0);
                float alpha = u_alpha * 0.7;
                
                // テキスト領域表示
                if (v_texcoord.x < 0.98 && v_texcoord.y < 0.98) {
                    fragColor = vec4(bg_color, alpha);
                } else {
                    fragColor = vec4(1.0, 1.0, 1.0, alpha * 0.5);  // 境界線
                }
            }
            """
            
            self.ui_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            # UI クアッド作成
            vertices = np.array([
                # Position    TexCoord
                [0.0, 0.0,   0.0, 0.0],
                [1.0, 0.0,   1.0, 0.0],
                [1.0, 1.0,   1.0, 1.0],
                [0.0, 1.0,   0.0, 1.0],
            ], dtype=np.float32)
            
            indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            
            vertex_buffer = self.ctx.buffer(vertices.tobytes())
            index_buffer = self.ctx.buffer(indices.tobytes())
            
            self.ui_vao = self.ctx.vertex_array(
                self.ui_program,
                [(vertex_buffer, '2f 2f', 'position', 'texcoord')],
                index_buffer
            )
            
            self.logger.info("✅ デバッグUI レンダリング初期化完了")
            
        except Exception as e:
            self.logger.error(f"❌ デバッグUI 初期化失敗: {e}")
    
    def update_stats(self, stats_update: Dict[str, Any]):
        """統計データ更新"""
        self.stats_data.update(stats_update)
    
    def render_debug_info(self, window_size: Tuple[int, int]):
        """デバッグ情報描画"""
        if not self.ui_program or not self.ui_vao:
            return
        
        try:
            # ブレンド設定
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            # UI 背景描画
            self.ui_program['u_screen_size'] = window_size
            self.ui_program['u_position'] = self.ui_position
            self.ui_program['u_alpha'] = 0.8
            
            self.ui_vao.render()
            
            # テキスト情報をログ出力（簡易実装）
            self._log_debug_info()
            
        except Exception as e:
            self.logger.error(f"デバッグUI 描画エラー: {e}")
    
    def _log_debug_info(self):
        """デバッグ情報ログ出力"""
        debug_lines = []
        
        # FPS情報
        if self.show_fps:
            debug_lines.append(f"FPS: {self.stats_data['fps']:.1f}")
            debug_lines.append(f"Frame Time: {self.stats_data['frame_time']:.1f}ms")
        
        # GPU統計
        if self.show_gpu_stats:
            debug_lines.append(f"GPU Usage: {self.stats_data['gpu_usage']:.1f}%")
            debug_lines.append(f"Memory: {self.stats_data['memory_usage']:.1f}MB")
        
        # カメラ情報
        if self.show_camera_info:
            debug_lines.append(f"Camera FPS: {self.stats_data['camera_fps']:.1f}")
        
        # AI統計
        if self.show_ai_stats:
            debug_lines.append(f"AI FPS: {self.stats_data['ai_fps']:.1f}")
            debug_lines.append(f"Emotion: {self.stats_data['emotion']}")
            debug_lines.append(f"Intensity: {self.stats_data['emotion_intensity']:.2f}")
        
        # 品質レベル
        debug_lines.append(f"Quality: {self.stats_data['quality_level']}")
        
        # エフェクト状態
        if self.show_effect_params and self.stats_data['effect_status']:
            for effect_name, status in self.stats_data['effect_status'].items():
                debug_lines.append(f"{effect_name}: {'ON' if status else 'OFF'}")
        
        # 5秒ごとにログ出力
        current_time = time.time()
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = current_time
        
        if current_time - self._last_log_time >= 5.0:
            self.logger.info("📊 デバッグ情報: " + " | ".join(debug_lines))
            self._last_log_time = current_time
    
    def toggle_fps_display(self):
        """FPS表示切り替え"""
        self.show_fps = not self.show_fps
    
    def toggle_gpu_stats(self):
        """GPU統計表示切り替え"""
        self.show_gpu_stats = not self.show_gpu_stats
    
    def toggle_ai_stats(self):
        """AI統計表示切り替え"""
        self.show_ai_stats = not self.show_ai_stats
    
    def toggle_effect_params(self):
        """エフェクトパラメータ表示切り替え"""
        self.show_effect_params = not self.show_effect_params
    
    def get_debug_text_lines(self) -> List[str]:
        """デバッグテキスト行取得（外部描画用）"""
        lines = []
        
        if self.show_fps:
            lines.append(f"🎮 FPS: {self.stats_data['fps']:.1f} ({self.stats_data['frame_time']:.1f}ms)")
        
        if self.show_gpu_stats:
            lines.append(f"🖥️ GPU: {self.stats_data['gpu_usage']:.1f}% | RAM: {self.stats_data['memory_usage']:.1f}MB")
        
        if self.show_camera_info:
            lines.append(f"📹 Camera: {self.stats_data['camera_fps']:.1f}fps")
        
        if self.show_ai_stats:
            lines.append(f"🧠 AI: {self.stats_data['ai_fps']:.1f}fps | {self.stats_data['emotion']} ({self.stats_data['emotion_intensity']:.2f})")
        
        lines.append(f"⚙️ Quality: {self.stats_data['quality_level']}")
        
        if self.show_effect_params and self.stats_data['effect_status']:
            effect_states = [f"{name}={'ON' if status else 'OFF'}" 
                           for name, status in self.stats_data['effect_status'].items()]
            lines.append(f"✨ Effects: {' | '.join(effect_states)}")
        
        return lines
    
    def cleanup(self):
        """リソース解放"""
        if self.ui_vao:
            self.ui_vao.release()
        
        if self.ui_program:
            self.ui_program.release()
        
        self.logger.info("✅ デバッグUI リソース解放完了")

"""
ModernGL Aqua Mirror - メインアプリケーション
GPU加速感情認識インタラクティブアート
"""

import time
import json
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import moderngl
import glfw
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

# 内部モジュール（これから実装）
from src.core.config_loader import ConfigLoader
from src.core.gl_context_manager import GLContextManager
from src.core.performance_monitor import PerformanceMonitor

class ModernGLApp:
    """
    Aqua Mirror ModernGL版メインアプリケーション
    感情認識→GPU水面エフェクト統合システム
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        # コンソール・ログ設定
        self.console = Console()
        self._setup_logging()
        
        # 設定読み込み
        self.config = ConfigLoader(config_path)
        
        # 初期化フラグ
        self.initialized = False
        self.running = False
        
        # コアコンポーネント
        self.gl_context: Optional[GLContextManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # GPU/OpenGL関連
        self.ctx: Optional[moderngl.Context] = None
        self.window = None
        
        # 基本レンダリング
        self.quad_program: Optional[moderngl.Program] = None
        self.quad_vao: Optional[moderngl.VertexArray] = None
        
        # 状態管理
        self.frame_count = 0
        self.last_time = time.time()
        
        self.logger.info("🌊 Aqua Mirror ModernGL版 初期化中...")
    
    def _setup_logging(self):
        """ログシステム設定"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[RichHandler(console=self.console)]
        )
        self.logger = logging.getLogger("AquaMirror")
    
    def initialize(self) -> bool:
        """
        アプリケーション初期化
        Returns:
            bool: 初期化成功フラグ
        """
        try:
            self.logger.info("🔧 システム初期化開始...")
            
            # 1. GLFW初期化
            if not self._initialize_glfw():
                return False
            
            # 2. OpenGL/ModernGLコンテキスト作成
            if not self._initialize_opengl():
                return False
            
            # 3. コアコンポーネント初期化
            if not self._initialize_components():
                return False
            
            # 4. 基本シェーダー・ジオメトリ作成
            if not self._initialize_basic_rendering():
                return False
            
            self.initialized = True
            self.logger.info("✅ システム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初期化失敗: {e}")
            return False
    
    def _initialize_glfw(self) -> bool:
        if not glfw.init():
            self.logger.error("❌ GLFW初期化失敗")
            return False
    
        # OpenGL 4.1 + 互換プロファイル設定
        opengl_major = self.config.get('gpu.opengl_version.0', 4)
        opengl_minor = self.config.get('gpu.opengl_version.1', 1)
        profile_type = self.config.get('gpu.opengl_profile', 'compatibility')
    
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, opengl_major)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, opengl_minor)
    
        # プロファイル設定
        if profile_type == "core":
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        else:
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
    
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, False)  # 互換性優先
    
        # デバッグコンテキスト（開発時）
        if self.config.get('debug.enable_debug', False):
            glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, True)
    
        # ウィンドウ作成
        width = self.config.get('app.window.width', 1920)
        height = self.config.get('app.window.height', 1080)
        fullscreen = self.config.get('app.window.fullscreen', False)
    
        monitor = glfw.get_primary_monitor() if fullscreen else None
        self.window = glfw.create_window(width, height, "Aqua Mirror", monitor, None)
    
        if not self.window:
            self.logger.error("❌ ウィンドウ作成失敗")
            glfw.terminate()
            return False
    
        glfw.make_context_current(self.window)
    
        # VSync設定
        vsync = self.config.get('app.window.vsync', True)
        glfw.swap_interval(1 if vsync else 0)
    
        # コールバック設定
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_window_size_callback(self.window, self._resize_callback)
    
        self.logger.info(f"✅ GLFW初期化完了: {width}x{height}, OpenGL {opengl_major}.{opengl_minor}")
        return True
    
    def _initialize_opengl(self) -> bool:
        """OpenGL/ModernGL初期化"""
        try:
            # ModernGLコンテキスト作成
            self.ctx = moderngl.create_context()
            
            # GPU情報取得
            gpu_info = {
                'renderer': self.ctx.info['GL_RENDERER'],
                'version': self.ctx.info['GL_VERSION'],
                'vendor': self.ctx.info['GL_VENDOR']
            }
            
            self.logger.info(f"✅ OpenGL初期化完了")
            self.logger.info(f"GPU: {gpu_info['renderer']}")
            self.logger.info(f"OpenGL: {gpu_info['version']}")
            
            # OpenGL設定
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ OpenGL初期化失敗: {e}")
            return False
    
    def _initialize_components(self) -> bool:
        """コアコンポーネント初期化"""
        try:
            # GLコンテキストマネージャー
            self.gl_context = GLContextManager(self.ctx, self.config)
            
            # パフォーマンス監視
            self.performance_monitor = PerformanceMonitor(self.config)
            
            self.logger.info("✅ コアコンポーネント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ コンポーネント初期化失敗: {e}")
            return False
    
    def _initialize_basic_rendering(self) -> bool:
        """基本レンダリング初期化"""
        try:
            # 基本シェーダープログラム作成
            vertex_shader = """
            #version 410 core
            
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 texcoord;
            
            out vec2 v_texcoord;
            
            void main() {
                gl_Position = vec4(position, 1.0);
                v_texcoord = texcoord;
            }
            """
            
            fragment_shader = """
            #version 410 core
            
            in vec2 v_texcoord;
            out vec4 fragColor;
            
            uniform float u_time;
            uniform vec3 u_color;
            
            void main() {
                vec2 uv = v_texcoord;
                
                // 簡単なグラデーション
                float gradient = sin(uv.x * 3.14159 + u_time) * 0.5 + 0.5;
                vec3 color = mix(vec3(0.2, 0.6, 1.0), u_color, gradient);
                
                fragColor = vec4(color, 1.0);
            }
            """
            
            self.quad_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            # フルスクリーンクアッド作成
            vertices = np.array([
                # Position      TexCoord
                [-1.0, -1.0, 0.0,  0.0, 0.0],  # 左下
                [ 1.0, -1.0, 0.0,  1.0, 0.0],  # 右下
                [ 1.0,  1.0, 0.0,  1.0, 1.0],  # 右上
                [-1.0,  1.0, 0.0,  0.0, 1.0],  # 左上
            ], dtype=np.float32)
            
            indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
            
            self.quad_vao = self.gl_context.create_vertex_array(
                "fullscreen_quad", 
                self.quad_program, 
                vertices, 
                indices
            )
            
            self.logger.info("✅ 基本レンダリング初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 基本レンダリング初期化失敗: {e}")
            return False
    
    def run(self):
        """メインループ実行"""
        if not self.initialized:
            self.logger.error("❌ 初期化されていません")
            return
        
        self.logger.info("🚀 メインループ開始")
        self.running = True
        
        # パフォーマンス監視開始
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
        
        try:
            while not glfw.window_should_close(self.window) and self.running:
                self._update_frame()
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ ユーザーによる終了")
        except Exception as e:
            self.logger.error(f"❌ 実行エラー: {e}")
        finally:
            self.cleanup()
    
    def _update_frame(self):
        """フレーム更新"""
        frame_start = time.time()
        
        # イベント処理
        glfw.poll_events()
        
        # 描画
        self._render_frame()
        
        # ウィンドウバッファスワップ
        glfw.swap_buffers(self.window)
        
        # パフォーマンス監視
        frame_time = time.time() - frame_start
        if self.performance_monitor:
            self.performance_monitor.update_frame_stats(frame_time)
        
        # フレーム統計
        self.frame_count += 1
        current_time = time.time()
        
        # 5秒ごとに統計表示
        if current_time - self.last_time >= 5.0:
            if self.performance_monitor:
                stats = self.performance_monitor.get_current_stats()
                self.logger.info(f"FPS: {stats['average_fps']:.1f}, フレーム: {self.frame_count}")
            
            self.last_time = current_time
    
    def _render_frame(self):
        """フレーム描画"""
        # 画面クリア
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # 基本描画
        if self.quad_program and self.quad_vao:
            # ユニフォーム設定
            current_time = time.time()
            self.quad_program['u_time'] = current_time
            self.quad_program['u_color'] = (0.8, 0.4, 1.0)  # 紫色
            
            # 描画
            self.quad_vao.render()
    
    def _key_callback(self, window, key, scancode, action, mods):
        """キーボードイベント処理"""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.running = False
            elif key == glfw.KEY_F11:
                # フルスクリーン切り替え
                pass
            elif key == glfw.KEY_P and self.performance_monitor:
                # パフォーマンス統計表示
                summary = self.performance_monitor.get_performance_summary()
                print(summary)
    
    def _resize_callback(self, window, width, height):
        """ウィンドウリサイズイベント処理"""
        if self.ctx:
            self.ctx.viewport = (0, 0, width, height)
        self.logger.info(f"ウィンドウリサイズ: {width}x{height}")
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("🧹 リソース解放中...")
        
        # パフォーマンス監視停止
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        # GLリソース解放
        if self.gl_context:
            self.gl_context.cleanup_all()
        
        # ウィンドウ解放
        if self.window:
            glfw.destroy_window(self.window)
        
        glfw.terminate()
        
        self.logger.info("✅ リソース解放完了")


def main():
    """エントリーポイント"""
    app = ModernGLApp()
    
    if app.initialize():
        app.run()
    else:
        print("❌ アプリケーション初期化失敗")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

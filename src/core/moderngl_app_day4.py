"""
ModernGL Aqua Mirror - メインアプリケーション（カメラ統合版）
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

# 内部モジュール
from src.core.config_loader import ConfigLoader
from src.core.gl_context_manager import GLContextManager
from src.core.performance_monitor import PerformanceMonitor
from src.vision.camera_manager import CameraManager
from src.rendering.texture_manager import TextureManager

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
        self.camera: Optional[CameraManager] = None
        self.texture_manager: Optional[TextureManager] = None
        
        # GPU/OpenGL関連
        self.ctx: Optional[moderngl.Context] = None
        self.window = None
        
        # 基本レンダリング
        self.camera_program: Optional[moderngl.Program] = None
        self.quad_vao: Optional[moderngl.VertexArray] = None
        
        # 状態管理
        self.frame_count = 0
        self.last_time = time.time()
        self.camera_enabled = False
        
        self.logger.info("🌊 Aqua Mirror ModernGL版（カメラ統合）初期化中...")
    
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
            
            # 4. カメラシステム初期化
            if not self._initialize_camera():
                return False
            
            # 5. レンダリングシステム初期化
            if not self._initialize_rendering():
                return False
            
            self.initialized = True
            self.logger.info("✅ システム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初期化失敗: {e}")
            return False
    
    def _initialize_glfw(self) -> bool:
        """GLFW初期化 - OpenGL 4.1対応版"""
        if not glfw.init():
            self.logger.error("❌ GLFW初期化失敗")
            return False
        
        # OpenGL 4.1 + 互換プロファイル設定
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, False)
        
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
        
        self.logger.info(f"✅ GLFW初期化完了: {width}x{height}, OpenGL 4.1")
        return True
    
    def _initialize_opengl(self) -> bool:
        """OpenGL/ModernGL初期化"""
        try:
            # ModernGLコンテキスト作成
            self.ctx = moderngl.create_context()
            
            # GPU情報取得
            gpu_info = {
                'renderer': self.ctx.info.get('GL_RENDERER', 'Unknown'),
                'version': self.ctx.info.get('GL_VERSION', 'Unknown'),
                'vendor': self.ctx.info.get('GL_VENDOR', 'Unknown')
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
            
            # テクスチャマネージャー
            self.texture_manager = TextureManager(self.ctx, self.config)
            
            self.logger.info("✅ コアコンポーネント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ コンポーネント初期化失敗: {e}")
            return False
    
    def _initialize_camera(self) -> bool:
        """カメラシステム初期化"""
        try:
            # カメラマネージャー初期化
            self.camera = CameraManager(self.config)
            
            if self.camera.initialize():
                # カメラストリーミング開始
                if self.camera.start_streaming():
                    self.camera_enabled = True
                    self.logger.info("✅ カメラシステム初期化完了")
                else:
                    self.logger.warning("⚠️ カメラストリーミング開始失敗（シミュレーションモード）")
            else:
                self.logger.warning("⚠️ カメラ初期化失敗（シミュレーションモード）")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ カメラシステム初期化失敗: {e}")
            return False
    
    def _initialize_rendering(self) -> bool:
        """レンダリングシステム初期化"""
        try:
            # カメラ表示用シェーダープログラム作成
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
            
            uniform sampler2D u_camera_texture;
            uniform bool u_camera_enabled;
            uniform float u_time;
            
            void main() {
                vec2 uv = v_texcoord;
                
                if (u_camera_enabled) {
                    // カメラ画像表示
                    vec3 camera_color = texture(u_camera_texture, uv).rgb;
                    fragColor = vec4(camera_color, 1.0);
                } else {
                    // カメラ無効時のテストパターン
                    float gradient = sin(uv.x * 3.14159 + u_time) * 0.5 + 0.5;
                    vec3 color = mix(vec3(0.2, 0.6, 1.0), vec3(0.8, 0.4, 1.0), gradient);
                    fragColor = vec4(color, 1.0);
                }
            }
            """
            
            self.camera_program = self.ctx.program(
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
                self.camera_program, 
                vertices, 
                indices
            )
            
            self.logger.info("✅ レンダリングシステム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ レンダリング初期化失敗: {e}")
            return False
    
    def run(self):
        """メインループ実行"""
        if not self.initialized:
            self.logger.error("❌ 初期化されていません")
            return
        
        self.logger.info("🚀 メインループ開始（カメラ統合版）")
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
        
        # カメラフレーム処理
        self._process_camera_frame()
        
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
            self._print_stats()
            self.last_time = current_time
    
    def _process_camera_frame(self):
        """カメラフレーム処理・GPU転送"""
        if not self.camera_enabled or not self.camera:
            return
        
        # 最新フレーム取得
        frame = self.camera.get_frame()
        if frame is not None:
            # GPU テクスチャ転送
            if self.texture_manager:
                self.texture_manager.upload_camera_frame(frame)
    
    def _render_frame(self):
        """フレーム描画"""
        # 画面クリア
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # 描画
        if self.camera_program and self.quad_vao:
            # ユニフォーム設定
            current_time = time.time()
            self.camera_program['u_time'] = current_time
            self.camera_program['u_camera_enabled'] = self.camera_enabled
            
            # カメラテクスチャバインド
            if self.camera_enabled and self.texture_manager:
                camera_texture = self.texture_manager.get_camera_texture()
                if camera_texture:
                    camera_texture.use(0)
                    self.camera_program['u_camera_texture'] = 0
            
            # 描画
            self.quad_vao.render()
    
    def _print_stats(self):
        """統計情報表示"""
        stats_lines = [f"🎮 フレーム: {self.frame_count}"]
        
        # パフォーマンス統計
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_current_stats()
            stats_lines.append(f"FPS: {perf_stats['average_fps']:.1f}")
        
        # カメラ統計
        if self.camera:
            camera_stats = self.camera.get_camera_stats()
            stats_lines.append(f"カメラFPS: {camera_stats['actual_fps']:.1f}")
            stats_lines.append(f"ドロップ: {camera_stats['dropped_frames']}")
        
        # テクスチャ統計
        if self.texture_manager:
            texture_stats = self.texture_manager.get_upload_stats()
            stats_lines.append(f"GPU転送: {texture_stats['avg_upload_time_ms']:.1f}ms")
        
        self.logger.info(" | ".join(stats_lines))
    
    def _key_callback(self, window, key, scancode, action, mods):
        """キーボードイベント処理"""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.running = False
            elif key == glfw.KEY_C:
                # カメラオンオフ切り替え
                self.camera_enabled = not self.camera_enabled
                self.logger.info(f"カメラ: {'ON' if self.camera_enabled else 'OFF'}")
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
        
        # カメラ停止
        if self.camera:
            self.camera.cleanup()
        
        # テクスチャマネージャー解放
        if self.texture_manager:
            self.texture_manager.cleanup_all()
        
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

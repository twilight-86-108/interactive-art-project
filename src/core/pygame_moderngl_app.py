from __future__ import annotations
#!/usr/bin/env python3
"""
Aqua Mirror - Pygame + ModernGL完全版 (Windows対応)
既存のGLFWベースシステムをPygameに移行
NVIDIA GeForce RTX 4060 Laptop GPU最適化
"""

import os
import cv2
import sys
import time
import json
import logging
import traceback
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import pygame
import moderngl
import numpy as np
from rich.console import Console
from rich.logging import RichHandler

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 既存のコンポーネントをインポート
try:
    from src.core.config_loader import ConfigLoader
    from src.core.gl_context_manager import GLContextManager
    from src.core.performance_monitor import PerformanceMonitor
    from src.vision.camera_manager import CameraManager
    from src.rendering.texture_manager import TextureManager
    from src.emotion.mediapipe_processor import MediaPipeProcessor
    from src.emotion.emotion_analyzer import EmotionAnalyzer
    from src.effects.emotion_effects import EmotionEffectManager
except ImportError as e:
    print(f"⚠️ 既存コンポーネントのインポートエラー: {e}")
    print("基本機能のみで動作します")
    # ダミークラス定義
    class ConfigLoader:
        def __init__(self, path): self.data = {}
        def get(self, key, default=None): return default
    
    GLContextManager = None
    PerformanceMonitor = None
    CameraManager = None
    TextureManager = None
    MediaPipeProcessor = None
    EmotionAnalyzer = None
    EmotionEffectManager = None

class PygameModernGLApp:
    """
    Aqua Mirror Pygame + ModernGL版アプリケーション
    Windows環境 + NVIDIA GPU最適化対応
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        # Windows環境変数設定
        self._setup_windows_gpu_environment()
        
        # コンソール・ログ設定
        self.console = Console()
        self._setup_logging()
        
        # 設定読み込み
        try:
            self.config = ConfigLoader(config_path)
        except:
            self.config = self._create_default_config()
        
        # 初期化フラグ
        self.initialized = False
        self.running = False
        
        # Pygame関連
        self.screen = None
        self.clock = None
        
        # ModernGL関連
        self.ctx: Optional[moderngl.Context] = None
        
        # 既存コンポーネント（互換性保持）
        self.gl_context: Optional[GLContextManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.camera: Optional[CameraManager] = None
        self.texture_manager: Optional[TextureManager] = None
        self.mediapipe_processor: Optional[MediaPipeProcessor] = None
        self.emotion_analyzer: Optional[EmotionAnalyzer] = None
        self.emotion_effects: Optional[EmotionEffectManager] = None
        
        # レンダリング関連
        self.emotion_program: Optional[moderngl.Program] = None
        self.camera_program: Optional[moderngl.Program] = None
        self.basic_program: Optional[moderngl.Program] = None
        self.quad_vao: Optional[moderngl.VertexArray] = None
        
        # 状態管理
        self.frame_count = 0
        self.last_time = time.time()
        self.camera_enabled = False
        self.ai_enabled = True
        self.effects_enabled = True
        
        # GPU情報
        self.gpu_info = {}
        
        self.logger.info("🌊 Aqua Mirror Pygame + ModernGL版（Windows）初期化中...")
    
    def _setup_windows_gpu_environment(self):
        """Windows GPU環境変数設定"""
        gpu_env_vars = {
            'CUDA_VISIBLE_DEVICES': '0',
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
            'NVIDIA_VISIBLE_DEVICES': 'all',
            'NVIDIA_DRIVER_CAPABILITIES': 'all',
            '__GL_SYNC_TO_VBLANK': '1',
            'LIBGL_ALWAYS_INDIRECT': '0',
            'LIBGL_ALWAYS_SOFTWARE': '0'
        }
        
        for key, value in gpu_env_vars.items():
            os.environ[key] = value
    
    def _setup_logging(self):
        """ログシステム設定"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                RichHandler(console=self.console),
                logging.FileHandler('aqua_mirror_pygame.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger("AquaMirrorPygame")
    
    def _create_default_config(self):
        """デフォルト設定作成"""
        class DefaultConfig:
            def __init__(self):
                self.data = {
                    'app': {
                        'window': {
                            'width': 1920,
                            'height': 1080,
                            'fullscreen': False,
                            'title': 'Aqua Mirror - Pygame + ModernGL (Windows)',
                            'vsync': True
                        }
                    },
                    'gpu': {
                        'enable_nvidia': True,
                        'opengl_version': [4, 3],
                        'opengl_profile': 'core'
                    },
                    'rendering': {
                        'target_fps': 60
                    },
                    'camera': {
                        'width': 1920,
                        'height': 1080,
                        'fps': 30
                    },
                    'debug': {
                        'enable_debug': False
                    }
                }
            
            def get(self, key, default=None):
                keys = key.split('.')
                value = self.data
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
        
        return DefaultConfig()
    
    def initialize(self) -> bool:
        """アプリケーション初期化"""
        try:
            self.logger.info("🔧 システム初期化開始...")
            
            # 1. Pygame初期化
            if not self._initialize_pygame():
                return False
            
            # 2. ModernGL初期化
            if not self._initialize_moderngl():
                return False
            
            # 3. コンポーネント初期化
            if not self._initialize_components():
                return False
            
            # 4. カメラシステム初期化
            if not self._initialize_camera():
                return False
            
            # 5. AI システム初期化
            if not self._initialize_ai():
                return False
            
            # 6. エフェクトシステム初期化
            if not self._initialize_effects():
                return False
            
            # 7. レンダリングシステム初期化
            if not self._initialize_rendering():
                return False
            
            self.initialized = True
            self.logger.info("✅ システム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 初期化失敗: {e}")
            traceback.print_exc()
            return False
    
    def _initialize_pygame(self) -> bool:
        """Pygame初期化"""
        self.logger.info("🎮 Pygame初期化中...")
        
        try:
            # Pygame初期化
            pygame.init()
            
            # OpenGL属性設定（NVIDIA GPU最適化）
            self._configure_opengl_attributes()
            
            # ウィンドウ設定
            width = self.config.get('app.window.width', 1920)
            height = self.config.get('app.window.height', 1080)
            title = self.config.get('app.window.title', 'Aqua Mirror - Pygame + ModernGL (Windows)')
            
            # OpenGLウィンドウ作成
            flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.HWSURFACE
            if self.config.get('app.window.fullscreen', False):
                flags |= pygame.FULLSCREEN
            
            self.screen = pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption(title)
            
            # クロック作成
            self.clock = pygame.time.Clock()
            
            self.logger.info(f"✅ Pygame初期化完了: {width}x{height}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pygame初期化失敗: {e}")
            return False
    
    def _configure_opengl_attributes(self):
        """OpenGL属性設定（NVIDIA GPU最適化）"""
        # OpenGLバージョン設定
        major, minor = self.config.get('gpu.opengl_version', [4, 3])
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, major)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, minor)
        
        # プロファイル設定
        profile = self.config.get('gpu.opengl_profile', 'core')
        if profile == 'core':
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, 
                pygame.GL_CONTEXT_PROFILE_CORE
            )
        
        # NVIDIA GPU加速有効化
        pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 1)
        
        # バッファ設定
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
        
        # マルチサンプリング（アンチエイリアス）
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        
        self.logger.info("🔧 OpenGL属性設定完了")
    
    def _initialize_moderngl(self) -> bool:
        """ModernGL初期化"""
        self.logger.info("🌊 ModernGL初期化中...")
        
        try:
            # ModernGLコンテキスト作成
            self.ctx = moderngl.create_context()
            
            # GPU情報取得
            self.gpu_info = {
                'renderer': self.ctx.info.get('GL_RENDERER', 'Unknown'),
                'version': self.ctx.info.get('GL_VERSION', 'Unknown'),
                'vendor': self.ctx.info.get('GL_VENDOR', 'Unknown'),
                'shading_language': self.ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown')
            }
            
            # GPU情報ログ出力
            self.logger.info("🖥️ GPU情報:")
            for key, value in self.gpu_info.items():
                self.logger.info(f"  {key}: {value}")
            
            # NVIDIA GPU確認
            if 'NVIDIA' in self.gpu_info['renderer']:
                if '4060' in self.gpu_info['renderer']:
                    self.logger.info("🎯 GeForce RTX 4060 Laptop GPU 検出・有効化！")
                else:
                    self.logger.info("✅ NVIDIA GPU 有効化")
            else:
                self.logger.warning("⚠️ NVIDIA GPU使用を確認できません")
                self.logger.warning("統合グラフィックスが使用されている可能性があります")
            
            # OpenGL設定
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            # VSync設定
            if self.config.get('app.window.vsync', True):
                pygame.display.gl_set_swap_interval(1)
            
            self.logger.info("✅ ModernGL初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ModernGL初期化失敗: {e}")
            return False
    
    def _initialize_components(self) -> bool:
        """コアコンポーネント初期化"""
        self.logger.info("🔧 コアコンポーネント初期化中...")
        
        try:
            # GLコンテキストマネージャー
            if GLContextManager:
                self.gl_context = GLContextManager(self.ctx, self.config)
                self.logger.info("  ✅ GLContext管理")
            else:
                self.logger.info("  ⚠️ GLContext管理 (スキップ)")
            
            # パフォーマンス監視
            if PerformanceMonitor:
                self.performance_monitor = PerformanceMonitor(self.config)
                self.logger.info("  ✅ パフォーマンス監視")
            else:
                self.logger.info("  ⚠️ パフォーマンス監視 (スキップ)")
            
            # テクスチャマネージャー
            if TextureManager:
                self.texture_manager = TextureManager(self.ctx, self.config)
                self.logger.info("  ✅ テクスチャ管理")
            else:
                self.logger.info("  ⚠️ テクスチャ管理 (スキップ)")
            
            self.logger.info("✅ コアコンポーネント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ コンポーネント初期化失敗: {e}")
            return False
    
    def _initialize_camera(self) -> bool:
        """カメラシステム初期化"""
        self.logger.info("📹 カメラシステム初期化中...")
        
        try:
            if CameraManager:
                # カメラマネージャー初期化
                self.camera = CameraManager(self.config)
                
                if self.camera.initialize():
                    if self.camera.start_streaming():
                        self.camera_enabled = True
                        self.logger.info("✅ カメラシステム初期化完了")
                    else:
                        self.logger.warning("⚠️ カメラストリーミング開始失敗")
                else:
                    self.logger.warning("⚠️ カメラ初期化失敗（シミュレーションモード）")
            else:
                self.logger.info("⚠️ カメラマネージャー (スキップ)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ カメラシステム初期化失敗: {e}")
            return False
    
    def _initialize_ai(self) -> bool:
        """AI システム初期化"""
        self.logger.info("🤖 AI システム初期化中...")
        
        try:
            if MediaPipeProcessor:
                # MediaPipe プロセッサー
                self.mediapipe_processor = MediaPipeProcessor(self.config)
                self.logger.info("  ✅ MediaPipe処理")
            else:
                self.logger.info("  ⚠️ MediaPipe処理 (スキップ)")
            
            if EmotionAnalyzer:
                # 感情認識エンジン
                self.emotion_analyzer = EmotionAnalyzer(self.config)
                self.logger.info("  ✅ 感情認識")
            else:
                self.logger.info("  ⚠️ 感情認識 (スキップ)")
            
            self.logger.info("✅ AI システム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ AI システム初期化失敗: {e}")
            return False
    
    def _initialize_effects(self) -> bool:
        """エフェクトシステム初期化"""
        self.logger.info("✨ エフェクトシステム初期化中...")
        
        try:
            if EmotionEffectManager:
                # 感情エフェクト管理
                self.emotion_effects = EmotionEffectManager(self.ctx, self.config)
                self.logger.info("  ✅ 感情エフェクト")
            else:
                self.logger.info("  ⚠️ 感情エフェクト (スキップ)")
            
            self.logger.info("✅ エフェクトシステム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ エフェクトシステム初期化失敗: {e}")
            return False
    
    def _initialize_rendering(self) -> bool:
        """レンダリングシステム初期化"""
        self.logger.info("🎨 レンダリングシステム初期化中...")
        
        try:
            # シェーダープログラム作成
            self._create_shader_programs()
            
            # ジオメトリ作成
            self._create_geometry()
            
            self.logger.info("✅ レンダリングシステム初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ レンダリングシステム初期化失敗: {e}")
            return False
    
    def _create_shader_programs(self):
        """シェーダープログラム作成"""
        # 基本頂点シェーダー
        vertex_shader = """
        #version 430 core
        layout(location = 0) in vec2 position;
        out vec2 uv;
        
        void main() {
            uv = position * 0.5 + 0.5;
            gl_Position = vec4(position, 0.0, 1.0);
        }
        """
        
        # 基本フラグメントシェーダー
        basic_fragment_shader = """
        #version 430 core
        in vec2 uv;
        out vec4 fragColor;
        
        uniform float u_time;
        uniform vec3 u_color;
        
        void main() {
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            // 水面効果（基本版）
            float wave = sin(dist * 20.0 - u_time * 3.0) * 0.1;
            vec3 color = u_color + vec3(wave);
            
            fragColor = vec4(color, 1.0);
        }
        """
        
        # カメラ用フラグメントシェーダー
        camera_fragment_shader = """
        #version 430 core
        in vec2 uv;
        out vec4 fragColor;
        
        uniform sampler2D u_camera_texture;
        uniform float u_time;
        uniform bool u_camera_enabled;
        
        void main() {
            if (u_camera_enabled) {
                vec3 camera_color = texture(u_camera_texture, uv).rgb;
                
                // 水面効果適用
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(uv, center);
                vec2 wave_offset = vec2(
                    sin(dist * 15.0 - u_time * 2.0) * 0.01,
                    cos(dist * 12.0 - u_time * 2.5) * 0.01
                );
                
                vec3 distorted = texture(u_camera_texture, uv + wave_offset).rgb;
                fragColor = vec4(distorted, 1.0);
            } else {
                // デフォルト水面
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(uv, center);
                float wave = sin(dist * 20.0 - u_time * 3.0) * 0.1;
                vec3 color = vec3(0.3, 0.6, 1.0) + vec3(wave);
                fragColor = vec4(color, 1.0);
            }
        }
        """
        
        # 感情エフェクト用フラグメントシェーダー
        emotion_fragment_shader = """
        #version 430 core
        in vec2 uv;
        out vec4 fragColor;
        
        uniform sampler2D u_camera_texture;
        uniform float u_time;
        uniform bool u_camera_enabled;
        uniform vec3 u_emotion_color;
        uniform float u_emotion_intensity;
        
        void main() {
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            if (u_camera_enabled) {
                // カメラ映像 + 感情エフェクト
                vec2 wave_offset = vec2(
                    sin(dist * 15.0 - u_time * 2.0) * 0.01 * u_emotion_intensity,
                    cos(dist * 12.0 - u_time * 2.5) * 0.01 * u_emotion_intensity
                );
                
                vec3 camera_color = texture(u_camera_texture, uv + wave_offset).rgb;
                vec3 emotion_tint = u_emotion_color * u_emotion_intensity * 0.3;
                
                fragColor = vec4(camera_color + emotion_tint, 1.0);
            } else {
                // 感情色ベースの水面
                float wave = sin(dist * 20.0 - u_time * 3.0) * 0.1 * u_emotion_intensity;
                vec3 color = u_emotion_color + vec3(wave);
                fragColor = vec4(color, 1.0);
            }
        }
        """
        
        # プログラム作成
        try:
            self.basic_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=basic_fragment_shader
            )
            
            self.camera_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=camera_fragment_shader
            )
            
            self.emotion_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=emotion_fragment_shader
            )
            
            self.logger.info("  ✅ シェーダープログラム作成完了")
            
        except Exception as e:
            self.logger.error(f"  ❌ シェーダープログラム作成失敗: {e}")
            raise
    
    def _create_geometry(self):
        """ジオメトリ作成"""
        # フルスクリーンクワッド
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        
        # VBO/VAO作成
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        # すべてのプログラムでVAO作成
        self.quad_vao = self.ctx.vertex_array(
            self.emotion_program, [(vbo, '2f', 'position')], ibo
        )
        
        self.logger.info("  ✅ ジオメトリ作成完了")
    
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
            while self.running:
                self._update_frame()
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ ユーザーによる終了")
        except Exception as e:
            self.logger.error(f"❌ 実行エラー: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def _update_frame(self):
        """フレーム更新"""
        frame_start = time.time()
        
        # イベント処理
        self._handle_events()
        
        # AI処理
        self._process_ai_frame()
        
        # カメラフレーム処理
        self._process_camera_frame()
        
        # 描画
        self._render_frame()
        
        # バッファスワップ
        pygame.display.flip()
        
        # フレームレート制御
        target_fps = self.config.get('rendering.target_fps', 60)
        self.clock.tick(target_fps)
        
        # パフォーマンス統計
        frame_time = time.time() - frame_start
        if self.performance_monitor:
            self.performance_monitor.update_frame_stats(frame_time)
        
        # 統計表示
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 5.0:
            self._print_stats()
            self.last_time = current_time
    
    def _handle_events(self):
        """イベント処理（GLFW→Pygame変換）"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)
            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize(event)
    
    def _handle_keydown(self, event):
        """キーボードイベント処理（GLFWコールバック互換）"""
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_F11:
            # フルスクリーン切り替え
            pygame.display.toggle_fullscreen()
        elif event.key == pygame.K_p and self.performance_monitor:
            # パフォーマンス統計表示
            stats = self.performance_monitor.get_performance_summary()
            print(stats)
    
    def _handle_resize(self, event):
        """リサイズイベント処理（GLFWコールバック互換）"""
        self.ctx.viewport = (0, 0, event.w, event.h)
        self.logger.info(f"ウィンドウリサイズ: {event.w}x{event.h}")
    
    def _process_ai_frame(self):
        """AI処理フレーム"""
        if not self.ai_enabled:
            return
        
        # MediaPipe処理
        if self.mediapipe_processor and self.camera and self.camera_enabled:
            frame = self.camera.get_frame()
            if frame is not None:
                # MediaPipe処理実行
                mp_results = self.mediapipe_processor.process_frame(frame)
                
                # 感情分析
                if self.emotion_analyzer and mp_results:
                    emotion_results = self.emotion_analyzer.analyze_frame(frame, mp_results)
                    
                    # エフェクトシステムに感情情報渡す
                    if self.emotion_effects and emotion_results:
                        self.emotion_effects.update_emotion(emotion_results)
    
    def _process_camera_frame(self):
        """カメラフレーム処理（修正版）"""
        if not self.camera_enabled or not self.camera:
            return
    
        try:
            frame = self.camera.get_frame()
            if frame is not None:
                # フレーム前処理
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # BGR → RGB 変換
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                    # フレームサイズ調整
                    target_height, target_width = 1080, 1920
                    if frame_rgb.shape[:2] != (target_height, target_width):
                        frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
                
                    # テクスチャアップロード
                    if self.texture_manager:
                        self.texture_manager.upload_camera_frame(frame_rgb)
                        self.logger.debug(f"カメラフレーム更新: {frame_rgb.shape}")
                else:
                    self.logger.warning(f"未対応フレーム形状: {frame.shape}")
                
        except Exception as e:
            self.logger.error(f"カメラフレーム処理エラー: {e}")
    
    def _create_camera_texture_fixed(self):
        """カメラテクスチャ作成（修正版）"""
        if not self.ctx:
            return None
    
        try:
            # RGB テクスチャ作成
            camera_texture = self.ctx.texture((1920, 1080), 3)
            camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            camera_texture.wrap_x = moderngl.CLAMP_TO_EDGE
            camera_texture.wrap_y = moderngl.CLAMP_TO_EDGE
        
            # 初期データ（黒画面）
            import numpy as np
            initial_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
            camera_texture.write(initial_data.tobytes())
        
            return camera_texture
        
        except Exception as e:
            self.logger.error(f"カメラテクスチャ作成エラー: {e}")
            return None

    def _render_frame(self):
        """フレーム描画"""
        # 画面クリア
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        current_time = time.time()
        
        # プログラム選択
        if self.emotion_effects and self.effects_enabled:
            # 感情エフェクト描画
            program = self.emotion_program
            program['u_time'] = current_time
            program['u_camera_enabled'] = self.camera_enabled
            
            # 感情情報設定
            emotion_data = self.emotion_effects.get_current_emotion_data()
            if emotion_data:
                program['u_emotion_color'] = emotion_data.get('color', (0.3, 0.6, 1.0))
                program['u_emotion_intensity'] = emotion_data.get('intensity', 0.5)
            else:
                program['u_emotion_color'] = (0.3, 0.6, 1.0)
                program['u_emotion_intensity'] = 0.5
                
        elif self.camera_enabled and self.texture_manager:
            # カメラ描画
            program = self.camera_program
            program['u_time'] = current_time
            program['u_camera_enabled'] = True
        else:
            # 基本描画
            program = self.basic_program
            program['u_time'] = current_time
            program['u_color'] = (0.3, 0.6, 1.0)
        
        # テクスチャバインド
        if self.camera_enabled and self.texture_manager:
            camera_texture = self.texture_manager.get_camera_texture()
            if camera_texture:
                camera_texture.use(0)
                if hasattr(program, 'u_camera_texture'):
                    program['u_camera_texture'] = 0
        
        # レンダリング
        self.quad_vao.render()
    
    def _print_stats(self):
        """統計情報表示"""
        if self.performance_monitor:
            stats = self.performance_monitor.get_current_stats()
            gpu_name = self.gpu_info.get('renderer', 'Unknown')[:50]  # 長い名前を短縮
            self.logger.info(
                f"FPS: {stats.get('average_fps', 0):.1f}, "
                f"フレーム: {self.frame_count}, "
                f"GPU: {gpu_name}"
            )
        else:
            fps = self.clock.get_fps()
            gpu_name = self.gpu_info.get('renderer', 'Unknown')[:50]
            self.logger.info(f"FPS: {fps:.1f}, フレーム: {self.frame_count}, GPU: {gpu_name}")
    
    def cleanup(self):
        """リソース解放"""
        self.logger.info("🧹 リソース解放中...")
        
        # パフォーマンス監視停止
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        # カメラ停止
        if self.camera:
            self.camera.cleanup()
        
        # GLコンテキスト解放
        if self.gl_context:
            self.gl_context.cleanup_all()
        
        # Pygame終了
        pygame.quit()
        
        self.logger.info("✅ リソース解放完了")

def main():
    """エントリーポイント"""
    print("🌊 Aqua Mirror - Pygame + ModernGL完全版 (Windows)")
    print("=" * 60)
    
    # GPU環境変数確認
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        print("⚠️ GPU環境変数が設定されていません")
        print("gpu_enabler_windows.py を先に実行することを推奨します")
    
    # アプリケーション実行
    try:
        app = PygameModernGLApp()
        
        if app.initialize():
            print("✅ 初期化完了、メインループ開始")
            print("操作方法:")
            print("  ESC: 終了")
            print("  F11: フルスクリーン切り替え")
            print("  P: パフォーマンス統計表示")
            print("=" * 60)
            
            app.run()
            return 0
        else:
            print("❌ アプリケーション初期化失敗")
            input("Enterキーを押して終了...")
            return 1
            
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        traceback.print_exc()
        input("Enterキーを押して終了...")
        return 1

if __name__ == "__main__":
    sys.exit(main())
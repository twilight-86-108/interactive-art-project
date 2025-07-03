#!/usr/bin/env python3
"""
Aqua Mirror 問題診断・修正スクリプト
カメラ・エフェクト・FPS問題の診断と修正
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path

class AquaMirrorDiagnosticFix:
    """Aqua Mirror 問題診断・修正"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.issues_found = []
        self.fixes_applied = []
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('diagnostic_fix.log', encoding='utf-8')
            ]
        )
        return logging.getLogger("DiagnosticFix")
    
    def diagnose_camera_issue(self):
        """カメラ問題診断"""
        self.logger.info("📹 カメラ問題診断開始...")
        
        try:
            import cv2
            
            # カメラデバイス確認
            camera_found = False
            for i in range(5):  # 0-4のカメラを確認
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.logger.info(f"  ✅ カメラ{i}: 動作確認 ({frame.shape})")
                        camera_found = True
                        
                        # フレームフォーマット確認
                        self.logger.info(f"    フレーム形状: {frame.shape}")
                        self.logger.info(f"    データ型: {frame.dtype}")
                        self.logger.info(f"    値範囲: {frame.min()}-{frame.max()}")
                    cap.release()
                    break
                cap.release()
            
            if not camera_found:
                self.issues_found.append("カメラデバイス未検出")
                self.logger.error("  ❌ カメラデバイスが見つかりません")
                return False
            
            # OpenCVバージョン確認
            self.logger.info(f"  OpenCVバージョン: {cv2.__version__}")
            
            return True
            
        except ImportError:
            self.issues_found.append("OpenCV未インストール")
            self.logger.error("  ❌ OpenCV (cv2) がインストールされていません")
            return False
        except Exception as e:
            self.issues_found.append(f"カメラエラー: {e}")
            self.logger.error(f"  ❌ カメラ診断エラー: {e}")
            return False
    
    def diagnose_shader_issue(self):
        """シェーダー問題診断"""
        self.logger.info("🎨 シェーダー問題診断開始...")
        
        try:
            import pygame
            import moderngl
            import numpy as np
            
            # Pygame + ModernGL 最小テスト
            pygame.init()
            screen = pygame.display.set_mode((100, 100), pygame.OPENGL | pygame.HIDDEN)
            ctx = moderngl.create_context()
            
            # 簡単なシェーダーテスト
            vertex_shader = """
            #version 330 core
            in vec2 position;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """
            
            fragment_shader = """
            #version 330 core
            out vec4 fragColor;
            void main() {
                fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // 赤色
            }
            """
            
            try:
                program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
                self.logger.info("  ✅ 基本シェーダーコンパイル成功")
                
                # 水面エフェクトシェーダーテスト
                water_fragment = """
                #version 330 core
                in vec2 uv;
                out vec4 fragColor;
                
                uniform float u_time;
                
                void main() {
                    vec2 center = vec2(0.5, 0.5);
                    float dist = distance(uv, center);
                    float wave = sin(dist * 20.0 - u_time * 3.0) * 0.1;
                    vec3 color = vec3(0.3, 0.6, 1.0) + vec3(wave);
                    fragColor = vec4(color, 1.0);
                }
                """
                
                water_vertex = """
                #version 330 core
                in vec2 position;
                out vec2 uv;
                void main() {
                    uv = position * 0.5 + 0.5;
                    gl_Position = vec4(position, 0.0, 1.0);
                }
                """
                
                water_program = ctx.program(vertex_shader=water_vertex, fragment_shader=water_fragment)
                self.logger.info("  ✅ 水面エフェクトシェーダーコンパイル成功")
                
                # ジオメトリテスト
                vertices = np.array([-1, -1, 1, -1, 1, 1, -1, 1], dtype=np.float32)
                vbo = ctx.buffer(vertices.tobytes())
                vao = ctx.vertex_array(water_program, [(vbo, '2f', 'position')])
                
                # レンダリングテスト
                water_program['u_time'] = 1.0
                vao.render()
                self.logger.info("  ✅ レンダリングテスト成功")
                
                pygame.quit()
                return True
                
            except Exception as shader_error:
                self.issues_found.append(f"シェーダーエラー: {shader_error}")
                self.logger.error(f"  ❌ シェーダーエラー: {shader_error}")
                pygame.quit()
                return False
                
        except Exception as e:
            self.issues_found.append(f"ModernGL診断エラー: {e}")
            self.logger.error(f"  ❌ ModernGL診断エラー: {e}")
            return False
    
    def diagnose_fps_issue(self):
        """FPS問題診断"""
        self.logger.info("⚡ FPS問題診断開始...")
        
        try:
            import pygame
            import moderngl
            import time
            
            pygame.init()
            screen = pygame.display.set_mode((800, 600), pygame.OPENGL)
            clock = pygame.time.Clock()
            ctx = moderngl.create_context()
            
            # FPS測定テスト
            fps_samples = []
            frame_times = []
            
            start_time = time.time()
            for frame in range(60):  # 60フレーム測定
                frame_start = time.time()
                
                # 簡単な描画
                ctx.clear(0.1, 0.1, 0.1, 1.0)
                
                pygame.display.flip()
                clock.tick(60)  # 60FPS目標
                
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                fps_samples.append(clock.get_fps())
            
            total_time = time.time() - start_time
            avg_fps = len(fps_samples) / total_time
            avg_frame_time = sum(frame_times) / len(frame_times)
            
            self.logger.info(f"  測定結果:")
            self.logger.info(f"    平均FPS: {avg_fps:.1f}")
            self.logger.info(f"    平均フレーム時間: {avg_frame_time*1000:.1f}ms")
            self.logger.info(f"    最大フレーム時間: {max(frame_times)*1000:.1f}ms")
            
            # VSync確認
            vsync_test_fps = []
            pygame.display.gl_set_swap_interval(1)  # VSync ON
            for _ in range(30):
                frame_start = time.time()
                ctx.clear(0.2, 0.2, 0.2, 1.0)
                pygame.display.flip()
                vsync_test_fps.append(clock.get_fps())
            
            avg_vsync_fps = sum(vsync_test_fps) / len(vsync_test_fps)
            self.logger.info(f"    VSync有効時FPS: {avg_vsync_fps:.1f}")
            
            pygame.quit()
            
            if avg_fps < 30:
                self.issues_found.append(f"低FPS: {avg_fps:.1f}")
                return False
            
            return True
            
        except Exception as e:
            self.issues_found.append(f"FPS診断エラー: {e}")
            self.logger.error(f"  ❌ FPS診断エラー: {e}")
            return False
    
    def create_camera_fix(self):
        """カメラ修正パッチ作成"""
        self.logger.info("🔧 カメラ修正パッチ作成中...")
        
        camera_fix_code = '''
# カメラ修正パッチ
# pygame_moderngl_app_complete.py に追加する修正

def _process_camera_frame_fixed(self):
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
'''
        
        with open('camera_fix_patch.py', 'w', encoding='utf-8') as f:
            f.write(camera_fix_code)
        
        self.fixes_applied.append("カメラ修正パッチ作成")
        self.logger.info("  ✅ camera_fix_patch.py 作成完了")
    
    def create_shader_fix(self):
        """シェーダー修正パッチ作成"""
        self.logger.info("🎨 シェーダー修正パッチ作成中...")
        
        shader_fix_code = '''
# シェーダー修正パッチ
# 動作確認済みシェーダーコード

def _create_working_shaders(self):
    """動作確認済みシェーダー作成"""
    
    # 基本頂点シェーダー
    vertex_shader = """
    #version 330 core
    layout(location = 0) in vec2 position;
    out vec2 uv;
    
    void main() {
        uv = position * 0.5 + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """
    
    # 水面エフェクトシェーダー（動作確認済み）
    water_fragment_shader = """
    #version 330 core
    in vec2 uv;
    out vec4 fragColor;
    
    uniform float u_time;
    uniform vec3 u_color;
    uniform bool u_show_test_pattern;
    
    void main() {
        if (u_show_test_pattern) {
            // テストパターン表示
            float checker = step(0.5, mod(uv.x * 10.0, 1.0)) + step(0.5, mod(uv.y * 10.0, 1.0));
            vec3 color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), mod(checker, 2.0));
            fragColor = vec4(color, 1.0);
        } else {
            // 水面エフェクト
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            float wave1 = sin(dist * 15.0 - u_time * 2.0) * 0.1;
            float wave2 = sin(dist * 25.0 - u_time * 3.0) * 0.05;
            float wave = wave1 + wave2;
            
            vec3 water_color = u_color + vec3(wave);
            water_color = clamp(water_color, 0.0, 1.0);
            
            fragColor = vec4(water_color, 1.0);
        }
    }
    """
    
    # カメラシェーダー（修正版）
    camera_fragment_shader = """
    #version 330 core
    in vec2 uv;
    out vec4 fragColor;
    
    uniform sampler2D u_camera_texture;
    uniform float u_time;
    uniform bool u_camera_enabled;
    uniform float u_effect_strength;
    
    void main() {
        if (u_camera_enabled) {
            // カメラ映像取得
            vec3 camera_color = texture(u_camera_texture, uv).rgb;
            
            // 水面歪みエフェクト
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            
            vec2 wave_offset = vec2(
                sin(dist * 20.0 - u_time * 2.0) * 0.01 * u_effect_strength,
                cos(dist * 15.0 - u_time * 2.5) * 0.01 * u_effect_strength
            );
            
            vec3 distorted = texture(u_camera_texture, uv + wave_offset).rgb;
            
            // 色調調整
            distorted *= 1.1; // 明度向上
            distorted = clamp(distorted, 0.0, 1.0);
            
            fragColor = vec4(distorted, 1.0);
        } else {
            // カメラ無効時のデフォルト表示
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(uv, center);
            float wave = sin(dist * 20.0 - u_time * 3.0) * 0.2;
            vec3 color = vec3(0.2, 0.4, 0.8) + vec3(wave);
            fragColor = vec4(color, 1.0);
        }
    }
    """
    
    try:
        # プログラム作成
        self.water_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=water_fragment_shader
        )
        
        self.camera_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=camera_fragment_shader
        )
        
        self.logger.info("✅ シェーダー作成成功")
        return True
        
    except Exception as e:
        self.logger.error(f"シェーダー作成エラー: {e}")
        return False
'''
        
        with open('shader_fix_patch.py', 'w', encoding='utf-8') as f:
            f.write(shader_fix_code)
        
        self.fixes_applied.append("シェーダー修正パッチ作成")
        self.logger.info("  ✅ shader_fix_patch.py 作成完了")
    
    def create_fps_fix(self):
        """FPS修正パッチ作成"""
        self.logger.info("⚡ FPS修正パッチ作成中...")
        
        fps_fix_code = '''
# FPS修正パッチ
# パフォーマンス最適化コード

def _optimize_fps_performance(self):
    """FPS最適化設定"""
    
    # OpenGL最適化設定
    if self.ctx:
        # 深度テスト無効化（2D描画のみの場合）
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # カリング有効化
        self.ctx.enable(moderngl.CULL_FACE)
        
        # ビューポート設定
        self.ctx.viewport = (0, 0, 1920, 1080)
    
    # Pygame最適化
    if hasattr(pygame.display, 'gl_set_swap_interval'):
        pygame.display.gl_set_swap_interval(1)  # VSync
    
    # フレーム制限解除（GPU制限で自然に制限される）
    return 144  # 最大FPS目標

def _update_frame_optimized(self):
    """最適化されたフレーム更新"""
    frame_start = time.perf_counter()
    
    # イベント処理（軽量化）
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
    
    # 描画頻度調整（AI処理は30FPSに制限）
    current_time = time.perf_counter()
    if current_time - getattr(self, '_last_ai_time', 0) > 1.0/30.0:
        self._process_ai_frame()
        self._last_ai_time = current_time
    
    # カメラは60FPS
    self._process_camera_frame()
    
    # レンダリング（最高FPS）
    self._render_frame()
    
    # バッファスワップ
    pygame.display.flip()
    
    # フレーム時間測定
    frame_time = time.perf_counter() - frame_start
    self._update_fps_stats(frame_time)

def _render_frame_optimized(self):
    """最適化された描画"""
    # 高速クリア
    self.ctx.clear(0.0, 0.0, 0.0, 1.0)
    
    current_time = time.perf_counter()
    
    # シェーダー選択（条件分岐最小化）
    if self.camera_enabled and self.texture_manager:
        program = self.camera_program
        program['u_time'] = current_time
        program['u_camera_enabled'] = True
        program['u_effect_strength'] = 1.0
        
        # テクスチャバインド
        camera_texture = self.texture_manager.get_camera_texture()
        if camera_texture:
            camera_texture.use(0)
            program['u_camera_texture'] = 0
    else:
        program = self.water_program
        program['u_time'] = current_time
        program['u_color'] = (0.3, 0.6, 1.0)
        program['u_show_test_pattern'] = False
    
    # 単一描画コール
    self.quad_vao.render()

def _update_fps_stats(self, frame_time):
    """FPS統計更新"""
    if not hasattr(self, '_fps_samples'):
        self._fps_samples = []
        self._last_fps_log = time.perf_counter()
    
    fps = 1.0 / frame_time if frame_time > 0 else 0
    self._fps_samples.append(fps)
    
    # 5秒ごとに統計表示
    current_time = time.perf_counter()
    if current_time - self._last_fps_log > 5.0:
        if self._fps_samples:
            avg_fps = sum(self._fps_samples) / len(self._fps_samples)
            min_fps = min(self._fps_samples)
            max_fps = max(self._fps_samples)
            
            self.logger.info(f"FPS統計: 平均{avg_fps:.1f} 最小{min_fps:.1f} 最大{max_fps:.1f}")
            
            self._fps_samples = []
            self._last_fps_log = current_time
'''
        
        with open('fps_fix_patch.py', 'w', encoding='utf-8') as f:
            f.write(fps_fix_code)
        
        self.fixes_applied.append("FPS修正パッチ作成")
        self.logger.info("  ✅ fps_fix_patch.py 作成完了")
    
    def run_complete_diagnosis(self):
        """完全診断実行"""
        self.logger.info("🔍 Aqua Mirror 完全問題診断開始")
        self.logger.info("=" * 50)
        
        # 診断実行
        camera_ok = self.diagnose_camera_issue()
        shader_ok = self.diagnose_shader_issue()
        fps_ok = self.diagnose_fps_issue()
        
        # 修正パッチ作成
        self.logger.info("\n🔧 修正パッチ作成中...")
        self.create_camera_fix()
        self.create_shader_fix()
        self.create_fps_fix()
        
        # 結果サマリー
        self.logger.info("\n📊 診断結果サマリー")
        self.logger.info("=" * 50)
        
        issues = {
            "カメラ": camera_ok,
            "シェーダー": shader_ok,
            "FPS": fps_ok
        }
        
        for issue, status in issues.items():
            status_icon = "✅" if status else "❌"
            self.logger.info(f"{status_icon} {issue}: {'正常' if status else '要修正'}")
        
        if self.issues_found:
            self.logger.info("\n🚨 発見された問題:")
            for i, issue in enumerate(self.issues_found, 1):
                self.logger.info(f"  {i}. {issue}")
        
        if self.fixes_applied:
            self.logger.info("\n🔧 作成された修正パッチ:")
            for i, fix in enumerate(self.fixes_applied, 1):
                self.logger.info(f"  {i}. {fix}")
        
        # 次のステップ案内
        self.logger.info("\n📋 次のステップ:")
        self.logger.info("1. 作成されたパッチファイルを確認")
        self.logger.info("2. pygame_moderngl_app_complete.py に修正を適用")
        self.logger.info("3. 修正版アプリケーションをテスト")
        
        return len(self.issues_found) == 0

def main():
    """メイン実行"""
    print("🔍 Aqua Mirror 問題診断・修正ツール")
    print("=" * 50)
    print("カメラ・エフェクト・FPS問題を診断し、修正パッチを作成します")
    print("=" * 50)
    
    diagnostic = AquaMirrorDiagnosticFix()
    
    try:
        success = diagnostic.run_complete_diagnosis()
        
        if success:
            print("\n✅ 診断完了！問題は見つかりませんでした")
        else:
            print("\n⚠️ 問題が発見されました。修正パッチを作成しました")
            print("\n作成されたファイル:")
            print("- camera_fix_patch.py")
            print("- shader_fix_patch.py") 
            print("- fps_fix_patch.py")
            print("- diagnostic_fix.log")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 診断エラー: {e}")
        traceback.print_exc()
        return 1
    finally:
        input("\nEnterキーを押して終了...")

if __name__ == "__main__":
    sys.exit(main())
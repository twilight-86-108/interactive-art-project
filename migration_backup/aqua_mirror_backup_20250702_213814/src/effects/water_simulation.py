"""
基本水面シミュレーション - Week 2版
"""

import moderngl
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
import math

class WaveSource:
    """波源データクラス"""
    def __init__(self, position: Tuple[float, float], intensity: float):
        self.position = position
        self.intensity = intensity
        self.birth_time = time.time()
        self.radius = 0.0

class BasicWaterSimulation:
    """基本水面シミュレーション"""
    
    def __init__(self, ctx: moderngl.Context, config: Dict[str, Any]):
        self.logger = logging.getLogger("BasicWaterSimulation")
        self.ctx = ctx
        self.config = config
        
        # 水面設定
        self.resolution = 128  # 低解像度で開始
        self.water_size = 2.0
        self.wave_sources: List[WaveSource] = []
        self.max_wave_sources = 5
        
        # GPU リソース
        self.water_program: Optional[moderngl.Program] = None
        self.water_vao: Optional[moderngl.VertexArray] = None
        
        # 水面メッシュ
        self.water_vertices: Optional[np.ndarray] = None
        self.water_indices: Optional[np.ndarray] = None
        
        self.logger.info(f"🌊 基本水面シミュレーション初期化 (解像度: {self.resolution})")
    
    def initialize(self) -> bool:
        """水面シミュレーション初期化"""
        try:
            # 水面メッシュ生成
            if not self._create_water_mesh():
                return False
            
            # シェーダー作成
            if not self._create_water_shaders():
                return False
            
            # GPU リソース作成
            if not self._create_gpu_resources():
                return False
            
            self.logger.info("✅ 基本水面シミュレーション初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 水面シミュレーション初期化失敗: {e}")
            return False
    
    def _create_water_mesh(self) -> bool:
        """水面メッシュ生成"""
        try:
            vertices = []
            indices = []
            
            # グリッド生成
            for y in range(self.resolution + 1):
                for x in range(self.resolution + 1):
                    # 位置 (-1 to 1)
                    pos_x = (x / self.resolution) * 2.0 - 1.0
                    pos_z = (y / self.resolution) * 2.0 - 1.0
                    pos_y = 0.0
                    
                    # テクスチャ座標
                    tex_u = x / self.resolution
                    tex_v = y / self.resolution
                    
                    vertices.extend([pos_x, pos_y, pos_z, tex_u, tex_v])
            
            # インデックス生成
            for y in range(self.resolution):
                for x in range(self.resolution):
                    top_left = y * (self.resolution + 1) + x
                    top_right = top_left + 1
                    bottom_left = (y + 1) * (self.resolution + 1) + x
                    bottom_right = bottom_left + 1
                    
                    # 三角形
                    indices.extend([top_left, bottom_left, top_right])
                    indices.extend([top_right, bottom_left, bottom_right])
            
            self.water_vertices = np.array(vertices, dtype=np.float32)
            self.water_indices = np.array(indices, dtype=np.uint32)
            
            self.logger.info(f"✅ 水面メッシュ生成: {len(vertices)//5}頂点, {len(indices)//3}三角形")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 水面メッシュ生成失敗: {e}")
            return False
    
    def _create_water_shaders(self) -> bool:
        """水面シェーダー作成"""
        try:
            # 頂点シェーダー
            vertex_shader = """
            #version 410 core
            
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 texcoord;
            
            uniform mat4 mvp_matrix;
            uniform float u_time;
            uniform vec2 u_wave_positions[5];
            uniform float u_wave_intensities[5];
            uniform int u_active_waves;
            
            out vec2 v_texcoord;
            out float v_wave_height;
            
            float wave_function(vec2 pos, vec2 source, float intensity, float time) {
                float distance = length(pos - source);
                float amplitude = intensity * exp(-distance * 2.0);
                float phase = distance * 8.0 - time * 3.0;
                return amplitude * sin(phase);
            }
            
            void main() {
                vec2 world_pos = position.xz;
                float wave_height = 0.0;
                
                for(int i = 0; i < u_active_waves; i++) {
                    wave_height += wave_function(world_pos, u_wave_positions[i], u_wave_intensities[i], u_time);
                }
                
                vec3 displaced_pos = position;
                displaced_pos.y += wave_height * 0.1;
                
                v_texcoord = texcoord;
                v_wave_height = wave_height;
                
                gl_Position = mvp_matrix * vec4(displaced_pos, 1.0);
            }
            """
            
            # フラグメントシェーダー
            fragment_shader = """
            #version 410 core
            
            in vec2 v_texcoord;
            in float v_wave_height;
            
            uniform sampler2D u_camera_texture;
            uniform vec3 u_water_color;
            uniform float u_time;
            
            out vec4 fragColor;
            
            void main() {
                // 波による画面歪み
                vec2 distorted_uv = v_texcoord + vec2(v_wave_height * 0.02);
                distorted_uv = clamp(distorted_uv, 0.01, 0.99);
                
                // カメラ画像取得
                vec3 camera_color = texture(u_camera_texture, distorted_uv).rgb;
                
                // 水の色とブレンド
                vec3 water_blend = mix(camera_color, u_water_color, 0.3);
                
                // 波による反射効果
                float wave_intensity = abs(v_wave_height);
                water_blend += vec3(wave_intensity) * 0.2;
                
                fragColor = vec4(water_blend, 0.8);
            }
            """
            
            # プログラム作成
            self.water_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 水面シェーダー作成失敗: {e}")
            return False
    
    def _create_gpu_resources(self) -> bool:
        """GPU リソース作成"""
        try:
            # バッファ作成
            vertex_buffer = self.ctx.buffer(self.water_vertices.tobytes())
            index_buffer = self.ctx.buffer(self.water_indices.tobytes())
            
            # VAO作成
            self.water_vao = self.ctx.vertex_array(
                self.water_program,
                [(vertex_buffer, '3f 2f', 'position', 'texcoord')],
                index_buffer
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 水面GPU リソース作成失敗: {e}")
            return False
    
    def add_wave_source(self, position: Tuple[float, float], intensity: float = 1.0):
        """波源追加"""
        try:
            # 古い波源削除
            if len(self.wave_sources) >= self.max_wave_sources:
                self.wave_sources.pop(0)
            
            # 新しい波源追加
            wave_source = WaveSource(position, intensity)
            self.wave_sources.append(wave_source)
            
            self.logger.debug(f"💧 波源追加: {position}, 強度: {intensity}")
            
        except Exception as e:
            self.logger.error(f"❌ 波源追加失敗: {e}")
    
    def update(self, dt: float):
        """水面更新"""
        try:
            current_time = time.time()
            
            # 古い波源削除
            self.wave_sources = [
                source for source in self.wave_sources 
                if current_time - source.birth_time < 3.0
            ]
            
            # 波源減衰
            for source in self.wave_sources:
                age = current_time - source.birth_time
                source.intensity *= 0.995  # 減衰
                source.radius = age * 0.5
            
        except Exception as e:
            self.logger.error(f"❌ 水面更新失敗: {e}")
    
    def render(self, camera_texture: moderngl.Texture):
        """水面描画"""
        try:
            if not self.water_program or not self.water_vao:
                return
            
            # MVP行列（簡易版）
            mvp_matrix = np.eye(4, dtype=np.float32)
            
            # ユニフォーム設定
            self.water_program['mvp_matrix'].write(mvp_matrix.tobytes())
            self.water_program['u_time'] = time.time()
            
            # 波源データ設定
            positions = []
            intensities = []
            
            for i in range(5):
                if i < len(self.wave_sources):
                    source = self.wave_sources[i]
                    positions.extend(source.position)
                    intensities.append(source.intensity)
                else:
                    positions.extend([0.0, 0.0])
                    intensities.append(0.0)
            
            self.water_program['u_wave_positions'] = positions
            self.water_program['u_wave_intensities'] = intensities
            self.water_program['u_active_waves'] = len(self.wave_sources)
            
            # 水の色
            self.water_program['u_water_color'] = (0.1, 0.3, 0.5)
            
            # カメラテクスチャ
            if camera_texture:
                camera_texture.use(location=0)
                self.water_program['u_camera_texture'] = 0
            
            # 描画
            self.ctx.enable(moderngl.BLEND)
            self.water_vao.render()
            
        except Exception as e:
            self.logger.error(f"❌ 水面描画失敗: {e}")
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.water_vao:
                self.water_vao.release()
            self.wave_sources.clear()
            self.logger.info("✅ 水面シミュレーションリソース解放完了")
        except Exception as e:
            self.logger.error(f"❌ 水面シミュレーションリソース解放失敗: {e}")

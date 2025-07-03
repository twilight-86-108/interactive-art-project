"""
GPU加速高品質水面物理シミュレーション
"""

import moderngl
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import time
import math

class AdvancedWaterSimulation:
    """GPU加速高品質水面物理シミュレーション"""
    
    def __init__(self, ctx: moderngl.Context, config: Dict[str, Any]):
        self.logger = logging.getLogger("AdvancedWaterSimulation")
        self.ctx = ctx
        self.config = config
        
        # 高解像度設定
        self.resolution = config.get('rendering.water_simulation_resolution', 512)
        self.water_size = 4.0
        
        # GPU テクスチャ
        self.height_texture_current: Optional[moderngl.Texture] = None
        self.height_texture_previous: Optional[moderngl.Texture] = None
        self.velocity_texture: Optional[moderngl.Texture] = None
        self.normal_texture: Optional[moderngl.Texture] = None
        
        # コンピュートシェーダー
        self.water_physics_compute: Optional[moderngl.ComputeShader] = None
        self.normal_compute: Optional[moderngl.ComputeShader] = None
        
        # 描画用リソース
        self.water_render_program: Optional[moderngl.Program] = None
        self.water_vao: Optional[moderngl.VertexArray] = None
        
        # 物理パラメータ
        self.wave_speed = 0.8
        self.damping = 0.99
        self.dt = 0.016  # 60FPS想定
        
        # 波源管理
        self.wave_sources: List[Dict] = []
        self.max_wave_sources = 8
        
        self.logger.info(f"🌊 高品質水面シミュレーション初期化 (解像度: {self.resolution}x{self.resolution})")
    
    def initialize(self) -> bool:
        """高品質水面シミュレーション初期化"""
        try:
            # GPU テクスチャ作成
            if not self._create_gpu_textures():
                return False
            
            # コンピュートシェーダー作成
            if not self._create_compute_shaders():
                return False
            
            # 描画シェーダー作成
            if not self._create_render_shaders():
                return False
            
            # 水面メッシュ作成
            if not self._create_water_mesh():
                return False
            
            # 初期化完了
            self.logger.info("✅ 高品質水面シミュレーション初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 高品質水面シミュレーション初期化失敗: {e}")
            return False
    
    def _create_gpu_textures(self) -> bool:
        """GPU テクスチャ作成"""
        try:
            # 初期化用の空データを作成
            initial_data = np.zeros((self.resolution, self.resolution, 4), dtype='f4')

            # 高さフィールド（現在フレーム）
            self.height_texture_current = self.ctx.texture(
                (self.resolution, self.resolution), 4, dtype='f4'
            )
            self.height_texture_current.write(initial_data) # 初期データを書き込む
            self.height_texture_current.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # 高さフィールド（前フレーム）
            self.height_texture_previous = self.ctx.texture(
                (self.resolution, self.resolution), 4, dtype='f4'
            )
            self.height_texture_previous.write(initial_data) # 初期データを書き込む
            self.height_texture_previous.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # 速度フィールド
            self.velocity_texture = self.ctx.texture(
                (self.resolution, self.resolution), 4, dtype='f4'
            )
            self.velocity_texture.write(initial_data) # 初期データを書き込む
            self.velocity_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # 法線テクスチャ
            self.normal_texture = self.ctx.texture(
                (self.resolution, self.resolution), 4, dtype='f4'
            )
            # 法線は初期状態で上向き(0, 1, 0)に設定
            normal_initial_data = np.zeros((self.resolution, self.resolution, 4), dtype='f4')
            normal_initial_data[:, :, 1] = 1.0 # Y
            normal_initial_data[:, :, 3] = 1.0 # Alpha
            self.normal_texture.write(normal_initial_data)
            self.normal_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            self.logger.info("✅ GPU テクスチャ作成完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ GPU テクスチャ作成失敗: {e}")
            return False
    
    def _create_compute_shaders(self) -> bool:
        """コンピュートシェーダー作成"""
        try:
            # 水面物理シミュレーション
            water_physics_source = """
            #version 430
            layout(local_size_x = 16, local_size_y = 16) in;
            
            layout(rgba32f, binding = 0) uniform image2D height_current;
            layout(rgba32f, binding = 1) uniform image2D height_previous;
            layout(rgba32f, binding = 2) uniform image2D velocity_field;
            
            uniform float u_wave_speed;
            uniform float u_damping;
            uniform float u_dt;
            uniform vec2 u_wave_sources[8];
            uniform float u_wave_intensities[8];
            uniform int u_active_sources;
            uniform float u_time;
            
            void main() {
                ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(height_current);
                
                if (coord.x >= size.x || coord.y >= size.y) return;
                
                vec2 uv = vec2(coord) / vec2(size);
                
                // 現在の高さと速度
                float height = imageLoad(height_current, coord).r;
                float velocity = imageLoad(velocity_field, coord).r;
                
                // 近隣ピクセルの高さ取得
                float h_left = 0.0, h_right = 0.0, h_up = 0.0, h_down = 0.0;
                
                if (coord.x > 0) 
                    h_left = imageLoad(height_current, coord + ivec2(-1, 0)).r;
                if (coord.x < size.x - 1) 
                    h_right = imageLoad(height_current, coord + ivec2(1, 0)).r;
                if (coord.y > 0) 
                    h_up = imageLoad(height_current, coord + ivec2(0, -1)).r;
                if (coord.y < size.y - 1) 
                    h_down = imageLoad(height_current, coord + ivec2(0, 1)).r;
                
                // 波動方程式（ラプラシアン）
                float laplacian = h_left + h_right + h_up + h_down - 4.0 * height;
                
                // 加速度計算
                float acceleration = u_wave_speed * u_wave_speed * laplacian;
                
                // 波源からの影響
                for (int i = 0; i < u_active_sources; i++) {
                    float dist = distance(uv, u_wave_sources[i]);
                    if (dist < 0.3) {
                        float wave_phase = dist * 20.0 - u_time * 8.0;
                        float wave_intensity = u_wave_intensities[i] * exp(-dist * 8.0);
                        acceleration += sin(wave_phase) * wave_intensity * 0.5;
                    }
                }
                
                // 境界条件（減衰）
                float border_factor = 1.0;
                float border_width = 0.05;
                if (uv.x < border_width || uv.x > 1.0 - border_width ||
                    uv.y < border_width || uv.y > 1.0 - border_width) {
                    border_factor = 0.7;
                }
                
                // 速度更新
                velocity += acceleration * u_dt;
                velocity *= u_damping * border_factor;
                
                // 高さ更新
                height += velocity * u_dt;
                
                // 結果保存
                imageStore(height_previous, coord, vec4(height));
                imageStore(velocity_field, coord, vec4(velocity, 0.0, 0.0, 1.0));
            }
            """
            
            self.water_physics_compute = self.ctx.compute_shader(water_physics_source)
            
            # 法線計算
            normal_compute_source = """
            #version 430
            layout(local_size_x = 16, local_size_y = 16) in;
            
            layout(rgba32f, binding = 0) uniform image2D height_field;
            layout(rgba32f, binding = 1) uniform image2D normal_field;
            
            uniform float u_normal_strength;
            
            void main() {
                ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(height_field);
                
                if (coord.x >= size.x || coord.y >= size.y) return;
                
                // 近隣の高さ取得
                float h_left = imageLoad(height_field, coord + ivec2(-1, 0)).r;
                float h_right = imageLoad(height_field, coord + ivec2(1, 0)).r;
                float h_up = imageLoad(height_field, coord + ivec2(0, -1)).r;
                float h_down = imageLoad(height_field, coord + ivec2(0, 1)).r;
                
                // 勾配計算
                float dx = (h_right - h_left) * u_normal_strength;
                float dy = (h_down - h_up) * u_normal_strength;
                
                // 法線ベクトル計算
                vec3 normal = normalize(vec3(-dx, 1.0, -dy));
                
                // 結果保存
                imageStore(normal_field, coord, vec4(normal, 1.0));
            }
            """
            
            self.normal_compute = self.ctx.compute_shader(normal_compute_source)
            
            self.logger.info("✅ コンピュートシェーダー作成完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ コンピュートシェーダー作成失敗: {e}")
            return False
    
    def _create_render_shaders(self) -> bool:
        """描画シェーダー作成"""
        try:
            # 頂点シェーダー
            vertex_shader = """
            #version 330 core
            
            layout(location = 0) in vec3 position;
            layout(location = 1) in vec2 texcoord;
            
            uniform mat4 mvp_matrix;
            uniform sampler2D u_height_texture;
            uniform float u_height_scale;
            
            out vec2 v_texcoord;
            out vec3 v_world_pos;
            out float v_height;
            
            void main() {
                v_texcoord = texcoord;
                
                // 高さ取得
                float height = texture(u_height_texture, texcoord).r;
                v_height = height;
                
                // 頂点位置に高さ適用
                vec3 displaced_pos = position;
                displaced_pos.y += height * u_height_scale;
                
                v_world_pos = displaced_pos;
                gl_Position = mvp_matrix * vec4(displaced_pos, 1.0);
            }
            """
            
            # フラグメントシェーダー
            fragment_shader = """
            #version 330 core
            
            in vec2 v_texcoord;
            in vec3 v_world_pos;
            in float v_height;
            
            uniform sampler2D u_camera_texture;
            uniform sampler2D u_normal_texture;
            uniform vec3 u_light_direction;
            uniform vec3 u_camera_position;
            uniform vec3 u_water_color;
            uniform float u_time;
            uniform vec3 u_emotion_color;
            uniform float u_emotion_intensity;
            
            out vec4 fragColor;
            
            vec3 fresnel_effect(float cos_theta, vec3 f0) {
                return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
            }
            
            void main() {
                // 法線取得
                vec3 normal = texture(u_normal_texture, v_texcoord).xyz;
                normal = normalize(normal);
                
                // ビュー方向
                vec3 view_dir = normalize(u_camera_position - v_world_pos);
                
                // 反射・屈折UV計算
                vec2 distorted_uv = v_texcoord + normal.xz * 0.03;
                distorted_uv = clamp(distorted_uv, 0.01, 0.99);
                
                // カメラ画像取得（屈折）
                vec3 refraction_color = texture(u_camera_texture, distorted_uv).rgb;
                
                // 反射計算（簡易）
                vec3 reflection_dir = reflect(-view_dir, normal);
                vec3 reflection_color = refraction_color * 1.2; // 簡易反射
                
                // フレネル効果
                float cos_theta = max(dot(view_dir, normal), 0.0);
                vec3 fresnel = fresnel_effect(cos_theta, vec3(0.02));
                
                // 基本水面色
                vec3 water_surface = mix(refraction_color, reflection_color, fresnel.r);
                
                // ライティング
                float NdotL = max(dot(normal, u_light_direction), 0.0);
                vec3 diffuse = vec3(NdotL) * 0.2;
                
                // スペキュラ
                vec3 reflect_dir = reflect(-u_light_direction, normal);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
                vec3 specular = vec3(spec) * 0.8;
                
                // 水の色ブレンド
                water_surface = mix(water_surface, u_water_color, 0.3);
                
                // 感情色エフェクト
                if (u_emotion_intensity > 0.1) {
                    float emotion_factor = u_emotion_intensity * (0.5 + abs(v_height) * 2.0);
                    water_surface = mix(water_surface, u_emotion_color, emotion_factor * 0.4);
                    
                    // 感情による輝度変化
                    float glow = sin(u_time * 4.0 + v_height * 10.0) * emotion_factor * 0.2;
                    water_surface += u_emotion_color * glow;
                }
                
                // 波による泡効果
                float foam = smoothstep(0.05, 0.1, abs(v_height));
                water_surface += vec3(foam) * 0.3;
                
                // 最終色合成
                vec3 final_color = water_surface + diffuse + specular;
                
                fragColor = vec4(final_color, 0.85);
            }
            """
            
            self.water_render_program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
            
            self.logger.info("✅ 描画シェーダー作成完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 描画シェーダー作成失敗: {e}")
            return False
    
    def _create_water_mesh(self) -> bool:
        """水面メッシュ作成"""
        try:
            # 高解像度メッシュ生成
            vertices = []
            indices = []
            
            mesh_resolution = self.resolution // 4  # 描画メッシュは物理より低解像度
            
            for y in range(mesh_resolution + 1):
                for x in range(mesh_resolution + 1):
                    # 位置
                    pos_x = (x / mesh_resolution) * 2.0 - 1.0
                    pos_z = (y / mesh_resolution) * 2.0 - 1.0
                    pos_y = 0.0
                    
                    # テクスチャ座標
                    tex_u = x / mesh_resolution
                    tex_v = y / mesh_resolution
                    
                    vertices.extend([pos_x, pos_y, pos_z, tex_u, tex_v])
            
            # インデックス生成
            for y in range(mesh_resolution):
                for x in range(mesh_resolution):
                    top_left = y * (mesh_resolution + 1) + x
                    top_right = top_left + 1
                    bottom_left = (y + 1) * (mesh_resolution + 1) + x
                    bottom_right = bottom_left + 1
                    
                    indices.extend([top_left, bottom_left, top_right])
                    indices.extend([top_right, bottom_left, bottom_right])
            
            # VAO作成
            vertex_buffer = self.ctx.buffer(np.array(vertices, dtype=np.float32).tobytes())
            index_buffer = self.ctx.buffer(np.array(indices, dtype=np.uint32).tobytes())
            
            self.water_vao = self.ctx.vertex_array(
                self.water_render_program,
                [(vertex_buffer, '3f 2f', 'position', 'texcoord')],
                index_buffer
            )
            
            self.logger.info(f"✅ 水面メッシュ作成完了: {len(vertices)//5}頂点, {len(indices)//3}三角形")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 水面メッシュ作成失敗: {e}")
            return False
    
    def add_wave_source(self, position: Tuple[float, float], intensity: float = 1.0):
        """波源追加"""
        # 不正なデータ型が渡された場合に、エラーを出力して処理を中断するガード節を追加
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            self.logger.warning(f"不正な波源位置データを受け取りました: {position}。スキップします。")
            return
        try:
            # 古い波源削除
            current_time = time.time()
            self.wave_sources = [
                source for source in self.wave_sources 
                if current_time - source['birth_time'] < 2.0
            ]
            
            # 新しい波源追加
            if len(self.wave_sources) < self.max_wave_sources:
                wave_source = {
                    'position': position,
                    'intensity': intensity,
                    'birth_time': current_time
                }
                self.wave_sources.append(wave_source)
                
                self.logger.debug(f"💧 波源追加: {position}, 強度: {intensity}")
            
        except Exception as e:
            self.logger.error(f"❌ 波源追加失敗: {e}")
    
    def update_physics(self, dt: float):
        """物理シミュレーション更新"""
        try:
            if not self.water_physics_compute:
                return
            
            # テクスチャバインド
            self.height_texture_current.bind_to_image(0, read=True, write=False)
            self.height_texture_previous.bind_to_image(1, read=False, write=True)
            self.velocity_texture.bind_to_image(2, read=True, write=True)
            
            # ユニフォーム設定
            self.water_physics_compute['u_wave_speed'] = self.wave_speed
            self.water_physics_compute['u_damping'] = self.damping
            self.water_physics_compute['u_dt'] = dt
            self.water_physics_compute['u_time'] = time.time()
            
            # 波源データ設定
            positions = []
            intensities = []
            
            for i in range(self.max_wave_sources):
                if i < len(self.wave_sources):
                    source = self.wave_sources[i]
                    positions.extend(source['position'])
                    intensities.append(source['intensity'])
                else:
                    positions.extend([0.0, 0.0])
                    intensities.append(0.0)
            
            positions_bytes = np.array(positions, dtype='f4').tobytes()
            intensities_bytes = np.array(intensities, dtype='f4').tobytes()
            
            self.water_physics_compute['u_wave_sources'].write(positions_bytes)
            self.water_physics_compute['u_wave_intensities'].write(intensities_bytes)
            self.water_physics_compute['u_active_sources'] = len(self.wave_sources)
            
            # コンピュートシェーダー実行
            groups_x = (self.resolution + 15) // 16
            groups_y = (self.resolution + 15) // 16
            self.water_physics_compute.run(groups_x, groups_y)
            
            # テクスチャスワップ
            self.height_texture_current, self.height_texture_previous = \
                self.height_texture_previous, self.height_texture_current
            
            # 法線計算
            self._update_normals()
            
        except Exception as e:
            self.logger.error(f"❌ 物理シミュレーション更新失敗: {e}")
    
    def _update_normals(self):
        """法線更新"""
        try:
            if not self.normal_compute:
                return
            
            # テクスチャバインド
            self.height_texture_current.bind_to_image(0, read=True, write=False)
            self.normal_texture.bind_to_image(1, read=False, write=True)
            
            # ユニフォーム設定
            self.normal_compute['u_normal_strength'] = 5.0
            
            # コンピュートシェーダー実行
            groups_x = (self.resolution + 15) // 16
            groups_y = (self.resolution + 15) // 16
            self.normal_compute.run(groups_x, groups_y)
            
        except Exception as e:
            self.logger.error(f"❌ 法線更新失敗: {e}")
    
    def render(self, camera_texture: moderngl.Texture, mvp_matrix: np.ndarray, 
              emotion_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
              emotion_intensity: float = 0.0):
        """高品質水面描画"""
        try:
            if not self.water_render_program or not self.water_vao:
                return
            
            # ユニフォーム設定
            self.water_render_program['mvp_matrix'].write(mvp_matrix.astype(np.float32).tobytes())
            self.water_render_program['u_height_scale'] = 0.1
            self.water_render_program['u_time'] = time.time()
            
            # テクスチャ設定
            if camera_texture:
                camera_texture.use(location=0)
                self.water_render_program['u_camera_texture'] = 0
            
            self.height_texture_current.use(location=1)
            self.water_render_program['u_height_texture'] = 1
            
            self.normal_texture.use(location=2)
            self.water_render_program['u_normal_texture'] = 2
            
            # ライティング設定
            self.water_render_program['u_light_direction'] = (0.5, 1.0, 0.3)
            self.water_render_program['u_camera_position'] = (0.0, 2.0, 2.0)
            self.water_render_program['u_water_color'] = (0.1, 0.3, 0.5)
            
            # 感情エフェクト
            self.water_render_program['u_emotion_color'] = emotion_color
            self.water_render_program['u_emotion_intensity'] = emotion_intensity
            
            # 描画
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            
            self.water_vao.render()
            
        except Exception as e:
            self.logger.error(f"❌ 高品質水面描画失敗: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            'resolution': f"{self.resolution}x{self.resolution}",
            'active_wave_sources': len(self.wave_sources),
            'wave_speed': self.wave_speed,
            'damping': self.damping,
            'textures_memory_mb': self.resolution * self.resolution * 4 * 4 / (1024 * 1024)  # 4テクスチャ
        }
    
    def cleanup(self):
        """リソース解放"""
        try:
            if self.water_vao:
                self.water_vao.release()
            
            if self.height_texture_current:
                self.height_texture_current.release()
            if self.height_texture_previous:
                self.height_texture_previous.release()
            if self.velocity_texture:
                self.velocity_texture.release()
            if self.normal_texture:
                self.normal_texture.release()
            
            self.wave_sources.clear()
            
            self.logger.info("✅ 高品質水面シミュレーションリソース解放完了")
            
        except Exception as e:
            self.logger.error(f"❌ 高品質水面シミュレーションリソース解放失敗: {e}")

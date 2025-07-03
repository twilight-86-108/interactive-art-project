
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

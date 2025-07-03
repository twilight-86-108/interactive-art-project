#version 430 core

in vec2 v_texcoord;
in vec3 v_world_pos;
in vec3 v_view_pos;
in float v_height;

uniform sampler2D u_camera_texture;
uniform sampler2D u_normal_texture;
uniform sampler2D u_height_texture;
uniform samplerCube u_environment_map;

uniform vec3 u_light_direction;
uniform vec3 u_camera_position;
uniform vec3 u_light_color;
uniform float u_time;

// 感情エフェクト
uniform vec3 u_emotion_color;
uniform float u_emotion_intensity;

// 水面物性
uniform float u_water_roughness;
uniform float u_water_metallic;
uniform vec3 u_water_base_color;
uniform float u_refraction_index;

out vec4 fragColor;

// PBR関数群
vec3 fresnel_schlick(float cos_theta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = 3.14159265359 * denom * denom;
    
    return num / denom;
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    
    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;
    
    return num / denom;
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometry_schlick_ggx(NdotV, roughness);
    float ggx1 = geometry_schlick_ggx(NdotL, roughness);
    
    return ggx1 * ggx2;
}

// 水面コースティクス計算
vec3 calculate_caustics(vec2 uv, vec3 normal, float time) {
    // 光線の屈折計算
    vec3 light_dir = normalize(-u_light_direction);
    vec3 refracted = refract(light_dir, normal, 1.0 / u_refraction_index);
    
    // コースティクスパターン
    vec2 caustic_uv = uv + refracted.xy * 0.1;
    
    // 複数周波数の干渉パターン
    float caustic1 = sin(caustic_uv.x * 40.0 + time * 2.0) * sin(caustic_uv.y * 40.0 + time * 2.0);
    float caustic2 = sin(caustic_uv.x * 60.0 - time * 1.5) * sin(caustic_uv.y * 60.0 - time * 1.5);
    float caustic3 = sin(caustic_uv.x * 80.0 + time * 3.0) * sin(caustic_uv.y * 80.0 + time * 3.0);
    
    float caustic_intensity = (caustic1 + caustic2 * 0.7 + caustic3 * 0.5) / 2.2;
    caustic_intensity = pow(max(caustic_intensity, 0.0), 3.0);
    
    // 水面の高さによる集光効果
    float focus_factor = 1.0 + abs(v_height) * 3.0;
    caustic_intensity *= focus_factor;
    
    return vec3(caustic_intensity * 0.8, caustic_intensity * 0.9, caustic_intensity);
}

// 環境反射計算
vec3 calculate_environment_reflection(vec3 view_dir, vec3 normal) {
    vec3 reflect_dir = reflect(-view_dir, normal);
    return texture(u_environment_map, reflect_dir).rgb;
}

// 屈折計算
vec3 calculate_refraction(vec2 base_uv, vec3 normal, float distortion_strength) {
    vec2 distorted_uv = base_uv + normal.xy * distortion_strength;
    distorted_uv = clamp(distorted_uv, 0.01, 0.99);
    return texture(u_camera_texture, distorted_uv).rgb;
}

void main() {
    // 法線取得・正規化
    vec3 N = normalize(texture(u_normal_texture, v_texcoord).xyz);
    
    // ビュー・ライト方向
    vec3 V = normalize(u_camera_position - v_world_pos);
    vec3 L = normalize(-u_light_direction);
    vec3 H = normalize(V + L);
    
    // 屈折計算
    float distortion_strength = 0.05 + abs(v_height) * 0.03;
    vec3 refraction_color = calculate_refraction(v_texcoord, N, distortion_strength);
    
    // 環境反射
    vec3 reflection_color = calculate_environment_reflection(V, N);
    
    // フレネル計算
    vec3 F0 = mix(vec3(0.04), u_water_base_color, u_water_metallic);
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    
    // PBR計算
    float NDF = distribution_ggx(N, H, u_water_roughness);
    float G = geometry_smith(N, V, L, u_water_roughness);
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
    vec3 specular = numerator / denominator;
    
    // エネルギー保存
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - u_water_metallic;
    
    // 拡散反射
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = kD * u_water_base_color / 3.14159265359;
    
    // 基本ライティング
    vec3 Lo = (diffuse + specular) * u_light_color * NdotL;
    
    // 環境ライティング（簡易IBL）
    vec3 ambient = vec3(0.03) * u_water_base_color;
    
    // 反射・屈折合成
    vec3 surface_color = mix(refraction_color, reflection_color, F.r);
    
    // 水の色とブレンド
    surface_color = mix(surface_color, u_water_base_color, 0.4);
    
    // PBRライティング適用
    vec3 color = surface_color * (Lo + ambient);
    
    // コースティクス追加
    vec3 caustics = calculate_caustics(v_texcoord, N, u_time);
    color += caustics * 0.3;
    
    // 感情エフェクト
    if (u_emotion_intensity > 0.1) {
        float emotion_factor = u_emotion_intensity * (0.5 + abs(v_height) * 2.0);
        
        // 感情色ブレンド
        color = mix(color, u_emotion_color, emotion_factor * 0.3);
        
        // 感情による動的輝度変化
        float emotion_glow = sin(u_time * 5.0 + v_height * 15.0) * emotion_factor * 0.4;
        color += u_emotion_color * emotion_glow;
        
        // 特定感情の特殊効果
        // この部分は追加の感情別エフェクトを実装可能
    }
    
    // トーンマッピング（Reinhard）
    color = color / (color + vec3(1.0));
    
    // ガンマ補正
    color = pow(color, vec3(1.0/2.2));
    
    // 泡・白波効果
    float foam = smoothstep(0.08, 0.15, abs(v_height));
    color = mix(color, vec3(1.0), foam * 0.6);
    
    fragColor = vec4(color, 0.9);
}

#version 430 core

in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D u_scene_texture;
uniform sampler2D u_water_height;
uniform sampler2D u_emotion_texture;

uniform float u_emotion_weights[5];  // [happy, sad, angry, surprised, neutral]
uniform float u_time;
uniform vec2 u_resolution;

// 感情色彩定義
const vec3 EMOTION_COLORS[5] = vec3[](
    vec3(1.0, 0.85, 0.3),   // Happy - 暖かい黄色
    vec3(0.3, 0.5, 0.8),    // Sad - 冷たい青
    vec3(0.9, 0.2, 0.3),    // Angry - 情熱的な赤
    vec3(1.0, 0.4, 0.8),    // Surprised - 鮮やかなピンク
    vec3(0.5, 0.7, 0.6)     // Neutral - 落ち着いた緑
);

vec3 blend_emotions() {
    vec3 final_color = vec3(0.0);
    float total_weight = 0.0;
    
    for(int i = 0; i < 5; i++) {
        final_color += EMOTION_COLORS[i] * u_emotion_weights[i];
        total_weight += u_emotion_weights[i];
    }
    
    if(total_weight > 0.0) {
        final_color /= total_weight;
    } else {
        final_color = EMOTION_COLORS[4]; // Default to neutral
    }
    
    return final_color;
}

vec2 water_distortion() {
    // 水面高さから法線ベクトル計算
    float height_center = texture(u_water_height, v_texcoord).r;
    float height_right = textureOffset(u_water_height, v_texcoord, ivec2(1, 0)).r;
    float height_top = textureOffset(u_water_height, v_texcoord, ivec2(0, 1)).r;
    
    vec2 gradient = vec2(height_right - height_center, height_top - height_center);
    return gradient * 0.05; // 歪み強度調整
}

void main() {
    // 水面歪みによるUV座標調整
    vec2 distorted_uv = v_texcoord + water_distortion();
    
    // ベースシーン色
    vec3 scene_color = texture(u_scene_texture, distorted_uv).rgb;
    
    // 感情色彩ブレンド
    vec3 emotion_color = blend_emotions();
    
    // 時間による動的変化
    float time_factor = sin(u_time * 2.0) * 0.1 + 0.9;
    emotion_color *= time_factor;
    
    // 最終色合成
    vec3 final_color = mix(scene_color, emotion_color, 0.6);
    
    // 水面反射効果
    float water_height = texture(u_water_height, v_texcoord).r;
    float reflection_intensity = abs(water_height) * 0.3;
    final_color += vec3(reflection_intensity);
    
    fragColor = vec4(final_color, 1.0);
}
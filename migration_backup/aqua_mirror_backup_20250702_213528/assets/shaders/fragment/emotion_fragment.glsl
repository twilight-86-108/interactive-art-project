#version 410 core

in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D u_emotion_texture;
uniform sampler2D u_features_texture;
uniform float u_emotion_intensity[5];  // [happy, sad, angry, surprised, neutral]
uniform float u_time;
uniform int u_blend_mode;

// 感情色彩定義
const vec3 EMOTION_COLORS[5] = vec3[](
    vec3(1.0, 0.85, 0.3),   // Happy - 暖かい黄色
    vec3(0.3, 0.5, 0.8),    // Sad - 冷たい青
    vec3(0.9, 0.2, 0.3),    // Angry - 情熱的な赤
    vec3(1.0, 0.4, 0.8),    // Surprised - 鮮やかなピンク
    vec3(0.5, 0.7, 0.6)     // Neutral - 落ち着いた緑
);

vec3 blend_emotions_smooth() {
    vec3 final_color = vec3(0.0);
    float total_intensity = 0.0;
    
    for(int i = 0; i < 5; i++) {
        final_color += EMOTION_COLORS[i] * u_emotion_intensity[i];
        total_intensity += u_emotion_intensity[i];
    }
    
    if(total_intensity > 0.0) {
        final_color /= total_intensity;
    } else {
        // デフォルトでニュートラル
        final_color = EMOTION_COLORS[4];
    }
    
    return final_color;
}

vec3 blend_emotions_artistic() {
    vec3 final_color = vec3(0.0);
    
    // 芸術的なブレンド（非線形）
    for(int i = 0; i < 5; i++) {
        float weight = pow(u_emotion_intensity[i], 1.5);
        final_color += EMOTION_COLORS[i] * weight;
    }
    
    // 時間的変化の追加
    float time_modulation = sin(u_time * 2.0) * 0.1 + 0.9;
    final_color *= time_modulation;
    
    // 正規化
    if (length(final_color) > 0.0) {
        return normalize(final_color) * length(final_color);
    } else {
        return EMOTION_COLORS[4]; // デフォルト: ニュートラル
    }
}

void main() {
    vec3 emotion_color;
    
    if(u_blend_mode == 0) {
        emotion_color = blend_emotions_smooth();
    } else {
        emotion_color = blend_emotions_artistic();
    }
    
    // 特徴量テクスチャからの追加情報（存在する場合）
    vec4 features = texture(u_features_texture, v_texcoord);
    float intensity_modifier = features.w;
    
    emotion_color *= max(intensity_modifier, 0.1); // 最小値を保証
    
    fragColor = vec4(emotion_color, 1.0);
}
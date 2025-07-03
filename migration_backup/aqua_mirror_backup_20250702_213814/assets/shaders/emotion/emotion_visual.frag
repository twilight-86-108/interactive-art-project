#version 410 core

in vec2 v_texcoord;
out vec4 fragColor;

// ユニフォーム
uniform sampler2D u_camera_texture;
uniform bool u_camera_enabled;
uniform float u_time;

// 感情パラメータ
uniform int u_emotion_type;        // 0-6: HAPPY, SAD, ANGRY, SURPRISED, NEUTRAL, FEAR, DISGUST
uniform float u_emotion_intensity; // 0.0-1.0
uniform vec3 u_emotion_color;      // RGB色
uniform float u_emotion_confidence; // 0.0-1.0

// エフェクトパラメータ
uniform float u_ripple_strength;
uniform float u_color_blend_factor;
uniform float u_glow_intensity;

// 感情色彩定義
vec3 emotion_colors[7] = vec3[](
    vec3(1.0, 0.8, 0.0),   // HAPPY - 明るい黄色
    vec3(0.3, 0.5, 0.8),   // SAD - 青
    vec3(0.9, 0.1, 0.1),   // ANGRY - 赤  
    vec3(1.0, 0.1, 0.6),   // SURPRISED - ピンク
    vec3(0.5, 0.5, 0.5),   // NEUTRAL - グレー
    vec3(0.4, 0.2, 0.6),   // FEAR - 紫
    vec3(0.2, 0.7, 0.2)    // DISGUST - 緑
);

// 感情波紋エフェクト
float emotion_ripple(vec2 uv, float time, int emotion_type, float intensity) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    
    float ripple = 0.0;
    
    if (emotion_type == 0) { // HAPPY - 明るい同心円
        ripple = sin(dist * 15.0 - time * 3.0) * exp(-dist * 2.0);
        ripple *= 0.3 + 0.7 * intensity;
    }
    else if (emotion_type == 1) { // SAD - 落ち着いた波
        ripple = sin(dist * 8.0 - time * 1.5) * exp(-dist * 1.5);
        ripple *= 0.2 + 0.3 * intensity;
    }
    else if (emotion_type == 2) { // ANGRY - 激しい波
        ripple = sin(dist * 20.0 - time * 5.0) * exp(-dist * 3.0);
        ripple += sin(dist * 25.0 + time * 4.0) * exp(-dist * 2.0) * 0.5;
        ripple *= 0.4 + 0.6 * intensity;
    }
    else if (emotion_type == 3) { // SURPRISED - 急激な拡散
        float pulse = sin(time * 6.0) * 0.5 + 0.5;
        ripple = sin(dist * 30.0 - time * 8.0) * exp(-dist * 4.0 * pulse);
        ripple *= 0.5 + 0.5 * intensity;
    }
    else if (emotion_type == 5) { // FEAR - 不規則な震え
        float noise = sin(uv.x * 50.0 + time * 10.0) * sin(uv.y * 50.0 + time * 8.0);
        ripple = noise * exp(-dist * 3.0) * 0.1;
        ripple += sin(dist * 12.0 - time * 2.0) * exp(-dist * 2.0) * 0.2;
        ripple *= 0.3 + 0.4 * intensity;
    }
    else if (emotion_type == 6) { // DISGUST - ねじれ効果
        float angle = atan(uv.y - 0.5, uv.x - 0.5);
        ripple = sin(dist * 10.0 + angle * 5.0 - time * 2.0) * exp(-dist * 2.0);
        ripple *= 0.25 + 0.35 * intensity;
    }
    else { // NEUTRAL or default
        ripple = sin(dist * 5.0 - time * 1.0) * exp(-dist * 1.0) * 0.1;
    }
    
    return ripple * u_ripple_strength;
}

// 感情グロー効果
vec3 emotion_glow(vec2 uv, vec3 base_color, int emotion_type, float intensity, float time) {
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(uv, center);
    
    vec3 glow_color = emotion_colors[emotion_type];
    float glow_strength = u_glow_intensity * intensity;
    
    // 感情別グロー計算
    float glow = 0.0;
    
    if (emotion_type == 0) { // HAPPY - 温かいグロー
        glow = exp(-dist * 1.5) * (0.5 + 0.5 * sin(time * 2.0));
    }
    else if (emotion_type == 1) { // SAD - 冷たいグロー
        glow = exp(-dist * 2.0) * (0.3 + 0.2 * sin(time * 0.8));
    }
    else if (emotion_type == 2) { // ANGRY - 激しいグロー
        glow = exp(-dist * 1.0) * (0.7 + 0.3 * sin(time * 4.0));
    }
    else if (emotion_type == 3) { // SURPRISED - 脈動グロー
        float pulse = sin(time * 6.0) * 0.5 + 0.5;
        glow = exp(-dist * 2.0 * pulse) * (0.6 + 0.4 * pulse);
    }
    else if (emotion_type == 5) { // FEAR - 震えるグロー
        float tremor = sin(time * 15.0) * 0.1;
        glow = exp(-dist * (2.0 + tremor)) * (0.4 + 0.3 * sin(time * 3.0));
    }
    else if (emotion_type == 6) { // DISGUST - 波打つグロー
        glow = exp(-dist * 1.8) * (0.35 + 0.25 * sin(time * 1.5 + dist * 10.0));
    }
    else { // NEUTRAL
        glow = exp(-dist * 3.0) * 0.2;
    }
    
    return base_color + glow_color * glow * glow_strength;
}

// 色彩ブレンド関数
vec3 blend_emotion_color(vec3 original_color, vec3 emotion_color, float blend_factor) {
    return mix(original_color, emotion_color, blend_factor);
}

void main() {
    vec2 uv = v_texcoord;
    vec3 final_color = vec3(0.0);
    
    // ベース色取得
    if (u_camera_enabled) {
        // カメラ画像ベース
        vec3 camera_color = texture(u_camera_texture, uv).rgb;
        final_color = camera_color;
    } else {
        // デフォルトグラデーション
        float gradient = sin(uv.x * 3.14159 + u_time) * 0.5 + 0.5;
        final_color = mix(vec3(0.2, 0.6, 1.0), vec3(0.8, 0.4, 1.0), gradient);
    }
    
    // 感情エフェクト適用
    if (u_emotion_intensity > 0.01 && u_emotion_confidence > 0.3) {
        // 波紋エフェクト
        float ripple = emotion_ripple(uv, u_time, u_emotion_type, u_emotion_intensity);
        
        // 波紋による色調変化
        vec3 ripple_color = emotion_colors[u_emotion_type];
        final_color += ripple_color * ripple;
        
        // 感情色彩ブレンド
        final_color = blend_emotion_color(final_color, u_emotion_color, 
                                        u_color_blend_factor * u_emotion_intensity);
        
        // グローエフェクト
        final_color = emotion_glow(uv, final_color, u_emotion_type, 
                                 u_emotion_intensity, u_time);
    }
    
    // 最終出力
    fragColor = vec4(final_color, 1.0);
}
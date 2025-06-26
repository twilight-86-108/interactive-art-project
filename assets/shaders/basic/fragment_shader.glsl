#version 410 core

in vec2 v_texcoord;
out vec4 fragColor;

uniform float u_time;
uniform vec3 u_color;

void main() {
    vec2 uv = v_texcoord;
    
    // OpenGL 4.1対応の基本的なエフェクト
    float gradient = sin(uv.x * 3.14159 + u_time) * 0.5 + 0.5;
    vec3 color = mix(vec3(0.2, 0.6, 1.0), u_color, gradient);
    
    fragColor = vec4(color, 1.0);
}

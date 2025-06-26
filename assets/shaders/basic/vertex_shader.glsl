#version 410 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;

uniform mat4 mvp_matrix;

out vec2 v_texcoord;

void main() {
    gl_Position = mvp_matrix * vec4(position, 1.0);
    v_texcoord = texcoord;
}

#version 460 core

// Fullscreen single triangle (covers [-1,1]^2 with 3 vertices).
vec2 positions[] = {
    { -1.0f, -1.0f },
    { +3.0f, -1.0f },
    { -1.0f, +3.0f }
};

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0f, 1.0f);
}

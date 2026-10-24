#version 460 core

#include "../scene_graph/camera.glsl"

// S4-1a: converts the linear RGBA32F shading result into the RGBA8
// presentation image on the GPU. The rounding is EXPLICIT and matches
// the former host conversion bit-for-bit:
//   host:  (unsigned char)(clamp(c, 0, 1) * 255.0 + 0.5)   (trunc = floor)
//   gpu:   floor(clamp(c, 0, 1) * 255.0 + 0.5) / 255.0
// Quantizing to exact n/255 multiples makes the UNORM write exact.

layout(binding = 1) uniform sampler2D shaded;

layout(location = 0) out vec4 color;

void main() {
    vec2 uv = gl_FragCoord.xy / camera.resolution;
    vec4 c = texture(shaded, uv);
    vec4 clamped = clamp(c, 0.0f, 1.0f);
    color = floor(clamped * 255.0f + 0.5f) / 255.0f;
}

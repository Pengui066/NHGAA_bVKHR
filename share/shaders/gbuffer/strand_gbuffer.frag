#version 460 core

#include "../scene_graph/camera.glsl"
#include "../anti-aliasing/gpaa.glsl"

// Undersampled hair G-buffer fragment shader: rasterizes strand lines
// into an MRT of (coverage, tangent, motion) for the neural hair
// G-buffer anti-aliasing pipeline. The frontmost fragment wins per
// pixel (depth test + write are on), and missing pixels simply have
// their clear values (coverage = 0, i.e. no valid position sample).
//
// Coverage is the GPAA line coverage scaled by the per-vertex strand
// thickness (taper) — pure geometry. The material transparency
// (hair_alpha) and the level-of-detail fade are deliberately NOT
// folded in here: they belong to shading, not to the G-buffer.
// Note: no early fragment tests — fragments with negligible coverage
// are discarded before the depth write, so nearly edge-on strands
// cannot occlude visible strands behind them without contributing.

#define STRAND_SCALING (1.0 / 0.042)

layout(location = 0) in PipelineIn {
    vec4  position;
    vec3  tangent;
    float thickness;
    vec2  motion;
} fs_in;

layout(push_constant) uniform Object {
    mat4 model;
    float strand_width;
} object;

layout(location = 0) out vec4 coverage_out;
layout(location = 1) out vec4 tangent_out;
layout(location = 2) out vec4 motion_out;

void main() {
    float coverage = gpaa(gl_FragCoord.xy, fs_in.position,
                          camera.projection * camera.view,
                          camera.resolution, object.strand_width);

    coverage *= fs_in.thickness * STRAND_SCALING; // strand taper.
    coverage = clamp(coverage, 0.0f, 1.0f);

    if (coverage < 0.001f)
        discard; // treat as missing: clear values stay in the MRT.

    // Canonical orientation: sign-align the tangent towards the viewer
    // (a strand tangent is direction-ambiguous, T = -T). This keeps the
    // G-buffer and the ground-truth averages from cancelling out.
    vec3 tangent = normalize(fs_in.tangent);
    vec3 view_direction = normalize(camera.position - fs_in.position.xyz);

    if (dot(tangent, view_direction) < 0.0f)
        tangent = -tangent;

    coverage_out = vec4(coverage, 0.0f, 0.0f, 1.0f);
    tangent_out  = vec4(tangent,  1.0f);
    motion_out   = vec4(fs_in.motion, 0.0f, 1.0f);
}

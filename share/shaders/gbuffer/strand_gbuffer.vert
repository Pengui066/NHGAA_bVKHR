#version 460 core

#include "../scene_graph/camera.glsl"

// Previous frame camera, used to compute per-vertex backward motion
// vectors (i.e. "where did the content at this fragment come from").
// The struct layout must match the Camera block above exactly, since
// both are filled from a vkhr::ViewProjection on the host.
layout(binding = 1) uniform PreviousCamera {
    mat4 view;
    mat4 projection;
    vec3 position;
    float look_at_distance;
    float near, far;
    vec2 resolution;
} prev_camera;

layout(location = 0) in vec3  position;
layout(location = 1) in vec3  tangent;
layout(location = 2) in float thickness;

layout(push_constant) uniform Object {
    mat4 model;
    float strand_width;
} object;

layout(location = 0) out PipelineOut {
    vec4  position;
    vec3  tangent;
    float thickness;
    vec2  motion;
} vs_out;

void main() {
    mat4 projection_view = camera.projection * camera.view;

    vec4 world_position = object.model * vec4(position, 1.0f);
    vec4 world_tangent  = object.model * vec4(tangent,  0.0f);

    vs_out.position  = world_position;
    vs_out.tangent   = world_tangent.xyz;
    vs_out.thickness = thickness;

    vec4 curr_clip = projection_view * world_position;
    vec4 prev_clip = prev_camera.projection * prev_camera.view * world_position;

    vec2 curr_ndc = curr_clip.xy / curr_clip.w;
    vec2 prev_ndc = prev_clip.xy / prev_clip.w;

    vs_out.motion = prev_ndc - curr_ndc; // backward motion (NDC units).

    gl_Position = curr_clip;
}

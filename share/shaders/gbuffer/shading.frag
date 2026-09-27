#version 460 core

#include "../scene_graph/camera.glsl"
#include "../scene_graph/lights.glsl"
#include "../self-shadowing/approximate_deep_shadows.glsl"
#include "../shading/kajiya-kay.glsl"
#include "../volumes/local_ambient_occlusion.glsl"

#include "../scene_graph/params.glsl"
#include "../scene_graph/shadow_maps.glsl"
#include "../strands/strand.glsl"

// Deferred hair shading pass: shades the (possibly reconstructed)
// hair G-buffer and composites it over the background. This is used
// both to shade the raw undersampled G-buffer (the "Input" baseline)
// and the high-sample ground truth G-buffer (the "Reference"), so
// the shading stays perfectly consistent between the two.

layout(binding = 3) uniform sampler2D hair_coverage;
layout(binding = 5) uniform sampler2D hair_tangent;
layout(binding = 6) uniform sampler2D hair_depth;
layout(binding = 7) uniform sampler2D background_color;
layout(binding = 8) uniform sampler3D strand_density;

layout(push_constant) uniform Inverse {
    mat4 inv_view_projection;
} inverse_vp;

layout(location = 0) out vec4 color;

void main() {
    vec2 uv  = gl_FragCoord.xy / camera.resolution;
    vec2 ndc = uv * 2.0f - 1.0f;

    float coverage = texture(hair_coverage, uv).r;
    float depth    = texture(hair_depth,    uv).r;

    vec4 background = texture(background_color, uv);

    if (coverage < 0.001f) {
        color = background;
        return;
    }

    // Reconstruct the world-space position of the frontmost strand
    // fragment from its (hardware) depth value and the inverse view
    // projection. Depth is in [0, 1] with GLM_FORCE_DEPTH_ZERO_TO_ONE.
    vec4 clip  = vec4(ndc, depth, 1.0f);
    vec4 world = inverse_vp.inv_view_projection * clip;
    vec3 position = world.xyz / world.w;

    vec3 tangent = texture(hair_tangent, uv).xyz;
    tangent = normalize(tangent);

    vec3 light_direction   = normalize(lights[0].origin - position);
    vec3 eye_direction     = normalize(position - camera.position);
    vec3 light_bulb_color  = lights[0].intensity;

    vec3 shading = vec3(1.0);

    if (shading_model == KAJIYA_KAY) {
        shading = kajiya_kay(hair_color, light_bulb_color, hair_exponent,
                             tangent, light_direction, eye_direction);
    }

    float occlusion = 1.000f;

    if (deep_shadows_on == YES) {
        occlusion *= approximate_deep_shadows(shadow_maps[0],
                                              lights[0].matrix * vec4(position, 1.0f),
                                              deep_shadows_kernel_size,
                                              deep_shadows_stride_size,
                                              15000.0f, hair_alpha);
    }

    occlusion *= local_ambient_occlusion(strand_density,
                                         position,
                                         volume_bounds.origin,
                                         volume_bounds.size,
                                         2, occlusion_radius,
                                         ao_exponent, ao_max);

    float alpha = coverage * hair_alpha;

    color = vec4(mix(background.rgb, shading * occlusion, alpha), 1.0f);
}

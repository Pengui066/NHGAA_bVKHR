#ifndef VKHR_PARAMS_GLSL
#define VKHR_PARAMS_GLSL

#define YES  1
#define NO   0

layout(binding = 4) uniform Params {
    int shading_model;

    int deep_shadows_kernel_size;
    int deep_shadows_sampling_type;
    int deep_shadows_stride_size;
    int deep_shadows_on;

    int pcf_shadows_kernel_size;
    int pcf_shadows_sampling_type;
    float pcf_shadows_bias;
    int pcf_shadows_on;

    int shadow_technique;

    float isosurface;
    float raycast_steps;
    float occlusion_radius;
    float ao_exponent;
    float ao_max;

    float magnified_distance;
    int renderer;
    float minified_distance;

    int benchmarking;

    // P1: view-direction transmittance (the thickness approximation).
    // Appended at the tail so every pre-existing field keeps its offset.
    int p1_enabled;
    float p1_sigma;
    int p1_march_steps;
    float p1_t_max;
    float p1_deep_r;
    float p1_deep_g;
    float p1_deep_b;
};

#endif

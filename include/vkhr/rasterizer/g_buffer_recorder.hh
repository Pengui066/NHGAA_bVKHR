#ifndef VKHR_G_BUFFER_RECORDER_HH
#define VKHR_G_BUFFER_RECORDER_HH

#include <vkpp/vkpp.hh>

#include <vkhr/rasterizer/pipeline.hh>

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include <vkhr/scene_graph/camera.hh>

#include <string>

namespace vk = vkpp;

namespace vkhr {
    class SceneGraph;
    class Rasterizer;

    // Configuration for the offline hair G-buffer dump mode. It renders
    // the strand-based hair into an undersampled G-buffer (the network
    // input) and a supersampled ground-truth G-buffer, shades both with
    // the deferred hair shader, and writes everything to disk as raw
    // binary channel files + a meta.json with the camera information.
    struct GBufferDumpConfig {
        std::string output_directory { "dumps" };
        unsigned    frame_count      { 1 };
        unsigned    ssaa_factor      { 2 };      // GT supersampling per axis.
        std::string camera_script    { "static" }; // "static" or "orbit".
        float       strand_radius    { -1.0f };  // < 0 keeps the scene default.
    };

    namespace vulkan {
        // Renders and dumps the hair G-buffers. Needs friend access to
        // Rasterizer (device, hair styles, models, shadow maps, ...) and
        // to vulkan::HairStyle / vulkan::Model (buffers and parameters).
        class GBufferRecorder final {
        public:
            GBufferRecorder(Rasterizer& vulkan_renderer, const GBufferDumpConfig& config);

            void dump(SceneGraph& scene_graph);

        private:
            // One set of G-buffer + shading attachments per resolution
            // (base resolution for the network input, ssaa x for the GT).
            struct Attachment {
                vk::Image       image;
                vk::DeviceMemory memory;
                vk::ImageView   view;

                Attachment() = default;
                Attachment(Attachment&&) = default;
                Attachment& operator=(Attachment&&) = default;
            };

            struct Target {
                std::uint32_t width, height;

                Attachment coverage;         // R16_SFLOAT
                Attachment tangent;          // R16G16B16A16_SFLOAT
                Attachment motion;           // R16G16_SFLOAT
                Attachment depth;            // D32_SFLOAT
                Attachment background_depth; // D32_SFLOAT
                Attachment background;       // R32G32B32A32_SFLOAT
                Attachment shaded;           // R32G32B32A32_SFLOAT

                // The attachment views referenced by the framebuffers
                // must outlive them (Vulkan framebuffer requirement).
                std::vector<vk::ImageView> gbuffer_attachments;
                std::vector<vk::ImageView> background_attachments;
                std::vector<vk::ImageView> shading_attachments;

                vk::Framebuffer gbuffer_framebuffer;
                vk::Framebuffer background_framebuffer;
                vk::Framebuffer shading_framebuffer;

                vk::HostBuffer coverage_buffer, tangent_buffer, motion_buffer,
                               depth_buffer, background_buffer, shaded_buffer;

                VkViewport viewport;
                VkRect2D   scissor;
            };

            void create_render_passes();
            void create_pipelines();
            void create_target(Target& target, std::uint32_t width, std::uint32_t height);
            void create_targets(std::uint32_t width, std::uint32_t height, std::uint32_t ssaa);

            void script_camera(SceneGraph& scene_graph, unsigned frame_index);
            void update_camera_buffers();

            void draw_head_occlusion(const SceneGraph& scene_graph, vk::CommandBuffer& command_buffer, Target& target);
            void draw_hair_gbuffer(const SceneGraph& scene_graph, vk::CommandBuffer& command_buffer, Target& target);
            void draw_background(const SceneGraph& scene_graph, vk::CommandBuffer& command_buffer, Target& target);
            void draw_deferred_shading(vk::CommandBuffer& command_buffer, Target& target);

            void bake_shadow_maps(const SceneGraph& scene_graph, vk::CommandBuffer& command_buffer);
            void write_scene_descriptors();

            void record_frame(SceneGraph& scene_graph);
            void readback_frame(unsigned frame_index);

            void write_binary(const std::string& path, void* data, std::size_t size);
            void write_meta();

            static Attachment create_attachment(Rasterizer& renderer, std::uint32_t width,
                                                std::uint32_t height, VkFormat format,
                                                VkImageUsageFlags extra_usage);

            struct PushConstants {
                glm::mat4 model;
                float strand_width;
            };

            Rasterizer& renderer;
            GBufferDumpConfig config;

            vk::RenderPass gbuffer_pass;    // coverage + tangent + motion + depth.
            vk::RenderPass background_pass; // shaded head model (composited under hair).
            vk::RenderPass shading_pass;    // deferred hair shading over background.

            Pipeline hair_gbuffer_pipeline;
            Pipeline head_occlusion_pipeline;
            Pipeline background_pipeline;
            Pipeline shading_pipeline;

            std::vector<vk::UniformBuffer> camera_buffer;      // input resolution.
            std::vector<vk::UniformBuffer> camera_gt_buffer;   // GT resolution.
            std::vector<vk::UniformBuffer> prev_camera_buffer; // previous frame (motion).


            vk::Sampler gbuffer_sampler;

            Target input_target, gt_target;

            ViewProjection current_transform, previous_transform;
            unsigned current_frame { 0 };

            glm::vec3 orbit_look_at { 0.0f };
            glm::vec3 orbit_offset  { 0.0f };

            std::vector<nlohmann::json> frame_metadata;
        };
    }
}

#endif

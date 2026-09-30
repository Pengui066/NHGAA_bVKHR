#ifndef VKHR_SHADE_DUMP_HH
#define VKHR_SHADE_DUMP_HH

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

    // Options for the offline shading mode: shades G-buffer channel files
    // from a dump directory (input/ or recon/) with the deferred hair
    // shader and writes the composited linear-light image back to disk.
    struct ShadeDumpOptions {
        std::string dump_directory;              // raw dump dir (has meta.json)
        std::string source { "recon" };          // "input" (verification) or "recon"
        std::string output_file { "" };          // explicit output path
    };

    namespace vulkan {
        // Shades dumped G-buffer channel files. Reproduces the exact dump
        // state: per-frame randomized light directions are re-derived from
        // the recorded seed (identical mt19937 draws and distributions),
        // shadow maps are baked with the dump's pipelines, and the camera
        // comes from meta.json.
        class ShadeDump final {
        public:
            ShadeDump(Rasterizer& vulkan_renderer, SceneGraph& scene_graph,
                      const ShadeDumpOptions& options,
                      const nlohmann::json& meta);

            void run(); // shades every frame listed in meta.json

        private:
            void create_render_pass();
            void create_pipeline();
            void create_buffers();
            void bake_shadow_maps(vk::CommandBuffer& command_buffer);

            void setup_frame_parameters(const nlohmann::json& frame_meta);
            void setup_frame_light(const nlohmann::json& frame_meta,
                                   unsigned frame_index);
            void update_camera_buffer(const nlohmann::json& frame_meta);
            void upload_texture(vk::Image& image, const std::string& path);
            void shade_frame();
            void save_output(const std::string& path);

            struct Attachment {
                vk::Image       image;
                vk::DeviceMemory memory;
                vk::ImageView   view;

                Attachment() = default;
                Attachment(Attachment&&) = default;
                Attachment& operator=(Attachment&&) = default;
            };

            Attachment create_attachment(std::uint32_t w, std::uint32_t h,
                                         VkFormat format, VkImageUsageFlags extra_usage);

            Rasterizer& renderer;
            SceneGraph& scene_graph;
            ShadeDumpOptions options;
            nlohmann::json meta;

            unsigned width { 0 }, height { 0 };

            vk::RenderPass shading_pass;
            Pipeline shading_pipeline;
            vk::Sampler channel_sampler;

            std::vector<vk::UniformBuffer> camera_buffer;
            ViewProjection camera_transform;

            Attachment coverage, tangent, depth, background, shaded;
            vk::Framebuffer shading_framebuffer;
            std::vector<vk::ImageView> shading_attachments;

            vk::HostBuffer staging_buffer;   // file -> GPU upload
            vk::HostBuffer readback_buffer;  // GPU -> file download
        };
    }
}

#endif

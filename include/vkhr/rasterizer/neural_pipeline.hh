#ifndef VKHR_NEURAL_PIPELINE_HH
#define VKHR_NEURAL_PIPELINE_HH

#include <vkpp/vkpp.hh>

#include <vkhr/image.hh>
#include <vkhr/scene_graph/camera.hh>

#include <vkhr/rasterizer/pipeline.hh>
#include <vkhr/rasterizer/neural_reconstruct.hh>

#include <glm/glm.hpp>

#include <string>

namespace vk = vkpp;

namespace vkhr {
    class SceneGraph;
    class Rasterizer;

    namespace vulkan {
        // The real-time neural reconstruction path (S3): per frame it
        // renders the strand hair into an undersampled G-buffer, runs the
        // Psi network on the coverage/tangent channels (CPU relay through
        // ONNX Runtime), uploads the reconstructed channels, and runs the
        // deferred hair shading over the head-model background. The result
        // is a CPU image presented through Rasterizer::draw(Image&).
        //
        // Everything runs synchronously per frame (render -> readback ->
        // inference -> upload -> shade), so the framerate is dominated by
        // the network; overlapping the GPU/CPU work is an S4 optimization.
        //
        // The class mirrors the offline GBufferRecorder (same render pass
        // and pipeline construction), minus the GT target, motion-vector
        // readback and disk I/O. Needs friend access to Rasterizer,
        // vulkan::HairStyle and vulkan::Model.
        class NeuralPipeline final {
        public:
        // Options for the real-time neural reconstruction path. The debug
        // facilities exist to decompose the on-screen look into its
        // components (raw input channels vs network reconstruction vs
        // occlusion) and to audit the readback/refill chain against the
        // offline pipeline.
        struct Options {
            std::string model_path { "runs/psi_v1/psi.onnx" };

            // When non-empty, the first frame's channels are written to
            // this directory (coverage/tangent readback, reconstructed
            // channels, shaded result, meta.json) and the window closes.
            std::string debug_dump_directory;

            // Shade the RAW input coverage/tangent instead of the
            // reconstructed channels (isolates the network's effect).
            bool shade_input_channels { false };

            // Use a shading.frag variant with occlusion forced to 1
            // (isolates the ADSM x LAO contribution).
            bool disable_occlusion { false };

            // F-21 deferred alpha calibration (see shading.frag):
            // 0 = legacy coverage*hair_alpha, 1 = coverage only,
            // 2 = k-layer accumulation 1-(1-c*hair_alpha)^alpha_layers.
            float alpha_mode   { 2.0f };
            float alpha_layers { 16.0f };
        };

        struct FrameTiming {
            double render_ms       { 0.0 }; // shadows + G-buffer + background.
            double readback_ms     { 0.0 }; // host readback + fp16->fp32 + sanitize.
            NeuralReconstruct::Timing neural;
            double upload_shade_ms { 0.0 }; // upload recon + shading + result readback.
            double result_convert_ms { 0.0 }; // shaded RGBA32F -> RGBA8 on the host.
            double total_ms        { 0.0 };
        };

        NeuralPipeline(Rasterizer& renderer, const Options& options);

        // Renders, reconstructs and shades one frame into get_result().
        void run(const SceneGraph& scene_graph);

        Image& get_result() { return result; }

        const FrameTiming& last_timing() const { return last_timing_; }
        const NeuralReconstruct& get_neural() const { return neural; }

        // True once the --nn-debug-dump frame has been written (the
        // render loop should close the window afterwards).
        bool debug_dump_complete() const { return debug_dump_complete_; }
        const std::string& debug_dump_directory() const { return options.debug_dump_directory; }

        private:
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
                Attachment motion;           // R16G16_SFLOAT (written, unused)
                Attachment depth;            // D32_SFLOAT
                Attachment background_depth; // D32_SFLOAT
                Attachment background;       // R32G32B32A32_SFLOAT
                Attachment shaded;           // R32G32B32A32_SFLOAT

                // Reconstructed channels uploaded from the network output,
                // sampled by the deferred shading pass.
                Attachment coverage_recon;   // R16_SFLOAT
                Attachment tangent_recon;    // R16G16B16A16_SFLOAT

                vk::HostBuffer coverage_buffer;  // R16F readback.
                vk::HostBuffer tangent_buffer;   // RGBA16F readback.
                vk::HostBuffer shaded_buffer;    // RGBA32F readback.
                vk::HostBuffer coverage_upload;  // R16F upload (recon).
                vk::HostBuffer tangent_upload;   // RGBA16F upload (recon).

                // Persistently mapped pointers (mapped once at creation).
                // Mapping per frame stalls the driver on the BAR memory
                // (measured ~500 ms/frame on the first map), and the BAR
                // region must be treated carefully: wide reads are fine,
                // but wide vector stores fault on the write-combined
                // mapping, so uploads go through scalar element writes.
                void* coverage_mapped  { nullptr };
                void* tangent_mapped   { nullptr };
                void* shaded_mapped    { nullptr };
                void* coverage_upload_mapped { nullptr };
                void* tangent_upload_mapped  { nullptr };

                std::vector<vk::ImageView> gbuffer_attachments;
                std::vector<vk::ImageView> background_attachments;
                std::vector<vk::ImageView> shading_attachments;

                vk::Framebuffer gbuffer_framebuffer;
                vk::Framebuffer background_framebuffer;
                vk::Framebuffer shading_framebuffer;

                VkViewport viewport;
                VkRect2D   scissor;
            };

            void create_render_passes();
            void create_pipelines();
            void create_target(Target& target, std::uint32_t width, std::uint32_t height);
            void write_descriptors();
            void write_debug_dump(const SceneGraph& scene_graph,
                                  const std::vector<float>& coverage_f32,
                                  const std::vector<float>& tangent_f32,
                                  const std::vector<float>& recon_coverage,
                                  const std::vector<float>& recon_tangent);

            void update_camera_buffers();
            void bake_shadow_maps(const SceneGraph& scene_graph,
                                  vk::CommandBuffer& command_buffer);
            void draw_head_occlusion(const SceneGraph& scene_graph,
                                     vk::CommandBuffer& command_buffer);
            void draw_hair_gbuffer(const SceneGraph& scene_graph,
                                   vk::CommandBuffer& command_buffer);
            void draw_background(const SceneGraph& scene_graph,
                                 vk::CommandBuffer& command_buffer);
            void draw_deferred_shading(vk::CommandBuffer& command_buffer);

            static Attachment create_attachment(Rasterizer& renderer, std::uint32_t width,
                                                std::uint32_t height, VkFormat format,
                                                VkImageUsageFlags extra_usage);

            struct PushConstants {
                glm::mat4 model;
                float strand_width;
            };

            struct ShadingPushConstants {
                glm::mat4 inverse_view_projection;
                float alpha_mode   { 0.0f };
                float alpha_layers { 16.0f };
            };

            Rasterizer& renderer;
            Options options;
            NeuralReconstruct neural;

            vk::RenderPass gbuffer_pass;    // coverage + tangent + motion + depth.
            vk::RenderPass background_pass; // shaded head model.
            vk::RenderPass shading_pass;    // deferred hair shading over background.

            Pipeline hair_gbuffer_pipeline;
            Pipeline head_occlusion_pipeline;
            Pipeline background_pipeline;
            Pipeline shading_pipeline;

            std::vector<vk::UniformBuffer> camera_buffer;
            std::vector<vk::UniformBuffer> prev_camera_buffer;

            vk::Sampler gbuffer_sampler;

            Target target;

            ViewProjection current_transform, previous_transform;

            Image result; // CPU RGBA8 image presented to the swapchain.
            FrameTiming last_timing_;
            unsigned frame_index { 0 };
            bool debug_dump_complete_ { false };
            bool debug_dump_written_ { false };
        };
    }
}

#endif

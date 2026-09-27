#include <vkhr/rasterizer/g_buffer_recorder.hh>

#include <vkhr/rasterizer.hh>
#include <vkhr/rasterizer/hair_style.hh>
#include <vkhr/rasterizer/model.hh>

#include <vkhr/scene_graph/camera.hh>

#include <vkpp/debug_marker.hh>

#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <sstream>


namespace vkhr {
    namespace vulkan {
        static const float Pi { glm::pi<float>() };

        static constexpr auto WRITE_ALL  = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        static constexpr auto WRITE_NONE = 0;

        static void set_opaque_blending(vk::GraphicsPipeline::FixedFunction& fixed_stages,
                                        std::uint32_t attachment, VkColorComponentFlags write_mask) {
            if (fixed_stages.attachments.size() <= attachment)
                fixed_stages.attachments.resize(attachment + 1);

            fixed_stages.attachments[attachment].blendEnable    = VK_FALSE;
            fixed_stages.attachments[attachment].colorWriteMask = write_mask;

            fixed_stages.color_blending_state.attachmentCount = fixed_stages.attachments.size();
            fixed_stages.color_blending_state.pAttachments    = fixed_stages.attachments.data();
        }

        GBufferRecorder::GBufferRecorder(Rasterizer& vulkan_renderer,
                                         const GBufferDumpConfig& dump_config)
                                        : renderer { vulkan_renderer },
                                          config { dump_config } {
            camera_buffer      = vk::UniformBuffer::create(renderer.device, sizeof(ViewProjection), 1, "Dump Camera (Input)");
            camera_gt_buffer   = vk::UniformBuffer::create(renderer.device, sizeof(ViewProjection), 1, "Dump Camera (GT)");
            prev_camera_buffer = vk::UniformBuffer::create(renderer.device, sizeof(ViewProjection), 1, "Dump Previous Camera");

            create_render_passes();
            create_pipelines();

            gbuffer_sampler = vk::Sampler {
                renderer.device,
                VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            };

            std::uint32_t base_width   = renderer.swap_chain.get_width(),
                           base_height = renderer.swap_chain.get_height();

            create_target(input_target, base_width, base_height);
            create_target(gt_target, base_width * config.ssaa_factor,
                                    base_height * config.ssaa_factor);

        }

        void GBufferRecorder::create_render_passes() {
            // The hair G-buffer pass: (coverage, tangent, motion) + depth.
            // All attachments end in a shader-readable layout, since the
            // deferred shading pass samples them right afterwards.
            std::vector<vk::RenderPass::Attachment> gbuffer_attachments {
                { VK_FORMAT_R16_SFLOAT,          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { VK_FORMAT_R16G16_SFLOAT,       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { VK_FORMAT_D32_SFLOAT,          VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL }
            };

            std::vector<VkAttachmentReference> gbuffer_subpass {
                { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
                { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
                { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
                { 3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
            };

            vk::RenderPass::Dependency gbuffer_dependency {
                0, VK_SUBPASS_EXTERNAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_TRANSFER_READ_BIT
            };

            gbuffer_pass = vk::RenderPass {
                renderer.device,
                gbuffer_attachments,
                std::vector<vk::RenderPass::Subpass> { gbuffer_subpass },
                std::vector<vk::RenderPass::Dependency> { gbuffer_dependency }
            };

            // The background pass renders the shaded head model, which is
            // later sampled by the deferred shading pass to composite over.
            std::vector<vk::RenderPass::Attachment> background_attachments {
                { VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { VK_FORMAT_D32_SFLOAT,          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                  VK_ATTACHMENT_STORE_OP_DONT_CARE }
            };

            std::vector<VkAttachmentReference> background_subpass {
                { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
                { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL }
            };

            vk::RenderPass::Dependency background_dependency {
                0, VK_SUBPASS_EXTERNAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_TRANSFER_READ_BIT
            };

            background_pass = vk::RenderPass {
                renderer.device,
                background_attachments,
                std::vector<vk::RenderPass::Subpass> { background_subpass },
                std::vector<vk::RenderPass::Dependency> { background_dependency }
            };

            // The deferred shading pass composites the shaded hair over
            // the background into the final linear-light color buffer.
            std::vector<vk::RenderPass::Attachment> shading_attachments {
                { VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }
            };

            std::vector<VkAttachmentReference> shading_subpass {
                { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }
            };

            vk::RenderPass::Dependency shading_dependency {
                0, VK_SUBPASS_EXTERNAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT
            };

            shading_pass = vk::RenderPass {
                renderer.device,
                shading_attachments,
                std::vector<vk::RenderPass::Subpass> { shading_subpass },
                std::vector<vk::RenderPass::Dependency> { shading_dependency }
            };
        }

        void GBufferRecorder::create_pipelines() {
            std::uint32_t light_count = renderer.shadow_maps.size();

            struct Constants { std::uint32_t light_size; } constant_data { light_count };
            std::vector<VkSpecializationMapEntry> constants {
                { 0, 0, sizeof(std::uint32_t) }
            };

            VkExtent2D extent = renderer.swap_chain.get_extent();

            // The strand G-buffer pipeline: rasterizes lines into the
            // (coverage, tangent, motion) MRT with depth testing, so the
            // frontmost fragment wins. Viewport/scissor/line width are
            // dynamic, since we render at both base and GT resolutions.
            hair_gbuffer_pipeline = Pipeline { };

            hair_gbuffer_pipeline.fixed_stages.add_vertex_binding({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec3) });
            hair_gbuffer_pipeline.fixed_stages.add_vertex_binding({ 1, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec3) });
            hair_gbuffer_pipeline.fixed_stages.add_vertex_binding({ 2, 2, VK_FORMAT_R32_SFLOAT,       sizeof(float)     });

            hair_gbuffer_pipeline.fixed_stages.set_topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST);

            hair_gbuffer_pipeline.fixed_stages.set_scissor({ 0, 0, extent });
            hair_gbuffer_pipeline.fixed_stages.set_viewport({ 0.0f, 0.0f,
                                                             static_cast<float>(extent.width),
                                                             static_cast<float>(extent.height),
                                                             0.0f, 1.0f });

            hair_gbuffer_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
            hair_gbuffer_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);
            hair_gbuffer_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_LINE_WIDTH);

            hair_gbuffer_pipeline.fixed_stages.set_line_width(1.0f);
            hair_gbuffer_pipeline.fixed_stages.enable_depth_test(); // test + write.

            set_opaque_blending(hair_gbuffer_pipeline.fixed_stages, 0, WRITE_ALL);
            set_opaque_blending(hair_gbuffer_pipeline.fixed_stages, 1, WRITE_ALL);
            set_opaque_blending(hair_gbuffer_pipeline.fixed_stages, 2, WRITE_ALL);

            hair_gbuffer_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/strand_gbuffer.vert"));
            vk::DebugMarker::object_name(renderer.device, hair_gbuffer_pipeline.shader_stages[0], VK_OBJECT_TYPE_SHADER_MODULE, "Hair G-Buffer Vertex Shader");
            hair_gbuffer_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/strand_gbuffer.frag"));
            vk::DebugMarker::object_name(renderer.device, hair_gbuffer_pipeline.shader_stages[1], VK_OBJECT_TYPE_SHADER_MODULE, "Hair G-Buffer Fragment Shader");

            hair_gbuffer_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device,
                {
                    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, // current camera.
                    { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }  // previous camera.
                }
            };

            hair_gbuffer_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(2,
                                                                                     hair_gbuffer_pipeline.descriptor_set_layout,
                                                                                     "Hair G-Buffer Descriptor Set");

            hair_gbuffer_pipeline.pipeline_layout = vk::Pipeline::Layout {
                renderer.device,
                hair_gbuffer_pipeline.descriptor_set_layout,
                {
                    { VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants) }
                }
            };

            hair_gbuffer_pipeline.pipeline = vk::GraphicsPipeline {
                renderer.device,
                hair_gbuffer_pipeline.shader_stages,
                hair_gbuffer_pipeline.fixed_stages,
                hair_gbuffer_pipeline.pipeline_layout,
                gbuffer_pass
            };

            // The head occlusion pipeline: draws the mesh with color
            // writes disabled, so hair fragments behind it fail the
            // depth test (i.e. the head occludes the hair G-buffer).
            head_occlusion_pipeline = Pipeline { };

            head_occlusion_pipeline.fixed_stages.add_vertex_binding({ 0, sizeof(vkhr::Model::Vertex), VK_VERTEX_INPUT_RATE_VERTEX });
            head_occlusion_pipeline.fixed_stages.add_vertex_attribute({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 });

            head_occlusion_pipeline.fixed_stages.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

            head_occlusion_pipeline.fixed_stages.set_scissor({ 0, 0, extent });
            head_occlusion_pipeline.fixed_stages.set_viewport({ 0.0f, 0.0f,
                                                               static_cast<float>(extent.width),
                                                               static_cast<float>(extent.height),
                                                               0.0f, 1.0f });

            head_occlusion_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
            head_occlusion_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

            head_occlusion_pipeline.fixed_stages.set_culling_mode(VK_CULL_MODE_BACK_BIT);

            head_occlusion_pipeline.fixed_stages.enable_depth_test(); // test + write.

            set_opaque_blending(head_occlusion_pipeline.fixed_stages, 0, WRITE_NONE);
            set_opaque_blending(head_occlusion_pipeline.fixed_stages, 1, WRITE_NONE);
            set_opaque_blending(head_occlusion_pipeline.fixed_stages, 2, WRITE_NONE);

            head_occlusion_pipeline.shader_stages.emplace_back(renderer.device, SHADER("self-shadowing/depth_map.vert"));
            vk::DebugMarker::object_name(renderer.device, head_occlusion_pipeline.shader_stages[0], VK_OBJECT_TYPE_SHADER_MODULE, "Head Occlusion Vertex Shader");

            head_occlusion_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device
            };

            head_occlusion_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(1,
                                                                                        head_occlusion_pipeline.descriptor_set_layout,
                                                                                        "Head Occlusion Descriptor Set");

            head_occlusion_pipeline.pipeline_layout = vk::Pipeline::Layout {
                renderer.device,
                head_occlusion_pipeline.descriptor_set_layout,
                {
                    { VK_SHADER_STAGE_ALL, 0, sizeof(glm::mat4) }
                }
            };

            head_occlusion_pipeline.pipeline = vk::GraphicsPipeline {
                renderer.device,
                head_occlusion_pipeline.shader_stages,
                head_occlusion_pipeline.fixed_stages,
                head_occlusion_pipeline.pipeline_layout,
                gbuffer_pass
            };

            // The background pipeline: the head model shaded exactly as in
            // the normal color pass (same vertex/fragment shaders), but
            // rendered into our own linear-light background attachment.
            background_pipeline = Pipeline { };

            background_pipeline.fixed_stages.add_vertex_binding({ 0, sizeof(vkhr::Model::Vertex), VK_VERTEX_INPUT_RATE_VERTEX });
            background_pipeline.fixed_stages.add_vertex_attribute({ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 });
            background_pipeline.fixed_stages.add_vertex_attribute({ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, sizeof(glm::vec3) });
            background_pipeline.fixed_stages.add_vertex_attribute({ 2, 0, VK_FORMAT_R32G32_SFLOAT,    sizeof(glm::vec3) * 2 });

            background_pipeline.fixed_stages.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

            background_pipeline.fixed_stages.set_scissor({ 0, 0, extent });
            background_pipeline.fixed_stages.set_viewport({ 0.0f, 0.0f,
                                                            static_cast<float>(extent.width),
                                                            static_cast<float>(extent.height),
                                                            0.0f, 1.0f });

            background_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
            background_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

            background_pipeline.fixed_stages.enable_depth_test(); // head self-occlusion.

            set_opaque_blending(background_pipeline.fixed_stages, 0, WRITE_ALL);

            background_pipeline.shader_stages.emplace_back(renderer.device, SHADER("models/model.vert"));
            vk::DebugMarker::object_name(renderer.device, background_pipeline.shader_stages[0], VK_OBJECT_TYPE_SHADER_MODULE, "Background Model Vertex Shader");
            background_pipeline.shader_stages.emplace_back(renderer.device, SHADER("models/model.frag"), constants, &constant_data, sizeof(constant_data));
            vk::DebugMarker::object_name(renderer.device, background_pipeline.shader_stages[1], VK_OBJECT_TYPE_SHADER_MODULE, "Background Model Fragment Shader");

            std::vector<vk::DescriptorSet::Binding> background_bindings {
                { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },
                { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },
                { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }
            };

            for (std::uint32_t i { 0 }; i < light_count; ++i)
                background_bindings.push_back({ 9 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER });

            background_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device, background_bindings
            };

            background_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(2,
                                                                                    background_pipeline.descriptor_set_layout,
                                                                                    "Background Descriptor Set");

            background_pipeline.pipeline_layout = vk::Pipeline::Layout {
                renderer.device,
                background_pipeline.descriptor_set_layout,
                {
                    { VK_SHADER_STAGE_ALL, 0, sizeof(glm::mat4) }
                }
            };

            background_pipeline.pipeline = vk::GraphicsPipeline {
                renderer.device,
                background_pipeline.shader_stages,
                background_pipeline.fixed_stages,
                background_pipeline.pipeline_layout,
                background_pass
            };

            // The deferred hair shading pipeline: fullscreen pass that
            // shades the G-buffer and composites it over the background.
            shading_pipeline = Pipeline { };

            shading_pipeline.fixed_stages.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

            shading_pipeline.fixed_stages.set_scissor({ 0, 0, extent });
            shading_pipeline.fixed_stages.set_viewport({ 0.0f, 0.0f,
                                                         static_cast<float>(extent.width),
                                                         static_cast<float>(extent.height),
                                                         0.0f, 1.0f });

            shading_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
            shading_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

            shading_pipeline.fixed_stages.set_culling_mode(VK_CULL_MODE_NONE); // fullscreen triangle.
            shading_pipeline.fixed_stages.disable_depth_test();

            set_opaque_blending(shading_pipeline.fixed_stages, 0, WRITE_ALL);

            shading_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/fullscreen.vert"));
            vk::DebugMarker::object_name(renderer.device, shading_pipeline.shader_stages[0], VK_OBJECT_TYPE_SHADER_MODULE, "Deferred Shading Vertex Shader");
            shading_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/shading.frag"), constants, &constant_data, sizeof(constant_data));
            vk::DebugMarker::object_name(renderer.device, shading_pipeline.shader_stages[1], VK_OBJECT_TYPE_SHADER_MODULE, "Deferred Hair Shading Fragment Shader");

            std::vector<vk::DescriptorSet::Binding> shading_bindings {
                { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // camera.
                { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // lights.
                { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // strand parameters.
                { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // coverage.
                { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // rendering parameters.
                { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // tangent.
                { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // depth.
                { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // background.
                { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }  // density volume.
            };

            for (std::uint32_t i { 0 }; i < light_count; ++i)
                shading_bindings.push_back({ 9 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER });

            shading_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device, shading_bindings
            };

            shading_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(2,
                                                                                 shading_pipeline.descriptor_set_layout,
                                                                                 "Deferred Shading Descriptor Set");

            shading_pipeline.pipeline_layout = vk::Pipeline::Layout {
                renderer.device,
                shading_pipeline.descriptor_set_layout,
                {
                    { VK_SHADER_STAGE_ALL, 0, sizeof(glm::mat4) } // inverse view-projection.
                }
            };

            shading_pipeline.pipeline = vk::GraphicsPipeline {
                renderer.device,
                shading_pipeline.shader_stages,
                shading_pipeline.fixed_stages,
                shading_pipeline.pipeline_layout,
                shading_pass
            };

            // Write all descriptors that don't depend on the scene graph
            // (the camera buffers). The per-style buffers are written in
            // write_scene_descriptors() at dump time.
            for (std::uint32_t i { 0 }; i < 2; ++i) {
                auto& camera = i == 0 ? camera_buffer[0] : camera_gt_buffer[0];

                hair_gbuffer_pipeline.descriptor_sets[i].write(0, camera);
                hair_gbuffer_pipeline.descriptor_sets[i].write(1, prev_camera_buffer[0]);

                background_pipeline.descriptor_sets[i].write(0, camera);
                background_pipeline.descriptor_sets[i].write(1, renderer.lights[0]);
                background_pipeline.descriptor_sets[i].write(4, renderer.params[0]);

                for (std::uint32_t j { 0 }; j < light_count; ++j)
                    background_pipeline.descriptor_sets[i].write(9 + j,
                                                                 renderer.shadow_maps[j].get_image_view(),
                                                                 renderer.shadow_maps[j].get_sampler());

                shading_pipeline.descriptor_sets[i].write(0, camera);
                shading_pipeline.descriptor_sets[i].write(1, renderer.lights[0]);
                shading_pipeline.descriptor_sets[i].write(4, renderer.params[0]);

                for (std::uint32_t j { 0 }; j < light_count; ++j)
                    shading_pipeline.descriptor_sets[i].write(9 + j,
                                                              renderer.shadow_maps[j].get_image_view(),
                                                              renderer.shadow_maps[j].get_sampler());
            }
        }

        void GBufferRecorder::write_scene_descriptors() {
            // The dump mode currently shades a single hair style: the
            // first one found in the scene (its density volume and strand
            // parameters are bound into the deferred shading pass).
            auto& style = renderer.hair_styles.begin()->second;

            for (std::uint32_t i { 0 }; i < 2; ++i) {
                auto& target = i == 0 ? input_target : gt_target;

                shading_pipeline.descriptor_sets[i].write(2, style.parameter_buffer);
                shading_pipeline.descriptor_sets[i].write(3, target.coverage.view, gbuffer_sampler);
                shading_pipeline.descriptor_sets[i].write(5, target.tangent.view, gbuffer_sampler);
                shading_pipeline.descriptor_sets[i].write(6, target.depth.view, gbuffer_sampler);
                shading_pipeline.descriptor_sets[i].write(7, target.background.view, gbuffer_sampler);
                shading_pipeline.descriptor_sets[i].write(8, style.density_view, style.density_sampler);
            }
        }

        GBufferRecorder::Attachment GBufferRecorder::create_attachment(Rasterizer& vulkan_renderer,
                                                                       std::uint32_t width, std::uint32_t height,
                                                                       VkFormat format, VkImageUsageFlags extra_usage) {
            Attachment attachment;

            attachment.image = vk::Image {
                vulkan_renderer.device, width, height, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT |
                extra_usage
            };

            vk::DebugMarker::object_name(vulkan_renderer.device, attachment.image, VK_OBJECT_TYPE_IMAGE, "G-Buffer Attachment Image");

            attachment.memory = vk::DeviceMemory {
                vulkan_renderer.device,
                attachment.image.get_memory_requirements(),
                vk::DeviceMemory::Type::DeviceLocal
            };

            attachment.image.bind(attachment.memory);

            VkImageLayout read_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            if (format == VK_FORMAT_D32_SFLOAT)
                read_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

            attachment.view = vk::ImageView {
                vulkan_renderer.device,
                attachment.image,
                read_layout
            };

            vk::DebugMarker::object_name(vulkan_renderer.device, attachment.view, VK_OBJECT_TYPE_IMAGE_VIEW, "G-Buffer Attachment View");

            return attachment;
        }

        void GBufferRecorder::create_target(Target& target, std::uint32_t width, std::uint32_t height) {
            target.width  = width;
            target.height = height;

            target.coverage   = create_attachment(renderer, width, height, VK_FORMAT_R16_SFLOAT,          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.tangent    = create_attachment(renderer, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.motion     = create_attachment(renderer, width, height, VK_FORMAT_R16G16_SFLOAT,       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.depth      = create_attachment(renderer, width, height, VK_FORMAT_D32_SFLOAT,          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            target.background_depth = create_attachment(renderer, width, height, VK_FORMAT_D32_SFLOAT,    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            target.background = create_attachment(renderer, width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.shaded     = create_attachment(renderer, width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

            target.gbuffer_attachments.emplace_back(renderer.device, target.coverage.image);
            target.gbuffer_attachments.emplace_back(renderer.device, target.tangent.image);
            target.gbuffer_attachments.emplace_back(renderer.device, target.motion.image);
            target.gbuffer_attachments.emplace_back(renderer.device, target.depth.image,
                                                   VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

            target.gbuffer_framebuffer = vk::Framebuffer {
                renderer.device, gbuffer_pass, target.gbuffer_attachments, VkExtent2D { width, height }
            };

            target.background_attachments.emplace_back(renderer.device, target.background.image);
            target.background_attachments.emplace_back(renderer.device, target.background_depth.image,
                                                      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

            target.background_framebuffer = vk::Framebuffer {
                renderer.device, background_pass, target.background_attachments, VkExtent2D { width, height }
            };

            target.shading_attachments.emplace_back(renderer.device, target.shaded.image);

            target.shading_framebuffer = vk::Framebuffer {
                renderer.device, shading_pass, target.shading_attachments, VkExtent2D { width, height }
            };

            auto readback_buffer = [&](std::uint32_t bytes) {
                return vk::HostBuffer {
                    renderer.device, bytes,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT
                };
            };

            target.coverage_buffer    = readback_buffer(width * height * 2);  // R16F.
            target.tangent_buffer     = readback_buffer(width * height * 8);  // RGBA16F.
            target.motion_buffer      = readback_buffer(width * height * 4);  // RG16F.
            target.depth_buffer       = readback_buffer(width * height * 4);  // D32F.
            target.background_buffer  = readback_buffer(width * height * 16); // RGBA32F.
            target.shaded_buffer      = readback_buffer(width * height * 16); // RGBA32F.

            target.viewport = VkViewport {
                0.0f, 0.0f,
                static_cast<float>(width), static_cast<float>(height),
                0.0f, 1.0f
            };

            target.scissor = VkRect2D { { 0, 0 }, { width, height } };
        }

        void GBufferRecorder::script_camera(SceneGraph& scene_graph, unsigned frame_index) {
            auto& camera = scene_graph.get_camera();

            camera.set_resolution(input_target.width, input_target.height);

            if (config.camera_script == "orbit") {
                if (frame_index == 0) {
                    // The orbit script revolves around the initial look-at
                    // point, preserving the initial viewing distance.
                    orbit_look_at = camera.get_look_at_point();
                    orbit_offset  = camera.get_position() - orbit_look_at;
                }

                float angle = 2.0f * Pi * static_cast<float>(frame_index)
                                  / static_cast<float>(config.frame_count);

                glm::mat4 rotation { 1.0f };
                rotation = glm::rotate(rotation, angle, glm::vec3 { 0.0f, 1.0f, 0.0f });

                camera.set_look_at_point(orbit_look_at);
                camera.set_position(orbit_look_at + glm::vec3 { rotation * glm::vec4 { orbit_offset, 1.0f } });
            }
        }

        void GBufferRecorder::update_camera_buffers() {
            ViewProjection input_transform = current_transform;
            input_transform.resolution = glm::vec2 {
                static_cast<float>(input_target.width),
                static_cast<float>(input_target.height)
            };

            ViewProjection gt_transform = current_transform;
            gt_transform.resolution = glm::vec2 {
                static_cast<float>(gt_target.width),
                static_cast<float>(gt_target.height)
            };

            camera_buffer[0].update(input_transform);
            camera_gt_buffer[0].update(gt_transform);
            prev_camera_buffer[0].update(previous_transform);
        }

        void GBufferRecorder::draw_head_occlusion(const SceneGraph& scene_graph,
                                                  vk::CommandBuffer& command_buffer, Target& target) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            command_buffer.bind_pipeline(head_occlusion_pipeline);

            glm::mat4 projection_view = current_transform.projection * current_transform.view;

            for (auto& model_node : scene_graph.get_nodes_with_models()) {
                command_buffer.push_constant(head_occlusion_pipeline, 0,
                                             projection_view * model_node->get_model_matrix());
                for (auto& model_mesh : model_node->get_models()) {
                    auto& model = renderer.models[model_mesh];
                    command_buffer.bind_vertex_buffer(0, model.vertices, 0);
                    command_buffer.bind_index_buffer(model.elements, 0);
                    command_buffer.draw_indexed(model.elements.count());
                }
            }
        }

        void GBufferRecorder::draw_hair_gbuffer(const SceneGraph& scene_graph,
                                                vk::CommandBuffer& command_buffer, Target& target) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            command_buffer.bind_pipeline(hair_gbuffer_pipeline);

            bool input_resolution = (&target == &input_target);

            command_buffer.bind_descriptor_set(hair_gbuffer_pipeline.descriptor_sets[input_resolution ? 0 : 1],
                                               hair_gbuffer_pipeline);

            float width_scale = input_resolution ? 1.0f
                                : static_cast<float>(config.ssaa_factor);

            for (auto& hair_node : scene_graph.get_nodes_with_hair_styles()) {
                for (auto& hair_style : hair_node->get_hair_styles()) {
                    auto& style = renderer.hair_styles[hair_style];

                    float strand_width = style.parameters.strand_radius * width_scale;

                    command_buffer.set_line_width(strand_width);

                    PushConstants push_constants {
                        hair_node->get_model_matrix(), strand_width
                    };

                    command_buffer.push_constant(hair_gbuffer_pipeline, 0, push_constants);

                    command_buffer.bind_vertex_buffer(0, style.vertices,  0);
                    command_buffer.bind_vertex_buffer(1, style.tangents,  0);
                    command_buffer.bind_vertex_buffer(2, style.thickness, 0);

                    command_buffer.bind_index_buffer(style.segments);

                    command_buffer.draw_indexed(style.segments.count() * style.parameters.strand_ratio);
                }
            }
        }

        void GBufferRecorder::draw_background(const SceneGraph& scene_graph,
                                              vk::CommandBuffer& command_buffer, Target& target) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            bool input_resolution = (&target == &input_target);

            command_buffer.bind_pipeline(background_pipeline);

            vk::DescriptorSet& descriptor_set = background_pipeline.descriptor_sets[input_resolution ? 0 : 1];

            for (auto& model_node : scene_graph.get_nodes_with_models()) {
                command_buffer.push_constant(background_pipeline, 0, model_node->get_model_matrix());
                for (auto& model_mesh : model_node->get_models())
                    renderer.models[model_mesh].draw(background_pipeline, descriptor_set, command_buffer);
            }
        }

        void GBufferRecorder::draw_deferred_shading(vk::CommandBuffer& command_buffer, Target& target) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            bool input_resolution = (&target == &input_target);

            command_buffer.bind_pipeline(shading_pipeline);

            glm::mat4 inverse_view_projection = glm::inverse(current_transform.projection * current_transform.view);
            command_buffer.push_constant(shading_pipeline, 0, inverse_view_projection);

            command_buffer.bind_descriptor_set(shading_pipeline.descriptor_sets[input_resolution ? 0 : 1],
                                               shading_pipeline);

            command_buffer.draw(3); // fullscreen triangle.
        }

        void GBufferRecorder::bake_shadow_maps(const SceneGraph& scene_graph,
                                               vk::CommandBuffer& command_buffer) {
            for (auto& shadow_map : renderer.shadow_maps) {
                auto& view_projection = shadow_map.light->get_view_projection();

                command_buffer.begin_render_pass(renderer.depth_pass, shadow_map);
                shadow_map.update_dynamic_viewport_scissor_depth(command_buffer);

                if (renderer.imgui.parameters.adsm_on) {
                    command_buffer.bind_pipeline(renderer.hair_depth_pipeline);

                    for (auto& hair_node : scene_graph.get_nodes_with_hair_styles()) {
                        command_buffer.push_constant(renderer.hair_depth_pipeline, 0,
                                                     view_projection * hair_node->get_model_matrix());
                        for (auto& hair_style : hair_node->get_hair_styles())
                            renderer.hair_styles[hair_style].draw(renderer.hair_depth_pipeline,
                                                                  renderer.hair_depth_pipeline.descriptor_sets[0],
                                                                  command_buffer);
                    }
                }

                if (renderer.imgui.parameters.ctsm_on) {
                    command_buffer.bind_pipeline(renderer.mesh_depth_pipeline);

                    for (auto& model_node : scene_graph.get_nodes_with_models()) {
                        command_buffer.push_constant(renderer.mesh_depth_pipeline, 0,
                                                     view_projection * model_node->get_model_matrix());
                        for (auto& model_mesh : model_node->get_models())
                            renderer.models[model_mesh].draw(renderer.mesh_depth_pipeline,
                                                             renderer.mesh_depth_pipeline.descriptor_sets[0],
                                                             command_buffer);
                    }
                }

                command_buffer.end_render_pass();
            }
        }

        void GBufferRecorder::record_frame(SceneGraph& scene_graph) {
            update_camera_buffers();

            auto command_buffer = renderer.command_pool.allocate_and_begin();

            // Bake the shadow maps once per frame (the light and the hair
            // are static during the dump, but this keeps the flow correct
            // for the dynamic case later on). We cannot reuse
            // Rasterizer::draw_depth here, since it records its debug
            // markers into the (never-begun) main-loop command buffers.
            bake_shadow_maps(scene_graph, command_buffer);

            VkClearValue zero { }, depth_one { }, white { };
            zero.color   = { { 0.0f, 0.0f, 0.0f, 0.0f } };
            white.color  = { { 1.0f, 1.0f, 1.0f, 1.0f } };
            depth_one.depthStencil = { 1.0f, 0 };

            for (Target* target : { &input_target, &gt_target }) {
                // The hair G-buffer: head occlusion first, then strands.
                std::vector<VkClearValue> gbuffer_clears { zero, zero, zero, depth_one };

                command_buffer.begin_render_pass(gbuffer_pass, target->gbuffer_framebuffer, gbuffer_clears);
                draw_head_occlusion(scene_graph, command_buffer, *target);
                draw_hair_gbuffer(scene_graph, command_buffer, *target);
                command_buffer.end_render_pass();

                if (config.dump_shaded) {
                    // The background (the head shaded as usual).
                    std::vector<VkClearValue> background_clears { white, depth_one };

                    command_buffer.begin_render_pass(background_pass, target->background_framebuffer, background_clears);
                    draw_background(scene_graph, command_buffer, *target);
                    command_buffer.end_render_pass();

                    // The deferred hair shading over the background.
                    std::vector<VkClearValue> shading_clears { white };

                    command_buffer.begin_render_pass(shading_pass, target->shading_framebuffer, shading_clears);
                    draw_deferred_shading(command_buffer, *target);
                    command_buffer.end_render_pass();
                }
            }

            // Transition every attachment into a transfer source, then
            // copy them into host-visible buffers for the dump.
            auto transition_to_source = [&](Attachment& attachment) {
                VkImageLayout read_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                if (attachment.image.get_format() == VK_FORMAT_D32_SFLOAT)
                    read_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

                attachment.image.transition(command_buffer,
                                            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                            read_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT);
            };

            for (Target* target : { &input_target, &gt_target }) {
                transition_to_source(target->coverage);
                transition_to_source(target->tangent);
                transition_to_source(target->motion);
                transition_to_source(target->depth);
                if (config.dump_shaded) {
                    transition_to_source(target->background);
                    transition_to_source(target->shaded);
                }
            }

            auto copy_attachment = [&](Attachment& attachment, vk::HostBuffer& buffer) {
                command_buffer.copy_image_to_buffer(attachment.image, buffer);
            };

            for (Target* target : { &input_target, &gt_target }) {
                copy_attachment(target->coverage,   target->coverage_buffer);
                copy_attachment(target->tangent,    target->tangent_buffer);
                copy_attachment(target->motion,     target->motion_buffer);
                copy_attachment(target->depth,      target->depth_buffer);
                if (config.dump_shaded) {
                    copy_attachment(target->background, target->background_buffer);
                    copy_attachment(target->shaded,     target->shaded_buffer);
                }
            }

            command_buffer.end();

            renderer.device.get_graphics_queue().submit(command_buffer).wait_idle();
                    }

        void GBufferRecorder::write_binary(const std::string& path, void* data, std::size_t size) {
            std::ofstream file { path, std::ios::binary };
            if (!file) {
                std::cerr << "vkhr: couldn't open '" << path << "' for writing!\n";
                return;
            }
            file.write(reinterpret_cast<const char*>(data), size);
        }

        void GBufferRecorder::readback_frame(unsigned frame_index) {
            std::stringstream frame_tag;
            frame_tag << "frame_" << std::setw(5) << std::setfill('0') << frame_index;

            std::string input_directory = config.output_directory + "/input",
                        gt_directory    = config.output_directory + "/gt";

            auto save = [&](vk::HostBuffer& buffer, const std::string& path) {
                void* data { nullptr };
                buffer.get_device_memory().map(0, buffer.get_size(), &data);
                write_binary(path, data, buffer.get_size());
                buffer.get_device_memory().unmap();
            };

            save(input_target.coverage_buffer,   input_directory + "/" + frame_tag.str() + "_coverage.f16");
            save(input_target.tangent_buffer,    input_directory + "/" + frame_tag.str() + "_tangent.f16");
            save(input_target.motion_buffer,     input_directory + "/" + frame_tag.str() + "_motion.f16");
            save(input_target.depth_buffer,      input_directory + "/" + frame_tag.str() + "_depth.f32");
            save(gt_target.coverage_buffer,      gt_directory + "/" + frame_tag.str() + "_coverage.f16");
            save(gt_target.tangent_buffer,       gt_directory + "/" + frame_tag.str() + "_tangent.f16");
            save(gt_target.depth_buffer,         gt_directory + "/" + frame_tag.str() + "_depth.f32");

            if (config.dump_shaded) {
                save(input_target.background_buffer, input_directory + "/" + frame_tag.str() + "_background.f32");
                save(input_target.shaded_buffer,     input_directory + "/" + frame_tag.str() + "_shaded.f32");
                save(gt_target.background_buffer,    gt_directory + "/" + frame_tag.str() + "_background.f32");
                save(gt_target.shaded_buffer,        gt_directory + "/" + frame_tag.str() + "_shaded.f32");
            }
        }

        static void write_json_matrix(nlohmann::json& value, const glm::mat4& matrix) {
            for (int column { 0 }; column < 4; ++column)
                for (int row { 0 }; row < 4; ++row)
                    value.push_back(matrix[column][row]);
        }

        void GBufferRecorder::write_meta(bool complete) {
            nlohmann::json meta;

            // Written once as { "complete": false } before the first frame
            // and overwritten with { "complete": true } at the end, so an
            // interrupted dump is always detectable from the metadata.
            meta["complete"] = complete;

            meta["renderer"] = "vkhr G-buffer dump";
            meta["camera_script"] = config.camera_script;
            meta["frame_count"] = config.frame_count;
            meta["ssaa_factor"] = config.ssaa_factor;
            meta["camera_seed"] = config.camera_seed;
            meta["distance_range"] = { config.distance_min, config.distance_max };
            meta["elevation_range_deg"] = { config.elevation_min, config.elevation_max };
            meta["radius_range"] = { config.radius_min, config.radius_max };
            meta["dump_shaded"] = config.dump_shaded;
            meta["random_light"] = config.random_light;

            meta["input_resolution"] = { input_target.width, input_target.height };
            meta["gt_resolution"] = { gt_target.width, gt_target.height };

            // Conventions of the dumped channels, so downstream tooling
            // doesn't have to guess:
            meta["conventions"] = {
                { "coverage", "GPAA line coverage x per-vertex thickness taper, in [0,1]; 0 = no hair sample" },
                { "tangent", "world-space strand tangent, sign-aligned towards the camera (RGBA16F, A unused)" },
                { "motion", "backward motion in NDC units (prev_ndc - curr_ndc); pixel_motion = motion * resolution" },
                { "depth", "hardware depth in [0,1]; world_position = inv_view_projection * (ndc.xy, depth, 1)" },
                { "background", "linear-light shaded head model + clear color (RGBA32F)" },
                { "shaded", "deferred hair shading of THIS G-buffer composited over background (RGBA32F)" },
                { "hair_alpha", "not folded into coverage; applied at shading time (coverage * hair_alpha)" },
                { "matrices", "view/projection stored column-major, like glm" }
            };

            meta["frames"] = frame_metadata;

            std::ofstream file { config.output_directory + "/meta.json" };
            file << meta.dump(4) << std::endl;
        }

        void GBufferRecorder::dump(SceneGraph& scene_graph) {
            if (scene_graph.get_hair_styles().empty()) {
                std::cerr << "vkhr: the dump mode requires a scene with at least one hair style!\n";
                return;
            }

            // The dump mode always renders with the strand rasterizer, so
            // the shadow maps get baked by draw_depth.
            renderer.imgui.make_current_renderer(Renderer::Type::Rasterizer);

            write_scene_descriptors();

            if (config.strand_radius > 0.0f) {
                for (auto& hair_node : scene_graph.get_nodes_with_hair_styles())
                    for (auto& hair_style : hair_node->get_hair_styles())
                        renderer.hair_styles[hair_style].parameters.strand_radius = config.strand_radius;
            }

            std::filesystem::create_directories(config.output_directory + "/input");
            std::filesystem::create_directories(config.output_directory + "/gt");

            write_meta(false); // mark the dump as in-progress.

            previous_transform = current_transform = scene_graph.get_camera().get_transform();

            for (unsigned frame { 0 }; frame < config.frame_count; ++frame) {
                if (config.random_light) {
                    std::mt19937 light_generator { config.camera_seed * 15485u + frame };

                    std::uniform_real_distribution<float> azimuth { 0.0f, 2.0f * Pi };
                    std::uniform_real_distribution<float> elevation { glm::radians(-30.0f),
                                                                      glm::radians(+60.0f) };

                    float az = azimuth(light_generator);
                    float el = elevation(light_generator);

                    glm::vec3 direction { glm::cos(el) * glm::cos(az),
                                          glm::sin(el),
                                          glm::cos(el) * glm::sin(az) };

                    for (auto& light_source : scene_graph.light_sources) {
                        light_source.set_direction(direction);
                        light_source.update_view_matrix();
                    }

                    renderer.lights[0].update(scene_graph.fetch_light_source_buffers());
                }

                script_camera(scene_graph, frame);

                current_transform = scene_graph.get_camera().get_transform();

                std::cout << "vkhr: dumping frame " << (frame + 1) << " / "
                          << config.frame_count << "...\n" << std::flush;

                record_frame(scene_graph);
                readback_frame(frame);

                nlohmann::json frame_data {
                    { "frame", frame },
                    { "view", nlohmann::json::array() },
                    { "projection", nlohmann::json::array() },
                    { "camera_position", { current_transform.position.x,
                                           current_transform.position.y,
                                           current_transform.position.z } },
                    { "resolution", { input_target.width, input_target.height } }
                };

                write_json_matrix(frame_data["view"], current_transform.view);
                write_json_matrix(frame_data["projection"], current_transform.projection);

                frame_metadata.push_back(frame_data);

                previous_transform = current_transform;
            }

            write_meta(true);

            std::cout << "vkhr: dump complete, written to '" << config.output_directory << "'.\n";
        }
    }
}

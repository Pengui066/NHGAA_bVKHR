#include <vkhr/rasterizer/neural_pipeline.hh>

#include <vkhr/rasterizer.hh>
#include <vkhr/rasterizer/hair_style.hh>
#include <vkhr/rasterizer/model.hh>

#include <vkhr/scene_graph/camera.hh>

#include <vkpp/debug_marker.hh>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace vkhr {
    namespace vulkan {
        static constexpr auto WRITE_ALL  = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        static constexpr auto WRITE_NONE = 0;

        // -----------------------------------------------------------------
        // fp16 <-> fp32 conversion for the G-buffer readback/upload
        // (coverage R16F, tangent RGBA16F). The network works in fp32.
        // -----------------------------------------------------------------

        static float half_to_float(std::uint16_t half) {
            std::uint32_t sign = (half >> 15) & 0x1;
            std::uint32_t exp  = (half >> 10) & 0x1F;
            std::uint32_t mant = half & 0x3FF;
            std::uint32_t bits;

            if (exp == 0) {
                if (mant == 0) {
                    bits = sign << 31; // +/- zero.
                } else { // subnormal: normalize the float mantissa.
                    int e = 127 - 15;
                    std::uint32_t m = mant << 13;
                    while (!(m & 0x800000)) { m <<= 1; e--; }
                    bits = (sign << 31) | (static_cast<std::uint32_t>(e) << 23) | (m & 0x7FFFFF);
                }
            } else if (exp == 0x1F) {
                bits = (sign << 31) | 0x7F800000 | (mant << 13); // inf / nan.
            } else {
                bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
            }

            float value;
            std::memcpy(&value, &bits, sizeof(float));
            return value;
        }

        static std::uint16_t float_to_half(float value) {
            std::uint32_t bits;
            std::memcpy(&bits, &value, sizeof(std::uint32_t));

            std::uint32_t sign = (bits >> 16) & 0x8000;
            std::uint32_t exp  = (bits >> 23) & 0xFF;
            std::uint32_t mant = bits & 0x7FFFFF;

            if (exp == 0xFF) // inf / nan.
                return sign | 0x7C00 | (mant ? (0x200 | (mant >> 13)) : 0);

            int unbiased = static_cast<int>(exp) - 127 + 15;
            if (unbiased >= 0x1F) return sign | 0x7C00;   // overflow -> inf.
            if (unbiased <= 0) {
                if (unbiased < -10) return sign;           // too small -> 0.
                mant |= 0x800000;
                std::uint32_t shift = 14 - unbiased;
                std::uint32_t rounded = mant >> shift;
                if ((mant >> (shift - 1)) & 1) rounded += 1;
                return sign | rounded;
            }

            std::uint32_t rounded = (unbiased << 10) | (mant >> 13);
            if (mant & 0x1000) rounded += 1;               // round to nearest.
            return sign | rounded;
        }

        static void set_opaque_blending(vk::GraphicsPipeline::FixedFunction& fixed_stages,
                                        std::uint32_t attachment, VkColorComponentFlags write_mask) {
            if (fixed_stages.attachments.size() <= attachment)
                fixed_stages.attachments.resize(attachment + 1);

            fixed_stages.attachments[attachment].blendEnable    = VK_FALSE;
            fixed_stages.attachments[attachment].colorWriteMask = write_mask;

            fixed_stages.color_blending_state.attachmentCount = fixed_stages.attachments.size();
            fixed_stages.color_blending_state.pAttachments    = fixed_stages.attachments.data();
        }

        NeuralPipeline::NeuralPipeline(Rasterizer& vulkan_renderer, const Options& pipeline_options)
                                       : renderer { vulkan_renderer },
                                         options { pipeline_options },
                                         neural { pipeline_options.model_path,
                                                  vulkan_renderer.swap_chain.get_width(),
                                                  vulkan_renderer.swap_chain.get_height() } {
            if (renderer.hair_styles.empty())
                throw std::runtime_error {
                    "vkhr: the neural reconstruction mode requires a scene "
                    "with at least one hair style!"
                };

            camera_buffer      = vk::UniformBuffer::create(renderer.device, sizeof(ViewProjection), 1, "Neural Camera");
            prev_camera_buffer = vk::UniformBuffer::create(renderer.device, sizeof(ViewProjection), 1, "Neural Previous Camera");

            create_render_passes();
            create_pipelines();

            gbuffer_sampler = vk::Sampler {
                renderer.device,
                VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            };

            create_target(target, renderer.swap_chain.get_width(),
                                  renderer.swap_chain.get_height());

            write_descriptors();

            result = Image { target.width, target.height };
        }

        void NeuralPipeline::create_render_passes() {
            // The hair G-buffer pass: (coverage, tangent, motion) + depth.
            // Same attachment set as the offline dump mode so the shaders
            // are reused verbatim; motion is written but never read back.
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

            // The background pass renders the shaded head model.
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

            // The deferred shading pass composites the reconstructed hair
            // over the background into the final linear-light color buffer.
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

        void NeuralPipeline::create_pipelines() {
            std::uint32_t light_count = renderer.shadow_maps.size();

            struct Constants { std::uint32_t light_size; } constant_data { light_count };
            std::vector<VkSpecializationMapEntry> constants {
                { 0, 0, sizeof(std::uint32_t) }
            };

            VkExtent2D extent = renderer.swap_chain.get_extent();

            // The strand G-buffer pipeline (identical to the dump mode's).
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
            hair_gbuffer_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/strand_gbuffer.frag"));

            hair_gbuffer_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device,
                {
                    { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }, // current camera.
                    { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }  // previous camera.
                }
            };

            hair_gbuffer_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(1,
                                                                                     hair_gbuffer_pipeline.descriptor_set_layout,
                                                                                     "Neural Hair G-Buffer Descriptor Set");

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

            // The head occlusion pipeline: color writes disabled, so hair
            // behind the head fails the depth test.
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
            head_occlusion_pipeline.fixed_stages.enable_depth_test();

            set_opaque_blending(head_occlusion_pipeline.fixed_stages, 0, WRITE_NONE);
            set_opaque_blending(head_occlusion_pipeline.fixed_stages, 1, WRITE_NONE);
            set_opaque_blending(head_occlusion_pipeline.fixed_stages, 2, WRITE_NONE);

            head_occlusion_pipeline.shader_stages.emplace_back(renderer.device, SHADER("self-shadowing/depth_map.vert"));

            head_occlusion_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device
            };

            head_occlusion_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(1,
                                                                                        head_occlusion_pipeline.descriptor_set_layout,
                                                                                        "Neural Head Occlusion Descriptor Set");

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

            // The background pipeline: the head shaded as in the normal
            // color pass, into our linear-light background attachment.
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

            background_pipeline.fixed_stages.enable_depth_test();

            set_opaque_blending(background_pipeline.fixed_stages, 0, WRITE_ALL);

            background_pipeline.shader_stages.emplace_back(renderer.device, SHADER("models/model.vert"));
            background_pipeline.shader_stages.emplace_back(renderer.device, SHADER("models/model.frag"), constants, &constant_data, sizeof(constant_data));

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

            background_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(1,
                                                                                    background_pipeline.descriptor_set_layout,
                                                                                    "Neural Background Descriptor Set");

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

            // The deferred hair shading pipeline: fullscreen pass sampling
            // the RECONSTRUCTED coverage/tangent (bound in
            // write_descriptors), the hardware depth and the background.
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
            // SHADER() needs a literal (it string-concatenates the path
            // prefix), so the debug variant is selected with two branches.
            if (options.disable_occlusion)
                shading_pipeline.shader_stages.emplace_back(renderer.device,
                                                            SHADER("gbuffer/shading_no_occlusion.frag"),
                                                            constants, &constant_data, sizeof(constant_data));
            else
                shading_pipeline.shader_stages.emplace_back(renderer.device,
                                                            SHADER("gbuffer/shading.frag"),
                                                            constants, &constant_data, sizeof(constant_data));

            std::vector<vk::DescriptorSet::Binding> shading_bindings {
                { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // camera.
                { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // lights.
                { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // strand parameters.
                { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // reconstructed coverage.
                { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER },         // rendering parameters.
                { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // reconstructed tangent.
                { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // depth.
                { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }, // background.
                { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER }  // density volume.
            };

            for (std::uint32_t i { 0 }; i < light_count; ++i)
                shading_bindings.push_back({ 9 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER });

            shading_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device, shading_bindings
            };

            shading_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(1,
                                                                                 shading_pipeline.descriptor_set_layout,
                                                                                 "Neural Deferred Shading Descriptor Set");

            shading_pipeline.pipeline_layout = vk::Pipeline::Layout {
                renderer.device,
                shading_pipeline.descriptor_set_layout,
                {
                    { VK_SHADER_STAGE_ALL, 0, sizeof(ShadingPushConstants) } // inverse view-projection + alpha calibration.
                }
            };

            shading_pipeline.pipeline = vk::GraphicsPipeline {
                renderer.device,
                shading_pipeline.shader_stages,
                shading_pipeline.fixed_stages,
                shading_pipeline.pipeline_layout,
                shading_pass
            };

            // Write the descriptors that don't depend on the scene graph
            // (the camera buffers). Scene-dependent buffers are written in
            // write_descriptors().
            hair_gbuffer_pipeline.descriptor_sets[0].write(0, camera_buffer[0]);
            hair_gbuffer_pipeline.descriptor_sets[0].write(1, prev_camera_buffer[0]);

            background_pipeline.descriptor_sets[0].write(0, camera_buffer[0]);
            background_pipeline.descriptor_sets[0].write(1, renderer.lights[0]);
            background_pipeline.descriptor_sets[0].write(4, renderer.params[0]);

            shading_pipeline.descriptor_sets[0].write(0, camera_buffer[0]);
            shading_pipeline.descriptor_sets[0].write(1, renderer.lights[0]);
            shading_pipeline.descriptor_sets[0].write(4, renderer.params[0]);

            for (std::uint32_t i { 0 }; i < light_count; ++i) {
                background_pipeline.descriptor_sets[0].write(9 + i,
                                                             renderer.shadow_maps[i].get_image_view(),
                                                             renderer.shadow_maps[i].get_sampler());
                shading_pipeline.descriptor_sets[0].write(9 + i,
                                                          renderer.shadow_maps[i].get_image_view(),
                                                          renderer.shadow_maps[i].get_sampler());
            }
        }

        void NeuralPipeline::write_descriptors() {
            // Like the dump mode, the deferred shading pass shades a single
            // hair style: the first one found in the scene. The coverage/
            // tangent bindings point at the RECONSTRUCTED channels - or at
            // the raw input channels in the --nn-shade-input debug mode,
            // which isolates the network's effect on the shaded image.
            auto& style = renderer.hair_styles.begin()->second;

            shading_pipeline.descriptor_sets[0].write(2, style.parameter_buffer);
            if (options.shade_input_channels) {
                shading_pipeline.descriptor_sets[0].write(3, target.coverage.view, gbuffer_sampler);
                shading_pipeline.descriptor_sets[0].write(5, target.tangent.view, gbuffer_sampler);
            } else {
                shading_pipeline.descriptor_sets[0].write(3, target.coverage_recon.view, gbuffer_sampler);
                shading_pipeline.descriptor_sets[0].write(5, target.tangent_recon.view, gbuffer_sampler);
            }
            shading_pipeline.descriptor_sets[0].write(6, target.depth.view, gbuffer_sampler);
            shading_pipeline.descriptor_sets[0].write(7, target.background.view, gbuffer_sampler);
            shading_pipeline.descriptor_sets[0].write(8, style.density_view, style.density_sampler);
        }

        NeuralPipeline::Attachment NeuralPipeline::create_attachment(Rasterizer& vulkan_renderer,
                                                                     std::uint32_t width, std::uint32_t height,
                                                                     VkFormat format, VkImageUsageFlags extra_usage) {
            Attachment attachment;

            attachment.image = vk::Image {
                vulkan_renderer.device, width, height, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT |
                extra_usage
            };

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

            return attachment;
        }

        void NeuralPipeline::create_target(Target& target, std::uint32_t width, std::uint32_t height) {
            target.width  = width;
            target.height = height;

            target.coverage   = create_attachment(renderer, width, height, VK_FORMAT_R16_SFLOAT,          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.tangent    = create_attachment(renderer, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.motion     = create_attachment(renderer, width, height, VK_FORMAT_R16G16_SFLOAT,       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.depth      = create_attachment(renderer, width, height, VK_FORMAT_D32_SFLOAT,          VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            target.background_depth = create_attachment(renderer, width, height, VK_FORMAT_D32_SFLOAT,    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
            target.background = create_attachment(renderer, width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            target.shaded     = create_attachment(renderer, width, height, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

            // The reconstructed channels: transfer destinations filled from
            // the network output, sampled by the deferred shading pass.
            target.coverage_recon.image = vk::Image {
                renderer.device, width, height, VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            };
            target.coverage_recon.memory = vk::DeviceMemory {
                renderer.device,
                target.coverage_recon.image.get_memory_requirements(),
                vk::DeviceMemory::Type::DeviceLocal
            };
            target.coverage_recon.image.bind(target.coverage_recon.memory);
            target.coverage_recon.view = vk::ImageView {
                renderer.device, target.coverage_recon.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

            target.tangent_recon.image = vk::Image {
                renderer.device, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            };
            target.tangent_recon.memory = vk::DeviceMemory {
                renderer.device,
                target.tangent_recon.image.get_memory_requirements(),
                vk::DeviceMemory::Type::DeviceLocal
            };
            target.tangent_recon.image.bind(target.tangent_recon.memory);
            target.tangent_recon.view = vk::ImageView {
                renderer.device, target.tangent_recon.image,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

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

            target.coverage_buffer = vk::HostBuffer {
                renderer.device, width * height * 2,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
            };
            target.tangent_buffer = vk::HostBuffer {
                renderer.device, width * height * 8,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
            };
            target.depth_buffer = vk::HostBuffer {
                renderer.device, width * height * 4,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
            };
            target.shaded_buffer = vk::HostBuffer {
                renderer.device, width * height * 16,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
            };
            target.coverage_upload = vk::HostBuffer {
                renderer.device, width * height * 2,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            };
            target.tangent_upload = vk::HostBuffer {
                renderer.device, width * height * 8,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            };

            // Map the staging buffers once: per-frame vkMapMemory stalls
            // the driver on the BAR memory (measured ~500 ms per frame on
            // the first map), and the persistent pointers also let us do
            // single wide reads instead of element-wise BAR accesses.
            auto map_pointer = [&](vk::HostBuffer& buffer) {
                void* data { nullptr };
                buffer.get_device_memory().map(0, buffer.get_size(), &data);
                return data;
            };

            target.coverage_mapped  = map_pointer(target.coverage_buffer);
            target.tangent_mapped   = map_pointer(target.tangent_buffer);
            target.depth_mapped     = map_pointer(target.depth_buffer);
            target.shaded_mapped    = map_pointer(target.shaded_buffer);
            target.coverage_upload_mapped = map_pointer(target.coverage_upload);
            target.tangent_upload_mapped  = map_pointer(target.tangent_upload);

            target.viewport = VkViewport {
                0.0f, 0.0f,
                static_cast<float>(width), static_cast<float>(height),
                0.0f, 1.0f
            };

            target.scissor = VkRect2D { { 0, 0 }, { width, height } };
        }

        void NeuralPipeline::update_camera_buffers() {
            ViewProjection transform = current_transform;
            transform.resolution = glm::vec2 {
                static_cast<float>(target.width),
                static_cast<float>(target.height)
            };

            camera_buffer[0].update(transform);
            prev_camera_buffer[0].update(previous_transform);
        }

        void NeuralPipeline::bake_shadow_maps(const SceneGraph& scene_graph,
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

        void NeuralPipeline::draw_head_occlusion(const SceneGraph& scene_graph,
                                                 vk::CommandBuffer& command_buffer) {
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

        void NeuralPipeline::draw_hair_gbuffer(const SceneGraph& scene_graph,
                                               vk::CommandBuffer& command_buffer) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            command_buffer.bind_pipeline(hair_gbuffer_pipeline);
            command_buffer.bind_descriptor_set(hair_gbuffer_pipeline.descriptor_sets[0],
                                               hair_gbuffer_pipeline);

            for (auto& hair_node : scene_graph.get_nodes_with_hair_styles()) {
                for (auto& hair_style : hair_node->get_hair_styles()) {
                    auto& style = renderer.hair_styles[hair_style];

                    float strand_width = style.parameters.strand_radius;

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

        void NeuralPipeline::draw_background(const SceneGraph& scene_graph,
                                             vk::CommandBuffer& command_buffer) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            command_buffer.bind_pipeline(background_pipeline);

            vk::DescriptorSet& descriptor_set = background_pipeline.descriptor_sets[0];

            for (auto& model_node : scene_graph.get_nodes_with_models()) {
                command_buffer.push_constant(background_pipeline, 0, model_node->get_model_matrix());
                for (auto& model_mesh : model_node->get_models())
                    renderer.models[model_mesh].draw(background_pipeline, descriptor_set, command_buffer);
            }
        }

        void NeuralPipeline::draw_deferred_shading(vk::CommandBuffer& command_buffer) {
            command_buffer.set_viewport(target.viewport);
            command_buffer.set_scissor(target.scissor);

            command_buffer.bind_pipeline(shading_pipeline);

            ShadingPushConstants push_constants {
                glm::inverse(current_transform.projection * current_transform.view),
                options.alpha_mode, options.alpha_layers
            };
            command_buffer.push_constant(shading_pipeline, 0, push_constants);

            command_buffer.bind_descriptor_set(shading_pipeline.descriptor_sets[0],
                                               shading_pipeline);

            command_buffer.draw(3); // fullscreen triangle.
        }

        void NeuralPipeline::run(const SceneGraph& scene_graph) {
            using clock = std::chrono::steady_clock;
            const auto total_start = clock::now();

            current_transform = scene_graph.get_camera().get_transform();
            update_camera_buffers();

            VkClearValue zero { }, depth_one { }, white { };
            zero.color      = { { 0.0f, 0.0f, 0.0f, 0.0f } };
            white.color     = { { 1.0f, 1.0f, 1.0f, 1.0f } };
            depth_one.depthStencil = { 1.0f, 0 };

            // ---- command buffer 1: shadows + G-buffer + background ----
            auto command_buffer = renderer.command_pool.allocate_and_begin();

            bake_shadow_maps(scene_graph, command_buffer);

            std::vector<VkClearValue> gbuffer_clears { zero, zero, zero, depth_one };

            command_buffer.begin_render_pass(gbuffer_pass, target.gbuffer_framebuffer, gbuffer_clears);
            draw_head_occlusion(scene_graph, command_buffer);
            draw_hair_gbuffer(scene_graph, command_buffer);
            command_buffer.end_render_pass();

            std::vector<VkClearValue> background_clears { white, depth_one };

            command_buffer.begin_render_pass(background_pass, target.background_framebuffer, background_clears);
            draw_background(scene_graph, command_buffer);
            command_buffer.end_render_pass();

            auto transition_to_source = [&](Attachment& attachment) {
                attachment.image.transition(command_buffer,
                                            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                            VK_PIPELINE_STAGE_TRANSFER_BIT);
            };

            transition_to_source(target.coverage);
            transition_to_source(target.tangent);

            command_buffer.copy_image_to_buffer(target.coverage.image, target.coverage_buffer);
            command_buffer.copy_image_to_buffer(target.tangent.image, target.tangent_buffer);

            if (!options.debug_dump_directory.empty()) {
                target.depth.image.transition(command_buffer,
                                              VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                              VK_PIPELINE_STAGE_TRANSFER_BIT);
                command_buffer.copy_image_to_buffer(target.depth.image, target.depth_buffer);
            }

            command_buffer.end();
            renderer.device.get_graphics_queue().submit(command_buffer).wait_idle();

            const auto nn_start = clock::now();
            last_timing_.render_ms = std::chrono::duration<double, std::milli>
                                     (nn_start - total_start).count();

            // ---- host: readback, fp16 -> fp32 + sanitize, inference ----
            const std::size_t pixel_count = std::size_t(target.width) * target.height;

            std::vector<float> coverage_f32(pixel_count), tangent_f32(3 * pixel_count);
            std::vector<float> recon_coverage_f32(pixel_count), recon_tangent_f32(3 * pixel_count);
            std::vector<std::uint16_t> half_staging;

            // Stage the BAR-mapped buffers through one wide memcpy each:
            // element-wise reads from BAR memory are uncached-slow, while
            // a single wide read is fast (wide WRITES to the BAR mapping
            // fault, so the upload side stays scalar below).
            half_staging.resize(2 * pixel_count);
            std::memcpy(half_staging.data(), target.coverage_mapped,
                        half_staging.size() * sizeof(std::uint16_t));
            for (std::size_t i { 0 }; i < pixel_count; ++i) {
                float value = half_to_float(half_staging[i]);
                // Zero-length tangents normalize to NaN in the shader and
                // coverage has the same failure mode; sanitize both before
                // the network (contract: nan_to_num is the caller's
                // responsibility, and it applies to ALL input channels).
                coverage_f32[i] = std::isfinite(value) ? value : 0.0f;
            }

            half_staging.resize(4 * pixel_count);
            std::memcpy(half_staging.data(), target.tangent_mapped,
                        half_staging.size() * sizeof(std::uint16_t));
            for (std::size_t i { 0 }; i < pixel_count; ++i)
                for (std::uint32_t c { 0 }; c < 3; ++c) {
                    float value = half_to_float(half_staging[4 * i + c]);
                    tangent_f32[3 * i + c] = std::isfinite(value) ? value : 0.0f;
                }

            // The readback window ends HERE: it covers only the host-side
            // staging and conversion, not the network run.
            const auto readback_end = clock::now();
            last_timing_.readback_ms = std::chrono::duration<double, std::milli>
                                       (readback_end - nn_start).count();

            neural.run(coverage_f32.data(), tangent_f32.data(),
                       recon_coverage_f32.data(), recon_tangent_f32.data(),
                       last_timing_.neural);

            const auto upload_start = clock::now();
            last_timing_.nn_ms = std::chrono::duration<double, std::milli>
                                 (upload_start - readback_end).count();

            // Pack the reconstructed channels back into fp16, writing the
            // persistently mapped upload buffers element by element through
            // VOLATILE pointers: wide vector stores fault on the
            // write-combined BAR mapping, and volatile forbids the
            // optimizer from coalescing the writes into such stores
            // (observed to resurface after unrelated code changes, since
            // the vectorization decision depends on the surrounding code).
            {
                volatile std::uint16_t* halfs =
                    static_cast<volatile std::uint16_t*>(target.coverage_upload_mapped);
                for (std::size_t i { 0 }; i < pixel_count; ++i)
                    halfs[i] = float_to_half(recon_coverage_f32[i]);
            }

            {
                volatile std::uint16_t* halfs =
                    static_cast<volatile std::uint16_t*>(target.tangent_upload_mapped);
                for (std::size_t i { 0 }; i < pixel_count; ++i) {
                    halfs[4 * i + 0] = float_to_half(recon_tangent_f32[3 * i + 0]);
                    halfs[4 * i + 1] = float_to_half(recon_tangent_f32[3 * i + 1]);
                    halfs[4 * i + 2] = float_to_half(recon_tangent_f32[3 * i + 2]);
                    halfs[4 * i + 3] = 0;
                }
            }

            // ---- command buffer 2: upload recon + shade + readback ----
            auto upload_buffer = renderer.command_pool.allocate_and_begin();

            VkImageLayout recon_old_layout = frame_index == 0
                                           ? VK_IMAGE_LAYOUT_UNDEFINED
                                           : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            target.coverage_recon.image.transition(upload_buffer,
                                                   VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                   recon_old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                                   VK_PIPELINE_STAGE_TRANSFER_BIT);
            target.tangent_recon.image.transition(upload_buffer,
                                                  VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                  recon_old_layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                                  VK_PIPELINE_STAGE_TRANSFER_BIT);

            upload_buffer.copy_buffer_image(target.coverage_upload, target.coverage_recon.image);
            upload_buffer.copy_buffer_image(target.tangent_upload, target.tangent_recon.image);

            target.coverage_recon.image.transition(upload_buffer,
                                                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            target.tangent_recon.image.transition(upload_buffer,
                                                  VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

            // The --nn-shade-input debug mode samples the RAW input images
            // in the shading pass; bring them back from the transfer
            // layout cmd1 left them in.
            if (options.shade_input_channels) {
                target.coverage.image.transition(upload_buffer,
                                                 VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
                target.tangent.image.transition(upload_buffer,
                                                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
            }

            std::vector<VkClearValue> shading_clears { white };

            upload_buffer.begin_render_pass(shading_pass, target.shading_framebuffer, shading_clears);
            draw_deferred_shading(upload_buffer);
            upload_buffer.end_render_pass();

            target.shaded.image.transition(upload_buffer,
                                           VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                           VK_PIPELINE_STAGE_TRANSFER_BIT);

            upload_buffer.copy_image_to_buffer(target.shaded.image, target.shaded_buffer);

            upload_buffer.end();
            renderer.device.get_graphics_queue().submit(upload_buffer).wait_idle();

            const auto shade_end = clock::now();
            last_timing_.upload_shade_ms = std::chrono::duration<double, std::milli>
                                           (shade_end - upload_start).count();

            // ---- host: linear RGBA32F -> RGBA8 result image ----
            const auto convert_start = clock::now();
            {
                std::vector<float> shaded_f32(4 * pixel_count);
                std::memcpy(shaded_f32.data(), target.shaded_mapped,
                            shaded_f32.size() * sizeof(float));

                const float* rgba = shaded_f32.data();
                Color* pixels = result.get_pixels();

                for (std::size_t i { 0 }; i < pixel_count; ++i) {
                    for (std::uint32_t c { 0 }; c < 4; ++c) {
                        float channel = glm::clamp(rgba[4 * i + c], 0.0f, 1.0f);
                        pixels[i][c] = static_cast<unsigned char>(channel * 255.0f + 0.5f);
                    }
                }
            }
            const auto convert_end = clock::now();
            last_timing_.result_convert_ms = std::chrono::duration<double, std::milli>
                                             (convert_end - shade_end).count();

            last_timing_.total_ms = std::chrono::duration<double, std::milli>
                                    (convert_end - total_start).count();
            stats_.record(last_timing_);

            if (frame_index == 0 && !options.debug_dump_directory.empty()
                && !debug_dump_written_) {
                debug_dump_written_ = true;
                write_debug_dump(scene_graph, coverage_f32, tangent_f32,
                                  recon_coverage_f32, recon_tangent_f32);
                debug_dump_complete_ = true;
            }

            previous_transform = current_transform;
            frame_index += 1;

            if (frame_index % 60 == 0) {
                std::cerr << "[nn] frame " << frame_index
                          << " | render " << last_timing_.render_ms << " ms"
                          << " | readback " << last_timing_.readback_ms << " ms"
                          << " | nn " << last_timing_.nn_ms << " ms"
                          << " (pack " << last_timing_.neural.pack_ms
                          << " + inf " << last_timing_.neural.inference_ms
                          << " + unpack " << last_timing_.neural.unpack_ms << ")"
                          << " | upload+shade " << last_timing_.upload_shade_ms << " ms"
                          << " | result " << last_timing_.result_convert_ms << " ms"
                          << " | total " << last_timing_.total_ms << " ms" << std::endl;
            }

            if (frame_index % 100 == 0) {
                double sum = stats_.mean(stats_.render_sum) + stats_.mean(stats_.readback_sum)
                           + stats_.mean(stats_.nn_sum) + stats_.mean(stats_.upload_sum)
                           + stats_.mean(stats_.result_sum);
                std::cerr << "[nn] stats @" << frame_index << " frames: "
                          << "render " << stats_.mean(stats_.render_sum)
                          << " | readback " << stats_.mean(stats_.readback_sum)
                          << " | nn " << stats_.mean(stats_.nn_sum)
                          << " | upload+shade " << stats_.mean(stats_.upload_sum)
                          << " | result " << stats_.mean(stats_.result_sum)
                          << " | total " << stats_.mean(stats_.total_sum)
                          << " | segment-sum " << sum
                          << " | total p95 " << stats_.p95()
                          << " | fps " << stats_.fps() << std::endl;
            }
        }

        void NeuralPipeline::write_debug_dump(const SceneGraph& scene_graph,
                                              const std::vector<float>& coverage_f32,
                                              const std::vector<float>& tangent_f32,
                                              const std::vector<float>& recon_coverage,
                                              const std::vector<float>& recon_tangent) {
            const std::string& directory = options.debug_dump_directory;
            std::filesystem::create_directories(directory);

            auto write_raw = [&](const char* name, const void* data, std::size_t size) {
                std::ofstream file { directory + "/" + name, std::ios::binary };
                file.write(static_cast<const char*>(data), size);
            };

            const std::size_t pixel_count = std::size_t(target.width) * target.height;
            write_raw("coverage_readback.f16", target.coverage_mapped, pixel_count * 2);
            write_raw("tangent_readback.f16", target.tangent_mapped, pixel_count * 8);
            write_raw("depth_readback.f32", target.depth_mapped, pixel_count * 4);
            write_raw("coverage_decoded.f32", coverage_f32.data(), pixel_count * 4);
            write_raw("tangent_decoded.f32", tangent_f32.data(), 3 * pixel_count * 4);
            write_raw("recon_coverage.f32", recon_coverage.data(), pixel_count * 4);
            write_raw("recon_tangent.f32", recon_tangent.data(), 3 * pixel_count * 4);
            write_raw("shaded.f32", target.shaded_mapped, 4 * pixel_count * 4);

            auto& style = renderer.hair_styles.begin()->second;

            nlohmann::json meta;
            meta["resolution"] = { target.width, target.height };
            meta["view"] = nlohmann::json::array();
            meta["projection"] = nlohmann::json::array();
            for (int column { 0 }; column < 4; ++column)
                for (int row { 0 }; row < 4; ++row) {
                    meta["view"].push_back(current_transform.view[column][row]);
                    meta["projection"].push_back(current_transform.projection[column][row]);
                }
            meta["camera_position"] = { current_transform.position.x,
                                        current_transform.position.y,
                                        current_transform.position.z };
            meta["strand_radius"] = style.parameters.strand_radius;

            // F-23: the live shading parameters, for the two-process
            // comparison against the offline shade run.
            {
                auto& p = renderer.imgui.parameters;
                meta["live_params"] = {
                    { "shading_model", p.shading_model },
                    { "adsm_kernel_size", p.adsm_kernel_size },
                    { "adsm_stride_size", p.adsm_stride_size },
                    { "adsm_on", p.adsm_on },
                    { "ctsm_kernel_size", p.ctsm_kernel_size },
                    { "ctsm_on", p.ctsm_on },
                    { "occlusion_radius", p.occlusion_radius },
                    { "ao_exponent", p.ao_exponent },
                    { "ao_clamp", p.ao_clamp }
                };
            }

            // The live path shades with the SCENE light as loaded; record
            // it so the live-vs-offline comparison can be checked for a
            // light mismatch (the offline replay derives its light from
            // the dump meta's recorded azimuth/elevation).
            {
                nlohmann::json live_lights = nlohmann::json::array();
                for (auto& light_source : scene_graph.get_light_sources()) {
                    auto direction = light_source.get_direction();
                    auto intensity = light_source.get_intensity();
                    live_lights.push_back({
                        { "direction", { direction.x, direction.y, direction.z } },
                        { "intensity", { intensity.x, intensity.y, intensity.z } }
                    });
                }
                meta["live_scene_lights"] = live_lights;
            }
            meta["hair_color"] = { style.parameters.hair_color.x,
                                   style.parameters.hair_color.y,
                                   style.parameters.hair_color.z };
            meta["hair_opacity"] = style.parameters.hair_opacity;
            meta["hair_shininess"] = style.parameters.hair_shininess;
            meta["channels"] = {
                { "coverage_readback.f16", "raw R16F readback of the G-buffer coverage attachment" },
                { "tangent_readback.f16", "raw RGBA16F readback of the G-buffer tangent attachment" },
                { "recon_coverage.f32", "network output coverage, fp32, before upload" },
                { "recon_tangent.f32", "network output tangent, fp32, before upload" },
                { "shaded.f32", "deferred shading result, linear RGBA32F" }
            };

            std::ofstream meta_file { directory + "/meta.json" };
            meta_file << meta.dump(4) << std::endl;

            std::cerr << "[nn] debug dump written to '" << directory << "'" << std::endl;
        }
    }
}

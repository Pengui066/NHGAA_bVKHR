#include <vkhr/rasterizer/shade_dump.hh>

#include <vkhr/rasterizer.hh>
#include <vkhr/rasterizer/hair_style.hh>
#include <vkhr/rasterizer/model.hh>

#include <vkhr/scene_graph/camera.hh>
#include <vkhr/scene_graph/light_source.hh>

#include <vkpp/debug_marker.hh>

#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace vkhr {
    namespace vulkan {
        static constexpr auto WRITE_ALL = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        static void set_opaque_blending(vk::GraphicsPipeline::FixedFunction& fixed_stages,
                                        std::uint32_t attachment, VkColorComponentFlags write_mask) {
            if (fixed_stages.attachments.size() <= attachment)
                fixed_stages.attachments.resize(attachment + 1);

            fixed_stages.attachments[attachment].blendEnable    = VK_FALSE;
            fixed_stages.attachments[attachment].colorWriteMask = write_mask;

            fixed_stages.color_blending_state.attachmentCount = fixed_stages.attachments.size();
            fixed_stages.color_blending_state.pAttachments    = fixed_stages.attachments.data();
        }

        ShadeDump::ShadeDump(Rasterizer& vulkan_renderer, SceneGraph& scene,
                             const ShadeDumpOptions& dump_options,
                             const nlohmann::json& dump_meta)
                            : renderer { vulkan_renderer },
                              scene_graph { scene },
                              options(dump_options),
                              meta(dump_meta) {
            std::cerr << "[shade] dump dir: [" << options.dump_directory << "]" << std::endl;
            std::cerr << "[shade] meta type: " << meta.type_name()
                      << " | is_object: " << meta.is_object()
                      << " | has input_resolution: " << (meta.find("input_resolution") != meta.end())
                      << " | has frames: " << (meta.find("frames") != meta.end()) << std::endl;
            auto input_resolution = meta["input_resolution"];
            width  = input_resolution[0];
            height = input_resolution[1];

            std::cerr << "[shade] ctor: render pass done" << std::endl;
            create_render_pass();
            std::cerr << "[shade] ctor: creating pipeline" << std::endl;
            create_pipeline();
            create_buffers();
            std::cerr << "[shade] ctor: buffers done" << std::endl;
        }

        void ShadeDump::create_render_pass() {
            std::vector<vk::RenderPass::Attachment> attachments {
                { VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }
            };

            std::vector<VkAttachmentReference> subpass {
                { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }
            };

            vk::RenderPass::Dependency dependency {
                0, VK_SUBPASS_EXTERNAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_TRANSFER_READ_BIT
            };

            shading_pass = vk::RenderPass {
                renderer.device,
                attachments,
                std::vector<vk::RenderPass::Subpass> { subpass },
                std::vector<vk::RenderPass::Dependency> { dependency }
            };
        }

        void ShadeDump::create_pipeline() {
            std::uint32_t light_count = renderer.shadow_maps.size();

            struct Constants { std::uint32_t light_size; } constant_data { light_count };
            std::vector<VkSpecializationMapEntry> constants {
                { 0, 0, sizeof(std::uint32_t) }
            };

            VkExtent2D extent { width, height };

            shading_pipeline = Pipeline { };

            shading_pipeline.fixed_stages.set_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
            shading_pipeline.fixed_stages.set_scissor({ 0, 0, extent });
            shading_pipeline.fixed_stages.set_viewport({ 0.0f, 0.0f,
                                                         static_cast<float>(width),
                                                         static_cast<float>(height),
                                                         0.0f, 1.0f });
            shading_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
            shading_pipeline.fixed_stages.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);
            shading_pipeline.fixed_stages.set_culling_mode(VK_CULL_MODE_NONE);
            shading_pipeline.fixed_stages.disable_depth_test();

            set_opaque_blending(shading_pipeline.fixed_stages, 0, WRITE_ALL);

            shading_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/fullscreen.vert"));
            vk::DebugMarker::object_name(renderer.device, shading_pipeline.shader_stages[0], VK_OBJECT_TYPE_SHADER_MODULE, "Shade Dump Vertex Shader");
            shading_pipeline.shader_stages.emplace_back(renderer.device, SHADER("gbuffer/shading.frag"), constants, &constant_data, sizeof(constant_data));
            vk::DebugMarker::object_name(renderer.device, shading_pipeline.shader_stages[1], VK_OBJECT_TYPE_SHADER_MODULE, "Shade Dump Fragment Shader");

            std::vector<vk::DescriptorSet::Binding> bindings {
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
                bindings.push_back({ 9 + i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER });

            shading_pipeline.descriptor_set_layout = vk::DescriptorSet::Layout {
                renderer.device, bindings
            };

            shading_pipeline.descriptor_sets = renderer.descriptor_pool.allocate(1,
                                                                                 shading_pipeline.descriptor_set_layout,
                                                                                 "Shade Dump Descriptor Set");

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
        }

        ShadeDump::Attachment ShadeDump::create_attachment(std::uint32_t w, std::uint32_t h,
                                                           VkFormat format,
                                                           VkImageUsageFlags extra_usage) {
            Attachment attachment;

            attachment.image = vk::Image {
                renderer.device, w, h, format,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT |
                extra_usage
            };

            attachment.memory = vk::DeviceMemory {
                renderer.device,
                attachment.image.get_memory_requirements(),
                vk::DeviceMemory::Type::DeviceLocal
            };

            attachment.image.bind(attachment.memory);

            attachment.view = vk::ImageView {
                renderer.device,
                attachment.image
            };

            return attachment;
        }

        void ShadeDump::create_buffers() {
            channel_sampler = vk::Sampler {
                renderer.device,
                VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
            };

            camera_buffer = vk::UniformBuffer::create(renderer.device, sizeof(ViewProjection), 1, "Shade Camera");

            coverage   = create_attachment(width, height, VK_FORMAT_R16_SFLOAT,          {});
            tangent    = create_attachment(width, height, VK_FORMAT_R16G16B16A16_SFLOAT, {});
            depth      = create_attachment(width, height, VK_FORMAT_R32_SFLOAT,          {});
            background = create_attachment(width, height, VK_FORMAT_R32G32B32A32_SFLOAT, {});
            shaded     = create_attachment(width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);

            shading_attachments.emplace_back(renderer.device, shaded.image);
            shading_framebuffer = vk::Framebuffer {
                renderer.device, shading_pass, shading_attachments, VkExtent2D { width, height }
            };

            staging_buffer = vk::HostBuffer {
                renderer.device, static_cast<VkDeviceSize>(width) * height * 16,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            };

            readback_buffer = vk::HostBuffer {
                renderer.device, static_cast<VkDeviceSize>(width) * height * 16,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT
            };
        }

        void ShadeDump::setup_frame_parameters(const nlohmann::json& frame_meta) {
            // New dumps record the exact per-frame strand radius; old ones
            // fall back to re-deriving the RNG draw (may not match exactly).
            float strand_radius;
            if (frame_meta.find("strand_radius") != frame_meta.end()) {
                strand_radius = frame_meta["strand_radius"];
            } else {
                auto radius_range = meta["radius_range"];
                if (!radius_range.is_array() || radius_range[1] <= radius_range[0] ||
                    radius_range[0] <= 0.0f)
                    return;
                unsigned seed = meta.value("camera_seed", 0u);
                std::mt19937 radius_generator { seed * 7919u + frame_meta["frame"] };
                std::uniform_real_distribution<float> radius { radius_range[0], radius_range[1] };
                strand_radius = radius(radius_generator);
            }

            for (auto& hair_node : scene_graph.get_nodes_with_hair_styles())
                for (auto& hair_style : hair_node->get_hair_styles()) {
                    renderer.hair_styles[hair_style].parameters.strand_radius = strand_radius;
                    renderer.hair_styles[hair_style].update_parameters();
                }
        }

        void ShadeDump::setup_frame_light(const nlohmann::json& frame_meta,
                                          unsigned frame_index) {
            glm::vec3 direction;

            if (frame_meta.find("light_azimuth_rad") != frame_meta.end()) {
                // New dumps record the exact per-frame light direction.
                float az = frame_meta["light_azimuth_rad"];
                float el = frame_meta["light_elevation_rad"];
                direction = glm::vec3 { glm::cos(el) * glm::cos(az), glm::sin(el),
                                        glm::cos(el) * glm::sin(az) };
            } else if (meta.value("random_light", false)) {
                // Old dump: re-derive from the seed (may not match exactly).
                unsigned seed = meta.value("camera_seed", 0u);
                std::mt19937 generator { seed * 15485u + frame_index };
                std::uniform_real_distribution<float> azimuth { 0.0f, 2.0f * glm::pi<float>() };
                std::uniform_real_distribution<float> elevation { glm::radians(-30.0f),
                                                                  glm::radians(60.0f) };
                float az = azimuth(generator);
                float el = elevation(generator);
                direction = glm::vec3 { glm::cos(el) * glm::cos(az), glm::sin(el),
                                        glm::cos(el) * glm::sin(az) };
            } else {
                return; // static scene light as loaded.
            }

            for (auto& light_source : scene_graph.light_sources) {
                light_source.set_direction(direction);
                light_source.update_view_matrix();
            }
            renderer.lights[0].update(scene_graph.fetch_light_source_buffers());
        }

        void ShadeDump::bake_shadow_maps(vk::CommandBuffer& command_buffer) {
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

        void ShadeDump::update_camera_buffer(const nlohmann::json& frame_meta) {
            camera_transform.view = glm::make_mat4(frame_meta["view"].get<std::vector<float>>().data());
            camera_transform.projection = glm::make_mat4(frame_meta["projection"].get<std::vector<float>>().data());
            auto position = frame_meta["camera_position"].get<std::vector<float>>();
            camera_transform.position = glm::vec3 { position[0], position[1], position[2] };
            camera_transform.look_at_distance = 0.0f;
            camera_transform.near = 1.0f;
            camera_transform.far = 10000.0f;
            camera_transform.resolution = glm::vec2 { static_cast<float>(width),
                                                      static_cast<float>(height) };
            camera_buffer[0].update(camera_transform);
        }

        void ShadeDump::upload_texture(vk::Image& image, const std::string& path) {
            std::ifstream file { path, std::ios::binary | std::ios::ate };
            if (!file) {
                std::cerr << "shade: couldn't open '" << path << "'!\n";
                return;
            }
            std::size_t size = static_cast<std::size_t>(file.tellg());
            file.seekg(0);

            void* mapped { nullptr };
            staging_buffer.get_device_memory().map(0, staging_buffer.get_size(), &mapped);
            std::memset(mapped, 0, staging_buffer.get_size());
            file.read(reinterpret_cast<char*>(mapped), size);
            staging_buffer.get_device_memory().unmap();

            auto command_buffer = renderer.command_pool.allocate_and_begin();
            image.transition(command_buffer, VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            command_buffer.copy_buffer_image(staging_buffer, image);
            image.transition(command_buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            command_buffer.end();
            renderer.device.get_graphics_queue().submit(command_buffer).wait_idle();
        }

        void ShadeDump::shade_frame() {
            auto command_buffer = renderer.command_pool.allocate_and_begin();

            VkClearValue white { };
            white.color = { { 1.0f, 1.0f, 1.0f, 1.0f } };

            command_buffer.begin_render_pass(shading_pass, shading_framebuffer, white);
            command_buffer.set_viewport(shading_pipeline.fixed_stages.viewport);
            command_buffer.set_scissor(shading_pipeline.fixed_stages.scissor);
            command_buffer.bind_pipeline(shading_pipeline);

            glm::mat4 inverse_view_projection = glm::inverse(camera_transform.projection * camera_transform.view);
            command_buffer.push_constant(shading_pipeline, 0, inverse_view_projection);

            command_buffer.bind_descriptor_set(shading_pipeline.descriptor_sets[0], shading_pipeline);
            command_buffer.draw(3);
            command_buffer.end_render_pass();

            shaded.image.transition(command_buffer,
                                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT);
            command_buffer.copy_image_to_buffer(shaded.image, readback_buffer);
            command_buffer.end();

            renderer.device.get_graphics_queue().submit(command_buffer).wait_idle();
        }

        void ShadeDump::save_output(const std::string& path) {
            void* data { nullptr };
            readback_buffer.get_device_memory().map(0, readback_buffer.get_size(), &data);
            std::ofstream file { path, std::ios::binary };
            if (!file) {
                std::cerr << "shade: couldn't open '" << path << "' for writing!\n";
            } else {
                file.write(reinterpret_cast<const char*>(data), readback_buffer.get_size());
            }
            readback_buffer.get_device_memory().unmap();
        }

        void ShadeDump::run() {
            std::cerr << "[shade] run entered" << std::endl;
            std::cerr << "[shade] run: writing descriptors" << std::endl;
            auto& style = renderer.hair_styles.begin()->second;

            auto& descriptor_set = shading_pipeline.descriptor_sets[0];
            descriptor_set.write(2, style.parameter_buffer);
            descriptor_set.write(4, renderer.params[0]);
            descriptor_set.write(8, style.density_view, style.density_sampler);
            for (std::uint32_t j { 0 }; j < renderer.shadow_maps.size(); ++j)
                descriptor_set.write(9 + j, renderer.shadow_maps[j].get_image_view(),
                                     renderer.shadow_maps[j].get_sampler());

            std::string source_dir = options.dump_directory + "/" + options.source;
            std::string input_dir = options.dump_directory + "/input";

            std::string output_path = options.output_file;
            if (output_path.empty()) {
                // Default: never overwrite the dump's own shaded images.
                output_path = options.dump_directory + "/" + options.source + "/shaded_offline.f32";
            }

            std::cerr << "[shade] run: initial shadow bake" << std::endl;
            auto bake_commands = renderer.command_pool.allocate_and_begin();
            bake_shadow_maps(bake_commands);
            bake_commands.end();
            renderer.device.get_graphics_queue().submit(bake_commands).wait_idle();

            for (auto& frame_meta : meta["frames"]) {
                unsigned frame_index = frame_meta["frame"].get<unsigned>();

                std::stringstream tag;
                tag << "frame_" << std::setw(5) << std::setfill('0') << frame_index;

                // Per-frame output: <source>/frame_XXXXX_shaded_offline.f32
                // (or the explicit --shade-out path with the tag inserted).
                std::string frame_output = options.output_file;
                if (frame_output.empty()) {
                    frame_output = source_dir + "/" + tag.str() + "_shaded_offline.f32";
                } else {
                    auto dot = frame_output.rfind(".f32");
                    frame_output.insert(dot, "_" + tag.str());
                }

                std::cout << "shade: frame " << (frame_index + 1) << "...\n" << std::flush;

                setup_frame_light(frame_meta, frame_index);
                update_camera_buffer(frame_meta);

                bake_commands = renderer.command_pool.allocate_and_begin();
                bake_shadow_maps(bake_commands);
                bake_commands.end();
                renderer.device.get_graphics_queue().submit(bake_commands).wait_idle();

                upload_texture(coverage.image, source_dir + "/" + tag.str() + "_coverage.f16");
                upload_texture(tangent.image, source_dir + "/" + tag.str() + "_tangent.f16");
                upload_texture(depth.image, source_dir + "/" + tag.str() + "_depth.f32");
                upload_texture(background.image, input_dir + "/" + tag.str() + "_background.f32");

                auto& descriptor_set = shading_pipeline.descriptor_sets[0];
                descriptor_set.write(0, camera_buffer[0]);
                descriptor_set.write(1, renderer.lights[0]);
                descriptor_set.write(3, coverage.view, channel_sampler);
                descriptor_set.write(5, tangent.view, channel_sampler);
                descriptor_set.write(6, depth.view, channel_sampler);
                descriptor_set.write(7, background.view, channel_sampler);

                shade_frame();
                save_output(frame_output);
            }

            std::cout << "shade: output written to '" << output_path << "'.\n";
        }
    }
}

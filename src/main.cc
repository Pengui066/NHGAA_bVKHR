#include <vkhr/arg_parser.hh>
#include <vkhr/paths.hh>
#include <vkhr/image.hh>
#include <vkhr/window.hh>
#include <vkhr/input_map.hh>

#include <vkhr/rasterizer.hh>
#include <vkhr/scene_graph.hh>
#include <vkhr/benchmark.hh>
#include <vkhr/ray_tracer.hh>

#include <glm/glm.hpp>

int main(int argc, char** argv) {
    vkhr::ArgParser argp { vkhr::arguments };
    auto scene_file = argp.parse(argc, argv);

    if (scene_file.empty()) scene_file = SCENE("ponytail.vkhr");

    vkhr::SceneGraph scene_graph { scene_file };
    auto& camera { (scene_graph.get_camera()) };

    int width  = argp["x"].value.integer,
        height = argp["y"].value.integer;

    camera.set_resolution(width, height);

    vkhr::Raytracer ray_tracer { scene_graph };

    const vkhr::Image vulkan_icon { IMAGE("vulkan_icon.png") };
    vkhr::Window window { width, height, "VKHR", vulkan_icon };

    if (argp["fullscreen"].value.boolean)
        window.toggle_fullscreen();

    window.enable_vsync(argp["vsync"].value.boolean);

    if (argp["benchmark"].value.boolean == 1)
        window.enable_vsync(false);

    vkhr::InputMap input_map { window };

    input_map.bind("toggle_ui", vkhr::Input::Key::U);
    input_map.bind("grab", vkhr::Input::MouseButton::Left);
    input_map.bind("make_fullscreen", std::vector<vkhr::Input::Key> { vkhr::Input::Key::F11, vkhr::Input::Key::F });
    input_map.bind("take_screenshot", vkhr::Input::Key::S);
    input_map.bind("quit", std::vector<vkhr::Input::Key> { vkhr::Input::Key::Escape, vkhr::Input::Key::Q });
    input_map.bind("toggle_renderer", vkhr::Input::Key::T);
    input_map.bind("pan", vkhr::Input::MouseButton::Middle);
    input_map.bind("rotate_light", vkhr::Input::Key::L);
    input_map.bind("recompile", vkhr::Input::Key::R);

    vkhr::Rasterizer rasterizer { window, scene_graph }; // Rasterizer构造，建齐全部Vulkan资源/pipeline

    if (argp["ui"].value.boolean == 0)
        rasterizer.get_imgui().hide();

    auto& imgui = rasterizer.get_imgui();

    window.show();

    if (argp["benchmark"].value.boolean == 1) {
        vkhr::Benchmark::construct(rasterizer);
        rasterizer.run_benchmarks(scene_graph);
    }

    while (window.is_open()) { // render loop
        if (input_map.just_pressed("quit")) { // handle key events
            window.close();
        } else if (input_map.just_pressed("toggle_ui")) {
            imgui.toggle_visibility();
        } else if (input_map.just_pressed("make_fullscreen")) {
            window.toggle_fullscreen();
        } else if (input_map.just_pressed("take_screenshot")) {
            rasterizer.get_screenshot(scene_graph, ray_tracer)
                      .save_time(); // label using date/time.
        } else if (input_map.just_pressed("toggle_renderer")) {
            imgui.toggle_renderer();
        } else if (input_map.just_pressed("rotate_light")) {
            imgui.toggle_light_rotation();
        } else if (input_map.just_pressed("recompile")) {
            rasterizer.recompile();
        }

        camera.control(input_map, window.update_delta_time(),
                       rasterizer.get_imgui().wants_focus());

        scene_graph.traverse_nodes();

        imgui.transform(scene_graph, rasterizer, ray_tracer);

        if (window.surface_is_dirty() || rasterizer.swapchain_is_dirty()) {
            ray_tracer.recreate(window.get_width(), window.get_height());
            rasterizer.recreate_swapchain(window, scene_graph); // slow!?
        }

        if (imgui.raytracing_enabled()) { // 如果光追开启，默认关闭，可以忽略
            ray_tracer.draw(scene_graph);
            auto& framebuffer = ray_tracer.get_framebuffer();
            rasterizer.draw(framebuffer);
        } else {
            rasterizer.draw(scene_graph); // 真正的渲染操作只有这一句
        }

        // Benchmark the renderer and dump timings.
        if (argp["benchmark"].value.boolean == 1) {
            if (!rasterizer.benchmark(scene_graph))
                return 0; // benchmark is complete!
        }

        window.poll_events();
    }

    return 0;
}

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
#include <iostream>

#ifdef DEBUG
#ifdef WINDOWS
// Route CRT assertions to stderr instead of blocking message boxes, so
// automated (dump-mode) runs never wait for a human to click "Ignore".
#include <crtdbg.h>
#include <cstdlib>
static void disable_debug_crt_dialogs() {
    _CrtSetReportMode(_CRT_WARN,   _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_WARN,   _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR,  _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR,  _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _set_invalid_parameter_handler([](const wchar_t*, const wchar_t*,
                                      const wchar_t*, unsigned, uintptr_t) { });
}
#else
static void disable_debug_crt_dialogs() { }
#endif
#else
static void disable_debug_crt_dialogs() { }
#endif

int main(int argc, char** argv) {
    disable_debug_crt_dialogs();

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

    // The dump mode runs fully offline: keep the window hidden and exit
    // as soon as the last frame has been written to disk.
    bool dump_mode = argp["dump"].value.boolean == 1;

    if (!dump_mode)
        window.show();

    if (dump_mode) {
        vkhr::GBufferDumpConfig config;
        config.output_directory = argp["dump-dir"].value.string;
        config.frame_count      = argp["dump-frames"].value.integer;
        config.ssaa_factor      = argp["dump-ssaa"].value.integer;
        config.camera_script    = argp["camera-script"].value.string;
        config.strand_radius    = argp["strand-radius"].value.floating;
        config.camera_seed      = argp["camera-seed"].value.integer;
        config.distance_min     = argp["distance-min"].value.floating;
        config.distance_max     = argp["distance-max"].value.floating;
        config.elevation_min    = argp["elevation-min"].value.floating;
        config.elevation_max    = argp["elevation-max"].value.floating;
        config.radius_min       = argp["radius-min"].value.floating;
        config.radius_max       = argp["radius-max"].value.floating;
        config.dump_shaded      = argp["dump-shaded"].value.boolean == 1;
        config.random_light     = argp["light-random"].value.boolean == 1;

        rasterizer.dump_gbuffer(scene_graph, config);
        window.close();
        return 0;
    }

    // Offline shading of dumped G-buffer channel files (recon/input).
    std::cerr << "[trace] shade flag = " << argp["shade"].value.boolean << std::endl;
    if (argp["shade"].value.boolean == 1) {
        std::cerr << "[trace] entering shade branch" << std::endl;
        vkhr::ShadeDumpOptions options;
        options.dump_directory = argp["shade-dir"].value.string;
        options.source         = argp["shade-source"].value.string;
        options.output_file    = argp["shade-out"].value.string;

        try {
            rasterizer.shade_dump(scene_graph, options);
        } catch (const std::exception& error) {
            std::cerr << "[shade exception] " << error.what() << std::endl;
        }
        window.close();
        return 0;
    }

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

#ifndef VKHR_NEURAL_RECONSTRUCT_HH
#define VKHR_NEURAL_RECONSTRUCT_HH

#include <cstdint>
#include <memory>
#include <string>

namespace vkhr::vulkan {
    // Runs the exported Psi network (psi.onnx) through ONNX Runtime.
    //
    // This is the CPU-relay implementation agreed for S3: the caller hands
    // over planar CPU buffers, the session runs on the CUDA execution
    // provider (ONNX Runtime moves the tensors across the bus), and the
    // reconstructed channels come back on the CPU. Zero-copy Vulkan/ONNX
    // interop is a later optimization.
    //
    // Pre/post-processing contract: the input must already be cleaned
    // (nan_to_num) and packed into the planar layout documented below - the
    // clamp/renormalize/support-gating postprocess lives INSIDE the graph.
    // See nn/onnx_prepost_contract.md for the authoritative table.
    class NeuralReconstruct final {
    public:
        // S4-3: execution provider selection. TRT uses the ORT TensorRT EP
        // with an engine cache (the first build is slow and crash-prone on
        // this machine; every later run deserializes the cached engine).
        struct ExecutionOptions {
            bool tensorrt { false };
            bool fp16 { false };              // TRT only; judge by the v1.2 criteria.
            std::string cache_dir { "runs/trt_cache" };
        };

        struct Timing {
            double pack_ms       { 0.0 }; // planar -> NCHW packing.
            double inference_ms  { 0.0 }; // ONNX Runtime session run.
            double unpack_ms     { 0.0 }; // NCHW -> planar unpacking.
            double total_ms      { 0.0 };
        };

        NeuralReconstruct(const std::string& model_path,
                          std::uint32_t width, std::uint32_t height,
                          const std::string& profile_path = "",
                          const ExecutionOptions& execution = {});
        ~NeuralReconstruct();

        NeuralReconstruct(const NeuralReconstruct&) = delete;
        NeuralReconstruct& operator=(const NeuralReconstruct&) = delete;

        bool is_ready() const;

        // coverage/tangent: planar float arrays of width*height (tangent is
        // 3*width*height, xyz interleaved per pixel). The reconstructed,
        // support-gated channels are written to out_coverage/out_tangent
        // (same layout). The timing of the last run is returned in `timing`.
        void run(const float* coverage, const float* tangent,
                 float* out_coverage, float* out_tangent, Timing& timing);

        const Timing& last_timing() const { return last_timing_; }

        std::uint32_t get_width()  const { return width_;  }
        std::uint32_t get_height() const { return height_; }

    private:
        class Implementation; // hides onnxruntime_cxx_api.h from callers.
        std::unique_ptr<Implementation> implementation_;

        std::uint32_t width_, height_;
        Timing last_timing_;
    };
}

#endif

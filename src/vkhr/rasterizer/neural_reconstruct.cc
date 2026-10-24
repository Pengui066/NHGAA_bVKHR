#include <vkhr/rasterizer/neural_reconstruct.hh>

#include <onnxruntime_cxx_api.h>

// The CUDA provider options live in their own header (the distribution
// zip only ships the CPU-side provider_options.h).
#include <onnxruntime/core/providers/cuda/cuda_provider_options.h>

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace vkhr::vulkan {
    // -------------------------------------------------------------------
    // Layout conventions (see nn/onnx_prepost_contract.md):
    //
    //   ONNX input  "input" : fp32 [1, 4, H, W] NCHW
    //               ch0 = coverage, ch1..3 = tangent xyz (channel-planar)
    //   ONNX output "coverage" : fp32 [1, 1, H, W]  support-gated, clamped
    //   ONNX output "tangent"  : fp32 [1, 3, H, W]  support-gated, normalized
    //
    // The renderer side hands over planar buffers with the tangent xyz
    // interleaved per pixel (the natural layout of the G-buffer image), so
    // this module transposes between the two layouts on the way in and out.
    // -------------------------------------------------------------------

    class NeuralReconstruct::Implementation {
    public:
        Implementation(const std::string& model_path,
                       std::uint32_t width, std::uint32_t height,
                       const std::string& profile_path,
                       const NeuralReconstruct::ExecutionOptions& execution)
            : width_ { width }, height_ { height } {
            input_.resize(4 * std::size_t(width) * height);
            input_shape_ = { 1, 4, static_cast<std::int64_t>(height),
                             static_cast<std::int64_t>(width) };
            memory_info_ = Ort::MemoryInfo { "Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault };
            env_ = Ort::Env { ORT_LOGGING_LEVEL_WARNING, "vkhr-nn" };

            // The Session constructor takes ORTCHAR_T (wchar_t under the
            // Unicode character set), so widen the path on Windows.
            #ifdef _WIN32
                std::wstring model_wide { model_path.begin(), model_path.end() };
                const ORTCHAR_T* model_argument = model_wide.c_str();
            #else
                const ORTCHAR_T* model_argument = model_path.c_str();
            #endif

            Ort::SessionOptions cuda_options;
            cuda_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // S4-3 route A: the ORT TensorRT EP in front of CUDA, through
            // the key-value option API (no provider-struct layout risk).
            if (execution.tensorrt) {
                std::cerr << "[nn] TRT EP: creating provider options..." << std::endl;
                const OrtApi& api = Ort::GetApi();
                OrtTensorRTProviderOptionsV2* trt_options { nullptr };
                OrtStatus* create = api.CreateTensorRTProviderOptions(&trt_options);
                if (!create) {
                    std::vector<const char*> keys {
                        "device_id", "trt_fp16_enable",
                        "trt_engine_cache_enable", "trt_engine_cache_path",
                        "trt_timing_cache_enable", "trt_timing_cache_path",
                        "trt_builder_optimization_level"
                    };
                    std::string fp16 = execution.fp16 ? "True" : "False";
                    std::string level = "2"; // lighter builder stress than the default 3.
                    std::vector<const char*> values {
                        "0", fp16.c_str(),
                        "True", execution.cache_dir.c_str(),
                        "True", execution.cache_dir.c_str(),
                        level.c_str()
                    };
                    OrtStatus* update = api.UpdateTensorRTProviderOptions(
                        trt_options, keys.data(), values.data(), keys.size());
                    if (update) {
                        std::cerr << "[nn] TRT option update failed: "
                                  << api.GetErrorMessage(update) << std::endl;
                        api.ReleaseStatus(update);
                    }
                    std::cerr << "[nn] TRT EP: appending provider (loads "
                                 "onnxruntime_providers_tensorrt.dll)..." << std::endl;
                    OrtStatus* append = api.SessionOptionsAppendExecutionProvider_TensorRT_V2(
                        cuda_options, trt_options);
                    api.ReleaseTensorRTProviderOptions(trt_options);
                    if (append) {
                        std::cerr << "[nn] TRT EP append failed: "
                                  << api.GetErrorMessage(append) << std::endl;
                        api.ReleaseStatus(append);
                    } else {
                        std::cerr << "[nn] TRT EP requested"
                                  << (execution.fp16 ? " (fp16)" : " (fp32)")
                                  << ", engine cache: " << execution.cache_dir
                                  << std::endl;
                    }
                } else {
                    std::cerr << "[nn] CreateTensorRTProviderOptions failed: "
                              << api.GetErrorMessage(create) << std::endl;
                    api.ReleaseStatus(create);
                }
            }

            // S4-0.4: optional ORT profiling (chrome-trace JSON, finalized
            // when the session is destroyed).
            #ifdef _WIN32
                std::wstring profile_wide;
            #endif
            if (!profile_path.empty()) {
                #ifdef _WIN32
                    profile_wide.assign(profile_path.begin(), profile_path.end());
                    cuda_options.EnableProfiling(profile_wide.c_str());
                #else
                    cuda_options.EnableProfiling(profile_path.c_str());
                #endif
                std::cerr << "[nn] ORT profiling enabled: " << profile_path << std::endl;
            }

            // Append the CUDA provider through the C API: the C++ wrapper
            // only exposes the deprecated V1 options overload.
            OrtCUDAProviderOptionsV2 provider_options { };
            provider_options.device_id = 0;
            provider_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            provider_options.do_copy_in_default_stream = 1;

            const OrtApi& api = Ort::GetApi();
            OrtStatusPtr provider_status = api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                cuda_options, &provider_options);
            if (provider_status) {
                std::string message = api.GetErrorMessage(provider_status);
                api.ReleaseStatus(provider_status);
                throw std::runtime_error { "CUDA EP: " + message };
            }

            try {
                session_ = Ort::Session { env_, model_argument, cuda_options };
                std::cerr << "[nn] ONNX Runtime ready (CUDA EP, device 0): " << model_path
                          << " @ " << width << "x" << height << std::endl;
            } catch (const Ort::Exception& exception) {
                std::cerr << "[nn] CUDA EP unavailable (" << exception.what()
                          << "); falling back to CPU" << std::endl;
                Ort::SessionOptions cpu_options;
                cpu_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                if (!profile_path.empty()) {
                    #ifdef _WIN32
                        cpu_options.EnableProfiling(profile_wide.c_str());
                    #else
                        cpu_options.EnableProfiling(profile_path.c_str());
                    #endif
                }
                session_ = Ort::Session { env_, model_argument, cpu_options };
                std::cerr << "[nn] ONNX Runtime ready (CPU EP): " << model_path
                          << " @ " << width << "x" << height << std::endl;
            }

            ready_ = true;
        }

        void run(const float* coverage, const float* tangent,
                 float* out_coverage, float* out_tangent,
                 NeuralReconstruct::Timing& timing) {
            using clock = std::chrono::steady_clock;
            const auto total_start = clock::now();
            const std::size_t pixel_count = std::size_t(width_) * height_;

            // Pack: planar [C, Txyz-per-pixel] -> NCHW [1, 4, H, W].
            const auto pack_start = clock::now();
            for (std::size_t i = 0; i < pixel_count; ++i) {
                input_[i]                       = coverage[i];
                input_[1 * pixel_count + i]     = tangent[3 * i + 0];
                input_[2 * pixel_count + i]     = tangent[3 * i + 1];
                input_[3 * pixel_count + i]     = tangent[3 * i + 2];
            }
            const auto inference_start = clock::now();
            timing.pack_ms = std::chrono::duration<double, std::milli>
                             (inference_start - pack_start).count();

            Ort::Value input = Ort::Value::CreateTensor<float>(
                memory_info_, input_.data(), input_.size(),
                input_shape_.data(), input_shape_.size());
            const char* input_names[]  { "input" };
            const char* output_names[] { "coverage", "tangent" };
            Ort::RunOptions run_options;
            auto outputs = session_.Run(run_options, input_names, &input, 1,
                                        output_names, 2);

            const auto unpack_start = clock::now();
            timing.inference_ms = std::chrono::duration<double, std::milli>
                                  (unpack_start - inference_start).count();

            // Unpack: NCHW [1, 1|3, H, W] -> planar (xyz per pixel).
            const float* cov = outputs[0].GetTensorData<float>();
            const float* tan = outputs[1].GetTensorData<float>();
            std::memcpy(out_coverage, cov, pixel_count * sizeof(float));
            for (std::size_t i = 0; i < pixel_count; ++i) {
                out_tangent[3 * i + 0] = tan[i];
                out_tangent[3 * i + 1] = tan[pixel_count + i];
                out_tangent[3 * i + 2] = tan[2 * pixel_count + i];
            }
            const auto unpack_end = clock::now();
            timing.unpack_ms = std::chrono::duration<double, std::milli>
                               (unpack_end - unpack_start).count();
            timing.total_ms = std::chrono::duration<double, std::milli>
                              (unpack_end - total_start).count();
        }

        bool ready() const { return ready_; }

    private:
        std::uint32_t width_, height_;
        bool ready_ { false };

        Ort::Env env_ { nullptr };
        Ort::Session session_ { nullptr };

        std::vector<float> input_;
        std::array<std::int64_t, 4> input_shape_;
        Ort::MemoryInfo memory_info_ { nullptr };
    };

    NeuralReconstruct::NeuralReconstruct(const std::string& model_path,
                                         std::uint32_t width, std::uint32_t height,
                                         const std::string& profile_path,
                                         const ExecutionOptions& execution)
        : width_ { width }, height_ { height },
          implementation_ { std::make_unique<Implementation>(model_path, width, height,
                                                             profile_path, execution) } {
    }

    NeuralReconstruct::~NeuralReconstruct() = default;

    bool NeuralReconstruct::is_ready() const {
        return implementation_ && implementation_->ready();
    }

    void NeuralReconstruct::run(const float* coverage, const float* tangent,
                                float* out_coverage, float* out_tangent,
                                Timing& timing) {
        implementation_->run(coverage, tangent, out_coverage, out_tangent, timing);
        last_timing_ = timing;
    }
}

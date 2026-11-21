// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: see vk_xess.h.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>

#include "gow3_settings.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_xess.h"

#if defined(_WIN32) && __has_include(<xess_vk.h>)
#define GOW3_HAS_XESS 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <xess_vk.h>
#else
#define GOW3_HAS_XESS 0
#endif

namespace Vulkan {

bool Xess::SelectedAtStart() {
    return Gow3Settings::Get().startup_upscaler == Gow3Settings::UpscalerXess;
}

#if GOW3_HAS_XESS

namespace {

constexpr const char* MissingLibrary =
    "libxess.dll is not next to gow3-probe.exe (tools/fetch_aa_files.sh xess)";

/// libxess.dll and the entry points this file calls; loaded once, on first use.
struct Library {
    HMODULE module{};
    std::string error;
    xess_result_t (*get_version)(xess_version_t*){};
    xess_result_t (*create)(VkInstance, VkPhysicalDevice, VkDevice, xess_context_handle_t*){};
    xess_result_t (*init)(xess_context_handle_t, const xess_vk_init_params_t*){};
    xess_result_t (*execute)(xess_context_handle_t, VkCommandBuffer,
                             const xess_vk_execute_params_t*){};
    xess_result_t (*destroy)(xess_context_handle_t){};
    xess_result_t (*instance_extensions)(uint32_t*, const char* const**, uint32_t*){};
    xess_result_t (*device_extensions)(VkInstance, VkPhysicalDevice, uint32_t*,
                                       const char* const**){};
    xess_result_t (*device_features)(VkInstance, VkPhysicalDevice, void**){};
    xess_result_t (*set_logging)(xess_context_handle_t, xess_logging_level_t,
                                 xess_app_log_callback_t){};
    xess_result_t (*is_optimal_driver)(xess_context_handle_t){};
    xess_result_t (*optimal_input)(xess_context_handle_t, const xess_2d_t*,
                                   xess_quality_settings_t, xess_2d_t*, xess_2d_t*, xess_2d_t*){};
    xess_result_t (*set_max_responsive)(xess_context_handle_t, float){};

    bool Ok() const {
        return module && create && init && execute && destroy;
    }
};

/// The folder of the executable (empty when it cannot be told).
std::wstring ExecutableDirectory() {
    wchar_t path[MAX_PATH * 2];
    const DWORD length = GetModuleFileNameW(nullptr, path, DWORD(std::size(path)));
    if (length == 0 || length >= std::size(path)) {
        return {};
    }
    std::wstring wide(path, length);
    const auto slash = wide.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return {};
    }
    wide.resize(slash);
    return wide;
}

template <typename T>
void Resolve(HMODULE module, const char* name, T& out) {
    out = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(module, name)));
}

Library& GetLibrary() {
    static Library library = [] {
        Library lib;
        const std::wstring directory = ExecutableDirectory();
        // By full path from the game's folder only: never the current directory or PATH, where
        // another libxess.dll could be planted. Its dependencies come from beside it, then from
        // the system folder.
        constexpr DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
        const std::wstring path = directory + L"\\libxess.dll";
        if (!directory.empty()) {
            lib.module = LoadLibraryExW(path.c_str(), nullptr, flags);
        }
        if (!lib.module) {
            const DWORD code = GetLastError();
            std::error_code error;
            if (directory.empty() || !std::filesystem::is_regular_file(path, error)) {
                lib.error = MissingLibrary;
            } else {
                // Intel's library needs the Visual C++ runtime (MSVCP140, VCRUNTIME140), which
                // the port itself does not.
                char text[200];
                std::snprintf(text, sizeof(text),
                              "libxess.dll could not be loaded (Windows error %lu): it needs the "
                              "Microsoft Visual C++ 2015-2022 Redistributable (x64)",
                              static_cast<unsigned long>(code));
                lib.error = text;
            }
            return lib;
        }
        Resolve(lib.module, "xessGetVersion", lib.get_version);
        Resolve(lib.module, "xessVKCreateContext", lib.create);
        Resolve(lib.module, "xessVKInit", lib.init);
        Resolve(lib.module, "xessVKExecute", lib.execute);
        Resolve(lib.module, "xessDestroyContext", lib.destroy);
        Resolve(lib.module, "xessVKGetRequiredInstanceExtensions", lib.instance_extensions);
        Resolve(lib.module, "xessVKGetRequiredDeviceExtensions", lib.device_extensions);
        Resolve(lib.module, "xessVKGetRequiredDeviceFeatures", lib.device_features);
        Resolve(lib.module, "xessSetLoggingCallback", lib.set_logging);
        Resolve(lib.module, "xessIsOptimalDriver", lib.is_optimal_driver);
        Resolve(lib.module, "xessGetOptimalInputResolution", lib.optimal_input);
        Resolve(lib.module, "xessSetMaxResponsiveMaskValue", lib.set_max_responsive);
        if (!lib.Ok()) {
            lib.error = "libxess.dll has no Vulkan entry points (not Intel's XeSS SDK library?)";
        }
        return lib;
    }();
    return library;
}

std::atomic<bool> g_device_ready{false}; ///< the device was created for XeSS (SetDeviceReady)
std::mutex g_note_mutex;
std::string g_device_note;

void LogCallback(const char* message, xess_logging_level_t level) {
    std::printf("XeSS [%d]: %s\n", int(level), message ? message : "");
}

const char* ResultName(xess_result_t result) {
    switch (result) {
    case XESS_RESULT_ERROR_UNSUPPORTED_DEVICE: return "unsupported device";
    case XESS_RESULT_ERROR_UNSUPPORTED_DRIVER: return "unsupported driver";
    case XESS_RESULT_ERROR_UNINITIALIZED: return "not initialised";
    case XESS_RESULT_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case XESS_RESULT_ERROR_DEVICE_OUT_OF_MEMORY: return "out of GPU memory";
    case XESS_RESULT_ERROR_DEVICE: return "device error";
    case XESS_RESULT_ERROR_NOT_IMPLEMENTED: return "not implemented";
    case XESS_RESULT_ERROR_INVALID_CONTEXT: return "invalid context";
    case XESS_RESULT_ERROR_UNSUPPORTED: return "unsupported configuration";
    case XESS_RESULT_ERROR_WRONG_CALL_ORDER: return "wrong call order";
    default: return "error";
    }
}

const char* QualityName(xess_quality_settings_t quality) {
    switch (quality) {
    case XESS_QUALITY_SETTING_ULTRA_PERFORMANCE: return "Ultra Performance";
    case XESS_QUALITY_SETTING_PERFORMANCE: return "Performance";
    case XESS_QUALITY_SETTING_BALANCED: return "Balanced";
    case XESS_QUALITY_SETTING_QUALITY: return "Quality";
    case XESS_QUALITY_SETTING_ULTRA_QUALITY: return "Ultra Quality";
    case XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS: return "Ultra Quality Plus";
    case XESS_QUALITY_SETTING_AA: return "Native AA";
    default: return "?";
    }
}

bool HasStencil(vk::Format format) {
    return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD32SfloatS8Uint;
}

xess_vk_image_view_info ViewInfo(const Xess::Image& image) {
    xess_vk_image_view_info info{};
    info.imageView = static_cast<VkImageView>(image.view);
    info.image = static_cast<VkImage>(image.image);
    info.subresourceRange = {static_cast<VkImageAspectFlags>(image.aspect), 0, 1, 0, 1};
    info.format = static_cast<VkFormat>(image.format);
    info.width = image.width;
    info.height = image.height;
    return info;
}

} // namespace

struct Xess::Impl {
    Impl(const Instance& instance_, Scheduler& scheduler_)
        : instance(instance_), scheduler(scheduler_) {}
    ~Impl() {
        if (context && GetLibrary().destroy) {
            scheduler.Finish();
            GetLibrary().destroy(context);
        }
    }

    const Instance& instance;
    Scheduler& scheduler;
    xess_context_handle_t context{};
    u32 init_out_width = 0, init_out_height = 0;
    u32 input_width = 0, input_height = 0; ///< of the last frame
    xess_quality_settings_t init_quality{};
    u32 init_flags = 0;
    const char* init_colour = "";
    float mask_max = -1.0f; ///< the responsive mask's clip last set on the context
    bool initialised = false;
    bool fresh = true; ///< the history of the context is empty: the next frame resets it
    bool fatal = false;
    u32 failures = 0;
    std::string problem;
    bool logged_running = false;

    void Fail(const char* what, xess_result_t result, bool permanent) {
        char text[200];
        std::snprintf(text, sizeof(text), "XeSS %s: %s (%d)", what, ResultName(result),
                      int(result));
        problem = text;
        std::printf("%s\n", text);
        if (permanent) {
            fatal = true;
        }
    }

    /// The quality setting whose input range holds the render size: XeSS takes any input size
    /// between that setting's minimum and maximum, nearest to its optimum. The library's own table
    /// decides which: libxess 2.0.2 picks Quality for the 1.5 preset (3408x960 -> 5120x1440).
    bool PickQuality(u32 w, u32 h, u32 ow, u32 oh, xess_quality_settings_t& quality) {
        static constexpr xess_quality_settings_t settings[] = {
            XESS_QUALITY_SETTING_AA,           XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS,
            XESS_QUALITY_SETTING_ULTRA_QUALITY, XESS_QUALITY_SETTING_QUALITY,
            XESS_QUALITY_SETTING_BALANCED,      XESS_QUALITY_SETTING_PERFORMANCE,
            XESS_QUALITY_SETTING_ULTRA_PERFORMANCE,
        };
        const Library& lib = GetLibrary();
        if (lib.optimal_input) {
            const xess_2d_t output{ow, oh};
            u64 best_distance = ~u64{0};
            for (const auto setting : settings) {
                xess_2d_t optimal{}, minimum{}, maximum{};
                if (lib.optimal_input(context, &output, setting, &optimal, &minimum, &maximum) < 0 ||
                    w < minimum.x || h < minimum.y || w > maximum.x || h > maximum.y) {
                    continue;
                }
                const u64 distance = u64(std::abs(s64(optimal.x) - s64(w))) +
                                     u64(std::abs(s64(optimal.y) - s64(h)));
                if (distance < best_distance) {
                    best_distance = distance;
                    quality = setting;
                }
            }
            return best_distance != ~u64{0};
        }
        // A library without the query: the setting of the nearest scale.
        const float scale = float(oh) / float(std::max(1u, h));
        quality = scale <= 1.01f  ? XESS_QUALITY_SETTING_AA
                  : scale <= 1.4f ? XESS_QUALITY_SETTING_ULTRA_QUALITY_PLUS
                  : scale <= 1.6f ? XESS_QUALITY_SETTING_ULTRA_QUALITY
                  : scale <= 1.85f ? XESS_QUALITY_SETTING_QUALITY
                  : scale <= 2.15f ? XESS_QUALITY_SETTING_BALANCED
                  : scale <= 2.6f  ? XESS_QUALITY_SETTING_PERFORMANCE
                                   : XESS_QUALITY_SETTING_ULTRA_PERFORMANCE;
        return true;
    }

    void Barrier(vk::CommandBuffer cmdbuf, const Image& image, bool to_read) {
        vk::ImageAspectFlags aspect = image.aspect;
        if (HasStencil(image.format)) {
            aspect |= vk::ImageAspectFlagBits::eStencil; // the whole image changes layout together
        }
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = to_read ? vk::PipelineStageFlagBits2::eAllCommands
                                    : vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = to_read ? vk::AccessFlagBits2::eMemoryRead |
                                           vk::AccessFlagBits2::eMemoryWrite
                                     : vk::AccessFlagBits2::eNone,
            .dstStageMask = to_read ? vk::PipelineStageFlagBits2::eComputeShader
                                    : vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = to_read ? vk::AccessFlagBits2::eShaderSampledRead
                                     : vk::AccessFlagBits2::eMemoryRead |
                                           vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout = to_read ? vk::ImageLayout::eGeneral
                                 : vk::ImageLayout::eShaderReadOnlyOptimal,
            .newLayout = to_read ? vk::ImageLayout::eShaderReadOnlyOptimal
                                 : vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image.image,
            .subresourceRange = {aspect, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    }

    bool Record(const Frame& frame) {
        if (fatal) {
            return false;
        }
        const Library& lib = GetLibrary();
        if (!lib.Ok()) {
            problem = lib.error;
            fatal = true;
            std::printf("XeSS: %s\n", problem.c_str());
            return false;
        }
        if (!context && !g_device_ready) {
            // The extensions and features are enabled when the device is created, and only when
            // XeSS is the upscaler the game starts with: a context on another device is undefined.
            problem = DeviceProblem();
            if (problem.empty()) {
                problem = "XeSS was not chosen when the game started: it runs from the next start";
            }
            fatal = true;
            std::printf("XeSS: %s\n", problem.c_str());
            return false;
        }
        if (!context) {
            const xess_result_t result =
                lib.create(static_cast<VkInstance>(instance.GetInstance()),
                           static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice()),
                           static_cast<VkDevice>(instance.GetDevice()), &context);
            if (result < 0) {
                context = nullptr;
                Fail("context", result, true);
                return false;
            }
            if (lib.set_logging) {
                lib.set_logging(context, XESS_LOGGING_LEVEL_WARNING, &LogCallback);
            }
            if (lib.get_version) {
                xess_version_t version{};
                if (lib.get_version(&version) >= 0) {
                    std::printf("XeSS: library %u.%u.%u\n", version.major, version.minor,
                                version.patch);
                }
            }
            if (lib.is_optimal_driver &&
                lib.is_optimal_driver(context) == XESS_RESULT_WARNING_OLD_DRIVER) {
                std::printf("XeSS: the graphics driver is older than XeSS asks for (it may run "
                            "slower or look worse)\n");
            }
        }
        const u32 w = frame.input_width, h = frame.input_height;
        const u32 ow = frame.output.width, oh = frame.output.height;
        // Native AA: the lit scene before the post processing (linear HDR, by default exposed by
        // XeSS); upscaling: the game's finished, tonemapped frame. Motion vectors at the render
        // size, without the jitter; depth not inverted.
        const bool has_mask = frame.responsive.view && lib.set_max_responsive;
        const u32 flags = (!frame.hdr          ? u32(XESS_INIT_FLAG_LDR_INPUT_COLOR)
                           : frame.auto_exposure ? u32(XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE)
                                                 : 0u) |
                          (has_mask ? u32(XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK) : 0u);
        const bool sizes_changed = !initialised || w != input_width || h != input_height ||
                                   ow != init_out_width || oh != init_out_height ||
                                   flags != init_flags;
        xess_quality_settings_t quality = init_quality;
        if (sizes_changed && !PickQuality(w, h, ow, oh, quality)) {
            char text[160];
            std::snprintf(text, sizeof(text), "XeSS: no quality setting takes %ux%u up to %ux%u",
                          w, h, ow, oh);
            problem = text;
            fatal = true;
            std::printf("%s\n", text);
            return false;
        }
        input_width = w;
        input_height = h;
        // A new input size within the setting's range needs no initialisation.
        if (!initialised || ow != init_out_width || oh != init_out_height ||
            flags != init_flags || quality != init_quality) {
            // XeSS's memory may still be read by submitted work; `frame.cmdbuf` stays open.
            scheduler.WaitSubmitted();
            xess_vk_init_params_t init{};
            init.outputResolution = {ow, oh};
            init.qualitySetting = quality;
            init.initFlags = flags;
            init.creationNodeMask = 1;
            init.visibleNodeMask = 1;
            const xess_result_t result = lib.init(context, &init);
            if (result < 0) {
                initialised = false;
                Fail("initialisation", result, true);
                return false;
            }
            initialised = true;
            init_out_width = ow;
            init_out_height = oh;
            init_quality = quality;
            init_flags = flags;
            init_colour = frame.colour ? frame.colour : "";
            mask_max = -1.0f;
            fresh = true;
            logged_running = false;
        }

        xess_vk_execute_params_t params{};
        params.colorTexture = ViewInfo(frame.color);
        params.velocityTexture = ViewInfo(frame.motion);
        params.depthTexture = ViewInfo(frame.depth);
        params.outputTexture = ViewInfo(frame.output);
        params.jitterOffsetX = frame.jitter[0];
        params.jitterOffsetY = frame.jitter[1];
        params.exposureScale = 1.0f;
        params.resetHistory = frame.reset || fresh ? 1u : 0u;
        params.inputWidth = w;
        params.inputHeight = h;
        if (has_mask) {
            params.responsivePixelMaskTexture = ViewInfo(frame.responsive);
            const float clip = std::clamp(frame.responsive_max, 0.0f, 1.0f);
            if (clip != mask_max) {
                mask_max = clip;
                lib.set_max_responsive(context, clip);
            }
        }

        // The inputs in the layout XeSS documents (shader read only), the output in General;
        // back to General afterwards for the rest of the frame.
        const vk::MemoryBarrier2 output_ready{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                             vk::AccessFlagBits2::eShaderStorageWrite |
                             vk::AccessFlagBits2::eShaderSampledRead,
        };
        frame.cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &output_ready});
        Barrier(frame.cmdbuf, frame.color, true);
        Barrier(frame.cmdbuf, frame.depth, true);
        Barrier(frame.cmdbuf, frame.motion, true);
        if (has_mask) {
            Barrier(frame.cmdbuf, frame.responsive, true);
        }
        const xess_result_t result =
            lib.execute(context, static_cast<VkCommandBuffer>(frame.cmdbuf), &params);
        Barrier(frame.cmdbuf, frame.color, false);
        Barrier(frame.cmdbuf, frame.depth, false);
        Barrier(frame.cmdbuf, frame.motion, false);
        if (has_mask) {
            Barrier(frame.cmdbuf, frame.responsive, false);
        }
        if (result < 0) {
            Fail("execute", result, ++failures >= 3);
            return false;
        }
        failures = 0;
        problem.clear();
        fresh = false;
        if (!logged_running) {
            logged_running = true;
            std::printf("XeSS: %ux%u -> %ux%u (%s, %s%s)\n", w, h, ow, oh,
                        QualityName(init_quality), init_colour,
                        (init_flags & XESS_INIT_FLAG_RESPONSIVE_PIXEL_MASK) ? ", responsive mask"
                                                                            : "");
        }
        return true;
    }
};

bool Xess::LibraryAvailable(std::string* why) {
    // Loaded only for a session that runs XeSS; for the others the file being there is enough to
    // offer it in the menu for the next start.
    if (!SelectedAtStart()) {
        const std::wstring directory = ExecutableDirectory();
        std::error_code error;
        const bool present = !directory.empty() &&
            std::filesystem::is_regular_file(std::filesystem::path{directory} / L"libxess.dll", error);
        if (why) {
            *why = present ? std::string{} : MissingLibrary;
        }
        return present;
    }
    const Library& lib = GetLibrary();
    if (why) {
        *why = lib.Ok() ? std::string{} : lib.error;
    }
    return lib.Ok();
}

void Xess::AppendInstanceExtensions(std::vector<const char*>& enabled) {
    const Library& lib = GetLibrary();
    if (!lib.Ok() || !lib.instance_extensions) {
        return;
    }
    uint32_t count = 0;
    const char* const* names = nullptr;
    uint32_t api_version = 0;
    if (lib.instance_extensions(&count, &names, &api_version) < 0 || !names) {
        return;
    }
    const auto [result, available] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess) {
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const std::string_view name = names[i];
        if (std::ranges::any_of(enabled, [&](const char* have) { return name == have; })) {
            continue;
        }
        if (std::ranges::none_of(available, [&](const vk::ExtensionProperties& e) {
                return name == e.extensionName.data();
            })) {
            std::printf("XeSS: instance extension %s unavailable\n", names[i]);
            continue;
        }
        enabled.push_back(names[i]);
    }
}

std::vector<const char*> Xess::DeviceExtensions(vk::Instance instance,
                                                vk::PhysicalDevice physical_device) {
    std::vector<const char*> names;
    const Library& lib = GetLibrary();
    if (!lib.Ok() || !lib.device_extensions) {
        return names;
    }
    uint32_t count = 0;
    const char* const* list = nullptr;
    if (lib.device_extensions(static_cast<VkInstance>(instance),
                              static_cast<VkPhysicalDevice>(physical_device), &count, &list) < 0 ||
        !list) {
        return names;
    }
    names.assign(list, list + count);
    return names;
}

bool Xess::PatchDeviceFeatures(vk::Instance instance, vk::PhysicalDevice physical_device,
                               void*& chain) {
    const Library& lib = GetLibrary();
    if (!lib.Ok() || !lib.device_features) {
        return false;
    }
    void* head = chain;
    if (lib.device_features(static_cast<VkInstance>(instance),
                            static_cast<VkPhysicalDevice>(physical_device), &head) < 0 ||
        !head) {
        return false;
    }
    chain = head;
    return true;
}

void Xess::SetDeviceReady(bool ready, const char* why) {
    {
        std::scoped_lock lock{g_note_mutex};
        g_device_note = ready || !why ? "" : why;
    }
    g_device_ready = ready;
}

bool Xess::DeviceReady() {
    return g_device_ready;
}

std::string Xess::DeviceProblem() {
    std::scoped_lock lock{g_note_mutex};
    return g_device_note;
}

#else // no XeSS headers or not Windows

struct Xess::Impl {
    Impl(const Instance&, Scheduler&) {}
    bool Record(const Frame&) {
        return false;
    }
};

bool Xess::LibraryAvailable(std::string* why) {
    if (why) {
        *why = "XeSS is available only in the Windows build";
    }
    return false;
}

void Xess::AppendInstanceExtensions(std::vector<const char*>&) {}

std::vector<const char*> Xess::DeviceExtensions(vk::Instance, vk::PhysicalDevice) {
    return {};
}

bool Xess::PatchDeviceFeatures(vk::Instance, vk::PhysicalDevice, void*&) {
    return false;
}

void Xess::SetDeviceReady(bool, const char*) {}

bool Xess::DeviceReady() {
    return false;
}

std::string Xess::DeviceProblem() {
    return {};
}

#endif

Xess::Xess(const Instance& instance, Scheduler& scheduler)
    : impl(std::make_unique<Impl>(instance, scheduler)) {}

Xess::~Xess() = default;

bool Xess::Record(const Frame& frame) {
    return impl->Record(frame);
}

const char* Xess::Problem() const noexcept {
#if GOW3_HAS_XESS
    return impl->problem.empty() ? nullptr : impl->problem.c_str();
#else
    return "XeSS is available only in the Windows build";
#endif
}

bool Xess::Fatal() const noexcept {
#if GOW3_HAS_XESS
    return impl->fatal;
#else
    return true;
#endif
}

} // namespace Vulkan

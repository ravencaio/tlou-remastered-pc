// SPDX-FileCopyrightText: Copyright 2026 gow3_pc contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/host_shaders/vol_directional_integral_comp.h"
#include "video_core/host_shaders/vol_directional_resolve_frag.h"
#include "video_core/host_shaders/vol_projector_integral_array_comp.h"
#include "video_core/host_shaders/vol_projector_integral_comp.h"
#include "video_core/host_shaders/vol_spot_integral_array_comp.h"
#include "video_core/host_shaders/vol_spot_integral_comp.h"
#include "video_core/host_shaders/vol_spot_resolve_frag.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_volumetric_light.h"

namespace Vulkan::VolumetricLight {

namespace {

// The grid passes. For a spot or a flashlight, the first is for a 2D shadow map, the second for
// an array (the shader's second image, binding 2).
enum Variant : int { Directional, Spot2D, SpotArray, Projector2D, ProjectorArray, VariantCount };
const std::span<const u32> GridCode[VariantCount] = {
    VOL_DIRECTIONAL_INTEGRAL_COMP, VOL_SPOT_INTEGRAL_COMP, VOL_SPOT_INTEGRAL_ARRAY_COMP,
    VOL_PROJECTOR_INTEGRAL_COMP, VOL_PROJECTOR_INTEGRAL_ARRAY_COMP,
};

enum Resolve : int { DirectionalResolve, SpotResolve, ResolveCount };
struct ResolveShader {
    u64 hash;
    std::span<const u32> code;
};
const ResolveShader Resolves[ResolveCount] = {
    {0xad16e118ULL, VOL_DIRECTIONAL_RESOLVE_FRAG}, // PS_VolumetricDirectLightDirectByShape
    {0xca25dbadULL, VOL_SPOT_RESOLVE_FRAG},
};

// The line samplers of update 1.11 (CS_VolDirectLight* and the spot ones), their grid variants
// and the resolve that reads their image.
struct LineSampler {
    u64 hash;
    Variant first, second; // second == first: one variant
    Resolve resolve;
};
constexpr LineSampler Samplers[] = {
    {0xbb1d430eULL, Directional, Directional, DirectionalResolve}, // sun
    {0x05866787ULL, Directional, Directional, DirectionalResolve}, // moon
    {0x06e1a806ULL, Spot2D, SpotArray, SpotResolve},               // spotlights (the truck)
    {0xcdecdf59ULL, Projector2D, ProjectorArray, SpotResolve},     // flashlights
};

std::atomic<bool> g_resolve_ready[ResolveCount]{};
// Per line sampler: the variants some translation of it fits, and whether one fit none.
std::atomic<u32> g_fitting[std::size(Samplers)]{};
std::atomic<bool> g_broken[std::size(Samplers)]{};

const LineSampler* FindSampler(u64 hash, size_t& index) {
    for (index = 0; index < std::size(Samplers); ++index) {
        if (Samplers[index].hash == hash) {
            return &Samplers[index];
        }
    }
    return nullptr;
}

void Report(const char* text, u64 pgm_hash) {
    std::printf("Volumetric light: %s (%#llx)\n", text, static_cast<unsigned long long>(pgm_hash));
    std::fflush(stdout);
}

// --- The resources a SPIR-V module declares, as comparable strings ---------------------------

struct Interface {
    std::map<u64, std::set<std::string>> descriptors; // set << 32 | binding -> declarations
    std::string push;
    std::map<u32, std::string> outputs, inputs; // by location
    bool valid = false;
};

Interface Reflect(std::span<const u32> code) {
    Interface out;
    if (code.size() < 5 || code[0] != 0x07230203u) {
        return out;
    }
    struct Type {
        u32 op = 0;
        std::vector<u32> args;
    };
    std::unordered_map<u32, Type> types;
    std::unordered_map<u32, u32> constants;
    std::unordered_map<u32, std::unordered_map<u32, u32>> decorations; // id -> decoration -> value
    std::unordered_map<u32, std::map<u32, u32>> member_offsets;
    std::unordered_map<u32, bool> member_builtin;
    struct Variable {
        u32 id, pointer, storage;
    };
    std::vector<Variable> variables;
    for (size_t at = 5; at < code.size();) {
        const u32 count = code[at] >> 16, op = code[at] & 0xFFFF;
        if (count == 0 || at + count > code.size()) {
            return out;
        }
        const std::span<const u32> w = code.subspan(at + 1, count - 1);
        switch (op) {
        case 71: // OpDecorate
            if (w.size() >= 2) {
                decorations[w[0]][w[1]] = w.size() >= 3 ? w[2] : 1;
            }
            break;
        case 72: // OpMemberDecorate
            if (w.size() >= 4 && w[2] == 35) { // Offset
                member_offsets[w[0]][w[1]] = w[3];
            } else if (w.size() >= 3 && w[2] == 11) { // BuiltIn
                member_builtin[w[0]] = true;
            }
            break;
        case 43: // OpConstant
            if (w.size() >= 3) {
                constants[w[1]] = w[2];
            }
            break;
        case 59: // OpVariable
            if (w.size() >= 3) {
                variables.push_back({w[1], w[0], w[2]});
            }
            break;
        default:
            if (op >= 19 && op <= 32 && !w.empty()) { // OpTypeVoid .. OpTypePointer
                types[w[0]] = Type{op, {w.begin() + 1, w.end()}};
            }
            break;
        }
        at += count;
    }
    const auto signature = [&](auto&& self, u32 id, int depth) -> std::string {
        const auto it = types.find(id);
        if (it == types.end() || depth > 16) {
            return "?";
        }
        const auto& [op, a] = it->second;
        const auto arg = [&](size_t i) { return i < a.size() ? a[i] : 0u; };
        switch (op) {
        case 19: return "void";
        case 20: return "bool";
        case 21: return "i" + std::to_string(arg(0)) + (arg(1) ? "s" : "u");
        case 22: return "f" + std::to_string(arg(0));
        case 23: return "v" + std::to_string(arg(1)) + self(self, arg(0), depth + 1);
        case 24: return "m" + std::to_string(arg(1)) + self(self, arg(0), depth + 1);
        case 25: // sampled type, dim, depth, arrayed, ms, sampled, format: depth and format left out
            return "img(" + self(self, arg(0), depth + 1) + "," + std::to_string(arg(1)) + "," +
                   std::to_string(arg(3)) + "," + std::to_string(arg(4)) + "," +
                   std::to_string(arg(5)) + ")";
        case 26: return "sampler";
        case 27: return "sampled(" + self(self, arg(0), depth + 1) + ")";
        case 28: {
            const auto length = constants.find(arg(1));
            return "a" + std::to_string(length == constants.end() ? 0 : length->second) +
                   self(self, arg(0), depth + 1);
        }
        case 29: {
            const auto stride = decorations[id].find(6); // ArrayStride
            return "ra" + std::to_string(stride == decorations[id].end() ? 0 : stride->second) +
                   self(self, arg(0), depth + 1);
        }
        case 30: {
            std::string s = "{";
            const auto& offsets = member_offsets[id];
            for (size_t i = 0; i < a.size(); ++i) {
                const auto offset = offsets.find(u32(i));
                s += (offset == offsets.end() ? std::string("-") : std::to_string(offset->second)) +
                     ":" + self(self, a[i], depth + 1) + ";";
            }
            return s + "}";
        }
        case 32: return self(self, arg(1), depth + 1);
        default: return "?";
        }
    };
    for (const auto& variable : variables) {
        const auto& decoration = decorations[variable.id];
        if (decoration.contains(11)) { // BuiltIn
            continue;
        }
        const auto pointer = types.find(variable.pointer);
        if (pointer == types.end() || pointer->second.args.size() < 2) {
            return out;
        }
        const u32 pointee = pointer->second.args[1];
        if (member_builtin[pointee]) {
            continue; // gl_PerVertex and the like
        }
        std::string sig = signature(signature, pointee, 0);
        const bool buffer_block = decorations[pointee].contains(3);
        switch (variable.storage) {
        case 0:  // UniformConstant
        case 2:  // Uniform
        case 12: { // StorageBuffer
            const char* kind = variable.storage == 0                    ? "rc:"
                               : variable.storage == 12 || buffer_block ? "ssbo:"
                                                                        : "ubo:";
            const auto set = decoration.find(34), binding = decoration.find(33);
            if (set == decoration.end() || binding == decoration.end()) {
                return out;
            }
            out.descriptors[u64(set->second) << 32 | binding->second].insert(kind + sig);
            break;
        }
        case 9: // PushConstant
            out.push = sig;
            break;
        case 1: // Input
        case 3: { // Output
            const auto location = decoration.find(30);
            if (location == decoration.end()) {
                return out;
            }
            (variable.storage == 1 ? out.inputs : out.outputs)[location->second] = sig;
            break;
        }
        default:
            break;
        }
    }
    out.valid = true;
    return out;
}

} // namespace

bool InterfaceFits(std::span<const u32> translated, std::span<const u32> replacement) {
    const Interface game = Reflect(translated), ours = Reflect(replacement);
    if (!game.valid || !ours.valid) {
        return false;
    }
    for (const auto& [slot, declarations] : ours.descriptors) {
        const auto it = game.descriptors.find(slot);
        if (it == game.descriptors.end()) {
            return false;
        }
        for (const auto& declaration : declarations) {
            if (!it->second.contains(declaration)) {
                return false;
            }
        }
    }
    if (!ours.push.empty() && ours.push != game.push) {
        return false;
    }
    for (const auto& [location, sig] : ours.outputs) {
        const auto it = game.outputs.find(location);
        if (it == game.outputs.end() || it->second != sig) {
            return false;
        }
    }
    // The pipeline's other stages feed the game's inputs; ours may read only those.
    for (const auto& [location, sig] : ours.inputs) {
        const auto it = game.inputs.find(location);
        if (it == game.inputs.end() || it->second != sig) {
            return false;
        }
    }
    return true;
}

bool On() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_VOLUMETRIC_RAYS");
        const bool enabled = env && std::strcmp(env, "1") == 0;
        if (enabled) {
            std::printf("Volumetric light: from a grid of camera rays (GOW3_VOLUMETRIC_RAYS=1)\n");
            std::fflush(stdout);
        }
        return enabled;
    }();
    return on;
}

std::span<const u32> ModuleCode(const Shader::Info& info, std::span<const u32> translated) {
    if (!On()) {
        return translated;
    }
    const u64 hash = info.pgm_hash;
    size_t index;
    if (const LineSampler* sampler = FindSampler(hash, index)) {
        u32 fits = 0;
        for (const Variant variant : {sampler->first, sampler->second}) {
            if (InterfaceFits(translated, GridCode[variant])) {
                fits |= 1u << variant;
            }
        }
        // A translation that fits no variant turns this light's grid off for the session.
        if (fits == 0) {
            if (!g_broken[index].exchange(true)) {
                Report("a line sampler's bindings changed: no grid pass, the game's lines", hash);
            }
            return translated;
        }
        const u32 before = g_fitting[index].fetch_or(fits);
        if ((before | fits) != before && !g_broken[index].load()) {
            Report(fits & ((1u << SpotArray) | (1u << ProjectorArray))
                       ? "the grid pass follows the line sampler (shadow map array)"
                       : "the grid pass follows the line sampler",
                   hash);
        }
        return translated;
    }
    for (int resolve = 0; resolve < ResolveCount; ++resolve) {
        if (hash != Resolves[resolve].hash) {
            continue;
        }
        if (info.render_size_literals != 0) {
            Report("the resolve holds render-size literals: left as translated", hash);
            return translated;
        }
        if (!InterfaceFits(translated, Resolves[resolve].code)) {
            Report("the resolve's bindings changed: left as translated (regenerate it with "
                   "tools/volumetric_resolve_source.py)",
                   hash);
            return translated;
        }
        if (!g_resolve_ready[resolve].exchange(true)) {
            Report("the resolve reads the grid", hash);
        }
        return Resolves[resolve].code;
    }
    return translated;
}

int GridVariant(const Shader::Info& cs) {
    if (!On()) {
        return -1;
    }
    size_t index;
    const LineSampler* sampler = FindSampler(cs.pgm_hash, index);
    if (!sampler || g_broken[index].load(std::memory_order_relaxed) ||
        !g_resolve_ready[sampler->resolve].load(std::memory_order_relaxed)) {
        return -1;
    }
    Variant variant = sampler->first;
    if (sampler->second != sampler->first) {
        // The shadow map: the shader's second image (binding 2), 2D or an array.
        if (cs.images.size() < 2) {
            return -1;
        }
        const auto& shadow = cs.images[1];
        const auto view = shadow.GetSharp(cs).GetViewType(shadow.is_array);
        if (view == AmdGpu::ImageType::Color2DArray) {
            variant = sampler->second;
        } else if (view != AmdGpu::ImageType::Color2D) {
            return -1;
        }
    }
    return g_fitting[index].load(std::memory_order_relaxed) & (1u << variant) ? variant : -1;
}

vk::Pipeline GridPipeline(vk::Device device, vk::PipelineLayout layout, int variant) {
    if (variant < 0 || variant >= VariantCount) {
        return {};
    }
    static std::mutex mutex;
    static std::map<std::pair<VkPipelineLayout, int>, vk::Pipeline> pipelines;
    std::scoped_lock lock{mutex};
    const auto key = std::make_pair(static_cast<VkPipelineLayout>(layout), variant);
    if (const auto it = pipelines.find(key); it != pipelines.end()) {
        return it->second;
    }
    const vk::ShaderModule module = CompileSPV(GridCode[variant], device);
    const auto [result, pipeline] = device.createComputePipeline(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = layout,
            });
    device.destroyShaderModule(module);
    const vk::Pipeline handle = result == vk::Result::eSuccess ? pipeline : vk::Pipeline{};
    static const char* const names[VariantCount] = {"sun or moon", "spotlight",
                                                    "spotlight (shadow map array)", "flashlight",
                                                    "flashlight (shadow map array)"};
    std::printf("Volumetric light: %s grid pass %s\n", names[variant],
                handle ? "started (the light's volume drawn from camera rays)"
                       : "could not be created");
    std::fflush(stdout);
    // Kept for the session: the layouts belong to the pipeline cache's pipelines, which live as
    // long.
    pipelines.emplace(key, handle);
    return handle;
}

} // namespace Vulkan::VolumetricLight

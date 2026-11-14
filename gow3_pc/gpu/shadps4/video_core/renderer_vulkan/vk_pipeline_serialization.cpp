// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <map>
#ifdef _WIN32
#include <windows.h>
#endif
#include "common/hash.h"
#include "common/serdes.h"
#include "common/scope_exit.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "common/thread.h"
#include "gow3_overlay.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/render_size_literals.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_warmup_inbox.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 11u; // MUBUF ADDR64 loads (gow3)
static constexpr u32 ShaderMetaVersion = 13u; // gow3: the hash of the binary it was written with
/// gow3 21 Oct 2026: metas of earlier builds (no binary hash) still load when their binary was
/// written together with them (DataBase::WrittenTogether).
static constexpr u32 LegacyShaderMetaVersion = 12u; // BufferResource::access_extent
static constexpr u32 PipelineKeyVersion = 5u; // gow3: Info layout (ImageResource::needs_native)
/// gow3 10 Nov 2026: starts the trailing WorkgroupSplit record of a compute shader's meta
/// ("WGSP"); older metas end without it (or with zero padding).
static constexpr u32 WorkgroupSplitMagic = 0x50534757u;
/// gow3 10 Nov 2026: starts the record of the direct memory access setting a shader was translated
/// under ("GDMA"), in every stage's meta.
static constexpr u32 DirectMemoryMagic = 0x414d4447u;
/// gow3 10 Nov 2026: starts the record of whether a compute shader's LDS accesses were
/// bounds-checked (SharedMemoryBoundsPass), "GLDB".
static constexpr u32 LdsBoundsMagic = 0x42444c47u;
/// gow3 12 Nov 2026: starts the record of the render size whose literals the game wrote into a
/// shader's code ("GRS2", a u32: width | height << 16, 0 for none; render_size_literals.h), for
/// a shader translated with the materials' ambient shadow grid read from the texture
/// (quarter_grid_pass.cpp). Metas with the first version of the record ("GRSZ") count as having
/// none.
static constexpr u32 RenderSizeMagic = 0x32535247u;
/// gow3 14 Nov 2026 (#55): starts the record of whether a compute shader was translated with
/// workgroup barriers made uniform (UniformBarrierPass, GOW3_UNIFORM_BARRIERS), "GUB3". Its
/// presence also marks a compute shader translated with its full VGPR count and with waterfall
/// loops over a relative register index served per lane (Translator::WaterfallLaneValue).
static constexpr u32 UniformBarriersMagic = 0x33425547u;

/// gow3: the cache name of a shader's SPIR-V. Shaders that gained multi-wave LDS barriers get a
/// new name, so binaries cached before the fix are not found and only those shaders recompile.
static std::string ShaderBinaryName(u64 pgm_hash, size_t perm_idx) {
    // gow3 11 Nov 2026: and shaders compiled with lanes (lane_coarsening.cpp), "_lanes2".
    const u32 lanes = Shader::Optimization::LaneCoarsening(pgm_hash);
    // gow3 12 Nov 2026: and The Last of Us' quarter-size depth written at the ambient shadows'
    // pixels (quarter_depth_pass.cpp), "_qd2".
    return fmt::format("{:#018x}_{}{}{}{}", pgm_hash, perm_idx,
                       Shader::Optimization::MultiwaveLdsBarriers(pgm_hash) ? "_lds4" : "",
                       lanes > 1 ? fmt::format("_lanes{}", lanes) : std::string{},
                       Shader::Optimization::QuarterDepthPointSample(pgm_hash) ? "_qd2" : "");
}
} // namespace Serialization

namespace Vulkan {

namespace PipelineBinaryCache {
bool ValidKey(const Key& key) {
    return !key.empty() && key.size() <= VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;
}

// gow3 25 Oct 2026: the shared store, <directory>/shared/<driver key>_<XXH3>.blob, the raw
// binary. Binaries read or written this session stay in memory (one 2 MiB binary for The Last of
// Us), up to a budget.
namespace {
std::mutex g_shared_mutex;
std::unordered_map<std::filesystem::path::string_type, std::shared_ptr<const std::vector<u8>>>
    g_shared_kept;
size_t g_shared_kept_bytes = 0;
constexpr size_t SharedKeptBudget = 256 * 1024 * 1024;
std::atomic<u64> g_shared_stored{0}, g_shared_read{0}, g_shared_damaged{0};

std::filesystem::path SharedPath(const std::filesystem::path& directory, const Key& key, u64 hash) {
    std::string name;
    for (const auto byte : key) {
        name += fmt::format("{:02x}", byte);
    }
    return directory / "shared" / fmt::format("{}_{:016x}.blob", name, hash);
}

std::shared_ptr<const std::vector<u8>> KeepShared(const std::filesystem::path& path,
                                                  std::shared_ptr<const std::vector<u8>> data) {
    std::scoped_lock lock{g_shared_mutex};
    const auto [it, inserted] = g_shared_kept.try_emplace(path.native());
    if (!inserted) {
        return it->second; // another thread read it first
    }
    if (g_shared_kept_bytes + data->size() > SharedKeptBudget) {
        g_shared_kept.erase(it);
        return data;
    }
    g_shared_kept_bytes += data->size();
    it->second = data;
    return data;
}

std::shared_ptr<const std::vector<u8>> LoadShared(const std::filesystem::path& directory,
                                                  const Key& key, u32 size, u64 hash) {
    const auto path = SharedPath(directory, key, hash);
    {
        std::scoped_lock lock{g_shared_mutex};
        if (const auto it = g_shared_kept.find(path.native()); it != g_shared_kept.end()) {
            return it->second->size() == size ? it->second : nullptr;
        }
    }
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return nullptr;
    }
    auto data = std::make_shared<std::vector<u8>>(size);
    const bool whole = stream.tellg() == std::streamoff(size) && stream.seekg(0) &&
                       stream.read(reinterpret_cast<char*>(data->data()), size);
    stream.close();
    if (!whole || XXH3_64bits(data->data(), data->size()) != hash) {
        ++g_shared_damaged;
        std::error_code error;
        std::filesystem::remove(path, error);
        return nullptr;
    }
    ++g_shared_read;
    return KeepShared(path, std::move(data));
}
} // namespace
std::vector<u8> Encode(const Key& global, const Key& pipeline, u64 graphics_hash,
                       const std::vector<Binary>& binaries) {
    if (!ValidKey(global) || !ValidKey(pipeline) || binaries.empty() ||
        binaries.size() > MaxBinaries) {
        return {};
    }
    std::vector<u8> bytes;
    const auto append = [&](const auto& value) {
        const auto* p = reinterpret_cast<const u8*>(&value);
        bytes.insert(bytes.end(), p, p + sizeof(value));
    };
    const auto blob = [&](const auto& data) {
        append(u32(data.size()));
        bytes.insert(bytes.end(), data.begin(), data.end());
    };
    // Version 2: each binary has a kind, 0 inline (size, data) or 1 shared (size, hash).
    const bool any_shared =
        std::ranges::any_of(binaries, [](const Binary& binary) { return binary.shared_hash != 0; });
    append(u32{0x31425047}); // GPB1, little-endian Windows/Linux cache format.
    append(u32{any_shared ? 2u : 1u});
    append(graphics_hash);
    blob(global);
    blob(pipeline);
    append(u32(binaries.size()));
    for (const auto& binary : binaries) {
        const bool shared = binary.shared_hash != 0;
        if (!ValidKey(binary.key) ||
            (shared ? !binary.shared_size || binary.shared_size > MaxFileSize
                    : binary.data.empty() || binary.data.size() > MaxFileSize) ||
            bytes.size() + binary.key.size() + (shared ? 0 : binary.data.size()) + 24 >
                MaxFileSize) {
            return {};
        }
        blob(binary.key);
        if (any_shared) {
            append(u32{shared ? 1u : 0u});
        }
        if (shared) {
            append(binary.shared_size);
            append(binary.shared_hash);
        } else {
            blob(binary.data);
        }
    }
    append(u64{XXH3_64bits(bytes.data(), bytes.size())});
    return bytes;
}
std::optional<std::vector<Binary>> Decode(std::span<const u8> bytes, const Key& global,
                                          const Key& pipeline, u64 graphics_hash) {
    if (bytes.size() < 8 || bytes.size() > MaxFileSize || !ValidKey(global) ||
        !ValidKey(pipeline)) {
        return std::nullopt;
    }
    u64 checksum;
    std::memcpy(&checksum, bytes.data() + bytes.size() - 8, 8);
    bytes = bytes.first(bytes.size() - 8);
    if (checksum != XXH3_64bits(bytes.data(), bytes.size())) {
        return std::nullopt;
    }
    size_t offset = 0;
    const auto read = [&](auto& value) {
        if (sizeof(value) > bytes.size() - offset) {
            return false;
        }
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        offset += sizeof(value);
        return true;
    };
    const auto blob = [&](auto& data, size_t limit) {
        u32 size;
        if (!read(size) || !size || size > limit || size > bytes.size() - offset) {
            return false;
        }
        data.assign(bytes.begin() + offset, bytes.begin() + offset + size);
        offset += size;
        return true;
    };
    u32 magic, version, count;
    u64 stored_hash;
    Key stored_global, stored_pipeline;
    if (!read(magic) || magic != 0x31425047 || !read(version) || (version != 1 && version != 2) ||
        !read(stored_hash) || stored_hash != graphics_hash || !blob(stored_global, 32) ||
        stored_global != global || !blob(stored_pipeline, 32) || stored_pipeline != pipeline ||
        !read(count) || !count || count > MaxBinaries) {
        return std::nullopt;
    }
    std::vector<Binary> binaries(count);
    for (auto& binary : binaries) {
        if (!blob(binary.key, 32)) {
            return std::nullopt;
        }
        u32 kind = 0;
        if (version == 2 && (!read(kind) || kind > 1)) {
            return std::nullopt;
        }
        if (kind == 1) {
            if (!read(binary.shared_size) || !binary.shared_size ||
                binary.shared_size > MaxFileSize || !read(binary.shared_hash) ||
                !binary.shared_hash) {
                return std::nullopt;
            }
        } else if (!blob(binary.data, MaxFileSize)) {
            return std::nullopt;
        }
    }
    return offset == bytes.size() ? std::optional{std::move(binaries)} : std::nullopt;
}
bool Save(const std::filesystem::path& file, std::span<const u8> bytes) {
    if (bytes.empty() || bytes.size() > MaxFileSize) {
        return false;
    }
    std::error_code error;
    std::filesystem::create_directories(file.parent_path(), error);
    if (error) {
        return false;
    }
    if (auto* pack = Storage::DataBase::Instance().PackFor(file.parent_path())) {
        pack->Invalidate(file.filename().string());
    }
    // Unique siblings avoid concurrent writers truncating one another's temporary data.
    static std::atomic<u64> sequence{0};
    auto temp = file;
    temp += "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "." + std::to_string(sequence++) + ".tmp";
    bool written = false;
    {
        std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
        if (stream) {
            stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            stream.flush();
            written = bool(stream);
            stream.close();
            written = written && !stream.fail();
        }
    }
    bool renamed = false;
    if (written) {
#ifdef _WIN32
        renamed = MoveFileExW(temp.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
        std::filesystem::rename(temp, file, error);
        renamed = !error;
#endif
    }
    if (!renamed) {
        std::filesystem::remove(temp, error);
    }
    return renamed;
}
std::optional<std::vector<Binary>> Load(const std::filesystem::path& file, const Key& global,
                                        const Key& pipeline, u64 graphics_hash) {
    std::vector<u8> bytes;
    // gow3 13 Nov 2026 (blob_pack.h): the store's pack first, as the file is now.
    auto* pack = Storage::DataBase::Instance().PackFor(file.parent_path());
    const auto file_name = pack ? file.filename().string() : std::string{};
    if (!pack || !pack->Read(file_name, bytes) || bytes.size() > MaxFileSize) {
        std::ifstream stream(file, std::ios::binary | std::ios::ate);
        if (!stream) {
            return std::nullopt;
        }
        const auto size = stream.tellg();
        if (size <= 0 || size > std::streamoff(MaxFileSize)) {
            return std::nullopt;
        }
        bytes.resize(static_cast<size_t>(size));
        stream.seekg(0);
        if (!stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) {
            return std::nullopt;
        }
        if (pack) {
            pack->Keep(file_name, bytes);
        }
    }
    auto binaries = Decode(bytes, global, pipeline, graphics_hash);
    if (!binaries) {
        return std::nullopt;
    }
    for (auto& binary : *binaries) {
        if (binary.shared_hash &&
            !(binary.shared = LoadShared(file.parent_path(), binary.key, binary.shared_size,
                                         binary.shared_hash))) {
            return std::nullopt;
        }
    }
    return binaries;
}

bool HasLargeInline(const std::vector<Binary>& binaries) {
    return std::ranges::any_of(binaries, [](const Binary& binary) {
        return !binary.shared_hash && binary.data.size() >= SharedMinSize;
    });
}

bool StoreShared(const std::filesystem::path& directory, std::vector<Binary>& binaries) {
    bool stored_all = true;
    for (auto& binary : binaries) {
        if (binary.shared_hash || binary.data.size() < SharedMinSize ||
            binary.data.size() > MaxFileSize) {
            continue;
        }
        const u64 hash = XXH3_64bits(binary.data.data(), binary.data.size());
        if (!hash) {
            continue;
        }
        const u32 size = u32(binary.data.size());
        // A stored copy is used when it reads back whole (a damaged one is written again).
        auto data = LoadShared(directory, binary.key, size, hash);
        if (!data) {
            auto owned = std::make_shared<std::vector<u8>>(std::move(binary.data));
            if (!Save(SharedPath(directory, binary.key, hash), *owned)) {
                binary.data = std::move(*owned);
                stored_all = false;
                continue;
            }
            ++g_shared_stored;
            data = KeepShared(SharedPath(directory, binary.key, hash), std::move(owned));
        }
        binary.data.clear();
        binary.data.shrink_to_fit();
        binary.shared = std::move(data);
        binary.shared_size = size;
        binary.shared_hash = hash;
    }
    return stored_all;
}
} // namespace PipelineBinaryCache

namespace {
std::atomic<u64> binary_hits{}, binary_misses{}, binary_rejected{}, binary_saved{}, binary_errors{};
std::atomic<u64> binary_restore_ns{}, binary_compile_ns{};
std::atomic<u64> binary_restore_calls{}, binary_compile_calls{};
std::atomic<u64> binary_key_ns{}, binary_load_ns{}, binary_handles_ns{};
std::atomic<u64> binary_moved_to_shared{};
u64 ElapsedNs(std::chrono::steady_clock::time_point started) {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - started).count());
}
PipelineBinaryCache::Key BinaryKey(const vk::PipelineBinaryKeyKHR& key) {
    if (!key.keySize || key.keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR) {
        return {};
    }
    return {key.key.begin(), key.key.begin() + key.keySize};
}

std::optional<std::vector<PipelineBinaryCache::Binary>> LoadLegacyBinary(
    const std::filesystem::path& directory, const std::string& name,
    const PipelineBinaryCache::Key& global, const PipelineBinaryCache::Key& pipeline) {
    using Index = std::unordered_multimap<std::string, std::pair<std::filesystem::path, u64>>;
    static std::mutex mutex;
    static std::map<std::filesystem::path, Index> indexes;
    std::vector<std::pair<std::filesystem::path, u64>> candidates;
    {
        std::scoped_lock lock{mutex};
        auto [it, inserted] = indexes.try_emplace(directory);
        if (inserted) {
            std::error_code error;
            for (std::filesystem::directory_iterator files(directory, error), end;
                 !error && files != end; files.increment(error)) {
                const auto stem = files->path().stem().string();
                if (files->path().extension() != ".bin" || stem.size() < 18 || stem[16] != '_') {
                    continue;
                }
                try {
                    size_t used = 0;
                    const auto hash = std::stoull(stem.substr(0, 16), &used, 16);
                    if (used == 16) {
                        it->second.emplace(stem.substr(17), std::pair{files->path(), hash});
                    }
                } catch (const std::exception&) {
                }
            }
        }
        const auto [first, last] = it->second.equal_range(name);
        for (auto entry = first; entry != last; ++entry) {
            candidates.push_back(entry->second);
        }
    }
    for (const auto& [file, hash] : candidates) {
        if (auto binaries = PipelineBinaryCache::Load(file, global, pipeline, hash)) {
            return binaries;
        }
    }
    return std::nullopt;
}
} // namespace

void ReportPipelineBinaryCache() {
    const auto hits = binary_hits.load();
    const auto misses = binary_misses.load();
    std::printf("Pipeline binary cache: %llu hits, %llu compiled, %llu rejected, %llu saved, "
                "%llu cache errors (restore %.3f ms, compile %.3f ms each); shared binaries: "
                "%llu stored, %llu read, %llu damaged, %llu files moved to them\n",
                static_cast<unsigned long long>(hits), static_cast<unsigned long long>(misses),
                static_cast<unsigned long long>(binary_rejected.load()),
                static_cast<unsigned long long>(binary_saved.load()),
                static_cast<unsigned long long>(binary_errors.load()),
                binary_restore_ns.load() / 1e6 / std::max(u64{1}, binary_restore_calls.load()),
                binary_compile_ns.load() / 1e6 / std::max(u64{1}, binary_compile_calls.load()),
                static_cast<unsigned long long>(PipelineBinaryCache::g_shared_stored.load()),
                static_cast<unsigned long long>(PipelineBinaryCache::g_shared_read.load()),
                static_cast<unsigned long long>(PipelineBinaryCache::g_shared_damaged.load()),
                static_cast<unsigned long long>(binary_moved_to_shared.load()));
}

vk::ResultValue<vk::UniquePipeline> CreateCachedGraphicsPipeline(
    vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info,
    const GraphicsPipelineKey& key, const vk::PipelineBinaryKeyKHR& global_key,
    const std::filesystem::path& directory) {
    using namespace PipelineBinaryCache;
    const auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
    const bool enabled = global_key.keySize && EmulatorSettings.IsPipelineCacheEnabled();
    const auto create = [&](vk::PipelineCache used_cache, const auto& used_info, bool restored) {
        const auto started = std::chrono::steady_clock::now();
        auto result = device.createGraphicsPipelineUnique(used_cache, used_info);
        const auto ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count());
        extern std::atomic<u64> g_gow3_pipeline_driver_ns;
        g_gow3_pipeline_driver_ns += ns;
        if (enabled) {
            (restored ? binary_restore_ns : binary_compile_ns) += ns;
            ++(restored ? binary_restore_calls : binary_compile_calls);
        }
        return result;
    };
    if (!enabled) {
        return create(cache, info, false);
    }
    SCOPE_EXIT {
        const auto count = binary_hits.load() + binary_misses.load();
        if (count && count % 256 == 0) {
            ReportPipelineBinaryCache();
        }
    };
    const auto global = BinaryKey(global_key);
    vk::PipelineCreateFlags2CreateInfoKHR flags{
        .pNext = info.pNext,
        .flags = vk::PipelineCreateFlags2{static_cast<VkPipelineCreateFlags>(info.flags)} |
                 vk::PipelineCreateFlagBits2::eCaptureDataKHR,
    };
    auto captured_info = info;
    captured_info.pNext = &flags;
    vk::PipelineCreateInfoKHR key_info{.pNext = &captured_info};
    vk::PipelineBinaryKeyKHR pipeline_key;
    const auto key_started = std::chrono::steady_clock::now();
    const auto key_result = device.getPipelineKeyKHR(&key_info, &pipeline_key);
    binary_key_ns += ElapsedNs(key_started);
    if (key_result != vk::Result::eSuccess ||
        !ValidKey(global) || !ValidKey(BinaryKey(pipeline_key))) {
        ++binary_errors;
        ++binary_misses;
        return create(cache, info, false);
    }
    const auto pipeline_id = BinaryKey(pipeline_key);
    std::string name;
    for (const auto byte : pipeline_id) {
        name += fmt::format("{:02x}", byte);
    }
    // The driver key identifies the pipeline independently of emulator draw-state keys.
    const auto file = directory / (name + ".bin");
    std::vector<VkPipelineBinaryKHR> handles;
    const auto destroy = [&] {
        for (auto handle : handles) {
            if (handle) {
                dispatch.vkDestroyPipelineBinaryKHR(device, handle, nullptr);
            }
        }
        handles.clear();
    };
    SCOPE_EXIT { destroy(); };
    try {
        const auto load_started = std::chrono::steady_clock::now();
        auto binaries = Load(file, global, pipeline_id, 0);
        if (!binaries) {
            binaries = LoadLegacyBinary(directory, name, global, pipeline_id);
            if (binaries) {
                StoreShared(directory, *binaries);
                Save(file, Encode(global, pipeline_id, 0, *binaries));
            }
        } else if (HasLargeInline(*binaries) && StoreShared(directory, *binaries) &&
                   Save(file, Encode(global, pipeline_id, 0, *binaries))) {
            ++binary_moved_to_shared; // gow3 25 Oct 2026: a file of before the shared store
        }
        binary_load_ns += ElapsedNs(load_started);
        if (binaries) {
            std::vector<VkPipelineBinaryKeyKHR> keys(binaries->size());
            std::vector<VkPipelineBinaryDataKHR> data(binaries->size());
            for (size_t i = 0; i < binaries->size(); ++i) {
                keys[i].sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
                keys[i].keySize = u32((*binaries)[i].key.size());
                std::ranges::copy((*binaries)[i].key, keys[i].key);
                const auto bytes = (*binaries)[i].Bytes();
                data[i] = {bytes.size(), const_cast<u8*>(bytes.data())};
            }
            VkPipelineBinaryKeysAndDataKHR stored{u32(keys.size()), keys.data(), data.data()};
            VkPipelineBinaryCreateInfoKHR binary_info{
                VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
            binary_info.pKeysAndDataInfo = &stored;
            handles.resize(keys.size());
            VkPipelineBinaryHandlesInfoKHR output{
                VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
            output.pipelineBinaryCount = u32(handles.size());
            output.pPipelineBinaries = handles.data();
            const auto handles_started = std::chrono::steady_clock::now();
            const auto handles_result =
                dispatch.vkCreatePipelineBinariesKHR(device, &binary_info, nullptr, &output);
            binary_handles_ns += ElapsedNs(handles_started);
            if (handles_result == VK_SUCCESS) {
                VkPipelineBinaryInfoKHR restored{VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR};
                restored.pNext = &flags;
                restored.binaryCount = u32(handles.size());
                restored.pPipelineBinaries = handles.data();
                // CAPTURE_DATA is only for compilation, never for binary-backed creation.
                flags.flags =
                    vk::PipelineCreateFlags2{static_cast<VkPipelineCreateFlags>(info.flags)};
                auto restored_info = info;
                restored_info.pNext = &restored;
                auto result = create(nullptr, restored_info, true);
                if (result.result == vk::Result::eSuccess) {
                    ++binary_hits;
                    return result;
                }
            }
            ++binary_rejected;
            destroy();
        }
    } catch (const std::exception&) {
        ++binary_errors;
        destroy();
    }
    ++binary_misses;
    flags.flags = vk::PipelineCreateFlags2{static_cast<VkPipelineCreateFlags>(info.flags)} |
                  vk::PipelineCreateFlagBits2::eCaptureDataKHR;
    auto result = create(nullptr, captured_info, false);
    if (result.result != vk::Result::eSuccess) {
        ++binary_errors;
        return create(cache, info, false);
    }
    VkReleaseCapturedPipelineDataInfoKHR release{
        VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR};
    release.pipeline = *result.value;
    SCOPE_EXIT {
        if (dispatch.vkReleaseCapturedPipelineDataKHR(device, &release, nullptr) != VK_SUCCESS) {
            ++binary_errors;
        }
    };
    try {
        VkPipelineBinaryCreateInfoKHR binary_info{
            VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
        binary_info.pipeline = *result.value;
        VkPipelineBinaryHandlesInfoKHR output{VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
        if (dispatch.vkCreatePipelineBinariesKHR(device, &binary_info, nullptr, &output) !=
                VK_SUCCESS ||
            !output.pipelineBinaryCount || output.pipelineBinaryCount > MaxBinaries) {
            ++binary_errors;
            return result;
        }
        handles.resize(output.pipelineBinaryCount);
        output.pPipelineBinaries = handles.data();
        if (dispatch.vkCreatePipelineBinariesKHR(device, &binary_info, nullptr, &output) !=
            VK_SUCCESS) {
            ++binary_errors;
            return result;
        }
        std::vector<Binary> binaries;
        size_t total_size = 0;
        for (const auto handle : handles) {
            VkPipelineBinaryDataInfoKHR data_info{VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};
            data_info.pipelineBinary = handle;
            vk::PipelineBinaryKeyKHR binary_key;
            size_t size = 0;
            auto* raw_key = reinterpret_cast<VkPipelineBinaryKeyKHR*>(&binary_key);
            if (dispatch.vkGetPipelineBinaryDataKHR(device, &data_info, raw_key, &size, nullptr) !=
                    VK_SUCCESS ||
                !size || size > MaxFileSize || total_size + size > MaxFileSize ||
                !ValidKey(BinaryKey(binary_key))) {
                ++binary_errors;
                return result;
            }
            total_size += size;
            Binary binary{BinaryKey(binary_key), std::vector<u8>(size)};
            if (dispatch.vkGetPipelineBinaryDataKHR(device, &data_info, raw_key, &size,
                                                    binary.data.data()) != VK_SUCCESS ||
                size > binary.data.size()) {
                ++binary_errors;
                return result;
            }
            binary.key = BinaryKey(binary_key);
            binary.data.resize(size);
            binaries.push_back(std::move(binary));
        }
        StoreShared(directory, binaries);
        const auto encoded = Encode(global, pipeline_id, 0, binaries);
        if (Save(file, encoded)) {
            ++binary_saved;
        } else {
            ++binary_errors;
        }
    } catch (const std::exception&) {
        ++binary_errors;
    }
    return result;
}

// gow3: GOW3_PARALLEL_WARMUP=1. The cache store is still read and the shader modules created in
// order on the calling thread (program_cache and its permutation checks are not thread-safe);
// only the pipeline objects, the slow part, are built here. The runtime compiler builds
// pipelines on worker threads the same way (vk_pipeline_cache.cpp).
struct PipelineCache::WarmupPool {
    explicit WarmupPool(u32 count, bool low_priority = false) {
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this, low_priority] {
                if (low_priority) {
                    Common::SetCurrentThreadName("GoW3:ShaderWarmup");
                    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
                }
                Run();
            });
        }
    }
    ~WarmupPool() {
        {
            std::scoped_lock lock{mutex};
            jobs.clear(); // only the running ones finish
            closing = true;
        }
        cv.notify_all();
        for (auto& thread : threads) {
            thread.join();
        }
    }
    void Push(std::function<bool()> job) {
        {
            std::scoped_lock lock{mutex};
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }
    /// Waits for every job; `tick` runs on this thread meanwhile (the loading screen).
    void Wait(const std::function<void()>& tick) {
        std::unique_lock lock{mutex};
        while (!jobs.empty() || running) {
            idle.wait_for(lock, std::chrono::milliseconds(50));
            lock.unlock();
            tick();
            lock.lock();
        }
    }
    /// Drops the queued jobs and waits for the running ones; the threads stay parked.
    void Drain() {
        std::unique_lock lock{mutex};
        jobs.clear();
        idle.wait(lock, [this] { return running == 0; });
    }
    bool Idle() {
        std::scoped_lock lock{mutex};
        return jobs.empty() && !running;
    }
    static u32 Threads() {
        const char* env = std::getenv("GOW3_PARALLEL_WARMUP");
        if (!env || !*env || *env == '0') {
            return 0;
        }
        const u32 cores = std::max(1u, std::thread::hardware_concurrency());
        return std::clamp(cores > 2 ? cores - 2 : 1u, 1u, 6u);
    }

    std::atomic<u32> built{0};
    std::atomic<u32> failed{0};

private:
    void Run() {
        std::unique_lock lock{mutex};
        while (true) {
            cv.wait(lock, [this] { return closing || !jobs.empty(); });
            if (jobs.empty()) {
                return;
            }
            auto job = std::move(jobs.front());
            jobs.pop_front();
            ++running;
            lock.unlock();
            bool ok = false;
            try {
                ok = job();
            } catch (const std::exception& e) {
                LOG_ERROR(Render_Vulkan, "gow3: warm-up pipeline failed: {}", e.what());
            } catch (...) {
                LOG_ERROR(Render_Vulkan, "gow3: warm-up pipeline failed");
            }
            (ok ? built : failed).fetch_add(1, std::memory_order_relaxed);
            lock.lock();
            --running;
            idle.notify_all();
        }
    }

    std::mutex mutex;
    std::condition_variable cv, idle;
    std::deque<std::function<bool()>> jobs;
    std::vector<std::thread> threads;
    u32 running = 0;
    bool closing = false;
};

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

// gow3 21 Oct 2026 (a tiled floor drawn flat red on one PC only): a meta and
// the binary of the same permutation are separate files. On 20 Oct a session rewrote
// 0x91304408_4.spv with another permutation's code while the rename of its meta failed (the
// warm-up was reading the old one; only that .meta.tmp was left), and every later start drew the
// floor with the wrong shader. A meta now carries the hash of the binary written just before it,
// and a binary that does not match its meta is not used (the permutation compiles again).
std::mutex g_binary_hash_mutex;
std::unordered_map<u64, u64> g_binary_hashes; // HashCombine(program, permutation) -> XXH3 of SPIR-V

u64 TakeBinaryHash(u64 perm_hash) {
    std::scoped_lock lock{g_binary_hash_mutex};
    const auto it = g_binary_hashes.find(perm_hash);
    if (it == g_binary_hashes.end()) {
        return 0;
    }
    const u64 hash = it->second;
    g_binary_hashes.erase(it);
    return hash;
}

/// Whether `spv` (the binary named by `pgm_hash`/`perm_idx`) is the one its meta was written
/// with. Metas of earlier builds have no hash (0): their files must have been written together.
bool BinaryMatchesMeta(const std::vector<u32>& spv, u64 binary_hash, u64 pgm_hash,
                       size_t perm_idx) {
    const auto meta_name = fmt::format("{:#018x}", HashCombine(pgm_hash, u64(perm_idx)));
    const auto binary_name = Serialization::ShaderBinaryName(pgm_hash, perm_idx);
    auto& db = Storage::DataBase::Instance();
    const bool matches =
        binary_hash != 0 ? XXH3_64bits(spv.data(), spv.size() * sizeof(u32)) == binary_hash
                         : db.WrittenTogether(Storage::BlobType::ShaderMeta, meta_name,
                                              Storage::BlobType::ShaderBinary, binary_name);
    if (!matches) {
        // Neither file is trusted again: the permutation compiles and saves anew when used.
        db.Remove(Storage::BlobType::ShaderMeta, meta_name);
        db.Remove(Storage::BlobType::ShaderBinary, binary_name);
        static std::atomic<u32> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 32) {
            std::printf("Pipeline cache: shader %#llx permutation %zu: its binary is not the one "
                        "its description was written with (%s); compiled again\n",
                        static_cast<unsigned long long>(pgm_hash), perm_idx,
                        binary_hash != 0 ? "hash differs" : "written at another time");
            std::fflush(stdout);
        }
    }
    return matches;
}

/// gow3 12 Nov 2026 (render_size_literals.h, #5/#66): whether this cached shader was translated
/// from code the game had written another render size into. Its meta records that size; a meta
/// of an earlier build has no record and its binary is searched for another size's literals
/// (every cache before 11 Nov 2026 was translated at 1080p), and in a pixel shader for this
/// size's too: they are the ambient shadows' grid as fixed numbers, which QuarterGridPass now
/// reads from the texture (drawn at half size by tools/probe_tlou.py). Such a shader compiles
/// again.
bool StaleRenderSize(const Shader::Info& info, const std::vector<u32>& spv) {
    // GOW3_RENDER_SIZE_CACHE=0 (diagnostics): cached shaders load whatever size they were
    // translated for, as before.
    static const bool on = [] {
        const char* env = std::getenv("GOW3_RENDER_SIZE_CACHE");
        return !(env && env[0] == '0' && env[1] == 0);
    }();
    if (!on) {
        return false;
    }
    const u32 size = Shader::GameRenderSize();
    const auto fixed_grid = [&] {
        if (info.sw_stage != Shader::SwStage::Fragment ||
            !Shader::Optimization::QuarterGridOn()) {
            return false;
        }
        const auto words = Shader::RenderSizeLiteralWords();
        return std::ranges::any_of(spv, [&](u32 word) {
            return std::ranges::find(words, word) != words.end();
        });
    };
    const bool stale = info.render_size_record
                           ? info.render_size_literals != 0 && info.render_size_literals != size
                           : Shader::HoldsOtherRenderSizeLiterals(spv) || fixed_grid();
    if (stale) {
        static std::atomic<u32> count{0};
        const u32 n = count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n == 1 || n == 100 || n == 1000) {
            std::printf("Pipeline cache: %u shader(s) cached for another render size than %ux%u "
                        "compile again (latest: %#llx)\n",
                        n, size & 0xFFFF, size >> 16,
                        static_cast<unsigned long long>(info.pgm_hash));
            std::fflush(stdout);
        }
    }
    return stale;
}

// gow3 21 Oct 2026 (#92, #95): before the compile-storm fix, two compute shaders
// (8f067540, 7d7e7206) built a new version for every set of garbage floats in buffer slots
// they don't use, and the cache kept each one: 1,268 and 383 versions here, 23,845 compute
// pipelines in one reporter's cache. The warm-up built all of them at every start (332 s in #95;
// an RX 9070 XT's driver crashed near the end in #92), and none is ever used again: those slots
// now count as empty. A shader with this many cached versions (normal ones have a few dozen at
// most, 35 here) is not warmed up and its files are removed; the versions it needs compile on
// first use and are cached anew.
constexpr size_t CrowdedShaderVersions = 100;
// Filled before a warm-up reads the store, then read by the same thread only.
std::unordered_set<u64> g_crowded_programs;

void FindCrowdedShaders() {
    g_crowded_programs.clear();
    auto& db = Storage::DataBase::Instance();
    std::unordered_map<u64, std::vector<std::string>> binaries;
    db.ForEachBlobName(Storage::BlobType::ShaderBinary, [&](const std::string& name) {
        unsigned long long pgm_hash{};
        if (std::sscanf(name.c_str(), "%llx_", &pgm_hash) == 1) {
            binaries[pgm_hash].push_back(name);
        }
    });
    for (const auto& [pgm_hash, names] : binaries) {
        if (names.size() < CrowdedShaderVersions) {
            continue;
        }
        g_crowded_programs.insert(pgm_hash);
        // Each version's binary, description and compute pipeline key (named by the same hash).
        for (const auto& name : names) {
            unsigned long long pgm{};
            size_t perm_idx{};
            if (std::sscanf(name.c_str(), "%llx_%zu", &pgm, &perm_idx) != 2) {
                continue;
            }
            const u64 perm_hash = HashCombine(pgm_hash, u64(perm_idx));
            db.Remove(Storage::BlobType::ShaderBinary, name);
            db.Remove(Storage::BlobType::ShaderMeta, fmt::format("{:#018x}", perm_hash));
            db.Remove(Storage::BlobType::PipelineKey, fmt::format("c_{:#018x}", perm_hash));
        }
        std::printf("Pipeline cache: shader %#llx has %zu cached versions (left by a compile "
                    "storm): not warmed up, removed; the ones in use compile again\n",
                    static_cast<unsigned long long>(pgm_hash), names.size());
    }
    std::fflush(stdout);
}

bool IsCrowdedShader(u64 pgm_hash) {
    return g_crowded_programs.contains(pgm_hash);
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);
    meta.Write(TakeBinaryHash(perm_hash));

    spec.Serialize(ar);
    info.Serialize(ar);
    // gow3 10 Nov 2026: trailing records, absent from older metas, which load as before.
    // The direct memory access switch the shader was translated under: a shader translated with
    // it off has no DMA buffers and uses_dma = false, which said nothing about the switch, so
    // turning it back on kept those binaries (the PS4 menu's translucent surfaces drawn wrong, and
    // pipelines of one program mixing permutations of both kinds: layout errors 07988/07990).
    {
        Serialization::Writer tail{ar};
        tail.Write(Serialization::DirectMemoryMagic);
        tail.Write(static_cast<u8>(EmulatorSettings.IsDirectMemoryAccessEnabled()));
    }
    // WorkgroupSplit: only a hint for LoadShaderMeta (whether to compile again); the dispatch
    // takes the split from the binary's LocalSize.
    if (info.sw_stage == Shader::SwStage::Compute) {
        Serialization::Writer tail{ar};
        tail.Write(Serialization::WorkgroupSplitMagic);
        tail.Write(static_cast<u8>(info.cs_split_eligible));
        tail.Write(static_cast<u8>(Shader::WorkgroupSplit(info.cs_split_eligible,
                                                          spec.runtime_info.hw.cs)));
        // SharedMemoryBoundsPass (GOW3_LDS_BOUNDS): whether its LDS accesses were checked.
        tail.Write(Serialization::LdsBoundsMagic);
        tail.Write(static_cast<u8>(Shader::Optimization::SharedMemoryBounds(info.pgm_hash)));
    }
    // The render size the game wrote into the shader's code (StaleRenderSize), in every stage.
    {
        Serialization::Writer tail{ar};
        tail.Write(Serialization::RenderSizeMagic);
        tail.Write(info.render_size_literals);
    }
    // UniformBarrierPass (GOW3_UNIFORM_BARRIERS): whether it ran. Last, so earlier builds, which
    // stop at a record they do not know, still read the ones above.
    if (info.sw_stage == Shader::SwStage::Compute) {
        Serialization::Writer tail{ar};
        tail.Write(Serialization::UniformBarriersMagic);
        tail.Write(static_cast<u8>(Shader::Optimization::UniformBarriersOn()));
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    {
        std::scoped_lock lock{g_binary_hash_mutex};
        g_binary_hashes[HashCombine(pgm_hash, u64(perm_idx))] =
            XXH3_64bits(spv.data(), spv.size() * sizeof(u32));
    }
    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       Serialization::ShaderBinaryName(pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx, u64& binary_hash) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion &&
        meta_version != Serialization::LegacyShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);
    binary_hash = 0;
    if (meta_version == Serialization::ShaderMetaVersion) {
        meta.Read(binary_hash);
    }

    spec.Deserialize(ar);
    info.Deserialize(ar);

    // gow3 10 Nov 2026: the trailing records (RegisterShaderMeta), in any order; zero padding or
    // anything else ends them.
    bool has_tail = false, has_dma_record = false, has_uniform_record = false;
    u8 eligible = 0, applied = 1, dma_at_compile = 0, lds_bounds = 0, uniform_barriers = 0;
    info.render_size_record = false;
    info.render_size_literals = 0;
    while (ar.Remaining() >= sizeof(u32)) {
        Serialization::Reader tail{ar};
        u32 magic{};
        tail.Read(magic);
        if (magic == Serialization::WorkgroupSplitMagic && ar.Remaining() >= 2 * sizeof(u8)) {
            tail.Read(eligible);
            tail.Read(applied);
            has_tail = true;
        } else if (magic == Serialization::DirectMemoryMagic && ar.Remaining() >= sizeof(u8)) {
            tail.Read(dma_at_compile);
            has_dma_record = true;
        } else if (magic == Serialization::LdsBoundsMagic && ar.Remaining() >= sizeof(u8)) {
            tail.Read(lds_bounds);
        } else if (magic == Serialization::UniformBarriersMagic && ar.Remaining() >= sizeof(u8)) {
            tail.Read(uniform_barriers);
            has_uniform_record = true;
        } else if (magic == Serialization::RenderSizeMagic && ar.Remaining() >= sizeof(u32)) {
            tail.Read(info.render_size_literals);
            info.render_size_record = true;
        } else {
            break;
        }
    }
    // A shader translated under the other direct memory access setting compiles again, either way.
    if (has_dma_record &&
        (dma_at_compile != 0) != EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        return false;
    }
    // A compute shader translated with the LDS bounds checks the other way (or before them: no
    // record) compiles again once (SharedMemoryBoundsPass, GOW3_LDS_BOUNDS).
    if (info.sw_stage == Shader::SwStage::Compute &&
        (lds_bounds != 0) != Shader::Optimization::SharedMemoryBounds(info.pgm_hash)) {
        return false;
    }
    // gow3 14 Nov 2026 (#55): a compute shader translated with UniformBarrierPass the other way
    // compiles again. Metas from before the record compile again too: they were translated with
    // 4 VGPRs too few (FillComputeRuntimeInfo), which the permutation key cannot tell apart.
    if (info.sw_stage == Shader::SwStage::Compute &&
        (!has_uniform_record ||
         (uniform_barriers != 0) != Shader::Optimization::UniformBarriersOn())) {
        return false;
    }

    // WorkgroupSplit (shader_recompiler/runtime_info.h): a compute shader cached before the split
    // (no record), or cached with the switch the other way, compiles again when the split would
    // change its binary. Other shaders load as before.
    if (info.sw_stage == Shader::SwStage::Compute) {
        info.cs_split_eligible = has_tail && eligible != 0;
        const auto& cs = spec.runtime_info.hw.cs;
        const bool would_split = Shader::WorkgroupSplitSize(cs.workgroup_size) && !cs.indirect;
        if (would_split && !has_tail) {
            return false; // eligibility unknown: translated again once
        }
        if (has_tail && Shader::WorkgroupSplit(info.cs_split_eligible, cs) != applied) {
            return false;
        }
    }

    // Motion vertex shaders embed session-local buffer device addresses. They must be
    // recompiled for the current allocation, never loaded from a previous process.
    if (info.hw_stage == Shader::HwStage::Vertex && spec.runtime_info.hw.vs.motion_vectors) {
        return false;
    }
    // gow3 3 Nov 2026: shaders cached with direct memory access read guest memory by address;
    // with GOW3_DIRECT_MEMORY_ACCESS=0 they are compiled again without it (the cache holds no
    // trace of the setting, so turning it off in the launcher changed nothing).
    if (info.uses_dma && !EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        return false;
    }
    // gow3 2 Nov 2026: the rain's alpha is scaled by the recompiler (GOW3_RAIN_SCALE); caches made
    // before that hold the original, and this small shader compiles in no time.
    if (info.pgm_hash == 0x2288df23) {
        return false;
    }

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    if (warmup_pool) {
        // The slot stays null until WarmUp moves the built pipeline in (the map may rehash).
        warmup_pool->Push([this, key = compute_key, info = sel.infos[0], module = sel.modules[0],
                           sdata]() mutable {
            auto pipeline = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap,
                                                              profile, *pipeline_cache, key, *info,
                                                              module, sdata, true);
            std::scoped_lock lock{warmup_results_mutex};
            warmup_compute.emplace_back(key, std::move(pipeline));
            return true;
        });
    } else {
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key,
                                                       *sel.infos[0], sel.modules[0], sdata, true);
    }

    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    sel.graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = sel.graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(sel.graphics_key);
    ASSERT(is_new);

    if (warmup_pool) {
        warmup_pool->Push([this, key = sel.graphics_key, infos = sel.infos,
                           runtime_infos = sel.runtime_infos, fetch = sel.fetch_shader,
                           modules = sel.modules, sdata]() mutable {
            auto pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, key, *pipeline_cache, infos,
                runtime_infos, fetch, modules, sdata, true);
            std::scoped_lock lock{warmup_results_mutex};
            warmup_graphics.emplace_back(key, std::move(pipeline));
            return true;
        });
    } else {
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, sel.graphics_key, *pipeline_cache, sel.infos,
            sel.runtime_infos, sel.fetch_shader, sel.modules, sdata, true);
    }

    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);
    sel.fetch_shader.reset();

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    u64 binary_hash{};
    if (!LoadShaderMeta(ar, program->info, sel.fetch_shader, spec, perm_idx, binary_hash) ||
        IsCrowdedShader(program->info.pgm_hash)) {
        return false;
    }

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       Serialization::ShaderBinaryName(program->info.pgm_hash, perm_idx),
                                       spv);
    if (spv.empty() ||
        !BinaryMatchesMeta(spv, binary_hash, program->info.pgm_hash, perm_idx) ||
        StaleRenderSize(program->info, spv)) {
        return false;
    }

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::move(program);
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            // A matching permutation is valid only at its original index. A different index means
            // the store holds entries from more than one cache generation, so this pipeline is
            // left to compile at runtime.
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} conflicts with index {}, skipping "
                            "preload",
                            perm_idx, program->info.hw_stage, program->info.pgm_hash, idx);
                return false;
            }
            module = it->module;
        } else {
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);

    sel.infos[stage] = &it_pgm.value()->info;
    sel.modules[stage] = module;

    return true;
}

bool PipelineCache::CheckCacheProfile() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return false;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return false;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render, "Pipeline cache profile has unexpected size ({} != {})",
                    profile_data.size(), sizeof(Shader::Profile));
    }
    Shader::Profile cached_profile{};
    if (profile_data.size() == sizeof(Shader::Profile)) {
        std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    }
    if (profile_data.size() != sizeof(Shader::Profile) || cached_profile != profile) {
        // gow3: upstream closed the cache for the session here, so it was never rewritten
        // and every later session compiled every shader again (stutters on each new area).
        // Start a fresh cache for this build and GPU instead.
        LOG_WARNING(Render, "Pipeline cache isn't compatible with current system: rebuilding it");
        // gow3 21 Oct 2026: a profile of the same build but another GPU (hybrid laptops) is kept
        // aside instead, and this GPU's own cache comes back if it was kept (SwapCache).
        if (profile_data.size() == sizeof(Shader::Profile)) {
            const bool restored = Storage::DataBase::Instance().SwapCache(
                [&](const std::vector<u8>& data) {
                    Shader::Profile stored{};
                    if (data.size() != sizeof(stored)) {
                        return false;
                    }
                    std::memcpy(&stored, data.data(), sizeof(stored));
                    return stored == profile;
                });
            if (restored) {
                return true;
            }
        } else {
            Storage::DataBase::Instance().Clear();
        }
        Storage::DataBase::Instance().FinishPreload();
        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return false;
    }
    return true;
}

void PipelineCache::WarmUp(const std::function<void(u32, u32)>& progress) {
    if (!CheckCacheProfile()) {
        return;
    }

    FindCrowdedShaders();
    u32 num_pipelines{};
    u32 num_total_pipelines{};
    const u32 expected =
        u32(Storage::DataBase::Instance().CountBlobs(Storage::BlobType::PipelineKey));
    const auto started = std::chrono::steady_clock::now();
    const u32 warmup_threads = WarmupPool::Threads();
    std::unique_ptr<WarmupPool> pool;
    if (warmup_threads) {
        pool = std::make_unique<WarmupPool>(warmup_threads);
        warmup_pool = pool.get();
    }
    // In parallel mode the screen follows the pipelines built, not the ones read.
    const auto report = [&] {
        if (progress) {
            const u32 done = pool ? pool->built + pool->failed : num_total_pipelines;
            progress(done, std::max(expected, num_total_pipelines));
        }
    };

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;
            report();

            // gow3 (SpartanPort c22329f): a damaged record is skipped, and the stage selection
            // it left half filled is cleared.
            try {
                Serialization::Archive ar{std::move(data)};
                Serialization::Reader pldata{ar};

                u32 version{};
                pldata.Read(version);
                if (version != Serialization::PipelineKeyVersion) {
                    return;
                }

                u32 is_compute{};
                pldata.Read(is_compute);

                bool result{};
                if (is_compute) {
                    result = LoadComputePipeline(ar);
                } else {
                    result = LoadGraphicsPipeline(ar);
                }

                if (result) {
                    ++num_pipelines;
                }
            } catch (const std::exception&) {
                sel.infos.fill(nullptr);
                sel.modules.fill(nullptr);
                sel.fetch_shader.reset();
            }
        });

    if (pool) {
        pool->Wait(report);
        warmup_pool = nullptr;
        // Every slot was reserved by the loaders; the ones whose build failed stay out.
        for (auto& [key, pipeline] : warmup_graphics) {
            graphics_pipelines[key] = std::move(pipeline);
        }
        for (auto& [key, pipeline] : warmup_compute) {
            compute_pipelines[key] = std::move(pipeline);
        }
        for (auto it = graphics_pipelines.begin(); it != graphics_pipelines.end();) {
            it = it->second ? std::next(it) : graphics_pipelines.erase(it);
        }
        for (auto it = compute_pipelines.begin(); it != compute_pipelines.end();) {
            it = it->second ? std::next(it) : compute_pipelines.erase(it);
        }
        num_pipelines -= pool->failed;
        warmup_graphics.clear();
        warmup_compute.clear();
        pool.reset();
    }
    std::printf("Pipeline warm-up: %u of %u pipelines in %.2f s (%u threads)\n", num_pipelines,
                num_total_pipelines,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                warmup_threads);
    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    ReportPipelineBinaryCache();
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::StartOnDemand() {
    CheckCacheProfile();
    Storage::DataBase::Instance().FinishPreload();
    std::printf("Pipeline startup: on demand, no bulk warm-up\n");
}

// gow3: one cached shader stage read by the background warm-up, shared by every cached
// pipeline that uses it. A pool thread fills it; the GPU thread merges it into program_cache
// once (MergeWarmStage).
struct PipelineCache::WarmStage {
    vk::Device device;
    std::unique_ptr<Program> program;
    Shader::StageSpecialization spec{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    size_t perm_idx{};
    vk::ShaderModule module{};
    std::once_flag loaded; ///< the first reader that needs it reads it; the others wait
    // GPU thread, from the merge on.
    bool merged = false;
    bool module_kept = false;           ///< the module now belongs to a Program
    const Shader::Info* info = nullptr; ///< null: not usable (read failed or conflict)
    vk::ShaderModule result{};

    ~WarmStage() {
        if (module && !module_kept) {
            device.destroyShaderModule(module);
        }
    }
};

// gow3: StartBackgroundWarmUp. The pool reads the store and builds the pipelines at low
// priority; the GPU thread merges the stages and publishes the pipelines a few at a time.
struct PipelineCache::BackgroundWarmup {
    struct Entry {
        std::chrono::steady_clock::time_point queued;
        bool compute = false;
        GraphicsPipelineKey graphics_key{};
        ComputePipelineKey compute_key{};
        GraphicsPipeline::SerializationSupport sdata{};
        std::array<std::shared_ptr<WarmStage>, MaxShaderStages> stages{};
    };
    struct Built {
        GraphicsPipelineKey graphics_key{};
        ComputePipelineKey compute_key{};
        std::unique_ptr<GraphicsPipeline> graphics;
        std::unique_ptr<ComputePipeline> compute;
    };

    std::unordered_map<u64, std::shared_ptr<WarmStage>> stages; ///< readers, under stages_mutex
    std::mutex stages_mutex;
    std::vector<std::string> order; ///< PipelineKey names, in reading order
    std::atomic<size_t> next_read{0};
    std::atomic<u32> readers_running{0};
    std::chrono::steady_clock::time_point reader_started{};
    WarmupInbox<Entry> read;
    WarmupInbox<Built> built;
    std::atomic<u32> total{0}, done{0};
    std::atomic<bool> reading{true}, stopping{false};
    std::atomic<u64> reader_ns{0}, entry_ns{0}, shader_io_ns{0}, module_ns{0};
    std::atomic<u64> read_queue_ns{0}, build_queue_ns{0}, build_ns{0};
    std::atomic<u32> entries{0}, modules{0}, submitted{0}, completed{0};
    std::atomic<u32> in_order{0}; ///< read first, in the last session's order of use
    u64 pump_ns = 0, pump_max_ns = 0, pump_gap_max_ns = 0;
    u32 batches = 0;
    bool finished = false;
    u32 loaded = 0, threads = 0, readers = 1;
    std::chrono::steady_clock::time_point started, last_batch;
    std::chrono::steady_clock::time_point last_pump{}, last_report{};
    std::unique_ptr<WarmupPool> pool; ///< last: its jobs use the members above
};

bool PipelineCache::StartBackgroundWarmUp() {
    if (!Storage::DataBase::Instance().ConcurrentReads()) {
        return false;
    }
    if (!CheckCacheProfile()) {
        return true; // nothing cached yet
    }
    background = new BackgroundWarmup;
    auto& warm = *background;
    warm.started = std::chrono::steady_clock::now();
    warm.threads = WarmupThreadCount(std::max(1u, std::thread::hardware_concurrency()));
    warm.pool = std::make_unique<WarmupPool>(warm.threads, true);
    // The first job reads the whole store; with one thread the builds queue behind it.
    warm.pool->Push([this, &warm] {
        const auto reader_started = std::chrono::steady_clock::now();
        auto& db = Storage::DataBase::Instance();
        FindCrowdedShaders();
        // gow3 25 Oct 2026: the pipelines the game used last come first (SaveWarmupOrder). In
        // directory order the ones a start needs (menu, the area of the save) were spread over
        // the whole read (~16 s), and the game built ~100 of them itself in its first seconds
        // (frames of 40-130 ms).
        std::vector<std::string> names;
        db.ForEachBlobName(Storage::BlobType::PipelineKey,
                           [&](const std::string& name) { names.push_back(name); });
        warm.total = u32(names.size());
        previous_warm_order = ReadWarmupOrder();
        std::unordered_set<std::string> unread{names.begin(), names.end()};
        std::vector<std::string> order;
        order.reserve(names.size());
        for (const auto& name : previous_warm_order) {
            if (unread.erase(name)) {
                order.push_back(name);
            }
        }
        warm.in_order = u32(order.size());
        for (auto& name : names) {
            if (unread.contains(name)) {
                order.push_back(std::move(name));
            }
        }
        warm.order = std::move(order);
        warm.reader_started = reader_started;
        previous_warm_order_read = true;
        // gow3 26 Oct 2026: several readers (half the pool, GOW3_WARMUP_READERS=<n>): one read
        // only ~280-400 entries a second while the game loaded (two small files per new shader
        // stage, ~2.8 ms each), and the session's first pipelines waited behind it.
        static const u32 readers_wanted = [] {
            const char* env = std::getenv("GOW3_WARMUP_READERS");
            return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
        }();
        const u32 readers = std::clamp(readers_wanted ? readers_wanted : warm.threads / 2, 1u,
                                       warm.threads);
        warm.readers = readers;
        warm.readers_running = readers;
        for (u32 i = 1; i < readers; ++i) {
            warm.pool->Push([this, &warm] {
                ReadWarmEntries(warm);
                return true;
            });
        }
        ReadWarmEntries(warm);
        return true;
    });
    std::printf("Pipeline warm-up: loading in the background (%u threads)\n", warm.threads);
    return true;
}

void PipelineCache::ReadWarmEntries(BackgroundWarmup& warm) {
    auto& db = Storage::DataBase::Instance();
    for (size_t i = warm.next_read.fetch_add(1, std::memory_order_relaxed); i < warm.order.size();
         i = warm.next_read.fetch_add(1, std::memory_order_relaxed)) {
        if (warm.stopping.load(std::memory_order_relaxed)) {
            break;
        }
        std::vector<u8> data;
        db.Load(Storage::BlobType::PipelineKey, warm.order[i], data);
        if (!data.empty()) {
            // gow3: a damaged record is skipped; the reader keeps going.
            try {
                ReadWarmEntry(warm, std::move(data));
            } catch (const std::exception&) {
                warm.done.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    // The last reader ends the reading (the GPU thread waits for `reading` to finish up).
    if (warm.readers_running.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        {
            std::scoped_lock lock{warm.stages_mutex};
            warm.stages.clear();
        }
        warm.reader_ns = ElapsedNs(warm.reader_started);
        warm.reading = false;
    }
}

void PipelineCache::ReadWarmEntry(BackgroundWarmup& warm, std::vector<u8>&& data) {
    const auto entry_started = std::chrono::steady_clock::now();
    SCOPE_EXIT { warm.entry_ns += ElapsedNs(entry_started); ++warm.entries; };
    const auto skip = [&] { warm.done.fetch_add(1, std::memory_order_relaxed); };
    Serialization::Archive ar{std::move(data)};
    Serialization::Reader pldata{ar};
    u32 version{};
    pldata.Read(version);
    if (version != Serialization::PipelineKeyVersion) {
        return skip();
    }
    u32 is_compute{};
    pldata.Read(is_compute);

    BackgroundWarmup::Entry entry{};
    entry.compute = is_compute != 0;
    std::array<u64, MaxShaderStages> hashes{};
    if (entry.compute) {
        entry.compute_key.Deserialize(ar);
        ComputePipeline::SerializationSupport sdata{};
        sdata.Deserialize(ar);
        hashes[0] = entry.compute_key.value;
    } else {
        entry.graphics_key.Deserialize(ar);
        entry.sdata.Deserialize(ar);
        for (u32 i = 0; i < MaxShaderStages; ++i) {
            hashes[i] = entry.graphics_key.stage_hashes[i];
        }
    }

    auto& db = Storage::DataBase::Instance();
    for (u32 i = 0; i < MaxShaderStages; ++i) {
        if (!hashes[i]) {
            continue;
        }
        std::shared_ptr<WarmStage> stage;
        {
            std::scoped_lock lock{warm.stages_mutex};
            auto& slot = warm.stages[hashes[i]];
            if (!slot) {
                slot = std::make_shared<WarmStage>();
            }
            stage = slot;
        }
        std::call_once(stage->loaded, [&] {
            stage->device = instance.GetDevice();
            stage->program = std::make_unique<Program>();
            stage->spec.info = &stage->program->info;
            const auto io_started = std::chrono::steady_clock::now();
            u64 module_time = 0;
            std::vector<u8> meta_blob;
            db.Load(Storage::BlobType::ShaderMeta, fmt::format("{:#018x}", hashes[i]), meta_blob);
            if (!meta_blob.empty()) {
                Serialization::Archive meta_ar{std::move(meta_blob)};
                u64 binary_hash{};
                // gow3: a damaged meta leaves the stage unusable (no module), not thrown out of
                // call_once to every record that shares it.
                bool meta_ok = false;
                try {
                    meta_ok = LoadShaderMeta(meta_ar, stage->program->info, stage->fetch,
                                             stage->spec, stage->perm_idx, binary_hash);
                } catch (const std::exception&) {
                }
                if (meta_ok && !IsCrowdedShader(stage->program->info.pgm_hash)) {
                    std::vector<u32> spv;
                    db.Load(Storage::BlobType::ShaderBinary,
                            Serialization::ShaderBinaryName(stage->program->info.pgm_hash,
                                                            stage->perm_idx),
                            spv);
                    if (!spv.empty() &&
                        BinaryMatchesMeta(spv, binary_hash, stage->program->info.pgm_hash,
                                          stage->perm_idx) &&
                        !StaleRenderSize(stage->program->info, spv)) {
                        const auto module_started = std::chrono::steady_clock::now();
                        stage->module = CompileSPV(spv, stage->device);
                        module_time = ElapsedNs(module_started);
                        warm.module_ns += module_time;
                        ++warm.modules;
                    }
                }
            }
            warm.shader_io_ns += ElapsedNs(io_started) - module_time;
            if (!stage->module) {
                stage->merged = true; // unusable: info stays null
            }
        });
        entry.stages[i] = stage;
    }
    entry.queued = std::chrono::steady_clock::now();
    warm.read.Push(std::move(entry));
}

bool PipelineCache::MergeWarmStage(WarmStage& stage) {
    stage.merged = true;
    const u64 hash = stage.program->info.pgm_hash;
    const auto it_pgm = program_cache.find(hash);
    if (it_pgm == program_cache.end()) {
        Program* program = stage.program.get();
        program->InsertPermut(stage.module, std::move(stage.spec), stage.perm_idx);
        program_cache.emplace(hash, std::move(stage.program));
        stage.module_kept = true;
        stage.result = stage.module;
        stage.info = &program->info;
        return true;
    }
    // The game already uses this shader: its permutations are the ones that count.
    Program& program = *it_pgm->second;
    if (program.pending) {
        return false; // the game is compiling a permutation that takes the next index
    }
    const auto it = std::ranges::find(program.modules, stage.spec, &Program::Module::spec);
    if (it != program.modules.end()) {
        if (size_t(std::distance(program.modules.begin(), it)) != stage.perm_idx) {
            return false;
        }
        stage.result = it->module;
    } else {
        if (stage.perm_idx < program.modules.size() && program.modules[stage.perm_idx].module) {
            return false;
        }
        stage.spec.info = &program.info;
        program.InsertPermut(stage.module, std::move(stage.spec), stage.perm_idx);
        stage.module_kept = true;
        stage.result = stage.module;
    }
    stage.info = &program.info;
    return true;
}

void PipelineCache::PumpWarmUp() {
    if (!background || background->finished) {
        return;
    }
    auto& warm = *background;
    using namespace std::chrono_literals;
    const auto pump_started = std::chrono::steady_clock::now();
    if (warm.last_pump.time_since_epoch().count()) {
        warm.pump_gap_max_ns = std::max(warm.pump_gap_max_ns,
            u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    pump_started - warm.last_pump).count()));
    }
    warm.last_pump = pump_started;
    SCOPE_EXIT {
        const auto ns = ElapsedNs(pump_started);
        warm.pump_ns += ns;
        warm.pump_max_ns = std::max(warm.pump_max_ns, ns);
    };

    // Built pipelines: the first one ready for a key wins (the game may have compiled it).
    for (auto& built : warm.built.Take(64)) {
        const bool kept =
            built.graphics
                ? PublishFirst(graphics_pipelines, built.graphics_key, std::move(built.graphics))
                : PublishFirst(compute_pipelines, built.compute_key, std::move(built.compute));
        warm.loaded += kept;
        warm.done.fetch_add(1, std::memory_order_relaxed);
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - warm.last_batch < 4ms) {
        return;
    }
    warm.last_batch = now;
    ++warm.batches;
    const u32 done = warm.done + warm.pool->failed;
    Gow3Overlay::SetBackgroundLoading(done, std::max(warm.total.load(), done));

    static const std::array<Shader::RuntimeInfo, MaxShaderStages> no_runtime_infos{};
    for (auto& entry : warm.read.Take(32)) {
        warm.read_queue_ns += ElapsedNs(entry.queued);
        if (entry.compute ? compute_pipelines.contains(entry.compute_key)
                          : graphics_pipelines.contains(entry.graphics_key)) {
            warm.done.fetch_add(1, std::memory_order_relaxed); // the game compiled it first
            continue;
        }
        std::array<const Shader::Info*, MaxShaderStages> infos{};
        std::array<vk::ShaderModule, MaxShaderStages> modules{};
        std::optional<Shader::Gcn::FetchShaderData> fetch;
        bool usable = true;
        {
            std::unique_lock lock{programs_mutex};
            for (u32 i = 0; i < MaxShaderStages && usable; ++i) {
                if (!entry.stages[i]) {
                    continue;
                }
                auto& stage = *entry.stages[i];
                if (!stage.merged) {
                    MergeWarmStage(stage);
                }
                usable = stage.info != nullptr;
                infos[i] = stage.info;
                modules[i] = stage.result;
                fetch = stage.fetch; // the blocking load keeps the last stage's too
            }
        }
        if (!usable) {
            warm.done.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        ++warm.submitted;
        warm.pool->Push([this, &warm, queued = std::chrono::steady_clock::now(),
                          compute = entry.compute, graphics_key = entry.graphics_key,
                         compute_key = entry.compute_key, sdata = entry.sdata, infos, modules,
                         fetch]() mutable {
            const auto build_started = std::chrono::steady_clock::now();
            warm.build_queue_ns += ElapsedNs(queued);
            SCOPE_EXIT { warm.build_ns += ElapsedNs(build_started); ++warm.completed; };
            BackgroundWarmup::Built built{graphics_key, compute_key};
            if (compute) {
                ComputePipeline::SerializationSupport compute_data{};
                built.compute = std::make_unique<ComputePipeline>(
                    instance, scheduler, desc_heap, profile, *pipeline_cache, compute_key,
                    *infos[0], modules[0], compute_data, true);
            } else {
                built.graphics = std::make_unique<GraphicsPipeline>(
                    instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
                    no_runtime_infos, fetch, modules, sdata, true);
            }
            warm.built.Push(std::move(built));
            return true;
        });
    }

    if (now - warm.last_report >= 1s) {
        warm.last_report = now;
        std::printf("Warm-up progress: %.2f s, read %u, submitted %u, completed %u, done %u/%u, "
                    "batches %u, reading %u\n",
                    std::chrono::duration<double>(now - warm.started).count(),
                    warm.entries.load(), warm.submitted.load(), warm.completed.load(),
                    warm.done.load(), warm.total.load(), warm.batches, unsigned(warm.reading.load()));
    }

    if (warm.reading || !warm.read.Empty() || !warm.pool->Idle() || !warm.built.Empty()) {
        return;
    }
    std::printf("Pipeline warm-up: %u of %u pipelines in %.2f s in the background (%u threads, "
                "%u reading), %u read first in the order of their last use\n",
                warm.loaded, warm.total.load(),
                std::chrono::duration<double>(now - warm.started).count(), warm.threads,
                warm.readers, warm.in_order.load());
    LOG_INFO(Render, "Preloaded {} pipelines in the background", warm.loaded);
    ReportPipelineBinaryCache();
    std::printf("Warm-up profile: reader wall %.3f s, entries %.3f s, shader IO/meta %.3f s, "
                "modules %.3f s (%u); GPU pump %.3f s, max %.3f ms, gap max %.3f ms, "
                "batches %u; builds %.3f thread-s, read queue %.3f item-s, "
                "build queue %.3f item-s\n",
                warm.reader_ns.load() / 1e9, warm.entry_ns.load() / 1e9,
                warm.shader_io_ns.load() / 1e9, warm.module_ns.load() / 1e9, warm.modules.load(),
                warm.pump_ns / 1e9, warm.pump_max_ns / 1e6, warm.pump_gap_max_ns / 1e6,
                warm.batches, warm.build_ns.load() / 1e9, warm.read_queue_ns.load() / 1e9,
                warm.build_queue_ns.load() / 1e9);
    std::printf("Pipeline binary profile: keys %.3f thread-s, disk/decode/migration %.3f thread-s, "
                "binary handles %.3f thread-s, driver restore %.3f thread-s, "
                "driver compile %.3f thread-s\n",
                binary_key_ns.load() / 1e9, binary_load_ns.load() / 1e9,
                binary_handles_ns.load() / 1e9, binary_restore_ns.load() / 1e9,
                binary_compile_ns.load() / 1e9);
    Storage::DataBase::Instance().ReportPacks();
    // gow3 13 Nov 2026: the packs written from the warm-up's reads, for the next start.
    Storage::DataBase::Instance().FinishPacks();
    extern std::atomic<u64> g_gow3_pipeline_total_ns, g_gow3_pipeline_driver_ns;
    extern std::atomic<u32> g_gow3_pipelines_built;
    const u32 built = std::max(1u, g_gow3_pipelines_built.load());
    std::printf("Pipeline build time: %u graphics pipelines, %.1f ms each (driver %.1f ms, the rest "
                "%.1f ms), %.1f s in all\n",
                built, g_gow3_pipeline_total_ns / 1e6 / built, g_gow3_pipeline_driver_ns / 1e6 / built,
                (g_gow3_pipeline_total_ns - g_gow3_pipeline_driver_ns) / 1e6 / built,
                g_gow3_pipeline_total_ns / 1e9);
    // The idle threads stay parked (StopWarmUp).
    warm.finished = true;
    Gow3Overlay::SetBackgroundLoading(0, 0);
    Storage::DataBase::Instance().FinishPreload();
}

namespace {
constexpr std::string_view WarmupOrderHeader = "gow3 warm-up order 1\n";
} // namespace

// gow3 25 Oct 2026: warmup-order.bin, a text file of PipelineKey blob names, one a line, the
// pipelines used most recently first.
std::vector<std::string> PipelineCache::ReadWarmupOrder() {
    auto& db = Storage::DataBase::Instance();
    std::vector<std::string> names;
    if (!db.Exists(Storage::BlobType::ShaderProfile, "warmup-order")) {
        return names;
    }
    std::vector<u8> data;
    db.Load(Storage::BlobType::ShaderProfile, "warmup-order", data);
    const std::string_view text{reinterpret_cast<const char*>(data.data()), data.size()};
    if (!text.starts_with(WarmupOrderHeader)) {
        return names;
    }
    for (size_t at = WarmupOrderHeader.size(); at < text.size();) {
        const size_t end = std::min(text.find('\n', at), text.size());
        if (end > at) {
            names.emplace_back(text.substr(at, end - at));
        }
        at = end + 1;
    }
    return names;
}

void PipelineCache::NoteWarmupOrder() {
    // Every 4096 lookups a clock read; a save a minute after the warm-up and every minute on.
    if ((++warm_order_calls & 4095) != 0 || (background && !background->finished)) {
        return;
    }
    auto& db = Storage::DataBase::Instance();
    if (!db.IsOpened() || !db.ConcurrentReads()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (warm_order_saved == std::chrono::steady_clock::time_point{}) {
        warm_order_saved = now;
        return;
    }
    if (now - warm_order_saved < std::chrono::seconds(60)) {
        return;
    }
    warm_order_saved = now;
    SaveWarmupOrder();
    // gow3 13 Nov 2026: without a background warm-up, the packs written from the reads so far are
    // completed a minute in (blob_pack.h; once a session).
    db.FinishPacks();
    SaveDriverPipelineCache(false);
}

void PipelineCache::SaveWarmupOrder() {
    const auto started = std::chrono::steady_clock::now();
    if (!previous_warm_order_read) {
        previous_warm_order = ReadWarmupOrder();
        previous_warm_order_read = true;
    }
    std::vector<std::pair<u64, std::string>> used;
    used.reserve(graphics_pipelines.size() + compute_pipelines.size());
    for (const auto& [key, pipeline] : graphics_pipelines) {
        if (pipeline && pipeline->last_use) {
            used.emplace_back(pipeline->last_use,
                              fmt::format("g_{:#018x}", std::hash<GraphicsPipelineKey>{}(key)));
        }
    }
    for (const auto& [key, pipeline] : compute_pipelines) {
        if (pipeline && pipeline->last_use) {
            used.emplace_back(pipeline->last_use, fmt::format("c_{:#018x}", key.value));
        }
    }
    std::ranges::stable_sort(used, std::greater{}, &std::pair<u64, std::string>::first);
    // The ones this session did not use keep their place after it.
    std::unordered_set<std::string_view> listed;
    std::string text{WarmupOrderHeader};
    for (const auto& [tick, name] : used) {
        if (listed.insert(name).second) {
            text += name;
            text += '\n';
        }
    }
    u32 kept = 0;
    for (const auto& name : previous_warm_order) {
        if (listed.insert(name).second) {
            text += name;
            text += '\n';
            ++kept;
        }
    }
    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "warmup-order",
                                       std::vector<u8>(text.begin(), text.end()));
    static bool reported = false;
    if (!reported) {
        reported = true;
        std::printf("Pipeline warm-up order: %zu used this session, %u kept from before (%.1f ms; "
                    "saved every minute)\n",
                    used.size(), kept, ElapsedNs(started) / 1e6);
    }
}

void PipelineCache::StopWarmUp() {
    if (!background) {
        return;
    }
    background->stopping = true;
    background->pool->Drain();
    // ponytail: leaked with its parked threads (this runs at exit only): a thread that ended
    // while the game ran crashed in ntdll (RtlWakeAllConditionVariable). Join them once the
    // cause is known.
    background = nullptr;
    Gow3Overlay::SetBackgroundLoading(0, 0);
}

void PipelineCache::Sync() {
    StopWarmUp();
    FinishCompilations();
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        // gow3: truncated walker code is rejected before registering bytes past the record.
        if (walker_func_size > ar.Remaining()) {
            walker_func = nullptr;
            throw Serialization::CorruptData{};
        }
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (fetch_shader_data) {
        spec.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader

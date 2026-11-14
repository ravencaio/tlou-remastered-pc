#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdio>
#include <functional>
#include "common/thread.h"
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <string>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include "gow3_copy.h"
#include "gow3_toggles.h"
#include "common/alignment.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_low_vram.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
extern std::function<void(u64)> g_describe_gpu_address;
void DumpGpuCheckpoints(const Instance& instance);
void DumpDeviceFault(const Instance& instance);
}

#ifdef _WIN32
extern "C" char __ImageBase;
extern "C" __declspec(dllimport) unsigned short __stdcall RtlCaptureStackBackTrace(
    unsigned long frames_to_skip, unsigned long frames_to_capture, void** back_trace,
    unsigned long* back_trace_hash);
#endif

namespace VideoCore {

// gow3 21 Oct 2026 (GPU worker): GOW3_GPU_PROFILE segments for the cache's own GPU copies
// (uploads of CPU-written guest memory, readbacks); the interrupted segment continues after
// them. Nothing is recorded without the profiler.
namespace {
class ProfileCopy {
public:
    ProfileCopy(Vulkan::Scheduler& scheduler, u64 key, const char* what) {
        profiler = Vulkan::GpuProfiler::Get();
        if (!profiler || !profiler->Records(&scheduler)) {
            profiler = nullptr;
            return;
        }
        resume = profiler->Current();
        profiler->Mark(key, [what] { return std::string{what}; });
    }
    ~ProfileCopy() {
        if (profiler && resume) {
            profiler->Resume(resume);
        }
    }
    ProfileCopy(const ProfileCopy&) = delete;
    ProfileCopy& operator=(const ProfileCopy&) = delete;

private:
    Vulkan::GpuProfiler* profiler = nullptr;
    u64 resume = 0;
};

// gow3 21 Oct 2026 (GPU worker): GOW3_READBACK_LATENCY=1 (diagnostics) - how long after the GPU
// finished a readback's submission the command thread landed it (the deferred ray-cast fences
// the guest waits for are written then). A waiter thread (the scheduler's priority operations)
// notes when each readback tick completes; landings compare against that.
bool ReadbackLatencyEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_READBACK_LATENCY");
        return env && env[0] == '1';
    }();
    return on;
}
std::array<std::atomic<u64>, 256> g_done_tick{};
std::array<std::atomic<s64>, 256> g_done_ns{};
s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
void NoteReadbackQueued(Vulkan::Scheduler& scheduler, u64 tick) {
    static u64 last = 0;
    if (!ReadbackLatencyEnabled() || tick == last) {
        return;
    }
    last = tick;
    scheduler.DeferPriorityOperation([tick] {
        g_done_ns[tick & 255].store(NowNs(), std::memory_order_relaxed);
        g_done_tick[tick & 255].store(tick, std::memory_order_release);
    });
}
void NoteReadbackLanded(u64 tick) {
    if (!ReadbackLatencyEnabled()) {
        return;
    }
    static u64 landed = 0, before_waiter = 0, sum_us = 0, max_us = 0;
    static std::array<u64, 5> buckets{}; // < 0.25, < 0.5, < 1, < 2, >= 2 ms
    static std::map<std::array<u64, 6>, std::pair<u64, u64>> late_sites; // stack -> count, us
    static u64 last_tick = 0;
    static s64 last_print = NowNs();
    if (tick == last_tick) {
        return; // one per submission
    }
    last_tick = tick;
    const s64 now = NowNs();
    ++landed;
    if (g_done_tick[tick & 255].load(std::memory_order_acquire) != tick) {
        ++before_waiter;
    } else {
        const u64 us = u64(std::max<s64>(0, now - g_done_ns[tick & 255].load(std::memory_order_relaxed))) / 1000;
        sum_us += us;
        max_us = std::max(max_us, us);
        ++buckets[us < 250 ? 0 : us < 500 ? 1 : us < 1000 ? 2 : us < 2000 ? 3 : 4];
#ifdef _WIN32
        // Late landings (2 ms or more): who landed them (image offsets of the callers of
        // CompleteReadbacks; llvm-symbolizer --obj=gow3-probe.exe 0x140000000+offset).
        if (us >= 2000) {
            void* frames[6]{};
            const unsigned n = RtlCaptureStackBackTrace(2, 6, frames, nullptr);
            std::array<u64, 6> key{};
            for (unsigned i = 0; i < n && i < 6; ++i) {
                key[i] = reinterpret_cast<u64>(frames[i]) - reinterpret_cast<u64>(&__ImageBase);
            }
            auto& entry = late_sites[key];
            ++entry.first;
            entry.second += us;
        }
#endif
    }
    if (now - last_print >= 5'000'000'000) {
        last_print = now;
        const u64 timed = landed - before_waiter;
        std::printf("Readback landing latency (5 s): %llu submissions landed, %llu before the waiter "
                    "woke; after the GPU finished: %.2f ms on average, %.2f ms at most; <0.25 ms %llu, "
                    "<0.5 %llu, <1 %llu, <2 %llu, >=2 %llu\n",
                    (unsigned long long)landed, (unsigned long long)before_waiter,
                    timed ? sum_us / 1e3 / timed : 0.0, max_us / 1e3, (unsigned long long)buckets[0],
                    (unsigned long long)buckets[1], (unsigned long long)buckets[2],
                    (unsigned long long)buckets[3], (unsigned long long)buckets[4]);
        std::vector<std::pair<u64, std::array<u64, 6>>> order;
        for (const auto& [key, entry] : late_sites) {
            order.emplace_back(entry.first, key);
        }
        std::ranges::sort(order, std::greater{});
        for (size_t i = 0; i < std::min<size_t>(order.size(), 5); ++i) {
            const auto& entry = late_sites[order[i].second];
            std::printf("  late landings: %llu x, %.1f ms on average, by",
                        (unsigned long long)entry.first, entry.second / 1e3 / entry.first);
            for (const u64 offset : order[i].second) {
                std::printf(" %#llx", (unsigned long long)offset);
            }
            std::printf("\n");
        }
        std::fflush(stdout);
        late_sites.clear();
        landed = before_waiter = sum_us = max_us = 0;
        buckets = {};
    }
}
} // namespace

// GOW3_READ_SOURCES=1 (with GOW3_FAULT_STATS=1): CPU reads of GPU-written bytes and the shaders
// that wrote their windows. Off with the default diagnostics: a lock and map update per GPU write.
bool ReadbackStatsEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_READBACK_STATS");
        return env && env[0] == '1';
    }();
    return on;
}

static bool FaultStatsEnabled() {
    static const bool on = [] {
        const char* stats = std::getenv("GOW3_FAULT_STATS");
        const char* sources = std::getenv("GOW3_READ_SOURCES");
        return (stats && stats[0] == '1' && sources && sources[0] == '1') || ReadbackStatsEnabled();
    }();
    return on;
}

static std::mutex g_rb_stats_mutex;
static std::unordered_map<u64, std::array<u64, 2>> g_rb_window_bytes; ///< window -> bytes, pieces
static std::unordered_map<u64, u32> g_rb_window_kinds;                ///< window -> ReadbackKind bits

void NoteReadbackKind(VAddr address, u64 size, u32 kind) {
    if (!ReadbackStatsEnabled() || size == 0) {
        return;
    }
    std::scoped_lock lock{g_rb_stats_mutex};
    for (u64 window = address >> 20; window <= (address + size - 1) >> 20; ++window) {
        g_rb_window_kinds[window] |= kind;
    }
}
static std::mutex g_writers_mutex;
static std::unordered_map<u64, std::map<u64, u32>> g_window_writers; ///< window -> shader -> writes
/// CPU reads that hit GPU-written bytes (not only their page): window -> (reads, addresses).
static std::unordered_map<u64, std::pair<u32, std::map<VAddr, u32>>> g_gpu_byte_reads;

void NoteGpuWriter(VAddr address, u64 size, u64 shader_hash) {
    if (!FaultStatsEnabled() || size == 0) {
        return;
    }
    std::scoped_lock lock{g_writers_mutex};
    const u64 first = address >> 20;
    const u64 last = std::min<u64>((address + size - 1) >> 20, first + 63);
    for (u64 window = first; window <= last; ++window) {
        ++g_window_writers[window][shader_hash];
    }
}

// gow3 24 Oct 2026 diagnostics (GOW3_FAULT_STATS=1): the most recent GPU writes (a shader's
// written binding, a DMA copy or fill), to name the writer of the GPU bytes a CPU write waits
// beside (GOW3_FETCH_NO_BLOCK deferrals).
struct RecentGpuWrite {
    VAddr address;
    u64 size;
    u64 writer;
};
static constexpr size_t RecentGpuWrites = 65536;
static std::mutex g_recent_writes_mutex;
static std::vector<RecentGpuWrite> g_recent_writes;
static u64 g_recent_writes_next = 0;
static bool RecentWritesEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_FAULT_STATS");
        return env && env[0] == '1';
    }();
    return on;
}

void NoteRecentGpuWrite(VAddr address, u64 size, u64 writer) {
    if (!RecentWritesEnabled() || size == 0) {
        return;
    }
    std::scoped_lock lock{g_recent_writes_mutex};
    if (g_recent_writes.empty()) {
        g_recent_writes.resize(RecentGpuWrites);
    }
    g_recent_writes[g_recent_writes_next++ % RecentGpuWrites] = {address, size, writer};
}

/// The newest recorded write into [lo, hi), with its size; writer ~0: none recorded.
static RecentGpuWrite NewestGpuWriteInto(VAddr lo, VAddr hi) {
    std::scoped_lock lock{g_recent_writes_mutex};
    const u64 count = std::min<u64>(g_recent_writes_next, g_recent_writes.size());
    for (u64 i = 1; i <= count; ++i) {
        const auto& write = g_recent_writes[(g_recent_writes_next - i) % RecentGpuWrites];
        if (write.address < hi && lo < write.address + write.size) {
            return write;
        }
    }
    return {0, 0, ~0ull};
}

static std::string GpuWriterName(u64 writer) {
    if (writer == ~0ull) {
        return "none recorded";
    }
    if (writer == GpuWriterDmaCopy) {
        return "DMA copy";
    }
    if (writer == GpuWriterFill) {
        return "DMA fill";
    }
    return fmt::format("{:08x}:{}", writer >> 8, writer & 0xff);
}

// gow3 26 Oct 2026 diagnostics (GOW3_FAULT_STATS=1): uploads of guest memory, for a binding the
// GPU only reads, over bytes the GPU wrote (gpu_modified_ranges): the GPU then reads the guest's
// copy instead of its own output. By the newest recorded writer of those bytes, per 5 s. (It
// relies on gpu_modified_ranges: the hair's state lost through RangeSet never showed here.)
static void NoteUploadOverGpu(VAddr address, u64 size, u64 gpu_bytes, u64 writer) {
    struct Entry {
        u32 pieces = 0;
        u64 bytes = 0, gpu_bytes = 0;
        VAddr example = 0;
        u64 example_size = 0;
    };
    static std::mutex mutex;
    static std::map<u64, Entry> by_writer;
    static auto last = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    auto& entry = by_writer[writer];
    ++entry.pieces;
    entry.bytes += size;
    entry.gpu_bytes += gpu_bytes;
    entry.example = address;
    entry.example_size = size;
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(5)) {
        return;
    }
    last = now;
    std::string text;
    for (const auto& [who, e] : by_writer) {
        text += fmt::format(" {} x{} ({} of {} bytes GPU-written, last {:#x}+{});",
                            GpuWriterName(who), e.pieces, e.gpu_bytes, e.bytes, e.example,
                            e.example_size);
    }
    std::printf("Uploads over GPU-written bytes for read bindings (5 s):%s\n", text.c_str());
    std::fflush(stdout);
    by_writer.clear();
}

struct WaitWriter {
    u32 waits = 0;
    VAddr example = 0;
    u64 example_size = 0;
    VAddr write_address = 0;
    u64 write_size = 0;
};
static std::mutex g_wait_writers_mutex;
static std::map<u64, WaitWriter> g_wait_writers; ///< writer -> waits beside its GPU bytes
static std::atomic<u32> g_wait_tick_window{0}, g_wait_tick_page{0};

/// "; written by <shader>x<writes> ..." for the statistics, empty unless GOW3_READ_SOURCES=1.
static std::string WindowWriters(u64 window) {
    if (!FaultStatsEnabled()) {
        return {};
    }
    std::scoped_lock lock{g_writers_mutex};
    const auto it = g_window_writers.find(window);
    if (it == g_window_writers.end()) {
        return "; written by no shader";
    }
    std::string out = "; written by";
    for (const auto& [hash, n] : it->second) {
        out += fmt::format(" {:08x}x{}", hash, n);
    }
    return out;
}

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
// gow3: 512 MiB (GOW3_STREAM_MB overrides; 64-2048): with two frames queued heavy
// scenes wrapped the 128 MiB ring in under two frames and waited for the GPU (1.3 s per 5 s).
static size_t StreamBufferSize() {
    const char* env = std::getenv("GOW3_STREAM_MB");
    const size_t mb = env ? std::clamp<size_t>(std::strtoull(env, nullptr, 10), 64, 2048) : 512;
    return mb * 1_MB;
}

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

// gow3 18 Oct 2026: AMD's Windows driver needs ~0.6 ms per MiB to allocate video memory (it
// clears it): one allocation per newly resident arena range stalled the GPU thread for 25-110 ms
// per frame in the first seconds of a level (Radeon 780M). Resident ranges now come from 64 MiB
// chunks, two of them allocated ahead on this thread; arena memory is never freed, so a chunk is
// used front to back. Larger ranges, or none left in reserve: one allocation, as before.
// GOW3_RESIDENCY_CHUNKS=0: always one allocation per range.
// gow3 14 Nov 2026: in the low video memory mode (vk_low_vram.h) chunks of 16 MiB and one ahead:
// 128 MiB of reserve was a sixteenth of a 2 GB card.
struct BufferCache::ResidencyPool {
    const u64 ChunkSize = u64{Vulkan::LowVram::Enabled() ? 16u : 64u} << 20;
    const size_t Spares = Vulkan::LowVram::Enabled() ? 1 : 2;

    ResidencyPool(vk::Device device_, u32 memory_type_, bool map_)
        : device{device_}, memory_type{memory_type_}, map{map_} {
        thread = std::thread{[this] { Run(); }};
    }
    ~ResidencyPool() {
        {
            std::scoped_lock lock{mutex};
            stop = true;
        }
        cv.notify_all();
        thread.join();
    }

    /// `bytes` (whole sparse blocks) from the current chunk; false when the caller allocates.
    bool Take(u64 bytes, vk::DeviceMemory& memory, u8*& mapped, u64& offset) {
        if (bytes > ChunkSize / 2) {
            return false;
        }
        std::unique_lock lock{mutex};
        if (current.used + bytes > current.size) {
            if (spares.empty()) {
                if (++misses <= 8) {
                    std::printf("GPU: arena memory reserve empty (a range is allocated now)\n");
                }
                return false;
            }
            current = spares.front();
            spares.pop_front();
            cv.notify_one();
        }
        memory = current.memory;
        mapped = current.mapped;
        offset = current.used;
        current.used += bytes;
        return true;
    }

private:
    struct Chunk {
        vk::DeviceMemory memory{};
        u8* mapped = nullptr;
        u64 size = 0;
        u64 used = 0;
    };

    void Run() {
        Common::SetCurrentThreadName("GPU:ArenaMemory");
        for (;;) {
            {
                std::unique_lock lock{mutex};
                cv.wait(lock, [this] { return stop || (spares.size() < Spares && !failed); });
                if (stop) {
                    return;
                }
            }
            const vk::MemoryAllocateInfo info = {
                .allocationSize = ChunkSize,
                .memoryTypeIndex = memory_type,
            };
            auto [result, memory] = device.allocateMemory(info);
            if (result != vk::Result::eSuccess) {
                std::printf("GPU: arena memory chunk not allocated (%s); ranges are allocated one "
                            "by one\n",
                            vk::to_string(result).c_str());
                std::scoped_lock lock{mutex};
                failed = true;
                continue;
            }
            Gow3Stats::arena_memory_bytes += ChunkSize;
            u8* mapped = nullptr;
            if (map) {
                auto [map_result, pointer] = device.mapMemory(memory, 0, VK_WHOLE_SIZE);
                if (map_result == vk::Result::eSuccess) {
                    mapped = static_cast<u8*>(pointer);
                }
            }
            std::scoped_lock lock{mutex};
            spares.push_back({memory, mapped, ChunkSize, 0});
        }
    }

    vk::Device device;
    u32 memory_type;
    bool map;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Chunk> spares;
    Chunk current{};
    u32 misses = 0;
    bool failed = false;
    bool stop = false;
    std::thread thread;
};

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, StreamBufferSize()},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    page_manager = &tracker;
    static BufferCache* wait_label_cache = nullptr;
    wait_label_cache = this;
    AmdGpu::g_read_wait_label = [](uintptr_t address) {
        return wait_label_cache->ReadWaitLabel(static_cast<VAddr>(address));
    };
    while (arena_page_bits > MIN_ARENA_PAGE_BITS &&
           (u64{2} << arena_page_bits) > instance.GetMaxBufferSize()) {
        --arena_page_bits;
    }
    arena_page_size = u64{1} << arena_page_bits;
    num_arena_pages = u64{1} << (ADDRESS_SPACE_BITS - arena_page_bits);
    if (arena_page_bits != MAX_ARENA_PAGE_BITS) {
        LOG_INFO(Render_Vulkan, "Buffer arenas of {} MiB (driver buffer limit {} MiB)",
                 arena_page_size >> 20, instance.GetMaxBufferSize() >> 20);
    }
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = arena_page_size,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = arena_page_size / block_size;
    blocks_per_arena_page_shift = arena_page_bits - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();
    // gow3 17 Oct 2026: a CPU-mappable type of the same video memory heap when the GPU has one
    // (Resizable BAR, integrated GPUs): GPU data whose writes have finished is then read straight
    // from the arena (ReadFinishedGpuPage) instead of a copy that waits for all queued GPU work.
    // GOW3_HOST_ARENAS=0: device-local only, as before.
    {
        static const bool wanted = [] {
            const char* env = std::getenv("GOW3_HOST_ARENAS");
            return !(env && env[0] == '0');
        }();
        const auto& props = instance.GetMemoryProperties();
        const u32 heap = props.memoryTypes[arena_memory_type_index].heapIndex;
        const auto need = vk::MemoryPropertyFlagBits::eDeviceLocal |
                          vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent;
        const auto avoid = vk::MemoryPropertyFlagBits::eDeviceCoherentAMD |
                           vk::MemoryPropertyFlagBits::eDeviceUncachedAMD;
        for (u32 i = 0; wanted && i < props.memoryTypeCount; ++i) {
            const auto flags = props.memoryTypes[i].propertyFlags;
            if (((reqs.memoryTypeBits >> i) & 1) && (flags & need) == need && !(flags & avoid) &&
                props.memoryTypes[i].heapIndex == heap) {
                arena_memory_type_index = i;
                arena_host_mapped = true;
                break;
            }
        }
        // gow3 20 Oct 2026 (Radeon 780M): an integrated GPU's "video memory" heap is a small
        // slice of system memory set aside for it (256 MiB on a Ryzen 7840HS laptop); the arenas
        // (1-2 GB) went there and the driver kept moving them. Its memory
        // is the system's anyway: the arenas go to the largest CPU-mappable system memory heap
        // (CPU-cached if offered), so they need no moving and finished GPU data is read in place.
        // Measured on the 780M (720p, twice each): 43.9/45.1 -> 47.2 FPS in the
        // same view, 46-47K -> 48.5-53.5K draws/s, the GPU command thread waiting for the GPU
        // 3-3.7% -> 0.1-0.5%, CPU writes into GPU data read in place instead of a full download.
        // Integrated GPUs only; GOW3_IGPU_ARENAS=local: the video memory heap, as before. (The
        // subtitle / pause menu flicker the user saw on the 780M is not this: it
        // shows with local arenas and with the build from before 20 Oct as well.)
        if (const char* env = std::getenv("GOW3_IGPU_ARENAS");
            instance.IsIntegrated() && !(env && std::string_view{env} == "local")) {
            const auto host = vk::MemoryPropertyFlagBits::eHostVisible |
                              vk::MemoryPropertyFlagBits::eHostCoherent;
            const auto avoid = vk::MemoryPropertyFlagBits::eDeviceCoherentAMD |
                               vk::MemoryPropertyFlagBits::eDeviceUncachedAMD;
            // gow3 21 Oct 2026 (GPU worker): GOW3_IGPU_ARENAS=uncached prefers the system memory
            // type without HostCached. GPU accesses to CPU-cached memory are snooped (kept
            // coherent with the CPU caches), which on AMD APUs costs GPU bandwidth; every buffer
            // the game's shaders read and write (vertices, indices, storage) lives in the arenas.
            // The CPU reads arena memory only for finished GPU data (ReadFinishedGpuPage).
            // =system / =uncached also apply when the video memory heap is the larger one (this
            // desktop's Radeon iGPU shows a 15.6 GiB one; the 780M laptops 256 MiB), for tests.
            const bool prefer_uncached = env && std::string_view{env} == "uncached";
            const bool forced = env && (prefer_uncached || std::string_view{env} == "system");
            s32 best = -1;
            u64 best_size = 0;
            bool best_cached = false;
            for (u32 i = 0; i < props.memoryTypeCount; ++i) {
                const auto flags = props.memoryTypes[i].propertyFlags;
                if (!((reqs.memoryTypeBits >> i) & 1) || (flags & host) != host || (flags & avoid)) {
                    continue;
                }
                if (forced && (props.memoryHeaps[props.memoryTypes[i].heapIndex].flags &
                               vk::MemoryHeapFlagBits::eDeviceLocal)) {
                    continue;
                }
                const u64 size = props.memoryHeaps[props.memoryTypes[i].heapIndex].size;
                const bool cached = bool(flags & vk::MemoryPropertyFlagBits::eHostCached);
                if (size > best_size ||
                    (size == best_size && (prefer_uncached ? !cached && best_cached
                                                           : cached && !best_cached))) {
                    best = static_cast<s32>(i);
                    best_size = size;
                    best_cached = cached;
                }
            }
            if (best >= 0 && (forced || best_size > props.memoryHeaps[heap].size)) {
                arena_memory_type_index = static_cast<u32>(best);
                arena_host_mapped = true;
                std::printf("GPU: integrated GPU: buffer arenas in system memory (type %d, %llu MiB "
                            "heap%s; GOW3_IGPU_ARENAS=local: the video memory heap)\n",
                            best, static_cast<unsigned long long>(best_size >> 20),
                            best_cached ? ", CPU-cached" : "");
            }
        }
        std::printf("GPU: buffer arenas in memory type %u (heap %u, %s)\n", arena_memory_type_index,
                    props.memoryTypes[arena_memory_type_index].heapIndex,
                    arena_host_mapped ? "CPU-mappable: finished GPU data is read directly"
                                      : "device-local only");
        // TEMP diagnostics: the memory types the driver allows for sparse arenas.
        for (u32 i = 0; i < props.memoryTypeCount; ++i) {
            if ((reqs.memoryTypeBits >> i) & 1) {
                const auto t = props.memoryTypes[i];
                std::printf("GPU: arena memory type %u allowed: heap %u (%llu MiB, %s), flags %s\n", i,
                            t.heapIndex,
                            (unsigned long long)(props.memoryHeaps[t.heapIndex].size >> 20),
                            (props.memoryHeaps[t.heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal)
                                ? "video memory" : "system memory",
                            vk::to_string(t.propertyFlags).c_str());
            }
        }
        std::fflush(stdout);
        if (arena_host_mapped) {
            scheduler.EnableHostReadBarrier();
        }
    }
    if (const char* env = std::getenv("GOW3_RESIDENCY_CHUNKS"); !(env && env[0] == '0')) {
        residency_pool = std::make_unique<ResidencyPool>(instance.GetDevice(),
                                                         arena_memory_type_index, arena_host_mapped);
    }

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * num_arena_pages) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * num_arena_pages);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
    // Diagnostics: name the arena and guest address of a GPU fault address.
    Vulkan::g_describe_gpu_address = [this](u64 gpu_address) {
        for (const Buffer& arena : arenas) {
            const u64 base = arena.BufferDeviceAddress();
            if (gpu_address < base || gpu_address >= base + arena.SizeBytes()) {
                continue;
            }
            const VAddr guest = arena.CpuAddr() + (gpu_address - base);
            const u64 block = guest >> block_shift;
            bool resident = true;
            resident_ranges.ForEachGap(block, block + 1, [&](u64, u64) { resident = false; });
            std::fprintf(stderr, "GPU fault address is arena %#llx+%#llx: guest %#llx, block %llu %s"
                         " (block size %#llx)\n",
                         (unsigned long long)arena.CpuAddr(), (unsigned long long)arena.SizeBytes(),
                         (unsigned long long)guest, (unsigned long long)block,
                         resident ? "resident" : "NOT resident", (unsigned long long)block_size);
        }
        const auto describe = [&](const char* name, const Buffer& buffer) {
            const u64 base = buffer.BufferDeviceAddress();
            if (gpu_address >= base && gpu_address < base + buffer.SizeBytes()) {
                std::fprintf(stderr, "GPU fault address is in %s at %#llx\n", name,
                             (unsigned long long)(gpu_address - base));
            }
        };
        describe("stream buffer", stream_buffer);
        describe("GDS buffer", gds_buffer);
        describe("BDA page table", *bda_pagetable_buffer);
    };
}

BufferCache::~BufferCache() = default;

// gow3 19 Oct 2026: downloads made before a range is invalidated (count, time), read by
// Rasterizer::UnmapMemory's timing line (a cutscene hitch).
std::atomic<u64> g_invalidate_downloads{0}, g_invalidate_download_ns{0};

static bool FaultTimesEnabled(); // GOW3_FAULT_TIMES, defined with the fault statistics
// gow3 26 Oct 2026 (GOW3_FAULT_TIMES): when this thread's InvalidateMemory started (the
// tracker's region lock comes before ReadMemory).
static thread_local std::chrono::steady_clock::time_point t_invalidate_start;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    if (FaultTimesEnabled()) {
        t_invalidate_start = std::chrono::steady_clock::now();
    }
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        const auto start = std::chrono::steady_clock::now();
        ReadMemory(device_addr, size, true, assume_locks);
        g_invalidate_downloads.fetch_add(1, std::memory_order_relaxed);
        g_invalidate_download_ns.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
                .count(),
            std::memory_order_relaxed);
    });
}

// gow3: the game fills its per-frame buffers (constants, skinning output) sequentially and
// every 4 KiB page cost a protection fault (~110k/s in the test scene, a fifth of each
// render worker's time in the kernel). A fault unprotects the aligned window around it instead;
// pages marked CPU-modified without being written only cost an upload when bound.
// Item: context = BufferCache, source = guest address, destination = host pointer, size,
// extra = the Buffer whose mapping holds the destination (flushed after the copy).
void BufferCache::RunGuestCopy(const Gow3Copy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    auto* dst = reinterpret_cast<u8*>(item.destination);
    cache->memory->CopySparseMemory(item.source, dst, item.size);
    auto* buffer = reinterpret_cast<Buffer*>(item.extra);
    buffer->Flush(dst - buffer->mapped_data.data(), item.size);
}

// Small guest copies run on the recording thread (it spins for work: no wakeup, and it is
// idle most of the time); PoolSmallCopies (toggle 524288) batches them for the copy threads.
void BufferCache::SmallGuestCopy(const Gow3Copy::Item& item) {
    if (scheduler.IsRecordingDeferred() && !Gow3Toggle::Disabled(Gow3Toggle::PoolSmallCopies)) {
        scheduler.RecordHostCopy([item] { item.run(item); });
        return;
    }
    Gow3Copy::QueueCopy(item);
}

void BufferCache::ReleaseStuckPage(VAddr page, bool is_write, bool assume_locks) {
    // gow3 18 Oct 2026: a game froze on a rooftop level. It cleared
    // a field of its next ray-cast batch (a read-modify-write) in a page a pending readback still
    // read-protected. The write fault found no GPU bytes under the written ones and returned; the
    // page stayed protected and the instruction faulted again and again, while the readback, long
    // done on the GPU, was never landed: only the GPU thread polls readbacks, and it was idle,
    // waiting for the game (one thread in the kernel, the game waiting on its ray-cast lock).
    // Shorter versions of the same loop, until the next frame's poll landed the readback, were
    // hitches. Now the readbacks holding the page land at once (polled, or waited for: they are
    // submitted, the GPU finishes them); nothing is dropped.
    const auto release = [this, page, is_write] {
        const auto holders = [this, page](u64& newest) {
            u32 count = 0;
            newest = 0;
            for (const auto& readback : pending_readbacks) {
                for (const auto& [address, size] : readback.watched) {
                    if (page >= address && page < address + size) {
                        ++count;
                        newest = std::max(newest, readback.tick);
                    }
                }
            }
            return count;
        };
        static std::atomic<u32> events{0}, landed{0}, waited{0}, unwatched{0}, other{0};
        const u32 event = events.fetch_add(1, std::memory_order_relaxed) + 1;
        u64 newest = 0;
        const u32 held = holders(newest);
        const char* outcome = "not a buffer readback";
        if (held) {
            PollReadbacksNow();
            if (!holders(newest)) {
                outcome = "landed (polled)";
                landed.fetch_add(1, std::memory_order_relaxed);
            } else if (newest < scheduler.CurrentTick()) {
                CompleteReadbacks(true, newest);
                outcome = "landed (waited for the GPU)";
                waited.fetch_add(1, std::memory_order_relaxed);
            } else {
                // Not submitted yet (not read-protected normally): stop protecting the page.
                UnwatchPage(page);
                outcome = "unprotected (not submitted)";
                unwatched.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            ReleaseOrphanReadWatch(page);
            other.fetch_add(1, std::memory_order_relaxed);
        }
        if (event <= 12 || event % 500 == 0) {
            std::printf("Stuck fault %u: %s of page %#llx, %u readbacks held it: %s (so far: %u "
                        "polled, %u waited, %u unprotected, %u other)\n",
                        event, is_write ? "write" : "read", static_cast<unsigned long long>(page),
                        held, outcome, landed.load(), waited.load(), unwatched.load(), other.load());
            std::fflush(stdout);
        }
    };
    if (assume_locks) {
        release();
    } else {
        liverpool->SendCommand<true>(release);
    }
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    static const u64 window = [] {
        const char* env = std::getenv("GOW3_FAULT_WINDOW");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        return std::bit_ceil(std::clamp<u64>(kib, 4, 1024)) * 1024;
    }();
    if (window <= TRACKER_BYTES_PER_PAGE || Gow3Toggle::Disabled(Gow3Toggle::FaultWindow)) {
        return;
    }
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
}

static std::atomic<u32> g_write_fault_overlap, g_write_fault_same_page, g_write_fault_window;
static std::atomic<u32> g_shared_finished_writes; // GOW3_SHARED_FINISHED_WRITES
bool BufferCache::SharedFinishedWrites() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_SHARED_FINISHED_WRITES");
        return env && env[0] == '1';
    }();
    return on;
}
static std::atomic<u64> g_write_fault_gpu_bytes;
static std::atomic<u32> g_lazy_writes, g_lazy_cpu_wins, g_readback_waits, g_async_readbacks;
static std::atomic<u32> g_readback_read_waits;
// gow3 statistics (GOW3_FAULT_STATS=1): read faults on pages with GPU data on its way back that
// hit those GPU bytes, or only other data on their page; time the faulting thread waited for the
// GPU thread to take the request, and for the readback itself (microseconds).
static std::atomic<u32> g_read_on_gpu_bytes, g_read_same_page;
static std::atomic<u32> g_read_no_gpu_bytes; // gow3 25 Oct 2026: read faults beside no GPU bytes
static std::atomic<u32> g_poisoned_page_writes; // gow3 25 Oct 2026: writes a poisoned page took
static bool FaultTimesEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_FAULT_TIMES");
        return env && env[0] == '1';
    }();
    return on;
}
static thread_local std::string t_write_fault_note;
std::string BufferCache::TakeWriteFaultNote() {
    return std::exchange(t_write_fault_note, {});
}
// gow3 26 Oct 2026 diagnostics (GOW3_FAULT_STATS=1): second-queue copies (SideQueueFetch) and
// the time their submitters waited for them, in microseconds.
static std::atomic<u32> g_side_fetches, g_side_fetch_max_us;
static std::atomic<u64> g_side_fetch_us;
static std::atomic<u32> g_label_backing_reads;  // gow3 25 Oct 2026: BufferCache::ReadWaitLabel
static std::atomic<u32> g_pieces_recurring, g_pieces_protected;
static std::atomic<u64> g_readback_bytes;
// gow3 statistics: fences deferred to ray-cast readbacks (label, delay until written).
static std::atomic<u32> g_deferred_fences;
static std::atomic<u64> g_deferred_us, g_deferred_max_us;
static std::atomic<u32> g_swept_pages, g_swept_windows;
static constexpr auto RecurringWindow = std::chrono::milliseconds{500};
static std::atomic<u64> g_read_queue_us, g_read_wait_us;
static std::mutex g_read_pieces_mutex;
static std::unordered_map<u64, u32> g_read_pieces; ///< (piece address << 16 | size) -> reads
static std::atomic<u32> g_readback_yields;

// gow3 18 Oct 2026, readback guard: GPU data comes back to guest memory late, up to a frame or
// two after the guest saw the fence of the work that wrote it (readbacks land when the GPU is
// done; most fences do not wait for them). Where the CPU can write without a fault (pages shared
// with GPU data in flight, lazy writes), the guest may reuse those bytes in between. A game
// freed the cloth buffers of an area it unloaded, its allocator wrote free-list links into them,
// and the late cloth readback put float data over the links: a crash in the allocator after
// ~40 minutes of play (18 Oct). Late GPU data now never overwrites a 64-byte line the guest
// changed since: async readbacks compare with the guest bytes when they were queued
// (PendingReadback::before), lazy and shared pages with their record (LazyPage::bytes).
// GOW3_READBACK_GUARD=0 turns it off (the GPU's bytes win, as before).
static constexpr u64 GuardLine = 64;
static bool ReadbackGuardEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_READBACK_GUARD");
        const bool on = !(env && env[0] == '0');
        if (!on) {
            std::printf("Readback guard: off (GOW3_READBACK_GUARD=0)\n");
            std::fflush(stdout);
        }
        return on;
    }();
    return enabled;
}
// gow3 20 Oct 2026 (laptop CPUs): every submission queues a readback of the small GPU writes
// (SyncSmallGpuWrites; ~135 submissions a frame) with a snapshot of the guest bytes,
// and its landing copies the guest bytes once more (LandGuarded). The snapshot vector was zeroed,
// then copied over, then freed at landing; on a Ryzen 7840HS (laptop, 47 FPS with the
// GPU command thread CPU-bound) the zeroing and fresh pages were 1.5% of that thread (1.8% with
// the Radeon 780M), freeing them 0.4-0.6% and the landing buffers 0.2-0.3%. Snapshots of landed
// readbacks are now reused and never zeroed (every byte is copied in: TryReadBacking, or
// CopySparseMemory, which zero-fills unmapped parts), as are the landing buffers.
// GOW3_READBACK_POOL=0: zeroed, allocated and freed each time, as before.
static bool ReadbackPoolEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_READBACK_POOL");
        return !(env && env[0] == '0');
    }();
    return enabled;
}
/// Sizes the pool keeps: at most 8 buffers, 4 MiB each, 16 MiB in all.
static constexpr size_t SnapshotPoolCount = 8;
static constexpr u64 SnapshotPoolMaxBytes = 4_MB;
static constexpr u64 SnapshotPoolTotalBytes = 16_MB;

/// gow3 (GOW3_READBACK_POOL): a buffer of `n` bytes, uninitialized when the pool is on (the caller
/// fills all of it) and zeroed otherwise.
static void SizeScratch(SnapshotBytes& bytes, u64 n) {
    if (ReadbackPoolEnabled()) {
        bytes.resize(n);
    } else {
        bytes.resize(n, 0);
    }
}

static std::atomic<u32> g_guard_lines, g_guard_lazy_lines, g_guard_logged;
static std::atomic<u64> g_guard_bytes;
// The first 16 kept lines, then the 32nd, 64th, ...: where and whether the page was CPU-writable.
static bool GuardLogNow() {
    const u32 seen = g_guard_logged.fetch_add(1, std::memory_order_relaxed);
    return seen < 16 || (seen & (seen + 1)) == 0;
}

// GOW3_ASYNC_READBACK=1: pages with a readback in flight are read-protected, so the CPU (the game,
// or the SRT walker) never reads their stale data: a read waits for that readback only. =2 is
// the earlier unprotected mode (stale reads: "dcb full", garbage sharps).
static bool SafeReadbacks() {
    // GOW3_ASYNC_READBACK=1: pages of pending readbacks are read-protected (a CPU read waits for
    // its readback). Default (unset, 2): not protected, the CPU reads the previous value.
    static const bool safe = [] {
        const char* env = std::getenv("GOW3_ASYNC_READBACK");
        return env && env[0] == '1';
    }();
    return safe;
}

// GOW3_LAZY_WRITES=1: a CPU write into a GPU-written page goes ahead without a download; the
// GPU bytes are merged in later where guest memory still matches a snapshot. Off by default
// since 15 Oct: a byte the CPU rewrites with its previous value is indistinguishable from an
// untouched one, so the merge put stale GPU bytes back into rewritten data (the
// skinning tables in the per-frame ring: half-float V#s, a GPU hang and game asserts at the
// treasure screen).
static std::atomic<u32> g_direct_page_reads{0};
static std::atomic<u32> g_waited_page_reads{0}; // gow3: integrated GPUs (GOW3_WAIT_PAGE_READ)
static std::array<std::atomic<u32>, 64> g_download_reasons{};
static std::atomic<u32> g_old_byte_fetches{0};
static std::array<std::atomic<u32>, 4> g_fetch_refused{}; // lazy, readback, no tick, open buffer
// gow3 21 Oct 2026 (GOW3_FETCH_NO_BLOCK=1): fetches a guest thread waited for itself.
static std::atomic<u32> g_fetch_deferrals{0};
// gow3 24 Oct 2026 (GOW3_SHARE_UNFINISHED_WRITES): writes beside running GPU writes, shared;
// such writes on physics result pages (waited for); poisoned pages a GPU write took back (or
// could not: the page was CPU-writable).
static std::atomic<u32> g_shared_unfinished_writes{0};
bool BufferCache::DmaEopReadbacks() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_DMA_EOP_READBACK");
        const bool enabled = env && env[0] == '1' && ShareUnfinishedWrites();
        if (enabled) {
            std::printf("GPU: small DMA copies into guest memory come back with the next "
                        "end-of-pipe fence (GOW3_DMA_EOP_READBACK=1)\n");
            std::fflush(stdout);
        }
        return enabled;
    }();
    return on;
}
static std::atomic<u32> g_result_page_waits{0}, g_poison_resolved{0}, g_poison_writable{0};
static std::atomic<u32> g_poison_restored{0}; // RestorePoison: pages given their stale bytes back
// gow3 24 Oct 2026 diagnostics (GOW3_SHARE_CHECK=1): the last operations on each poisoned page,
// printed with the share check's poison examples.
static bool ShareCheckEnabled();
static std::mutex g_poison_history_mutex;
static std::unordered_map<u64, std::deque<std::string>> g_poison_history;
static void NotePoisonEvent(VAddr page, std::string event) {
    if (!ShareCheckEnabled()) {
        return;
    }
    std::scoped_lock lock{g_poison_history_mutex};
    if (g_poison_history.size() > 8192) {
        g_poison_history.clear();
    }
    auto& history = g_poison_history[page >> 12];
    if (history.size() >= 12) {
        history.pop_front();
    }
    history.push_back(std::move(event));
}
static std::string PoisonHistory(VAddr page);
/// A poisoned page's bytes went back to their pre-poison values (GPU ownership lost without
/// data): where, with the page's history, 16 lines at most (GOW3_SHARE_CHECK=1).
static void ReportPoisonRestore(const char* where, VAddr page) {
    if (!ShareCheckEnabled()) {
        return;
    }
    static std::atomic<int> printed{0};
    if (printed.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::printf("Poison restore (%s): page %#llx; history:%s\n", where,
                    static_cast<unsigned long long>(page), PoisonHistory(page).c_str());
        std::fflush(stdout);
    }
}
static std::string PoisonHistory(VAddr page) {
    std::scoped_lock lock{g_poison_history_mutex};
    std::string out;
    if (const auto it = g_poison_history.find(page >> 12); it != g_poison_history.end()) {
        for (const auto& event : it->second) {
            out += ' ';
            out += event;
        }
    }
    return out;
}
bool BufferCache::ShareUnfinishedWrites() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_SHARE_UNFINISHED_WRITES");
        const bool enabled = env && env[0] == '1';
        if (enabled) {
            std::printf("GPU: CPU writes beside GPU writes still running go ahead, the page's GPU "
                        "bytes read back after the submission (GOW3_SHARE_UNFINISHED_WRITES=1)\n");
            std::fflush(stdout);
        }
        return enabled;
    }();
    return on;
}
static bool UrgentCopiesEnabled(); // GOW3_URGENT_COPIES, defined with SideQueueFetch

static bool LazyWritesEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_LAZY_WRITES");
        return env && env[0] == '1';
    }();
    // Precise readbacks read-protect GPU-written pages: the records would be read through a
    // protected page.
    return enabled && EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Relaxed;
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    const auto entered = FaultTimesEnabled() ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
    const auto invalidate_start = std::exchange(t_invalidate_start, {}); // taken: not from an older fault
    // gow3: GOW3_FAULT_STATS=1 — every GPU-to-guest download request (CPU read of GPU data,
    // or CPU write into GPU-written memory), by 1 MiB window, every 5 s.
    static const bool stats = [] {
        const char* env = std::getenv("GOW3_FAULT_STATS");
        return env && env[0] == '1';
    }();
    {
        if (stats) {
            static std::mutex mutex;
            static std::unordered_map<u64, u32> counted_windows;
            static u32 counted_reads, counted_writes;
            static auto last = std::chrono::steady_clock::now();
            std::unique_lock lock{mutex};
            ++counted_windows[(device_addr >> 20) | (is_write ? 1ULL << 40 : 0)];
            ++(is_write ? counted_writes : counted_reads);
            if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
                last = now;
                // gow3 26 Oct 2026: printed without the lock. Every fault request takes it, and
                // the summary held it ~25 ms: faults of other threads arriving meanwhile (the
                // hair's joint slots of The Last of Us among them) waited for it.
                const auto windows = std::exchange(counted_windows, {});
                const u32 reads = std::exchange(counted_reads, 0);
                const u32 writes = std::exchange(counted_writes, 0);
                lock.unlock();
                std::vector<std::pair<u32, u64>> sorted;
                for (const auto& [key, n] : windows) {
                    sorted.emplace_back(n, key);
                }
                std::ranges::sort(sorted, std::greater{});
                std::printf("Download stats (5 s): %u for CPU reads, %u for CPU writes into GPU data\n", reads,
                            writes);
                const u32 overlap = g_write_fault_overlap.exchange(0);
                const u32 same_page = g_write_fault_same_page.exchange(0);
                const u32 window_only = g_write_fault_window.exchange(0);
                const u64 gpu_bytes = g_write_fault_gpu_bytes.exchange(0);
                std::printf("  CPU write faults: %u on GPU-written bytes, %u elsewhere in a GPU-written page, "
                            "%u elsewhere in the window; %.1f GPU-written bytes per faulting page\n",
                            overlap, same_page, window_only,
                            double(gpu_bytes) / std::max(1u, overlap + same_page + window_only));
                std::printf("  lazy writes (no download): %u; GPU-written bytes the CPU overwrote later: %u; "
                            "writes into pages shared with GPU data: %llu (%u beside final GPU bytes); "
                            "pages fetched without waiting for queued GPU work: %u (%u after waiting "
                            "for their own write only)\n",
                            g_lazy_writes.exchange(0), g_lazy_cpu_wins.exchange(0),
                            static_cast<unsigned long long>(std::exchange(g_shared_writes_count, 0)),
                            g_shared_finished_writes.exchange(0),
                            g_direct_page_reads.exchange(0), g_waited_page_reads.exchange(0));
                {
                    const u32 lazy_lines = g_guard_lazy_lines.exchange(0);
                    const u32 lines = g_guard_lines.exchange(0) + lazy_lines;
                    std::printf("  readback guard: %u lines the game wrote after the GPU kept from late GPU "
                                "data (%llu bytes), %u of them on shared/lazy pages\n",
                                lines, static_cast<unsigned long long>(g_guard_bytes.exchange(0)),
                                lazy_lines);
                }
                {
                    std::string reasons;
                    for (u32 bits = 0; bits < 64; ++bits) {
                        if (const u32 n = g_download_reasons[bits].exchange(0)) {
                            reasons += fmt::format(" [{}{}{}{}{}{}]={}", bits & 1 ? "write" : "read",
                                                   bits & 2 ? ",onGPU" : "", bits & 4 ? ",lazy" : "",
                                                   bits & 8 ? ",pending" : "", bits & 16 ? ",inflight" : "",
                                                   bits & 32 ? ",open" : "", n);
                        }
                    }
                    std::printf("  full downloads by case:%s; writes into poisoned shared pages that "
                                "went ahead: %u\n",
                                reasons.c_str(), g_poisoned_page_writes.exchange(0));
                }
                std::printf("  pages with old GPU bytes fetched on the second queue: %u\n",
                            g_old_byte_fetches.exchange(0));
                {
                    const u32 fetches = g_side_fetches.exchange(0);
                    const double waited_ms = g_side_fetch_us.exchange(0) / 1e3;
                    std::printf("  second-queue copies: %u, waited for %.1f ms (%.2f ms each, %.2f "
                                "at most)\n",
                                fetches, waited_ms, fetches ? waited_ms / fetches : 0.0,
                                g_side_fetch_max_us.exchange(0) / 1e3);
                }
                std::printf("  page fetches refused: lazy page %u, pending readback %u, no GPU write "
                            "tick %u, written in the open command buffer %u\n",
                            g_fetch_refused[0].exchange(0), g_fetch_refused[1].exchange(0),
                            g_fetch_refused[2].exchange(0), g_fetch_refused[3].exchange(0));
                const u32 deferrals = g_fetch_deferrals.exchange(0);
                const u32 shared_unfinished = g_shared_unfinished_writes.exchange(0);
                if (deferrals || shared_unfinished) {
                    std::printf("  write fetches waited for on the faulting thread: %u "
                                "(GOW3_FETCH_NO_BLOCK); writes beside running GPU writes that "
                                "went ahead: %u (GOW3_SHARE_UNFINISHED_WRITES; physics and DMA "
                                "copy pages waited: %u; shared pages a GPU write took back: %u, %u of them "
                                "CPU-writable; given their stale bytes back: %u)\n",
                                deferrals, shared_unfinished, g_result_page_waits.exchange(0),
                                g_poison_resolved.exchange(0), g_poison_writable.exchange(0),
                                g_poison_restored.exchange(0));
                }
                {
                    std::map<u64, WaitWriter> writers;
                    {
                        std::scoped_lock lock{g_wait_writers_mutex};
                        writers.swap(g_wait_writers);
                    }
                    const u32 page_ticks = g_wait_tick_page.exchange(0);
                    const u32 window_ticks = g_wait_tick_window.exchange(0);
                    if (page_ticks || window_ticks) {
                        std::vector<std::pair<u32, u64>> sorted_writers;
                        for (const auto& [writer, entry] : writers) {
                            sorted_writers.emplace_back(entry.waits, writer);
                        }
                        std::ranges::sort(sorted_writers, std::greater{});
                        std::string list;
                        for (size_t i = 0; i < std::min<size_t>(sorted_writers.size(), 8); ++i) {
                            const auto& entry = writers[sorted_writers[i].second];
                            list += fmt::format(" {} x{} ({:#x}+{}, written as {:#x}+{});",
                                                GpuWriterName(sorted_writers[i].second),
                                                entry.waits, entry.example, entry.example_size,
                                                entry.write_address, entry.write_size);
                        }
                        std::printf("  those waits: %u for a GPU write into their page, %u for a "
                                    "large one elsewhere in its 1 MiB window; GPU-written ranges "
                                    "beside them by newest writer:%s\n",
                                    page_ticks, window_ticks, list.c_str());
                    }
                }
                std::printf("  tracking: %zu lazy/shared pages, %zu pending readbacks, %zu pages with "
                            "GPU write records, %zu GPU-written ranges\n",
                            lazy_pages.size(), pending_readbacks.size(), page_writes.size(),
                            gpu_modified_ranges.Count());
                {
                    const u32 on_bytes = g_read_on_gpu_bytes.exchange(0);
                    const u32 same_page = g_read_same_page.exchange(0);
                    const u64 total_us = g_read_queue_us.exchange(0);
                    const u64 wait_us = g_read_wait_us.exchange(0);
                    std::printf("  CPU read faults on readback pages: %u on the GPU bytes, %u elsewhere on "
                                "their page; %.1f ms in all, %.1f ms of it waiting for the GPU; readback "
                                "pieces: %u recurring (readable), %u new (protected); reads of no GPU "
                                "bytes unprotected without a download: %u; wait labels read through "
                                "the backing view: %u\n",
                                on_bytes, same_page, total_us / 1e3, wait_us / 1e3,
                                g_pieces_recurring.exchange(0), g_pieces_protected.exchange(0),
                                g_read_no_gpu_bytes.exchange(0), g_label_backing_reads.exchange(0));
                    std::scoped_lock lock{g_read_pieces_mutex};
                    std::vector<std::pair<u32, u64>> pieces;
                    for (const auto& [key, n] : g_read_pieces) {
                        pieces.emplace_back(n, key);
                    }
                    std::ranges::sort(pieces, std::greater{});
                    for (size_t i = 0; i < std::min<size_t>(pieces.size(), 6); ++i) {
                        std::printf("    read piece %#llx+%llu: %u\n",
                                    static_cast<unsigned long long>(pieces[i].second >> 16),
                                    static_cast<unsigned long long>(pieces[i].second & 0xffff), pieces[i].first);
                    }
                    g_read_pieces.clear();
                }
                {
                    const u32 n = g_deferred_fences.exchange(0);
                    std::printf("  ray-cast fences written after their readback: %u, %.2f ms on average, "
                                "%.2f ms at most\n",
                                n, n ? g_deferred_us.exchange(0) / 1e3 / n : 0.0,
                                g_deferred_max_us.exchange(0) / 1e3);
                }
                std::printf("  readback bytes: %.1f MB; swept: %u pages with finished GPU writes, %u cooled "
                            "hot windows\n",
                            g_readback_bytes.exchange(0) / 1e6, g_swept_pages.exchange(0),
                            g_swept_windows.exchange(0));
                std::printf("  async readbacks: %u; CPU writes that waited for one: %u, CPU reads: %u; "
                            "CPU writes into pending ray-cast results: %u\n",
                            g_async_readbacks.exchange(0), g_readback_waits.exchange(0),
                            g_readback_read_waits.exchange(0), g_readback_yields.exchange(0));
                for (size_t i = 0; i < std::min<size_t>(sorted.size(), 10); ++i) {
                    const auto [n, key] = sorted[i];
                    const u64 window = key & ((1ULL << 40) - 1);
                    std::printf("  %s %#llx (1 MiB): %u%s\n", key >> 40 ? "write" : "read",
                                (unsigned long long)(window << 20), n, WindowWriters(window).c_str());
                }
                {
                    std::unordered_map<u64, std::pair<u32, std::map<VAddr, u32>>> byte_reads;
                    {
                        std::scoped_lock lock{g_writers_mutex};
                        byte_reads.swap(g_gpu_byte_reads);
                    }
                    for (const auto& [window, entry] : byte_reads) {
                        std::string top;
                        std::vector<std::pair<u32, VAddr>> sorted_addresses;
                        for (const auto& [address, n] : entry.second) {
                            sorted_addresses.emplace_back(n, address);
                        }
                        std::ranges::sort(sorted_addresses, std::greater{});
                        for (size_t i = 0; i < std::min<size_t>(sorted_addresses.size(), 8); ++i) {
                            top += fmt::format(" {:#x}x{}", sorted_addresses[i].second,
                                               sorted_addresses[i].first);
                        }
                        std::printf("  reads of GPU bytes in %#llx (1 MiB): %u at%s%s\n",
                                    static_cast<unsigned long long>(window << 20), entry.first,
                                    top.c_str(), WindowWriters(window).c_str());
                    }
                }
                std::fflush(stdout);
            }
        }
    }
    // gow3 21 Oct 2026 (GOW3_FETCH_NO_BLOCK=1, multithreaded rendering): a guest thread's write
    // into GPU data whose newest GPU write is submitted but not finished made the GPU command
    // thread wait for the GPU in the second-queue copy (SideQueueFetch -> waitSemaphores): 24-38
    // ms per such fault with multithreaded rendering (the GPU runs further behind), with no draw
    // processed meanwhile (44-77 ms frames). The request now returns that tick
    // instead (deferred_tick); the guest thread waits for it on the work semaphore and sends the
    // request again, which finds the page's GPU writes finished (re-checked from the start). The
    // command thread never waits for a guest thread, and a tick FetchableWindowTick returns is
    // submitted, so the wait ends. GPU-side faults and the fourth attempt keep the old path.
    static const bool no_block = [] {
        const char* env = std::getenv("GOW3_FETCH_NO_BLOCK");
        const bool on = env && env[0] == '1';
        if (on) {
            std::printf("GPU: write faults wait for unfinished GPU writes on the faulting thread, "
                        "not the GPU command thread (GOW3_FETCH_NO_BLOCK=1)\n");
            std::fflush(stdout);
        }
        return on;
    }();
    u64 deferred_tick = 0;
    bool allow_defer = no_block && is_write && !assume_locks;
    // gow3 26 Oct 2026 diagnostics (GOW3_FAULT_TIMES=1): how the request ended, and when the GPU
    // command thread took it (TakeWriteFaultNote).
    const char* path = "";
    const char* wait_note = "";
    std::chrono::steady_clock::time_point started{};
    const auto flush_request = [this, device_addr, size, is_write, &deferred_tick, &allow_defer,
                                &path, &wait_note, &started] {
        started = std::chrono::steady_clock::now();
        if (!is_write && FaultStatsEnabled() && gpu_modified_ranges.Intersects(device_addr, size)) {
            std::scoped_lock lock{g_writers_mutex};
            auto& [reads, addresses] = g_gpu_byte_reads[device_addr >> 20];
            ++reads;
            if (addresses.size() < 64 || addresses.contains(device_addr)) {
                ++addresses[device_addr];
            }
        }
        if (!is_write && !pending_readbacks.empty()) {
            // A read of a page whose GPU data is still on its way back (read-protected by
            // DownloadRangesAsync): that readback completes and unprotects it. Other GPU data
            // in the window stays as relaxed readbacks leave it.
            const VAddr page = Common::AlignDown(device_addr, 4096);
            const VAddr end = Common::AlignUp(device_addr + size, 4096);
            if (const u64 tick = PendingReadbackTick(page, end - page, false);
                tick && tick < scheduler.CurrentTick()) { // submitted (only those are read-protected)
                ++g_readback_read_waits;
                VAddr hit = 0;
                u64 hit_size = 0;
                for (const auto& readback : pending_readbacks) {
                    for (const auto& [piece, piece_size] : readback.pieces) {
                        if (piece < device_addr + size && device_addr < piece + piece_size) {
                            hit = piece;
                            hit_size = piece_size;
                        }
                    }
                }
                if (hit) {
                    ++g_read_on_gpu_bytes;
                    std::scoped_lock lock{g_read_pieces_mutex};
                    ++g_read_pieces[(hit << 16) | std::min<u64>(hit_size, 0xffff)];
                } else {
                    // False sharing: the CPU reads its own data beside GPU bytes still on their
                    // way back. Nothing to wait for: the page stops being read-protected, now
                    // and (noisy_pages) for the next readbacks of the coming 10 seconds.
                    ++g_read_same_page;
                    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds{10};
                    for (VAddr p = page; p < end; p += 4096) {
                        noisy_pages[p >> 12] = until;
                        UnwatchPage(p);
                    }
                    return;
                }
                const auto wait_start = std::chrono::steady_clock::now();
                CompleteReadbacks(true, tick);
                g_read_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - wait_start)
                                      .count();
                if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Relaxed) {
                    return;
                }
            }
        }
        // gow3 25 Oct 2026: a read of no GPU-written bytes needs no download. The GPU command
        // thread's memory waits read their labels (which the runtime writes into guest memory) on
        // pages read-protected for no submitted readback of theirs (merged watch ranges, or an
        // unsubmitted one): each read downloaded a 512 KiB window and waited for all GPU work
        // (~90 per 5 s in the Quarantine Zone, 7.5% of that thread). The page is unprotected.
        if (!is_write && EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Relaxed &&
            !gpu_modified_ranges.Intersects(device_addr, size)) {
            const VAddr page = Common::AlignDown(device_addr, 4096);
            const VAddr end = Common::AlignUp(device_addr + size, 4096);
            for (VAddr p = page; p < end; p += 4096) {
                UnwatchPage(p);
                ReleaseOrphanReadWatch(p);
            }
            g_read_no_gpu_bytes.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (is_write) {
            cpu_write_hot[device_addr >> 20] = std::chrono::steady_clock::now(); // GPU-side (command or inline)
            // gow3 statistics: does the CPU write hit GPU-written bytes, or only their page?
            // (Printed only with the download stats above; not looked up without them.)
            const VAddr page = Common::AlignDown(device_addr, 4096);
            if (stats) {
                u64 gpu_bytes_in_page = 0;
                gpu_modified_ranges.ForEachInRange(page, 4096,
                                                   [&](VAddr lo, VAddr hi) { gpu_bytes_in_page += hi - lo; });
                if (gpu_modified_ranges.Intersects(device_addr, size)) {
                    ++g_write_fault_overlap;
                } else if (gpu_bytes_in_page) {
                    ++g_write_fault_same_page;
                } else {
                    ++g_write_fault_window;
                }
                g_write_fault_gpu_bytes += gpu_bytes_in_page;
            }
            // gow3: ray-cast result readbacks (yield_to_cpu): a CPU write into one is the guest
            // refilling the buffer for its next batch, so the pending GPU values of that piece
            // are dropped, not waited for. With multithreaded rendering the recording thread
            // runs behind and every such write waited for the GPU (about 8 ms per
            // frame, a third of it).
            // 16 Oct: dropping them threw away nearly every ray-cast batch's results in
            // (~250 per 5 s): the guest had read the stale buffer at once (the fence
            // was written when recorded) and refilled it, and NPCs placed with those results fell
            // from the sky. Ray-cast fences now wait for their readback (RunEventWriteEop), so the
            // guest reads first; a write into pending results waits for them below (counted).
            for (auto& readback : pending_readbacks) {
                if (!readback.yield_to_cpu) {
                    continue;
                }
                for (const auto& [piece, piece_size] : readback.pieces) {
                    if (piece < device_addr + size && device_addr < piece + piece_size) {
                        ++g_readback_yields;
                    }
                }
            }
            // gow3: bulk GPU data read back for a CPU-hot window (pieces over 64 KiB: skinning
            // output and the like in the per-frame rings). A CPU write into one of its pages is
            // the guest reusing that ring memory: the page's part of the piece is dead and is
            // dropped instead of waited for. With multithreaded rendering every such write waited
            // about 18 ms for the GPU (20+ per second, a third of the frame). Small
            // pieces (constants the CPU reads back) still wait.
            {
                const VAddr pages_lo = Common::AlignDown(device_addr, 4096);
                const VAddr pages_hi = Common::AlignUp(device_addr + size, 4096);
                for (auto& readback : pending_readbacks) {
                    for (const auto& [piece, piece_size] : readback.pieces) {
                        if (piece_size <= 64_KB || piece >= device_addr + size ||
                            device_addr >= piece + piece_size) {
                            continue;
                        }
                        const VAddr lo = std::max(piece, pages_lo);
                        const VAddr hi = std::min(piece + piece_size, pages_hi);
                        // A poisoned shared page's bytes go back to the guest's stale ones, as
                        // without the poison (it was uploaded to the GPU from the game's bytes,
                        // 88 dwords at a time, GOW3_SHARE_CHECK).
                        RestorePoison(lo, hi);
                        readback.rewritten.Add(lo, hi - lo);
                        GpuRangesSubtract(lo, hi - lo);
                        ++g_readback_yields;
                    }
                }
            }
            // The newest GPU data of the written bytes is on its way back: wait for that
            // readback only, not for all GPU work.
            if (const u64 tick = PendingReadbackTick(device_addr, size, true)) {
                wait_note = lazy_pages.contains(page) ? "readback waited (shared page), "
                                                      : "readback waited, ";
                ++g_readback_waits;
                static std::atomic<u32> described{0};
                if (described.fetch_add(1) < 24) {
                    for (const auto& readback : pending_readbacks) {
                        for (const auto& [piece, piece_size] : readback.pieces) {
                            if (piece < device_addr + size && device_addr < piece + piece_size) {
                                std::printf("Readback wait: CPU write %#llx+%llu into piece %#llx+%llu "
                                            "(tick %llu, now %llu)\n",
                                            static_cast<unsigned long long>(device_addr),
                                            static_cast<unsigned long long>(size),
                                            static_cast<unsigned long long>(piece),
                                            static_cast<unsigned long long>(piece_size),
                                            static_cast<unsigned long long>(readback.tick),
                                            static_cast<unsigned long long>(scheduler.CurrentTick()));
                            }
                        }
                    }
                    std::fflush(stdout);
                }
                CompleteReadbacks(true, tick);
            }
            // gow3: finished readbacks of this page first (no wait): it may be clean already.
            CompleteReadbacks(false);
            if (!gpu_modified_ranges.Intersects(page, 4096)) {
                lazy_pages.erase(page);
                memory_tracker->UnmarkRegionAsGpuModified(page, 4096);
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                path = "clean";
                return;
            }
            // Diagnostics: GOW3_SHARED_WRITES=0 turns the shared path below off (the write
            // downloads first).
            static const bool shared_writes = [] {
                const char* env = std::getenv("GOW3_SHARED_WRITES");
                return !(env && env[0] == '0');
            }();
            // The page becomes the CPU's and the GPU's: its GPU bytes stay the GPU's (uploads skip
            // them) until a readback brings them, and the guest copy of those bytes is kept so the
            // readback guard can tell the CPU's later writes into them.
            const auto share_page = [&](std::vector<std::pair<VAddr, u64>>& readbacks) {
                auto& slot = lazy_pages[page];
                const bool new_slot = !slot;
                if (!slot) {
                    slot = std::make_unique<LazyPage>();
                }
                slot->gpu_wins = true;
                std::array<u8, LAZY_PAGE_SIZE> guest;
                const bool backed = memory->TryReadBacking(page, guest.data(), LAZY_PAGE_SIZE);
                if (!backed) {
                    memory->CopySparseMemory(page, guest.data(), LAZY_PAGE_SIZE);
                }
                // gow3 25 Oct 2026 (GOW3_SHARE_UNFINISHED_WRITES): a page shared here (a readback
                // of its GPU bytes pending: DMA counters, GOW3_DMA_EOP_READBACK) is poisoned too,
                // except the physics shaders' pages. Without it the readback guard's 64-byte
                // line kept the game's line beside a counter, and the counter's old value.
                if (new_slot && backed && DmaEopReadbacks() && !IsResultPage(page)) {
                    slot->poisoned = true;
                }
                // gow3 24 Oct 2026: a poisoned page (share_poisoned) poisons its new GPU bytes
                // too (this write fault: the guest cannot write the page meanwhile).
                std::array<u8, LAZY_PAGE_SIZE> poisoned = guest;
                bool poison_new = false;
                if (slot->poisoned && backed) {
                    gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
                        for (VAddr byte = lo; byte < hi; ++byte) {
                            if (!slot->tracked.test(byte - page)) {
                                poisoned[byte - page] = PoisonByte(byte - page);
                                poison_new = true;
                            }
                        }
                    });
                    poison_new = poison_new && memory->TryWriteBacking(std::bit_cast<u8*>(page),
                                                                       poisoned.data(), LAZY_PAGE_SIZE);
                }
                if (new_slot && slot->poisoned && !poison_new) {
                    slot->poisoned = false; // nothing poisoned: an ordinary shared page
                }
                if (poison_new) {
                    NotePoisonEvent(page, new_slot ? "share-pending" : "extend");
                }
                const auto& reference = poison_new ? poisoned : guest;
                gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
                    for (VAddr byte = lo; byte < hi; ++byte) {
                        if (!slot->tracked.test(byte - page)) {
                            slot->tracked.set(byte - page);
                            slot->original[byte - page] = guest[byte - page];
                            slot->bytes[byte - page] = reference[byte - page];
                        }
                    }
                });
                // A poisoned page's GPU bytes always come back (the shared readbacks of `readbacks`
                // are dropped without the constants sync).
                (slot->poisoned ? unfinished_shared_readbacks : readbacks).emplace_back(page, 4096);
                NoteReadbackKind(page, 4096, ReadbackSharedPage);
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                ++g_shared_writes_count;
            };
            // gow3 24 Oct 2026 (GOW3_SHARE_UNFINISHED_WRITES): share_page with the GPU bytes'
            // guest copy poisoned (LazyPage::poisoned), so that the game's later writes into them
            // are seen whatever value they write. Only for a page no record or readback holds
            // (FetchableWindowTick refused those before this is reached), while the guest cannot
            // write it (this fault), in one write through the backing view.
            const auto share_poisoned = [&] {
                if (lazy_pages.contains(page)) {
                    return false;
                }
                std::array<u8, LAZY_PAGE_SIZE> contents;
                if (!memory->TryReadBacking(page, contents.data(), LAZY_PAGE_SIZE)) {
                    return false;
                }
                auto record = std::make_unique<LazyPage>();
                record->gpu_wins = true;
                record->poisoned = true;
                gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
                    for (VAddr byte = lo; byte < hi; ++byte) {
                        const size_t i = byte - page;
                        record->tracked.set(i);
                        record->original[i] = contents[i];
                        record->bytes[i] = PoisonByte(i);
                        contents[i] = record->bytes[i];
                    }
                });
                if (!memory->TryWriteBacking(std::bit_cast<u8*>(page), contents.data(),
                                             LAZY_PAGE_SIZE)) {
                    return false;
                }
                NotePoisonEvent(page, fmt::format("share{}", record->tracked.count()));
                lazy_pages[page] = std::move(record);
                unfinished_shared_readbacks.emplace_back(page, 4096);
                NoteReadbackKind(page, 4096, ReadbackSharedPage);
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                ++g_shared_writes_count;
                return true;
            };
            // gow3 22 Oct 2026 (GOW3_SHARED_FINISHED_WRITES=1): a write beside the page's GPU bytes
            // (false sharing: 761-930 of 910-1,039 CPU writes into GPU data per 5 s in Sarah's
            // bedroom) is shared whether those bytes are in flight or final. Final ones were
            // copied back first (the batch or CopyFinishedGpuPage below), the guest thread waiting
            // for the GPU command thread and the copy, 1-20 ms each: ~3 s of guest time per 5 s.
            // Now the write goes ahead at once and an asynchronous readback of only the page's GPU
            // bytes, at the end of the submission (Rasterizer::SyncSmallGpuWrites), brings them.
            if (SharedFinishedWrites() && shared_writes &&
                !gpu_modified_ranges.Intersects(device_addr, size)) {
                share_page(shared_readbacks);
                g_shared_finished_writes.fetch_add(1, std::memory_order_relaxed);
                path = "shared beside final GPU bytes";
                return;
            }
            // gow3 17 Oct 2026: the page's GPU bytes are final (the last GPU write into its 1 MiB
            // window has finished) and the arenas are in CPU-mappable video memory (Resizable BAR,
            // integrated GPUs): copy them from there. The download below waited for all GPU work
            // queued since, the current frame's included (crowded street: ~150 per
            // 5 s, ~10 ms each at 1440p-4K, 60 -> 45 FPS).
            // gow3 21 Oct 2026 (GOW3_FETCH_BATCH_MAPPED=1, laptop): with CPU-mappable
            // arenas (Resizable BAR) the direct read below fetched one page per fault, so the
            // window batch of CopyFinishedGpuPage (GOW3_FETCH_BATCH_KB) never ran there. A game
            // streaming a level part into memory the GPU wrote long ago (a rooftop level):
            // ~7,700 write faults in 5 s on whole GPU-written pages, each an uncached 4 KiB read
            // of video memory plus a round trip to the GPU command thread: 50-180 ms frames. When
            // another page of the fault's window is finished GPU data too, the batched copy on the
            // second queue goes first (CopyFinishedGpuPage's own rules: finished windows only, no
            // lazy page, no pending readback, write-protected pages only), and ExtendWriteFault
            // then lets the rest of the window be written without faults. A lone page keeps the
            // direct read; without the second queue nothing changes.
            // On by default since 21 Oct 2026 (laptop A/B, two builds, two pairs each: the same
            // streaming bursts cost 31-46 frames over 40 ms off, 0-4 on, none of them from
            // write faults); GOW3_FETCH_BATCH_MAPPED=0 turns it off.
            static const bool batch_mapped = [] {
                const char* env = std::getenv("GOW3_FETCH_BATCH_MAPPED");
                const bool on = !(env && env[0] == '0');
                std::printf("GPU: write faults into GPU data in CPU-mappable arenas fetch "
                            "their window's finished pages in one copy%s\n",
                            on ? " (GOW3_FETCH_BATCH_MAPPED=0: one page)" : ": off");
                std::fflush(stdout);
                return on;
            }();
            static const u64 batch_window = [] {
                // As CopyFinishedGpuPage reads GOW3_FETCH_BATCH_KB (0 or < 8: no batch).
                const char* env = std::getenv("GOW3_FETCH_BATCH_KB");
                const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
                return kib < 8 ? u64{0} : std::bit_ceil(std::min<u64>(kib, 1024)) * 1024;
            }();
            // The pages CopyFinishedGpuPage would take along: only when there are some and the
            // GPU has finished all of them (the batch waits for the newest tick it takes; the
            // direct read of a finished page never waits).
            const auto other_finished_page = [&] {
                u64 newest = FetchableWindowTick(page, false);
                bool others = false;
                const VAddr window = Common::AlignDown(page, batch_window);
                for (VAddr other = window; newest && other < window + batch_window; other += 4096) {
                    if (other != page && gpu_modified_ranges.Intersects(other, 4096) &&
                        memory_tracker->IsRegionGpuModified(other, 4096) &&
                        !memory_tracker->IsRegionCpuModified(other, 4096)) {
                        if (const u64 tick = FetchableWindowTick(other, false)) {
                            newest = std::max(newest, tick);
                            others = true;
                        }
                    }
                }
                if (!others || !newest) {
                    return false;
                }
                auto* semaphore = scheduler.GetWorkSemaphore();
                if (!semaphore->IsFree(newest)) {
                    semaphore->Refresh();
                }
                return semaphore->IsFree(newest);
            };
            if (batch_mapped && batch_window && arena_host_mapped && instance.GetCopyQueue() &&
                other_finished_page() && CopyFinishedGpuPage(page)) {
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                path = "batched copy";
                return;
            }
            // gow3 21 Oct 2026 (GOW3_FETCH_NO_BLOCK=1): the second-queue copy below would wait for
            // this page's newest GPU write (submitted, not finished): the guest thread waits for
            // it instead (see ReadMemory's end) and asks again.
            if (allow_defer && instance.GetCopyQueue() && UrgentCopiesEnabled()) {
                if (const u64 tick = FetchableWindowTick(page, false)) {
                    auto* semaphore = scheduler.GetWorkSemaphore();
                    if (!semaphore->IsFree(tick)) {
                        semaphore->Refresh();
                    }
                    if (stats && !semaphore->IsFree(tick)) {
                        // Diagnostics: the GPU write this fetch waits for, and the newest
                        // recorded writer of each GPU-written range in the page.
                        u64 page_tick = 0;
                        {
                            std::shared_lock lock{gpu_ranges_mutex};
                            if (const auto it = gpu_write_page_tick.find(page >> 12);
                                it != gpu_write_page_tick.end()) {
                                page_tick = it->second;
                            }
                        }
                        ++(page_tick == tick ? g_wait_tick_page : g_wait_tick_window);
                        std::scoped_lock lock{g_wait_writers_mutex};
                        gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
                            const auto write = NewestGpuWriteInto(lo, hi);
                            auto& entry = g_wait_writers[write.writer];
                            if (entry.waits++ == 0) {
                                entry.example = lo;
                                entry.example_size = hi - lo;
                                entry.write_address = write.address;
                                entry.write_size = write.size;
                            }
                        });
                    }
                    // gow3 24 Oct 2026 (GOW3_SHARE_UNFINISHED_WRITES=1; off by default, on for The
                    // Last of Us in tools/probe_tlou.py): a write beside
                    // the page's GPU bytes (false sharing) waited here for the frame in flight:
                    // in the Quarantine Zone (the player's profile) ~205 such writes per 5 s on
                    // the game's job threads, ~37 ms each (85% of their write-fault time), into
                    // per-frame rings where the GPU writes a few hundred bytes per page. The
                    // write now goes ahead on a shared page (uploads skip its GPU bytes, the
                    // readback guard keeps the game's later changes) and those GPU bytes come
                    // back asynchronously after this submission (Rasterizer::SyncSmallGpuWrites),
                    // recorded after the running write, so guest memory gets them as the fetch
                    // gave them. Writes into the GPU bytes themselves still wait.
                    // The player's session with it on: no such wait left, but NPCs moved
                    // wrongly again. GOW3_SHARE_CHECK=1 then found a physics page (106beafd's)
                    // whose game-written bytes the GPU never got: the game rewrote GPU bytes with
                    // the value their stale guest copy already had, which the shared page's
                    // compare cannot see, so uploads skipped them. Now those bytes are poisoned
                    // in guest memory while shared (share_poisoned), and pages of the physics
                    // shaders and of DMA copies (counters: with them shared the NPCs walked
                    // oddly), whose GPU bytes the game reads, wait as before (IsResultPage).
                    // Five player sessions later: no stale or poisoned game bytes left, NPCs
                    // normal, frame time stddev ~15 -> ~10 ms. (The DMA page's differences the
                    // check reports are on the waited path too: labels the runtime writes into
                    // guest memory only.) 25 Oct 2026: with GOW3_DMA_EOP_READBACK the DMA
                    // copies come back with the next end-of-pipe fence and their pages are
                    // shared too (stddev ~8 ms).
                    if (!semaphore->IsFree(tick) && ShareUnfinishedWrites() && shared_writes &&
                        !gpu_modified_ranges.Intersects(device_addr, size)) {
                        if (IsResultPage(page)) {
                            g_result_page_waits.fetch_add(1, std::memory_order_relaxed);
                        } else if (share_poisoned()) {
                            g_shared_unfinished_writes.fetch_add(1, std::memory_order_relaxed);
                            NoteShareCheck(page, 1);
                            path = "shared poisoned beside running GPU writes";
                            return;
                        }
                    }
                    if (!semaphore->IsFree(tick)) {
                        deferred_tick = tick;
                        NoteShareCheck(page, 0);
                        path = "deferred";
                        return;
                    }
                }
            }
            if (ReadFinishedGpuPage(page) || CopyFinishedGpuPage(page, allow_defer)) {
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                path = "copied";
                return;
            }
            // gow3: the GPU bytes in this page come from work still in flight (recorded in this
            // submission or not finished): the guest, a frame ahead, writes next to them (false
            // sharing: ~90 per second in heavy scenes, each a download waiting for all GPU
            // work). The write goes ahead; the page's GPU bytes stay the GPU's (uploads skip
            // them) until the next readback brings them. No guest bytes are compared (the
            // lazy-write merge did, and put stale GPU bytes back into rewritten data).
            // gow3 17 Oct 2026: GPU bytes partly in flight, partly old (a crowded street: ~120
            // per 5 s, each a full download): the old ones are fetched on the second queue (they
            // are final), then the page takes the shared path.
            if (shared_writes && !gpu_modified_ranges.Intersects(device_addr, size) &&
                !GpuBytesInFlight(page)) {
                FetchOldGpuBytes(page);
            }
            if (shared_writes && !gpu_modified_ranges.Intersects(device_addr, size)) {
                if (GpuBytesInFlight(page)) {
                    share_page(shared_readbacks);
                    path = "shared beside GPU bytes in flight";
                    return;
                }
            }
            // gow3 25 Oct 2026: any write into a poisoned shared page (protected again after an
            // upload), into its GPU bytes too: the poison shows the uploads and write-backs which
            // bytes the game wrote (per 4-byte group), so it goes ahead; its GPU bytes are read
            // back after the submission. Each such write downloaded a 512 KiB window and waited
            // for all GPU work (20-47 per 5 s in the Quarantine Zone).
            if (const auto lazy = lazy_pages.find(page);
                lazy != lazy_pages.end() && lazy->second->poisoned && device_addr + size <= page + 4096) {
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                unfinished_shared_readbacks.emplace_back(page, 4096);
                g_poisoned_page_writes.fetch_add(1, std::memory_order_relaxed);
                NotePoisonEvent(page, gpu_modified_ranges.Intersects(device_addr, size) ? "write-into"
                                                                                          : "write-beside");
                path = "poisoned shared page, went ahead";
                return;
            }
            if (TryLazyWrite(device_addr, size)) {
                path = "lazy write";
                return;
            }
            // gow3 20 Oct 2026 (integrated GPUs): without a second queue of the graphics family
            // (the Radeon 780M has one) CopyFinishedGpuPage cannot fetch, and the download below
            // waits for all GPU work queued so far, the frame in flight included: 3.2% of the GPU
            // command thread on a 780M (720p, Ryzen 7840HS). With CPU-mappable
            // arenas (an integrated GPU's memory is the system's) waiting only for the submission
            // that last wrote the page's window, then reading the page in place, does the same.
            // GOW3_WAIT_PAGE_READ=0: the download, as before.
            static const bool wait_page_read = [] {
                const char* env = std::getenv("GOW3_WAIT_PAGE_READ");
                return !(env && env[0] == '0');
            }();
            if (wait_page_read && arena_host_mapped && !instance.GetCopyQueue() &&
                device_addr + size <= page + 4096) {
                if (const u64 tick = FetchableWindowTick(page, false)) {
                    scheduler.GetWorkSemaphore()->Wait(tick);
                    if (ReadFinishedGpuPage(page)) {
                        ++g_waited_page_reads;
                        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                        path = "waited page read";
                        return;
                    }
                }
            }
        }
        path = "full download";
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        {
            // gow3 diagnostics: why this fault ends in a full download (bits, 5 s statistics).
            const VAddr page = Common::AlignDown(device_addr, 4096);
            u32 bits = is_write ? 1 : 0;
            bits |= gpu_modified_ranges.Intersects(device_addr, size) ? 2 : 0;
            bits |= lazy_pages.contains(page) ? 4 : 0;
            bits |= PendingReadbackTick(page, 4096, false) ? 8 : 0;
            bits |= GpuBytesInFlight(page) ? 16 : 0;
            {
                std::shared_lock lock{gpu_ranges_mutex};
                const auto it = gpu_write_window_tick.find(page >> 20);
                const auto small = gpu_write_page_tick.find(page >> 12);
                const u64 tick = std::max(it != gpu_write_window_tick.end() ? it->second : 0,
                                          small != gpu_write_page_tick.end() ? small->second : 0);
                bits |= tick >= scheduler.CurrentTick() ? 32 : 0;
            }
            ++g_download_reasons[bits];
        }
        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        DownloadMemory(arena, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
        if (is_write && FaultTimesEnabled()) {
            t_write_fault_note = fmt::format("{}{}", wait_note, path);
        }
    } else {
        const auto queued = std::chrono::steady_clock::now();
        // gow3 21 Oct 2026: a fault request (GOW3_WAIT_SERVICES_FAULTS=1: the command thread's
        // frames-ahead wait serves it too; otherwise SendCommand<true> as before).
        liverpool->SendFaultCommand(flush_request);
        // (GOW3_FAULT_TIMES) the time the request waited for the command thread, the time it
        // ran there, the deferred GPU waits.
        using Ms = std::chrono::duration<double, std::milli>;
        const auto request_done = std::chrono::steady_clock::now();
        double queue_ms = Ms(started - queued).count();
        double run_ms = Ms(request_done - started).count();
        double defer_ms = 0.0;
        // gow3 21 Oct 2026 (GOW3_FETCH_NO_BLOCK=1): the GPU write the fetch needs finished is
        // waited for here; the request starts over (the page may have changed meanwhile). After
        // three such waits the old path (the command thread waits) ends it.
        u32 deferrals = 0;
        for (u32 attempt = 1; deferred_tick; ++attempt) {
            const u64 tick = std::exchange(deferred_tick, 0);
            const auto wait_start = std::chrono::steady_clock::now();
            scheduler.GetWorkSemaphore()->Wait(tick);
            const auto resent = std::chrono::steady_clock::now();
            defer_ms += Ms(resent - wait_start).count();
            allow_defer = attempt < 3;
            g_fetch_deferrals.fetch_add(1, std::memory_order_relaxed);
            liverpool->SendFaultCommand(flush_request);
            queue_ms += Ms(started - resent).count();
            run_ms += Ms(std::chrono::steady_clock::now() - started).count();
            ++deferrals;
        }
        if (!is_write) {
            g_read_queue_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now() - queued)
                                   .count();
        } else if (FaultTimesEnabled()) {
            // Before the request: the tracker's region lock (InvalidateRegion), then the
            // statistics above.
            const double tracker_ms = invalidate_start.time_since_epoch().count()
                                          ? Ms(entered - invalidate_start).count()
                                          : 0.0;
            t_write_fault_note = fmt::format(
                "{}{}{}; tracker lock {:.2f} ms, stats {:.2f} ms, command thread queue {:.2f} ms, "
                "ran {:.2f} ms{}",
                wait_note, path,
                deferrals ? fmt::format(" after {} deferral(s)", deferrals) : std::string{},
                tracker_ms, Ms(queued - entered).count(), queue_ms, run_ms,
                deferrals ? fmt::format(", GPU waits {:.2f} ms", defer_ms) : std::string{});
        }
    }
}

void BufferCache::DownloadRanges(std::span<const std::pair<VAddr, u64>> ranges) {
    struct Pending {
        const Buffer* arena;
        VAddr address;
        u64 size;
        u64 offset;
    };
    std::vector<Pending> pending;
    u64 total_size_bytes = 0;
    // Older asynchronous readbacks of these bytes go first (theirs is older data).
    u64 readback_tick = 0;
    for (const auto& [address, size] : ranges) {
        readback_tick = std::max(readback_tick, PendingReadbackTick(address, size, false));
    }
    if (readback_tick) {
        CompleteReadbacks(true, readback_tick);
    }
    for (const auto& [address, size] : ranges) {
        if (size == 0) {
            continue;
        }
        const u64 first_block = address >> block_shift;
        const u64 last_block = (address + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        memory_tracker->ForEachDownloadRange<false>(address, size, [&](u64 range, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(range, range_size, [&](VAddr start, VAddr end) {
                pending.push_back({arena, start, end - start, total_size_bytes});
                total_size_bytes += (end - start + 63) & ~u64(63);
            });
            GpuRangesSubtract(range, range_size);
        });
    }
    if (total_size_bytes == 0) {
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (size_t i = 0; i < pending.size();) {
        boost::container::small_vector<vk::BufferCopy, 16> copies;
        const Buffer* arena = pending[i].arena;
        for (; i < pending.size() && pending[i].arena == arena; ++i) {
            copies.push_back(vk::BufferCopy{
                .srcOffset = pending[i].address - arena->cpu_addr,
                .dstOffset = download.offset + pending[i].offset,
                .size = pending[i].size,
            });
        }
        runtime.CopyBuffer(arena, download.buffer, copies);
    }
    scheduler.Finish();
    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& p : pending) {
        WriteBackGpuData(p.address, download.mapped + p.offset, p.size);
    }
    for (const auto& [address, size] : ranges) {
        if (size != 0) {
            EraseLazyPages(address, size);
            memory_tracker->UnmarkRegionAsGpuModified(address, size);
        }
    }
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    // Older asynchronous readbacks of these bytes go first (theirs is older data).
    if (const u64 readback_tick = PendingReadbackTick(device_addr, size, false)) {
        CompleteReadbacks(true, readback_tick);
    }
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        GpuRangesSubtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    ++gow3_downloads;
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    scheduler.Finish();

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        WriteBackGpuData(arena_base + copy.srcOffset,
                         download.mapped + (copy.dstOffset - download.offset), copy.size);
    }
    EraseLazyPages(device_addr, size);
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

bool BufferCache::TryLazyWrite(VAddr device_addr, u64 size) {
    static_assert(LAZY_PAGE_SIZE == TRACKER_BYTES_PER_PAGE);
    // Fault-sized writes only (unmapping invalidates whole ranges).
    if (size > 64 || !LazyWritesEnabled() || gpu_modified_ranges.Intersects(device_addr, size)) {
        return false;
    }
    SnapshotLazyPages(device_addr, size);
    memory_tracker->MarkRegionAsCpuModified(device_addr, size);
    ++g_lazy_writes;
    return true;
}

bool BufferCache::PrepareHostWrite(VAddr address, u64 size) {
    if (gpu_modified_ranges.Intersects(address, size)) {
        return false;
    }
    if (!LazyWritesEnabled()) {
        if (memory_tracker->IsRegionGpuModified(address, size)) {
            return false;
        }
    } else {
        SnapshotLazyPages(address, size);
    }
    memory_tracker->MarkRegionAsCpuModified(address, size);
    return true;
}

bool BufferCache::GpuBytesInFlight(VAddr page) {
    std::bitset<4096> covered;
    if (const auto it = page_writes.find(page >> 12); it != page_writes.end()) {
        auto* semaphore = scheduler.GetWorkSemaphore();
        semaphore->Refresh(); // the cached value lags: finished work would count as running
        for (const auto& write : it->second) {
            // Recorded since the last readback (the next one brings it) or not finished yet.
            if (write.tick >= last_readback_tick || !semaphore->IsFree(write.tick)) {
                for (u32 byte = write.lo; byte < write.hi; ++byte) {
                    covered.set(byte);
                }
            }
        }
    }
    for (const auto& readback : pending_readbacks) {
        for (const auto& [address, size] : readback.pieces) {
            const VAddr lo = std::max(address, page);
            const VAddr hi = std::min(address + size, page + 4096);
            for (VAddr byte = lo; byte < hi; ++byte) {
                covered.set(byte - page);
            }
        }
    }
    bool all = true;
    gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
        for (VAddr byte = lo; byte < hi && all; ++byte) {
            all = covered.test(byte - page);
        }
    });
    return all;
}

static std::atomic<u32> g_labels_in_readbacks{0};

u32 BufferCache::TakeLabelsInReadbacks() {
    return g_labels_in_readbacks.exchange(0, std::memory_order_relaxed);
}

void BufferCache::NoteCpuWrite(VAddr address, u64 size) {
    bool inside = false;
    if (TouchesPendingReadback(address, size)) {
        for (auto& readback : pending_readbacks) {
            readback.NoteRewrite(address, size);
            for (const auto& [lo, n] : readback.pieces) {
                inside |= address < lo + n && lo < address + size;
            }
        }
    }
    if (inside) {
        g_labels_in_readbacks.fetch_add(1, std::memory_order_relaxed);
    }
    if (!gpu_modified_ranges.Intersects(address, size)) {
        return;
    }
    GpuRangesSubtract(address, size);
    for (VAddr page = Common::AlignDown(address, LAZY_PAGE_SIZE); page < address + size;
         page += LAZY_PAGE_SIZE) {
        if (!gpu_modified_ranges.Intersects(page, LAZY_PAGE_SIZE)) {
            lazy_pages.erase(page);
            memory_tracker->UnmarkRegionAsGpuModified(page, LAZY_PAGE_SIZE);
        }
    }
}

bool BufferCache::WriteThrough(VAddr address, const void* data, u64 size) {
    if (size == 0 || size > 64_KB || address % 4 != 0 || size % 4 != 0) {
        return false;
    }
    if (!memory->TryWriteBacking(reinterpret_cast<void*>(address), data, size)) {
        return false;
    }
    // The GPU's copy, in order with the work recorded so far (resident and synchronized first).
    const auto [arena, offset] = ObtainBuffer(address, static_cast<u32>(size), false);
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached, 4);
    std::memcpy(staging.mapped, data, size);
    staging.Flush();
    const vk::BufferCopy copy{
        .srcOffset = staging.offset,
        .dstOffset = offset,
        .size = size,
    };
    runtime.CopyBuffer(staging.buffer, arena, std::span{&copy, 1});
    // Newest value in both views: not GPU data any more; readbacks recorded earlier must not
    // write their older values over it.
    if (gpu_modified_ranges.Intersects(address, size)) {
        GpuRangesSubtract(address, size);
        for (VAddr page = Common::AlignDown(address, LAZY_PAGE_SIZE); page < address + size;
             page += LAZY_PAGE_SIZE) {
            if (!gpu_modified_ranges.Intersects(page, LAZY_PAGE_SIZE)) {
                lazy_pages.erase(page);
                memory_tracker->UnmarkRegionAsGpuModified(page, LAZY_PAGE_SIZE);
            }
        }
    }
    if (TouchesPendingReadback(address, size)) {
        for (auto& readback : pending_readbacks) {
            readback.NoteRewrite(address, size);
        }
    }
    return true;
}

void BufferCache::SnapshotLazyPages(VAddr address, u64 size) {
    const VAddr end = address + size;
    for (VAddr page = Common::AlignDown(address, LAZY_PAGE_SIZE); page < end;
         page += LAZY_PAGE_SIZE) {
        LazyPage* record = nullptr;
        std::array<u8, LAZY_PAGE_SIZE> guest;
        gpu_modified_ranges.ForEachInRange(page, LAZY_PAGE_SIZE, [&](VAddr lo, VAddr hi) {
            if (!record) {
                auto& slot = lazy_pages[page];
                if (!slot) {
                    slot = std::make_unique<LazyPage>();
                }
                record = slot.get();
                if (!memory->TryReadBacking(page, guest.data(), LAZY_PAGE_SIZE)) {
                    memory->CopySparseMemory(page, guest.data(), LAZY_PAGE_SIZE);
                }
            }
            for (VAddr byte = lo; byte < hi; ++byte) {
                const size_t i = byte - page;
                // A byte recorded by an earlier lazy write keeps that record.
                if (!record->tracked.test(i)) {
                    record->tracked.set(i);
                    record->bytes[i] = guest[i];
                }
            }
        });
    }
}

template <typename Func>
void BufferCache::ForEachLazyUploadPiece(VAddr address, u64 size, Func&& func) {
    const VAddr end = address + size;
    auto it = lazy_pages.lower_bound(Common::AlignDown(address, LAZY_PAGE_SIZE));
    if (it == lazy_pages.end() || it->first >= end) {
        func(address, size);
        return;
    }
    boost::container::small_vector<std::pair<VAddr, VAddr>, 16> skips;
    while (it != lazy_pages.end() && it->first < end) {
        const VAddr page = it->first;
        LazyPage& record = *it->second;
        std::array<u8, LAZY_PAGE_SIZE> guest;
        if (!memory->TryReadBacking(page, guest.data(), LAZY_PAGE_SIZE)) {
            memory->CopySparseMemory(page, guest.data(), LAZY_PAGE_SIZE);
        }
        const VAddr lo_limit = std::max(page, address);
        const VAddr hi_limit = std::min(page + LAZY_PAGE_SIZE, end);
        boost::container::small_vector<std::pair<VAddr, VAddr>, 8> cpu_wins;
        const auto add_run = [](auto& runs, VAddr byte) {
            if (!runs.empty() && runs.back().second == byte) {
                ++runs.back().second;
            } else {
                runs.emplace_back(byte, byte + 1);
            }
        };
        // Records outside gpu_modified_ranges are stale (downloaded meanwhile): ignored.
        // Readback guard: decided per 64-byte line, shared pages too.
        const bool guard = ReadbackGuardEnabled();
        // Poisoned pages: per aligned 4-byte group (a 64-byte line would upload the poison of
        // the GPU bytes the game left alone next to one it wrote).
        const u64 granule = record.poisoned ? 4 : guard ? GuardLine : 1;
        const bool compare = guard || !record.gpu_wins || record.poisoned;
        if (record.poisoned) {
            // Tracked bytes no longer GPU data that still hold the poison would be uploaded
            // with it (the loop below only skips GPU data): their pre-poison bytes come back.
            std::bitset<LAZY_PAGE_SIZE> gpu_owned;
            gpu_modified_ranges.ForEachInRange(lo_limit, hi_limit - lo_limit, [&](VAddr a, VAddr b) {
                for (VAddr byte = a; byte < b; ++byte) {
                    gpu_owned.set(byte - page);
                }
            });
            u32 stale = 0;
            for (VAddr byte = lo_limit; byte < hi_limit; ++byte) {
                const size_t i = byte - page;
                if (record.tracked.test(i) && !gpu_owned.test(i)) {
                    if (record.bytes[i] == PoisonByte(i) && guest[i] == PoisonByte(i)) {
                        guest[i] = record.original[i];
                        ++stale;
                    }
                    record.tracked.reset(i);
                }
            }
            if (stale) {
                memory->TryWriteBacking(std::bit_cast<u8*>(lo_limit), guest.data() + (lo_limit - page),
                                        hi_limit - lo_limit);
                g_poison_restored.fetch_add(1, std::memory_order_relaxed);
                NotePoisonEvent(page, fmt::format("upload-stale{}", stale));
                ReportPoisonRestore("upload", page);
            }
        }
        u32 poisoned_cpu = 0, poisoned_skip = 0;
        gpu_modified_ranges.ForEachInRange(lo_limit, hi_limit - lo_limit, [&](VAddr lo, VAddr hi) {
            for (VAddr line = lo; line < hi;) {
                const VAddr line_end = std::min(Common::AlignDown(line, granule) + granule, hi);
                bool cpu_line = false;
                for (VAddr byte = line; compare && byte < line_end && !cpu_line; ++byte) {
                    const size_t i = byte - page;
                    cpu_line = record.tracked.test(i) && guest[i] != record.bytes[i];
                }
                for (VAddr byte = line; byte < line_end; ++byte) {
                    const size_t i = byte - page;
                    if (!record.tracked.test(i)) {
                        continue;
                    }
                    if (cpu_line) {
                        record.tracked.reset(i);
                        add_run(cpu_wins, byte);
                        ++poisoned_cpu;
                    } else {
                        add_run(skips, byte);
                        ++poisoned_skip;
                    }
                }
                line = line_end;
            }
        });
        if (record.poisoned) {
            NotePoisonEvent(page, fmt::format("up{}/{}{}", poisoned_cpu, poisoned_skip,
                                              record.tracked.none() ? "-gone" : ""));
        }
        // The CPU overwrote these after the GPU: they are uploaded and stop being GPU data.
        for (const auto& [lo, hi] : cpu_wins) {
            GpuRangesSubtract(lo, hi - lo);
            g_lazy_cpu_wins += u32(hi - lo);
        }
        it = record.tracked.none() ? lazy_pages.erase(it) : std::next(it);
    }
    VAddr cursor = address;
    for (const auto& [lo, hi] : skips) {
        if (lo > cursor) {
            func(cursor, lo - cursor);
        }
        cursor = hi;
    }
    if (cursor < end) {
        func(cursor, end - cursor);
    }
}

void BufferCache::WriteBackGpuData(VAddr address, const u8* data, u64 size,
                                   const RangeSet* rewritten) {
    const VAddr end = address + size;
    auto it = lazy_pages.lower_bound(Common::AlignDown(address, LAZY_PAGE_SIZE));
    VAddr cursor = address;
    while (it != lazy_pages.end() && it->first < end) {
        const VAddr page = it->first;
        LazyPage& record = *it->second;
        std::array<u8, LAZY_PAGE_SIZE> guest;
        // Through the backing view: the page may still be read-protected for this very readback
        // (CompleteReadbacks unwatches after the write-back); a read fault there completed the
        // same readback again, recursively, until the stack ran out (a silent exit, 15 Oct).
        if (!memory->TryReadBacking(page, guest.data(), LAZY_PAGE_SIZE)) {
            memory->CopySparseMemory(page, guest.data(), LAZY_PAGE_SIZE);
        }
        const VAddr lo = std::max(page, address);
        const VAddr hi = std::min(page + LAZY_PAGE_SIZE, end);
        if (record.poisoned) {
            // gow3 24 Oct 2026: per aligned 4-byte group. A group the game wrote (some tracked
            // byte no longer holds the poison) keeps its value (the page is CPU-modified, the
            // next upload sends it); the others take the GPU's bytes, except those a newer GPU
            // write covers (`rewritten`), which keep the poison and stay tracked.
            if (lo > cursor) {
                memory->TryWriteBacking(std::bit_cast<u8*>(cursor), data + (cursor - address),
                                        lo - cursor);
            }
            // The bytes taking the GPU's value, written back in runs.
            VAddr run = hi;
            const auto flush_run = [&](VAddr to) {
                if (run < to) {
                    memory->TryWriteBacking(std::bit_cast<u8*>(run), data + (run - address), to - run);
                }
                run = hi;
            };
            u32 n_data = 0, n_cpu = 0, n_still = 0;
            for (VAddr group = lo; group < hi;) {
                const VAddr group_end = std::min(Common::AlignDown(group, 4) + 4, hi);
                bool cpu_group = false;
                for (VAddr byte = group; byte < group_end && !cpu_group; ++byte) {
                    const size_t i = byte - page;
                    cpu_group = record.tracked.test(i) && guest[i] != record.bytes[i];
                }
                for (VAddr byte = group; byte < group_end; ++byte) {
                    const size_t i = byte - page;
                    bool write = !record.tracked.test(i);
                    if (!write) {
                        const bool still_gpu = rewritten && rewritten->Contains(byte, 1);
                        if (cpu_group) {
                            ++g_lazy_cpu_wins;
                            ++n_cpu;
                            if (still_gpu) {
                                record.bytes[i] = guest[i];
                            } else {
                                record.tracked.reset(i);
                            }
                        } else if (!still_gpu) {
                            write = true;
                            ++n_data;
                            record.tracked.reset(i);
                        } else {
                            ++n_still;
                        }
                    }
                    if (write) {
                        run = std::min(run, byte);
                    } else {
                        flush_run(byte);
                    }
                }
                group = group_end;
            }
            flush_run(hi);
            NotePoisonEvent(page, fmt::format("{}{}/{}/{}{}", rewritten ? "land" : "down", n_data,
                                              n_cpu, n_still, record.tracked.none() ? "-gone" : ""));
            cursor = hi;
            it = record.tracked.none() ? lazy_pages.erase(it) : std::next(it);
            continue;
        }
        // Readback guard: decided per 64-byte line, shared pages (gpu_wins) too. Without it,
        // per byte, and shared pages take the GPU's bytes unseen.
        const bool guard = ReadbackGuardEnabled();
        const u64 granule = guard ? GuardLine : 1;
        const bool compare = guard || !record.gpu_wins;
        for (VAddr line = lo; line < hi;) {
            const VAddr line_end = std::min(Common::AlignDown(line, granule) + granule, hi);
            // The CPU wrote GPU bytes of this line after the GPU: they differ from the record
            // (their guest copy when the page became CPU-writable) and from the GPU's value.
            bool cpu_line = false;
            for (VAddr byte = line; compare && byte < line_end && !cpu_line; ++byte) {
                const size_t i = byte - page;
                cpu_line = record.tracked.test(i) && guest[i] != record.bytes[i] &&
                           guest[i] != data[byte - address];
            }
            for (VAddr byte = line; byte < line_end; ++byte) {
                const size_t i = byte - page;
                if (!record.tracked.test(i)) {
                    continue;
                }
                const bool still_gpu = rewritten && rewritten->Contains(byte, 1);
                if (cpu_line) {
                    // The CPU value stays. The page is still CPU-modified, so the next upload
                    // sends it to the GPU copy too; newer GPU data still to come is compared
                    // with it.
                    ++g_lazy_cpu_wins;
                    if (still_gpu) {
                        record.bytes[i] = guest[i];
                    } else if (rewritten) {
                        record.tracked.reset(i);
                    }
                } else if (still_gpu) {
                    // Written now; the GPU's newer value is stale in guest memory again.
                    record.bytes[i] = data[byte - address];
                } else if (rewritten || record.gpu_wins) {
                    record.tracked.reset(i);
                }
            }
            if (cpu_line) {
                if (line > cursor) {
                    memory->TryWriteBacking(std::bit_cast<u8*>(cursor), data + (cursor - address),
                                            line - cursor);
                }
                cursor = line_end;
                if (guard) {
                    ++g_guard_lazy_lines;
                    g_guard_bytes += line_end - line;
                    if (GuardLogNow()) {
                        std::printf("Readback guard: kept %llu bytes at %#llx that the game wrote after "
                                    "the GPU (%s page, %s)\n",
                                    static_cast<unsigned long long>(line_end - line),
                                    static_cast<unsigned long long>(line),
                                    record.gpu_wins ? "shared" : "lazy",
                                    rewritten ? "readback" : "download");
                        std::fflush(stdout);
                    }
                }
            }
            line = line_end;
        }
        it = record.tracked.none() ? lazy_pages.erase(it) : std::next(it);
    }
    if (cursor < end) {
        memory->TryWriteBacking(std::bit_cast<u8*>(cursor), data + (cursor - address),
                                end - cursor);
    }
}

void BufferCache::LandGuarded(PendingReadback& readback, size_t piece, VAddr lo, VAddr hi) {
    const auto [address, size] = readback.pieces[piece];
    const u8* data = readback.mapped + readback.offsets[piece] + (lo - address);
    const u8* before = readback.before.data() + readback.offsets[piece] + (lo - address);
    const u64 n = hi - lo;
    // The guest bytes now, through the backing view (the page may still be read-protected for
    // this readback). A local buffer: a fault in here may land readbacks again. With
    // GOW3_READBACK_POOL a buffer kept by this thread (the outermost landing only; nested ones,
    // from a fault in here, take their own) that is not zeroed: it is filled completely below.
    static thread_local SnapshotBytes kept;
    static thread_local u32 depth = 0;
    SnapshotBytes own;
    SnapshotBytes& pre = ReadbackPoolEnabled() && depth == 0 ? kept : own;
    struct Depth {
        u32& value;
        ~Depth() {
            --value;
            if (value == 0 && kept.capacity() > SnapshotPoolMaxBytes) {
                SnapshotBytes{}.swap(kept);
            }
        }
    } depth_guard{++depth};
    SizeScratch(pre, n);
    if (!memory->TryReadBacking(lo, pre.data(), n)) {
        memory->CopySparseMemory(lo, pre.data(), n);
    }
    VAddr cursor = lo;
    for (VAddr line = lo; line < hi;) {
        const VAddr line_end = std::min(Common::AlignDown(line, GuardLine) + GuardLine, hi);
        const u64 k = line - lo;
        const u64 m = line_end - line;
        // A poisoned shared page decides per byte from its record (WriteBackGpuData): a line
        // kept here would keep the poison of its GPU bytes the game did not write.
        if (const auto lazy = lazy_pages.find(Common::AlignDown(line, LAZY_PAGE_SIZE));
            lazy != lazy_pages.end() && lazy->second->poisoned) {
            line = line_end;
            continue;
        }
        // Changed since the readback was queued, and not to the GPU's value: the game wrote it
        // after the GPU (its fence was out by then), so the game's value is the newer one.
        if (std::memcmp(pre.data() + k, before + k, m) != 0 &&
            std::memcmp(pre.data() + k, data + k, m) != 0) {
            if (line > cursor) {
                WriteBackGpuData(cursor, data + (cursor - lo), line - cursor, &readback.rewritten);
            }
            cursor = line_end;
            ++g_guard_lines;
            g_guard_bytes += m;
            if (GuardLogNow()) {
                std::printf("Readback guard: kept %llu bytes at %#llx that the game wrote after the GPU "
                            "(readback of %#llx+%llu, tick %llu, page %s)\n",
                            static_cast<unsigned long long>(m), static_cast<unsigned long long>(line),
                            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                            static_cast<unsigned long long>(readback.tick),
                            memory_tracker->IsRegionCpuModified(line, m) ? "CPU-writable" : "protected");
                std::fflush(stdout);
            }
        }
        line = line_end;
    }
    if (hi > cursor) {
        WriteBackGpuData(cursor, data + (cursor - lo), hi - cursor, &readback.rewritten);
    }
    UpdateLaterSnapshots(lo, hi, pre.data());
}

void BufferCache::UpdateLaterSnapshots(VAddr lo, VAddr hi, const u8* pre) {
    SnapshotBytes post;
    for (size_t r = 1; r < pending_readbacks.size(); ++r) {
        PendingReadback& later = pending_readbacks[r];
        if (later.before.empty()) {
            continue;
        }
        for (size_t j = 0; j < later.pieces.size(); ++j) {
            const auto [piece, piece_size] = later.pieces[j];
            const VAddr a = std::max(piece, lo);
            const VAddr b = std::min(piece + piece_size, hi);
            if (a >= b) {
                continue;
            }
            if (post.empty()) {
                SizeScratch(post, hi - lo); // filled completely below
                if (!memory->TryReadBacking(lo, post.data(), hi - lo)) {
                    memory->CopySparseMemory(lo, post.data(), hi - lo);
                }
            }
            // Bytes the later readback saw as they were before this landing take its values;
            // bytes the CPU changed in between stay as the later readback saw them.
            u8* snapshot = later.before.data() + later.offsets[j] + (a - piece);
            for (VAddr x = a; x < b;) {
                const VAddr x_end = std::min(Common::AlignDown(x, GuardLine) + GuardLine, b);
                const u64 m = x_end - x;
                if (std::memcmp(snapshot + (x - a), pre + (x - lo), m) == 0) {
                    std::memcpy(snapshot + (x - a), post.data() + (x - lo), m);
                } else {
                    for (VAddr y = x; y < x_end; ++y) {
                        if (snapshot[y - a] == pre[y - lo]) {
                            snapshot[y - a] = post[y - lo];
                        }
                    }
                }
                x = x_end;
            }
        }
    }
}

void BufferCache::RecycleSnapshot(SnapshotBytes& bytes) {
    if (!ReadbackPoolEnabled() || bytes.capacity() == 0 || bytes.capacity() > SnapshotPoolMaxBytes) {
        return;
    }
    u64 kept = bytes.capacity();
    size_t smallest = 0;
    for (size_t i = 0; i < snapshot_pool.size(); ++i) {
        kept += snapshot_pool[i].capacity();
        if (snapshot_pool[i].capacity() < snapshot_pool[smallest].capacity()) {
            smallest = i;
        }
    }
    // Full: it replaces the smallest kept buffer if it is larger (sizes vary little).
    if (snapshot_pool.size() >= SnapshotPoolCount || kept > SnapshotPoolTotalBytes) {
        if (snapshot_pool.empty() || snapshot_pool[smallest].capacity() >= bytes.capacity() ||
            kept - snapshot_pool[smallest].capacity() > SnapshotPoolTotalBytes) {
            return;
        }
        snapshot_pool[smallest] = std::move(bytes);
        snapshot_pool[smallest].clear();
        return;
    }
    bytes.clear();
    snapshot_pool.push_back(std::move(bytes));
}

void BufferCache::DownloadRangesAsync(std::span<const std::pair<VAddr, u64>> ranges, bool protect,
                                      bool yield_to_cpu, bool never_protect) {
    last_readback_tick = scheduler.CurrentTick();
    PendingReadback readback;
    struct Copy {
        const Buffer* arena;
        VAddr address;
        u64 size;
    };
    std::vector<Copy> pending;
    u64 total_size_bytes = 0;
    for (const auto& [address, size] : ranges) {
        if (size == 0) {
            continue;
        }
        const u64 first_block = address >> block_shift;
        const u64 last_block = (address + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        memory_tracker->ForEachDownloadRange<false>(address, size, [&](u64 range, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(range, range_size, [&](VAddr start, VAddr end) {
                pending.push_back({arena, start, end - start});
                readback.pieces.emplace_back(start, end - start);
                readback.offsets.push_back(total_size_bytes);
                total_size_bytes += (end - start + 63) & ~u64(63);
            });
        });
    }
    if (total_size_bytes == 0) {
        return;
    }
    for (const auto& [address, size] : readback.pieces) {
        readback.span_lo = std::min(readback.span_lo, address);
        readback.span_hi = std::max(readback.span_hi, address + size);
    }
    // Readback guard: the guest bytes now, to tell the game's later writes from older data.
    // A snapshot memory can't be had for (#47: a laptop out of memory) is skipped: that readback
    // lands unguarded, as before 0.3.0, instead of the game stopping.
    if (ReadbackGuardEnabled()) {
        try {
            // GOW3_READBACK_POOL: the smallest kept buffer that is large enough (else the
            // largest, which grows); every byte is copied in below, so it is not zeroed.
            if (ReadbackPoolEnabled() && !snapshot_pool.empty()) {
                size_t best = 0;
                for (size_t i = 1; i < snapshot_pool.size(); ++i) {
                    const u64 have = snapshot_pool[i].capacity();
                    const u64 chosen = snapshot_pool[best].capacity();
                    if (chosen < total_size_bytes ? have > chosen
                                                  : have >= total_size_bytes && have < chosen) {
                        best = i;
                    }
                }
                readback.before = std::move(snapshot_pool[best]);
                if (best + 1 != snapshot_pool.size()) {
                    snapshot_pool[best] = std::move(snapshot_pool.back());
                }
                snapshot_pool.pop_back();
                readback.before.clear();
            }
            SizeScratch(readback.before, total_size_bytes);
        } catch (const std::bad_alloc&) {
            readback.before = {};
        }
    }
    if (!readback.before.empty()) {
        for (size_t i = 0; i < readback.pieces.size(); ++i) {
            const auto [address, size] = readback.pieces[i];
            u8* snapshot = readback.before.data() + readback.offsets[i];
            if (!memory->TryReadBacking(address, snapshot, size)) {
                memory->CopySparseMemory(address, snapshot, size);
            }
        }
    }
    // Held until the readback completes (a ring allocation could be reused before).
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached,
                                               0, true);
    const ProfileCopy mark{scheduler, 0xB0F7'0002ull, "buffer readback (async)"};
    for (size_t i = 0; i < pending.size();) {
        boost::container::small_vector<vk::BufferCopy, 16> copies;
        const Buffer* arena = pending[i].arena;
        for (; i < pending.size() && pending[i].arena == arena; ++i) {
            copies.push_back(vk::BufferCopy{
                .srcOffset = pending[i].address - arena->cpu_addr,
                .dstOffset = download.offset + readback.offsets[i],
                .size = pending[i].size,
            });
        }
        runtime.CopyBuffer(arena, download.buffer, copies);
    }
    if (never_protect) {
        readback.open = true;
        for (const auto& [address, size] : readback.pieces) {
            open_readback_ranges.Add(address, size);
        }
    }
    if (!never_protect && (SafeReadbacks() || protect)) {
        std::vector<VAddr> pages;
        const auto now = std::chrono::steady_clock::now();
        for (const auto& [address, size] : readback.pieces) {
            if (!protect && open_readback_ranges.Contains(address, size)) {
                continue; // fence-gated results already on their way back, unprotected
            }
            if (!protect) {
                if (const auto it = landed_pieces.find(address);
                    it != landed_pieces.end() && it->second.first == size &&
                    now - it->second.second < RecurringWindow) {
                    ++g_pieces_recurring;
                    continue;
                }
                ++g_pieces_protected;
            }
            for (VAddr page = Common::AlignDown(address, 4096); page < address + size; page += 4096) {
                if (!protect) {
                    if (const auto it = noisy_pages.find(page >> 12); it != noisy_pages.end()) {
                        if (it->second > now) {
                            continue;
                        }
                        noisy_pages.erase(it);
                    }
                }
                pages.push_back(page);
            }
        }
        std::ranges::sort(pages);
        const auto [first, last] = std::ranges::unique(pages);
        pages.erase(first, last);
        for (size_t i = 0; i < pages.size();) {
            size_t j = i + 1;
            while (j < pages.size() && pages[j] == pages[j - 1] + 4096) {
                ++j;
            }
            readback.watched.emplace_back(pages[i], (pages[j - 1] + 4096) - pages[i]);
            i = j;
        }
    }
    readback.tick = scheduler.CurrentTick();
    NoteReadbackQueued(scheduler, readback.tick);
    readback.yield_to_cpu = yield_to_cpu;
    readback.buffer = download.buffer;
    readback.offset = download.offset;
    readback.size = download.size;
    readback.mapped = download.mapped;
    g_readback_bytes += readback.size;
    if (ReadbackStatsEnabled()) {
        static auto last = std::chrono::steady_clock::now();
        std::string text;
        {
            std::scoped_lock lock{g_rb_stats_mutex};
            for (const auto& [address, size] : readback.pieces) {
                auto& entry = g_rb_window_bytes[address >> 20];
                entry[0] += size;
                ++entry[1];
                if (protect) {
                    g_rb_window_kinds[address >> 20] |= ReadbackEop;
                }
            }
            if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
                last = now;
                std::vector<std::pair<u64, u64>> top;
                u64 total = 0;
                for (const auto& [window, entry] : g_rb_window_bytes) {
                    top.emplace_back(entry[0], window);
                    total += entry[0];
                }
                std::ranges::sort(top, std::greater{});
                text = fmt::format("Readback windows (5 s): {:.1f} MB in {} windows\n", total / 1e6, top.size());
                for (size_t i = 0; i < std::min<size_t>(top.size(), 16); ++i) {
                    const u64 window = top[i].second;
                    const u32 kinds = g_rb_window_kinds[window];
                    text += fmt::format("  {:#x}: {:.1f} MB, {} pieces, {}{}{}{}{}{}\n", window << 20,
                                        top[i].first / 1e6, g_rb_window_bytes[window][1],
                                        kinds & ReadbackSmallWrite ? "small " : "",
                                        kinds & ReadbackHotWindow ? "hot " : "",
                                        kinds & ReadbackCpWrite ? "cp " : "",
                                        kinds & ReadbackSharedPage ? "shared " : "",
                                        kinds & ReadbackEop ? "eop " : "", "");
                }
                g_rb_window_bytes.clear();
                g_rb_window_kinds.clear();
            }
        }
        if (!text.empty()) {
            // Writers per window (NoteGpuWriter is on with these statistics).
            std::string out;
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t end = text.find('\n', pos);
                std::string line = text.substr(pos, end - pos);
                if (line.starts_with("  0x")) {
                    const u64 window = std::strtoull(line.c_str() + 2, nullptr, 16) >> 20;
                    line += WindowWriters(window);
                }
                out += line + "\n";
                pos = end + 1;
            }
            std::fputs(out.c_str(), stdout);
            std::fflush(stdout);
        }
    }
    IndexReadbackWindows(readback, true);
    pending_readbacks.push_back(std::move(readback));
    ++g_async_readbacks;
    SweepTracking();
}

void BufferCache::IndexReadbackWindows(const PendingReadback& readback, bool add) {
    boost::container::small_vector<u64, 16> windows;
    for (const auto& [address, size] : readback.pieces) {
        if (size == 0) {
            continue;
        }
        for (u64 window = address >> 20; window <= (address + size - 1) >> 20; ++window) {
            windows.push_back(window);
        }
    }
    std::ranges::sort(windows);
    const auto [first, last] = std::ranges::unique(windows);
    windows.erase(first, last);
    for (const u64 window : windows) {
        if (add) {
            ++pending_readback_windows[window];
        } else if (const auto it = pending_readback_windows.find(window);
                   it != pending_readback_windows.end() && --it->second == 0) {
            pending_readback_windows.erase(it);
        }
    }
}

void BufferCache::PeekGpuRange(VAddr address, u64 size, u64 tag, u32 slot, u64 dispatch, u32 phase) {
    if (size == 0 || size > 1_MB || peeks.size() > 8192) {
        return;
    }
    const u64 first_block = address >> block_shift;
    const u64 last_block = (address + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    if (!arena) {
        return;
    }
    static const auto t0 = std::chrono::steady_clock::now();
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostCached, 0, true);
    const vk::BufferCopy copy = {
        .srcOffset = address - arena->cpu_addr,
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(arena, staging.buffer, std::span{&copy, 1});
    peeks.push_back({scheduler.CurrentTick(), staging.buffer, staging.offset, staging.size, staging.mapped,
                     address, size, tag, slot,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                     dispatch, phase});
}

void BufferCache::PeekGpuBuffer(const Buffer* source, u64 offset, VAddr address, u64 size, u64 tag,
                                u32 slot, u64 dispatch, u32 phase) {
    if (!source || size == 0 || size > 1_MB || peeks.size() > 8192) {
        return;
    }
    static const auto t0 = std::chrono::steady_clock::now();
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostCached, 0, true);
    const vk::BufferCopy copy = {
        .srcOffset = offset,
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(source, staging.buffer, std::span{&copy, 1});
    peeks.push_back({scheduler.CurrentTick(), staging.buffer, staging.offset, staging.size, staging.mapped,
                     address, size, tag, slot,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                     dispatch, phase});
}

// gow3 26 Oct 2026 diagnostics (GOW3_BONE_TRACE=1): see buffer_cache.h.
bool BufferCache::BoneTraceEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_BONE_TRACE");
        const bool enabled = env && (env[0] == '1' || env[0] == '2');
        if (enabled) {
            std::printf("GPU: %s (GOW3_BONE_TRACE=%c)\n",
                        env[0] == '2' ? "the skinning's vertex outputs are watched for stretches, "
                                        "with the bones they read (light trace)"
                                      : "the skinning's bone bindings (cs 45df1f62) are checked "
                                        "against guest memory once the GPU has run them",
                        env[0]);
            std::fflush(stdout);
        }
        return enabled;
    }();
    return on;
}

// GOW3_BONE_TRACE=2: the hair watch and the bones of slots 6 and 7 only (what the skinning read,
// not checked against guest memory); no copies of the physics shader's bindings.
bool BufferCache::BoneTraceLight() {
    static const bool light = [] {
        const char* env = std::getenv("GOW3_BONE_TRACE");
        return env && env[0] == '2';
    }();
    return light;
}

static s64 TraceUs() {
    static const auto t0 = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
        .count();
}
struct TracedFault {
    VAddr address = 0;
    s64 start_us = 0, end_us = 0;
    std::string note;
};
struct TracedJointWrite {
    VAddr address = 0;
    u64 size = 0;
    u32 slot = 0;
    u64 tick = 0;
    s64 us = 0;
};
static std::mutex g_trace_mutex;
static std::array<TracedFault, 4096> g_traced_faults;
static u32 g_traced_fault_next = 0;
static std::array<TracedJointWrite, 1024> g_traced_joint_writes;
static u32 g_traced_joint_next = 0;

void BufferCache::NoteTracedFault(VAddr address, u64 total_ns, const std::string& note) {
    const s64 end = TraceUs();
    std::scoped_lock lock{g_trace_mutex};
    auto& entry = g_traced_faults[g_traced_fault_next++ % g_traced_faults.size()];
    entry.address = address;
    entry.start_us = end - static_cast<s64>(total_ns / 1000);
    entry.end_us = end;
    entry.note = note;
}

void BufferCache::NoteJointWrite(VAddr address, u64 size, u32 slot, u64 tick) {
    const s64 now = TraceUs();
    std::scoped_lock lock{g_trace_mutex};
    g_traced_joint_writes[g_traced_joint_next++ % g_traced_joint_writes.size()] = {address, size, slot,
                                                                                   tick, now};
}

void BufferCache::PeekBones(const Buffer* source, u64 offset, VAddr address, u64 size, u32 slot,
                            u64 dispatch, u64 shader, bool written) {
    // The earlier ones the GPU has run are checked first: as soon as possible after the GPU ran
    // them (a later check takes the game's writes after that, harmless, for late ones).
    PollBonePeeks();
    if (!source || size == 0 || size > 16_KB || bone_peeks.size() > 1024) {
        return;
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostCached, 0, true);
    const vk::BufferCopy copy = {
        .srcOffset = offset,
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(source, staging.buffer, std::span{&copy, 1});
    BonePeek peek{
        .tick = scheduler.CurrentTick(),
        .buffer = staging.buffer,
        .offset = staging.offset,
        .staging_size = staging.size,
        .mapped = staging.mapped,
        .address = address,
        .size = size,
        .slot = slot,
        .bound_us = TraceUs(),
        .dispatch = dispatch,
        .shader = shader,
        .written = written,
        .source = source,
        .source_offset = offset,
    };
    // GOW3_BONE_TRACE=2: only what the skinning read is kept (for the bone changes of the hair
    // watch), not checked against guest memory.
    if (!BoneTraceLight()) {
        peek.guest.resize(size);
        if (!memory->TryReadBacking(address, peek.guest.data(), size)) {
            memory->CopySparseMemory(address, peek.guest.data(), size);
        }
        peek.gpu_owned.assign(size, 0);
        gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr lo, VAddr hi) {
            for (VAddr byte = lo; byte < hi; ++byte) {
                peek.gpu_owned[byte - address] = 1;
            }
        });
        peek.pending_readback = PendingReadbackTick(address, size, false) != 0;
        peek.shared = lazy_pages.contains(Common::AlignDown(address, LAZY_PAGE_SIZE)) ||
                      lazy_pages.contains(Common::AlignDown(address + size - 1, LAZY_PAGE_SIZE));
        peek.result_page = IsResultPage(Common::AlignDown(address, LAZY_PAGE_SIZE)) ||
                           IsResultPage(Common::AlignDown(address + size - 1, LAZY_PAGE_SIZE));
    }
    bone_peeks.push_back(std::move(peek));
    if (written) {
        // (A deque's push_back and pop_front keep references to the other elements.)
        bone_peeks_after.push_back(&bone_peeks.back());
    }
}

void BufferCache::PeekBonesAfter() {
    for (BonePeek* peek : bone_peeks_after) {
        const auto staging = staging_pool.Request(peek->size, VideoCore::MemoryType::HostCached, 0, true);
        const vk::BufferCopy copy = {
            .srcOffset = peek->source_offset,
            .dstOffset = staging.offset,
            .size = peek->size,
        };
        runtime.CopyBuffer(peek->source, staging.buffer, std::span{&copy, 1});
        peek->post_buffer = staging.buffer;
        peek->post_offset = staging.offset;
        peek->post_staging_size = staging.size;
        peek->post_mapped = staging.mapped;
    }
    bone_peeks_after.clear();
}

void BufferCache::PeekSkinOutput(const Buffer* source, u64 offset, VAddr address, u64 size,
                                 u64 dispatch) {
    if (!source || size == 0 || size > 256_KB || size % 52 != 0 || skin_peeks.size() > 512) {
        return;
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostCached, 0, true);
    const vk::BufferCopy copy = {
        .srcOffset = offset,
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(source, staging.buffer, std::span{&copy, 1});
    skin_peeks.push_back({scheduler.CurrentTick(), staging.buffer, staging.offset, staging.size,
                          staging.mapped, address, size, dispatch, TraceUs()});
}

// The positions (the first 3 floats of each 52-byte vertex) against the previous output of the
// same buffer: the extent of the mesh and how far vertices moved. A stretch: the extent or a
// vertex's move well past what that buffer usually shows.
void BufferCache::CheckSkinOutput(const SkinPeek& peek) {
    struct History {
        std::vector<float> previous;
        std::deque<float> extents, jumps;
        u64 previous_dispatch = 0;
    };
    static std::map<std::pair<VAddr, u64>, History> history;
    static u32 checked = 0, spikes = 0;
    static auto last = std::chrono::steady_clock::now();
    static u32 printed = 0;
    const u64 count = peek.size / 52;
    std::vector<float> positions(count * 3);
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    u64 finite = 0;
    for (u64 v = 0; v < count; ++v) {
        float p[3];
        std::memcpy(p, peek.mapped + v * 52, 12);
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            ok &= std::isfinite(p[k]) && std::abs(p[k]) < 1e5f;
        }
        for (int k = 0; k < 3; ++k) {
            positions[v * 3 + k] = ok ? p[k] : 0.0f;
            if (ok) {
                lo[k] = std::min(lo[k], p[k]);
                hi[k] = std::max(hi[k], p[k]);
            }
        }
        finite += ok;
    }
    if (finite == 0) {
        return;
    }
    const float extent = std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) +
                                   (hi[2] - lo[2]) * (hi[2] - lo[2]));
    auto& h = history[{peek.address, peek.size}];
    float max_jump = 0.0f;
    u64 max_vertex = 0;
    if (h.previous.size() == positions.size()) {
        for (u64 v = 0; v < count; ++v) {
            const float dx = positions[v * 3] - h.previous[v * 3];
            const float dy = positions[v * 3 + 1] - h.previous[v * 3 + 1];
            const float dz = positions[v * 3 + 2] - h.previous[v * 3 + 2];
            const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d > max_jump) {
                max_jump = d;
                max_vertex = v;
            }
        }
    }
    const auto median = [](const std::deque<float>& values) {
        std::vector<float> sorted(values.begin(), values.end());
        std::ranges::sort(sorted);
        return sorted.empty() ? 0.0f : sorted[sorted.size() / 2];
    };
    ++checked;
    // The median change of the bones' 16-byte rows (slots 6 and 7) since the previous skinning
    // of this buffer, for every check: how often the bones jump as a whole.
    static std::array<std::array<u32, 4>, 2> row_change{}; // [slot 6/7][<0.05, <0.2, <1, more]
    static std::array<std::array<u32, 4>, 2> row_change_stretched{};
    const auto slot_change = [&](u32 slot) -> float {
        const std::vector<BoneRange>* now_ranges = nullptr;
        const std::vector<BoneRange>* before_ranges = nullptr;
        for (const auto& [d, ranges] : recent_bone_ranges) {
            if (d == peek.dispatch) {
                now_ranges = &ranges;
            } else if (d == h.previous_dispatch) {
                before_ranges = &ranges;
            }
        }
        if (!now_ranges || !before_ranges) {
            return -1.0f;
        }
        for (const auto& range : *now_ranges) {
            for (const auto& old_range : *before_ranges) {
                if (range.slot != slot || old_range.slot != slot || range.read.empty() ||
                    old_range.read.size() != range.read.size()) {
                    continue;
                }
                std::vector<float> rows;
                for (u32 r = 0; r < range.read.size() / 16; ++r) {
                    float a[4], b[4], d = 0.0f;
                    std::memcpy(a, range.read.data() + r * 16, 16);
                    std::memcpy(b, old_range.read.data() + r * 16, 16);
                    for (int k = 0; k < 4; ++k) {
                        if (std::isfinite(a[k]) && std::isfinite(b[k])) {
                            d = std::max(d, std::abs(a[k] - b[k]));
                        }
                    }
                    rows.push_back(d);
                }
                if (rows.empty()) {
                    return -1.0f;
                }
                std::ranges::nth_element(rows, rows.begin() + rows.size() / 2);
                return rows[rows.size() / 2];
            }
        }
        return -1.0f;
    };
    const float change6 = slot_change(6), change7 = slot_change(7);
    const auto bucket = [](float c) { return c < 0.05f ? 0 : c < 0.2f ? 1 : c < 1.0f ? 2 : 3; };
    bool stretched_now = false;
    if (h.extents.size() >= 8) {
        const float usual_extent = median(h.extents);
        const float usual_jump = median(h.jumps);
        // The whole mesh jumping (a camera cut, a teleport) moves every vertex: only a stretch
        // grows the extent too.
        const bool stretched = extent > usual_extent * 1.3f && extent - usual_extent > 0.02f &&
                               max_jump > std::max(usual_jump * 4.0f, usual_extent * 0.2f);
        stretched_now = stretched;
        if (stretched) {
            ++spikes;
            // A frame dump (GOW3_FRAME_DUMP_TRIGGER, probe_tlou.py --capture) of the next frame
            // shown, to see the stretch: at most 12, a second apart.
            static const char* dump_trigger = std::getenv("GOW3_FRAME_DUMP_TRIGGER");
            static u32 dumps = 0;
            static s64 last_dump_us = 0;
            if (dump_trigger && dumps < 12 && TraceUs() - last_dump_us > 1'000'000) {
                ++dumps;
                last_dump_us = TraceUs();
                if (std::FILE* f = std::fopen(dump_trigger, "wb")) {
                    std::fclose(f);
                }
                std::printf("Hair watch: frame dump %u requested at %.3f s\n", dumps, last_dump_us / 1e6);
            }
            if (printed < 200) {
                ++printed;
                std::string bones = " (not kept)";
                for (const auto& [dispatch, text] : recent_bone_reads) {
                    if (dispatch == peek.dispatch) {
                        bones = text;
                    }
                }
                // The physics writes into each bone binding around the skinning's bind, by time
                // and tick: one recorded after the skinning (or in a later tick) means the
                // skinning read the slot before the physics wrote it this frame.
                std::string order;
                for (const auto& [dispatch, ranges] : recent_bone_ranges) {
                    if (dispatch != peek.dispatch) {
                        continue;
                    }
                    for (const auto& range : ranges) {
                        if (range.slot != 6 && range.slot != 7) {
                            continue;
                        }
                        std::vector<std::pair<s64, std::string>> writes;
                        {
                            std::scoped_lock lock{g_trace_mutex};
                            for (const auto& w : g_traced_joint_writes) {
                                if (w.us && w.address < range.address + range.size &&
                                    range.address < w.address + w.size &&
                                    w.us >= range.bound_us - 300'000 && w.us <= range.bound_us + 300'000) {
                                    writes.emplace_back(w.us, fmt::format(" {:+.1f} ms tick {} (slot {})",
                                                                          (w.us - range.bound_us) / 1e3,
                                                                          w.tick, w.slot));
                                }
                            }
                        }
                        std::ranges::sort(writes);
                        order += fmt::format(" [bones slot {} {:#x} bound at tick {}; physics writes:",
                                             range.slot, range.address, range.tick);
                        for (const auto& [t, text] : writes) {
                            order += text;
                        }
                        order += writes.empty() ? " none]" : "]";
                    }
                }
                bones += order;
                bones += PhysicsInputDiff(peek.dispatch, h.previous_dispatch);
                // The bones it read against those the previous skinning of this buffer read:
                // the 16-byte rows (a matrix row) that changed most.
                const auto find_ranges = [&](u64 dispatch) -> const std::vector<BoneRange>* {
                    for (const auto& [d, ranges] : recent_bone_ranges) {
                        if (d == dispatch) {
                            return &ranges;
                        }
                    }
                    return nullptr;
                };
                const auto* now_ranges = find_ranges(peek.dispatch);
                const auto* before_ranges = find_ranges(h.previous_dispatch);
                if (now_ranges && before_ranges) {
                    for (const auto& range : *now_ranges) {
                        for (const auto& old_range : *before_ranges) {
                            if (old_range.slot != range.slot || range.read.empty() ||
                                old_range.read.size() != range.read.size()) {
                                continue;
                            }
                            // Per 16-byte row: the largest change of its 4 floats.
                            std::vector<std::pair<float, u32>> rows;
                            const u32 n_rows = static_cast<u32>(range.read.size() / 16);
                            float typical = 0.0f;
                            for (u32 r = 0; r < n_rows; ++r) {
                                float a[4], b[4], d = 0.0f;
                                std::memcpy(a, range.read.data() + r * 16, 16);
                                std::memcpy(b, old_range.read.data() + r * 16, 16);
                                for (int k = 0; k < 4; ++k) {
                                    if (std::isfinite(a[k]) && std::isfinite(b[k])) {
                                        d = std::max(d, std::abs(a[k] - b[k]));
                                    }
                                }
                                rows.emplace_back(d, r);
                            }
                            std::ranges::sort(rows, std::greater{});
                            if (!rows.empty()) {
                                typical = rows[rows.size() / 2].first;
                            }
                            bones += fmt::format(" [slot {} rows changed most since the last skinning "
                                                 "of this buffer (median {:.4f}):",
                                                 range.slot, typical);
                            for (size_t i = 0; i < std::min<size_t>(rows.size(), 4); ++i) {
                                float a[4];
                                std::memcpy(a, range.read.data() + rows[i].second * 16, 16);
                                bones += fmt::format(" row {} by {:.3f} ({:.3f} {:.3f} {:.3f} {:.3f})",
                                                     rows[i].second, rows[i].first, a[0], a[1], a[2],
                                                     a[3]);
                            }
                            bones += "]";
                        }
                    }
                }
                std::printf("Hair watch: at %.1f s, skinning output %#llx+%llu (dispatch %llu, tick "
                            "%llu) stretched: extent %.3f (usually %.3f), vertex %llu moved %.3f "
                            "(usually at most %.3f); its bone bindings:%s\n",
                            TraceUs() / 1e6, static_cast<unsigned long long>(peek.address),
                            static_cast<unsigned long long>(peek.size),
                            static_cast<unsigned long long>(peek.dispatch),
                            static_cast<unsigned long long>(peek.tick), extent, usual_extent,
                            static_cast<unsigned long long>(max_vertex), max_jump, usual_jump,
                            bones.c_str());
                std::fflush(stdout);
            }
        }
    }
    // (GOW3_BONE_TRACE=1) the same physics comparison for a few unstretched hair skinnings.
    static u32 baseline_printed = 0, baseline_count = 0;
    if (!stretched_now && peek.size == 144300 && !BoneTraceLight() && baseline_printed < 4 &&
        ++baseline_count % 200 == 0) {
        ++baseline_printed;
        std::printf("Hair watch: baseline, skinning output %#llx+%llu (dispatch %llu) not "
                    "stretched:%s\n",
                    static_cast<unsigned long long>(peek.address),
                    static_cast<unsigned long long>(peek.size),
                    static_cast<unsigned long long>(peek.dispatch),
                    PhysicsInputDiff(peek.dispatch, h.previous_dispatch).c_str());
        std::fflush(stdout);
    }
    for (u32 i = 0; i < 2; ++i) {
        const float c = i == 0 ? change6 : change7;
        if (c >= 0.0f) {
            ++(stretched_now ? row_change_stretched : row_change)[i][bucket(c)];
        }
    }
    h.extents.push_back(extent);
    h.jumps.push_back(max_jump);
    if (h.extents.size() > 31) {
        h.extents.pop_front();
        h.jumps.pop_front();
    }
    h.previous = std::move(positions);
    h.previous_dispatch = peek.dispatch;
    if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
        last = now;
        std::string text;
        for (const auto& [key, entry] : history) {
            if (!entry.extents.empty()) {
                text += fmt::format(" {:#x}+{}: extent {:.3f}, move {:.3f};", key.first, key.second,
                                    median(entry.extents), median(entry.jumps));
            }
        }
        std::printf("Hair watch (5 s): %u skinning outputs checked, %u stretched; bones' median row "
                    "change since the last skinning of the buffer (<0.05/<0.2/<1/more): slot 6 "
                    "%u/%u/%u/%u, slot 7 %u/%u/%u/%u; when stretched: slot 6 %u/%u/%u/%u, slot 7 "
                    "%u/%u/%u/%u; usual by buffer:%s\n",
                    checked, spikes, row_change[0][0], row_change[0][1], row_change[0][2],
                    row_change[0][3], row_change[1][0], row_change[1][1], row_change[1][2],
                    row_change[1][3], row_change_stretched[0][0], row_change_stretched[0][1],
                    row_change_stretched[0][2], row_change_stretched[0][3],
                    row_change_stretched[1][0], row_change_stretched[1][1],
                    row_change_stretched[1][2], row_change_stretched[1][3], text.c_str());
        row_change = {};
        row_change_stretched = {};
        std::fflush(stdout);
        checked = spikes = 0;
        if (history.size() > 64) {
            history.clear();
        }
    }
}

// The bytes a physics dispatch read from a binding against what the last physics dispatch wrote
// there: the simulation's state lives in GPU memory from frame to frame, so a difference is
// someone else changing it (the game resetting it, or an upload of guest memory over it).
void BufferCache::CheckPhysicsState(const BonePeek& peek, const std::vector<u8>& guest_now) {
    static u32 checked = 0, changed = 0, printed = 0;
    static auto last = std::chrono::steady_clock::now();
    const auto key = std::make_pair(peek.address, peek.size);
    if (const auto it = last_physics_writes.find(key); it != last_physics_writes.end()) {
        const auto& before = it->second.bytes;
        u32 n = 0, first = 0, last_byte = 0, zero_now = 0, guest_equal = 0;
        for (u64 i = 0; i < peek.size && i < before.size(); ++i) {
            if (peek.mapped[i] != before[i]) {
                if (n++ == 0) {
                    first = static_cast<u32>(i);
                }
                last_byte = static_cast<u32>(i);
                zero_now += peek.mapped[i] == 0;
                // The read equals the guest's copy: guest memory went to the GPU (an upload).
                guest_equal += !peek.guest.empty() && peek.mapped[i] == peek.guest[i];
            }
        }
        ++checked;
        if (n) {
            ++changed;
            if (printed < 40) {
                ++printed;
                std::string faults;
                {
                    std::scoped_lock lock{g_trace_mutex};
                    std::vector<std::pair<s64, std::string>> list;
                    for (const auto& f : g_traced_faults) {
                        if (f.end_us && f.address >= Common::AlignDown(peek.address, 4096) &&
                            f.address < Common::AlignUp(peek.address + peek.size, 4096) &&
                            f.end_us >= it->second.us - 50'000 && f.start_us <= peek.bound_us) {
                            list.emplace_back(f.start_us,
                                              fmt::format(" [{:#x} {:+.1f} ms: {}]", f.address,
                                                          (f.start_us - peek.bound_us) / 1e3, f.note));
                        }
                    }
                    std::ranges::sort(list);
                    for (const auto& [t, text] : list) {
                        faults += text;
                    }
                }
                u32 read_word = 0, before_word = 0, guest_word = 0;
                const u32 word = first & ~3u;
                if (word + 4 <= peek.size && word + 4 <= before.size()) {
                    std::memcpy(&read_word, peek.mapped + word, 4);
                    std::memcpy(&before_word, before.data() + word, 4);
                    if (word + 4 <= peek.guest.size()) {
                        std::memcpy(&guest_word, peek.guest.data() + word, 4);
                    }
                }
                std::printf("Physics state: slot %u %#llx+%llu (tick %llu) read %u bytes unlike the last "
                            "physics write there (tick %llu, %.1f ms before; offsets %#x-%#x, %u now "
                            "zero, %u equal to the guest's copy at the bind; dword %#x: read %08x, "
                            "written %08x, guest %08x); %s; CPU write faults on its pages since:%s\n",
                            peek.slot, static_cast<unsigned long long>(peek.address),
                            static_cast<unsigned long long>(peek.size),
                            static_cast<unsigned long long>(peek.tick), n,
                            static_cast<unsigned long long>(it->second.tick),
                            (peek.bound_us - it->second.us) / 1e3, first, last_byte, zero_now,
                            guest_equal, word, read_word, before_word, guest_word,
                            peek.shared ? "shared page" : "not shared",
                            faults.empty() ? " none" : faults.c_str());
                std::fflush(stdout);
            }
        }
    }
    if (peek.written && peek.post_mapped) {
        last_physics_writes[key] = {std::vector<u8>(peek.post_mapped, peek.post_mapped + peek.size),
                                    peek.bound_us, peek.tick};
        if (last_physics_writes.size() > 4096) {
            last_physics_writes.clear();
        }
    }
    if (const auto now = std::chrono::steady_clock::now(); checked && now - last >= std::chrono::seconds{5}) {
        last = now;
        std::printf("Physics state (5 s): %u reads of bindings the physics wrote before, %u of them "
                    "unlike its last write\n",
                    checked, changed);
        std::fflush(stdout);
        checked = changed = 0;
    }
    (void)guest_now;
}

std::string BufferCache::PhysicsInputDiff(u64 skin_dispatch, u64 previous_skin_dispatch) {
    const auto slot7_address = [&](u64 dispatch) -> VAddr {
        for (const auto& [d, ranges] : recent_bone_ranges) {
            if (d == dispatch) {
                for (const auto& range : ranges) {
                    if (range.slot == 7) {
                        return range.address;
                    }
                }
            }
        }
        return 0;
    };
    const auto writers = [&](VAddr address) {
        std::vector<std::pair<u64, const std::vector<PhysicsBinding>*>> out;
        for (const auto& [d, bindings] : recent_physics) {
            for (const auto& b : bindings) {
                if (address && b.written && b.address == address) {
                    out.emplace_back(d, &bindings);
                    break;
                }
            }
        }
        return out;
    };
    const VAddr now_address = slot7_address(skin_dispatch);
    const VAddr before_address = slot7_address(previous_skin_dispatch);
    const auto now_writers = writers(now_address);
    const auto before_writers = writers(before_address);
    std::string text = fmt::format(" [physics writers of slot 7 {:#x}: {}, two frames before ({:#x}): {}",
                                   now_address, now_writers.size(), before_address,
                                   before_writers.size());
    const auto diff = [](const std::vector<u8>& a, const std::vector<u8>& b, float& max_diff,
                         u32& at, u32& big, float& value_now, float& value_before) {
        max_diff = 0.0f;
        at = 0;
        big = 0;
        for (u32 i = 0; i + 4 <= a.size() && i + 4 <= b.size(); i += 4) {
            float x, y;
            std::memcpy(&x, a.data() + i, 4);
            std::memcpy(&y, b.data() + i, 4);
            if (!std::isfinite(x) || !std::isfinite(y)) {
                continue;
            }
            const float d = std::abs(x - y);
            big += d > 0.5f;
            if (d > max_diff) {
                max_diff = d;
                at = i / 4;
                value_now = x;
                value_before = y;
            }
        }
    };
    for (size_t k = 0; k < std::min(now_writers.size(), before_writers.size()); ++k) {
        const auto& now_bindings = *now_writers[k].second;
        const auto& before_bindings = *before_writers[k].second;
        std::vector<std::pair<float, std::string>> slots;
        for (const auto& b : now_bindings) {
            for (const auto& o : before_bindings) {
                if (o.slot != b.slot || o.size != b.size) {
                    continue;
                }
                float max_pre = 0, now_pre = 0, before_pre = 0;
                u32 at_pre = 0, big_pre = 0;
                diff(b.pre, o.pre, max_pre, at_pre, big_pre, now_pre, before_pre);
                std::string entry = fmt::format(
                    " slot {} ({} B{}{}): read differs most at float {} ({:.4g} vs {:.4g}), {} "
                    "floats by > 0.5",
                    b.slot, b.size, b.written ? ", written" : "",
                    b.address == o.address ? ", same address" : "", at_pre, now_pre, before_pre,
                    big_pre);
                float max_post = 0;
                if (b.written && !b.post.empty() && !o.post.empty()) {
                    float now_post = 0, before_post = 0;
                    u32 at_post = 0, big_post = 0;
                    diff(b.post, o.post, max_post, at_post, big_post, now_post, before_post);
                    entry += fmt::format("; wrote: most at float {} ({:.4g} vs {:.4g}), {} by > 0.5",
                                         at_post, now_post, before_post, big_post);
                }
                slots.emplace_back(std::max(max_pre, max_post), std::move(entry));
            }
        }
        std::ranges::sort(slots, [](const auto& a, const auto& b) { return a.first > b.first; });
        text += fmt::format(" | dispatch {} vs {}:", now_writers[k].first, before_writers[k].first);
        for (size_t i = 0; i < std::min<size_t>(slots.size(), 8); ++i) {
            text += slots[i].second + ";";
        }
    }
    return text + "]";
}

void BufferCache::PollBonePeeks() {
    static u32 checked = 0, late = 0, late_shared = 0, late_pending = 0, late_gpu_bytes = 0;
    // Binds on physics result pages (the joint slots: the hair's bones) checked and late; and
    // late ones elsewhere whose changed bytes are all zero now (per-frame ring memory the game
    // cleared after the emulator's fence, which comes before the GPU runs: harmless).
    static u32 result_checked = 0, result_late = 0, cleared_late = 0, other_details = 0;
    static u64 late_bytes = 0;
    static std::map<u32, std::pair<u32, u32>> by_slot; // slot -> (checked, late)
    static std::vector<std::string> details;
    static u32 details_total = 0;
    // What each recent skinning dispatch's bone bindings read (for CheckSkinOutput).
    auto& recent = recent_bone_reads;
    static auto last = std::chrono::steady_clock::now();
    static s64 last_refresh_us = 0;
    while (!bone_peeks.empty()) {
        if (!scheduler.IsFree(bone_peeks.front().tick)) {
            // The cached GPU tick lags: refreshed at most every 0.5 ms (an ioctl).
            const s64 now_us = TraceUs();
            if (now_us - last_refresh_us < 500) {
                break;
            }
            last_refresh_us = now_us;
            scheduler.GetWorkSemaphore()->Refresh();
            if (!scheduler.IsFree(bone_peeks.front().tick)) {
                break;
            }
        }
        BonePeek& peek = bone_peeks.front();
        if (peek.written && !peek.post_mapped) {
            break; // its copy after the dispatch is not recorded yet
        }
        peek.buffer->Invalidate(peek.offset, peek.staging_size);
        if (peek.post_mapped) {
            peek.post_buffer->Invalidate(peek.post_offset, peek.post_staging_size);
        }
        std::vector<u8> now(peek.guest.empty() ? 0 : peek.size); // none: not checked (light)
        if (!now.empty() && !memory->TryReadBacking(peek.address, now.data(), peek.size)) {
            memory->CopySparseMemory(peek.address, now.data(), peek.size);
        }
        const s64 checked_us = TraceUs();
        // A byte the skinning read older than the guest's: it differs from what the GPU read,
        // unless it was GPU data whose guest copy has not changed since (its readback has not
        // come yet: the guest's copy is the older one), or the guest holds the poison of a shared
        // page there (GPU data too).
        u32 n = 0, first = 0, last_byte = 0, gpu_bytes = 0;
        bool cleared = true;
        for (u64 i = 0; i < now.size(); ++i) {
            const u8 used = peek.mapped[i];
            if (now[i] != used && !(peek.gpu_owned[i] && now[i] == peek.guest[i]) &&
                now[i] != PoisonByte((peek.address + i) & (LAZY_PAGE_SIZE - 1)) &&
                !(peek.post_mapped && now[i] == peek.post_mapped[i])) {
                if (n++ == 0) {
                    first = static_cast<u32>(i);
                }
                last_byte = static_cast<u32>(i);
                gpu_bytes += peek.gpu_owned[i];
                cleared &= now[i] == 0;
            }
        }
        const u32 slot_key = peek.shader == 0x106beafd ? 1000 + peek.slot : peek.slot;
        ++checked;
        ++by_slot[slot_key].first;
        result_checked += peek.result_page;
        if (n) {
            result_late += peek.result_page;
            cleared_late += !peek.result_page && cleared;
        }
        if (recent.empty() || recent.back().first != peek.dispatch) {
            recent.emplace_back(peek.dispatch, std::string{});
            if (recent.size() > 256) {
                recent.pop_front();
            }
        }
        if (recent_bone_ranges.empty() || recent_bone_ranges.back().first != peek.dispatch) {
            recent_bone_ranges.emplace_back(peek.dispatch, std::vector<BoneRange>{});
            if (recent_bone_ranges.size() > 256) {
                recent_bone_ranges.pop_front();
            }
        }
        if (peek.shader == 0x106beafd) {
            CheckPhysicsState(peek, now);
            if (recent_physics.empty() || recent_physics.back().first != peek.dispatch) {
                recent_physics.emplace_back(peek.dispatch, std::vector<PhysicsBinding>{});
                if (recent_physics.size() > 128) {
                    recent_physics.pop_front();
                }
            }
            recent_physics.back().second.push_back(
                {peek.slot, peek.address, peek.size, peek.written,
                 std::vector<u8>(peek.mapped, peek.mapped + peek.size),
                 peek.post_mapped ? std::vector<u8>(peek.post_mapped, peek.post_mapped + peek.size)
                                  : std::vector<u8>{}});
        }
        recent_bone_ranges.back().second.push_back(
            {peek.address, peek.size, peek.slot, peek.bound_us, peek.tick,
             peek.shader == 0x45df1f62 && (peek.slot == 6 || peek.slot == 7)
                 ? std::vector<u8>(peek.mapped, peek.mapped + peek.size)
                 : std::vector<u8>{}});
        recent.back().second += fmt::format(
            " [slot {} {:#x}+{}{}{}: {}]", peek.slot, peek.address, peek.size,
            peek.result_page ? " result page" : "", peek.pending_readback ? " readback pending" : "",
            n ? fmt::format("{} bytes older than the guest's at {:#x}-{:#x}", n, first, last_byte)
              : std::string{"as the guest"});
        // Details: every late one of the skinning on a result page (up to 10 per 5 s), two
        // others. (The physics shader's are counted only: it writes its joint slots in two
        // dispatches and their readbacks land after the check, so its late bytes are its own.)
        const bool detail = n && peek.shader != 0x106beafd &&
                            (peek.result_page || (!cleared && other_details < 2));
        if (n) {
            ++late;
            late_bytes += n;
            late_gpu_bytes += gpu_bytes;
            ++by_slot[slot_key].second;
            late_shared += peek.shared;
            late_pending += peek.pending_readback;
            if (detail && details.size() < 10 && details_total < 400) {
                ++details_total;
                other_details += !peek.result_page;
                std::vector<std::pair<s64, std::string>> faults, writes;
                {
                    std::scoped_lock lock{g_trace_mutex};
                    for (const auto& f : g_traced_faults) {
                        if (f.end_us && f.address >= peek.address && f.address < peek.address + peek.size &&
                            f.end_us >= peek.bound_us - 100'000 && f.start_us <= checked_us) {
                            faults.emplace_back(f.start_us,
                                                fmt::format(" [{:#x} {:+.1f}..{:+.1f} ms: {}]", f.address,
                                                            (f.start_us - peek.bound_us) / 1e3,
                                                            (f.end_us - peek.bound_us) / 1e3, f.note));
                        }
                    }
                    for (const auto& w : g_traced_joint_writes) {
                        if (w.us && w.address < peek.address + peek.size &&
                            peek.address < w.address + w.size && w.us >= peek.bound_us - 200'000 &&
                            w.us <= checked_us) {
                            writes.emplace_back(w.us, fmt::format(" [slot {} {:#x}+{} {:+.1f} ms, tick {}]",
                                                                  w.slot, w.address, w.size,
                                                                  (w.us - peek.bound_us) / 1e3, w.tick));
                        }
                    }
                }
                std::ranges::sort(faults);
                std::ranges::sort(writes);
                std::string fault_text, write_text;
                for (const auto& [t, text] : faults) {
                    fault_text += text;
                }
                for (const auto& [t, text] : writes) {
                    write_text += text;
                }
                const u32 word = first & ~3u;
                u32 used_word = 0, now_word = 0, bound_word = 0;
                if (word + 4 <= peek.size) {
                    std::memcpy(&used_word, peek.mapped + word, 4);
                    std::memcpy(&now_word, now.data() + word, 4);
                    std::memcpy(&bound_word, peek.guest.data() + word, 4);
                }
                details.push_back(fmt::format(
                    "Bone trace: {} slot {} {:#x}+{} (tick {}{}{}{}): {} bytes read older than the guest's "
                    "({} of them GPU data then), offsets {:#x}-{:#x} (dword {:#x}: read {:08x}, guest "
                    "at the bind {:08x}, now {:08x}); the GPU done by {:+.1f} ms after the bind; CPU write "
                    "faults into it:{}; physics writes into it:{}",
                    peek.shader == 0x106beafd ? "physics 106beafd" : "skinning", peek.slot,
                    peek.address, peek.size, peek.tick,
                    peek.result_page ? ", physics result page" : "", peek.shared ? ", shared page" : "",
                    peek.pending_readback ? ", readback pending" : "", n, gpu_bytes, first, last_byte,
                    word, used_word, bound_word, now_word, (checked_us - peek.bound_us) / 1e3,
                    fault_text.empty() ? " none" : fault_text, write_text.empty() ? " none" : write_text));
            }
        }
        staging_pool.FreeDeferred(Vulkan::StagingBufferRef{
            .buffer = peek.buffer,
            .offset = peek.offset,
            .size = peek.staging_size,
            .mapped = peek.mapped,
            .type = VideoCore::MemoryType::HostCached,
        });
        if (peek.post_mapped) {
            staging_pool.FreeDeferred(Vulkan::StagingBufferRef{
                .buffer = peek.post_buffer,
                .offset = peek.post_offset,
                .size = peek.post_staging_size,
                .mapped = peek.post_mapped,
                .type = VideoCore::MemoryType::HostCached,
            });
        }
        bone_peeks.pop_front();
    }
    while (!skin_peeks.empty() && scheduler.IsFree(skin_peeks.front().tick)) {
        SkinPeek& peek = skin_peeks.front();
        peek.buffer->Invalidate(peek.offset, peek.staging_size);
        CheckSkinOutput(peek);
        staging_pool.FreeDeferred(Vulkan::StagingBufferRef{
            .buffer = peek.buffer,
            .offset = peek.offset,
            .size = peek.staging_size,
            .mapped = peek.mapped,
            .type = VideoCore::MemoryType::HostCached,
        });
        skin_peeks.pop_front();
    }
    if (const auto now = std::chrono::steady_clock::now(); checked && now - last >= std::chrono::seconds{5}) {
        last = now;
        std::string slots;
        for (const auto& [slot, counts] : by_slot) {
            slots += fmt::format(" {}: {}/{}", slot, counts.second, counts.first);
        }
        std::printf("Bone trace (5 s): %u bone bindings of the skinning checked, %u read bytes older "
                    "than the guest's (%llu bytes, %u of them GPU data at the bind; %u on shared "
                    "pages, %u with a readback pending); on physics result pages (joint slots): "
                    "%u of %u; elsewhere cleared since (ring memory reused): %u; late/checked by "
                    "slot:%s\n",
                    checked, late, static_cast<unsigned long long>(late_bytes), late_gpu_bytes,
                    late_shared, late_pending, result_late, result_checked, cleared_late,
                    slots.c_str());
        for (const auto& line : details) {
            std::printf("%s\n", line.c_str());
        }
        std::fflush(stdout);
        checked = late = late_shared = late_pending = late_gpu_bytes = 0;
        result_checked = result_late = cleared_late = other_details = 0;
        late_bytes = 0;
        by_slot.clear();
        details.clear();
    }
}

void BufferCache::ResolvePoisonedPages(VAddr address, u64 size) {
    if (lazy_pages.empty() || size == 0) {
        return;
    }
    const VAddr end = address + size;
    for (auto it = lazy_pages.lower_bound(Common::AlignDown(address, LAZY_PAGE_SIZE));
         it != lazy_pages.end() && it->first < end; ++it) {
        LazyPage& record = *it->second;
        if (!record.poisoned) {
            continue;
        }
        const VAddr page = it->first;
        // A page the guest can still write keeps its record as it is (the new GPU bytes are
        // not tracked: as on a shared page before); poisoning them could hide a write.
        if (memory_tracker->IsRegionCpuModified(page, LAZY_PAGE_SIZE)) {
            g_poison_writable.fetch_add(1, std::memory_order_relaxed);
            NotePoisonEvent(page, "gpu-writable");
            continue;
        }
        // Uploaded just now (the bytes the game wrote went with it) and write-protected: the
        // new GPU bytes are poisoned too and the page's GPU bytes are read back after this
        // submission. Dropping the record here (first version) left the poison it held as
        // ordinary GPU data: a later share of the page (share_page, with a pending readback)
        // took the poison for the guest's copy and its 64-byte lines kept it in the game's
        // bytes (up to 1,760 dwords per 5 s, GOW3_SHARE_CHECK).
        std::array<u8, LAZY_PAGE_SIZE> contents;
        if (!memory->TryReadBacking(page, contents.data(), LAZY_PAGE_SIZE)) {
            continue;
        }
        const VAddr lo = std::max(page, address);
        const VAddr hi = std::min(page + LAZY_PAGE_SIZE, end);
        bool added = false;
        const std::array<u8, LAZY_PAGE_SIZE> before = contents;
        for (VAddr byte = lo; byte < hi; ++byte) {
            const size_t i = byte - page;
            if (!record.tracked.test(i)) {
                contents[i] = PoisonByte(i);
                added = true;
            }
        }
        if (added && memory->TryWriteBacking(std::bit_cast<u8*>(lo), contents.data() + (lo - page),
                                             hi - lo)) {
            for (VAddr byte = lo; byte < hi; ++byte) {
                const size_t i = byte - page;
                if (!record.tracked.test(i)) {
                    record.tracked.set(i);
                    record.original[i] = before[i];
                    record.bytes[i] = contents[i];
                }
            }
        }
        // Also when the write only covers bytes it tracks already: an earlier readback lands
        // them as rewritten (still poisoned), and only a later one brings them.
        unfinished_shared_readbacks.emplace_back(page, 4096);
        g_poison_resolved.fetch_add(1, std::memory_order_relaxed);
        NotePoisonEvent(page, fmt::format("gpu{}", added ? "+" : ""));
    }
}

void BufferCache::RestorePoison(VAddr lo, VAddr hi) {
    if (lazy_pages.empty() || lo >= hi) {
        return;
    }
    for (auto it = lazy_pages.lower_bound(Common::AlignDown(lo, LAZY_PAGE_SIZE));
         it != lazy_pages.end() && it->first < hi;) {
        LazyPage& record = *it->second;
        if (!record.poisoned) {
            ++it;
            continue;
        }
        const VAddr page = it->first;
        std::array<u8, LAZY_PAGE_SIZE> contents;
        if (!memory->TryReadBacking(page, contents.data(), LAZY_PAGE_SIZE)) {
            ++it;
            continue;
        }
        const VAddr a = std::max(page, lo);
        const VAddr b = std::min(page + LAZY_PAGE_SIZE, hi);
        bool restored = false;
        for (VAddr byte = a; byte < b; ++byte) {
            const size_t i = byte - page;
            if (!record.tracked.test(i)) {
                continue;
            }
            if (record.bytes[i] == PoisonByte(i) && contents[i] == PoisonByte(i)) {
                contents[i] = record.original[i];
                restored = true;
            }
            record.tracked.reset(i);
        }
        if (restored) {
            memory->TryWriteBacking(std::bit_cast<u8*>(a), contents.data() + (a - page), b - a);
            g_poison_restored.fetch_add(1, std::memory_order_relaxed);
        }
        NotePoisonEvent(page, "drop");
        if (restored) {
            ReportPoisonRestore("dropped readback piece", page);
        }
        it = record.tracked.none() ? lazy_pages.erase(it) : std::next(it);
    }
}

u32 BufferCache::ReadWaitLabel(VAddr address) const {
    if (page_manager && page_manager->IsReadWatched(address) &&
        !IsRangeGpuWrittenShared(address, sizeof(u32))) {
        u32 value;
        if (memory->TryReadBacking(address, &value, sizeof(value))) {
            g_label_backing_reads.fetch_add(1, std::memory_order_relaxed);
            return value;
        }
    }
    return *std::bit_cast<const u32*>(address);
}

bool BufferCache::IsResultPage(VAddr page) {
    std::scoped_lock lock{result_pages_mutex};
    const auto it = result_pages.find(page >> 12);
    return it != result_pages.end() &&
           std::chrono::steady_clock::now() - it->second < std::chrono::seconds{10};
}

void BufferCache::NoteResultPages(VAddr address, u64 size) {
    if (size == 0 || size > 1_MB) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{result_pages_mutex};
    if (result_pages.size() > 32768) {
        std::erase_if(result_pages, [&](const auto& entry) {
            return now - entry.second >= std::chrono::seconds{10};
        });
    }
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        result_pages[page] = now;
    }
}

static bool ShareCheckEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_SHARE_CHECK");
        return env && env[0] == '1';
    }();
    return on;
}

void BufferCache::NoteShareCheck(VAddr page, u32 kind) {
    if (!ShareCheckEnabled()) {
        return;
    }
    u64 writer = ~0ull;
    gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
        if (writer == ~0ull) {
            writer = NewestGpuWriteInto(lo, hi).writer;
        }
    });
    std::scoped_lock lock{share_checks_mutex};
    if (share_checks.size() < 1024) {
        share_checks.push_back({.page = page,
                                .since = std::chrono::steady_clock::now(),
                                .kind = kind,
                                .writer = writer});
    }
}

void BufferCache::RunShareChecks() {
    if (!ShareCheckEnabled()) {
        return;
    }
    struct Stats {
        u32 checked = 0, bad_pages = 0, never_clean = 0, poison_left = 0;
        u64 bad_bytes = 0;
    };
    static std::array<Stats, 2> stats{};
    static std::map<std::pair<u32, u64>, u32> bad_by_writer;
    static int printed = 0;
    static auto last = std::chrono::steady_clock::now();
    static const char* const kinds[] = {"waited", "shared"};
    // Copies that have landed: the CPU-owned bytes the GPU holds against the guest's.
    while (!share_checks_queued.empty() && scheduler.IsFree(share_checks_queued.front().tick)) {
        ShareCheck& check = share_checks_queued.front();
        check.buffer->Invalidate(check.offset, check.staging_size);
        u32 bad = 0, first = 0, last_byte = 0;
        for (u32 i = 0; i < 4096; ++i) {
            if (!check.gpu_owned.test(i) && check.mapped[i] != check.guest[i]) {
                if (bad++ == 0) {
                    first = i;
                }
                last_byte = i;
            }
        }
        auto& s = stats[check.kind];
        ++s.checked;
        // Poison left in the game's bytes (aligned 4-byte groups the GPU does not own).
        u32 poison_dwords = 0, first_poison = 0, poison_on_gpu = 0;
        for (u32 group = 0; group < 4096; group += 4) {
            bool poison = true;
            for (u32 i = group; i < group + 4 && poison; ++i) {
                poison = !check.gpu_owned.test(i) && check.guest[i] == PoisonByte(i);
            }
            if (poison) {
                if (poison_dwords++ == 0) {
                    first_poison = group;
                }
                poison_on_gpu += std::memcmp(check.mapped + group, check.guest.data() + group, 4) == 0;
            }
        }
        s.poison_left += poison_dwords;
        static int poison_printed = 0;
        if (poison_dwords && poison_printed < 16) {
            ++poison_printed;
            u32 gpu_owned = 0;
            for (u32 i = 0; i < 4096; ++i) {
                gpu_owned += check.gpu_owned.test(i);
            }
            std::printf("Share check: %s page %#llx (GPU bytes by %s, %u GPU-owned now): %u poison "
                        "dwords in the game's bytes from %#x, %u of them on the GPU too; history:%s\n",
                        kinds[check.kind], static_cast<unsigned long long>(check.page),
                        GpuWriterName(check.writer).c_str(), gpu_owned, poison_dwords,
                        first_poison, poison_on_gpu, PoisonHistory(check.page).c_str());
            std::fflush(stdout);
        }
        if (bad) {
            ++s.bad_pages;
            s.bad_bytes += bad;
            ++bad_by_writer[{check.kind, check.writer}];
            if (printed < 24) {
                ++printed;
                std::printf("Share check: %s page %#llx (GPU bytes by %s): %u CPU-owned bytes differ "
                            "on the GPU, offsets %#x-%#x, first guest %02x GPU %02x\n",
                            kinds[check.kind], static_cast<unsigned long long>(check.page),
                            GpuWriterName(check.writer).c_str(), bad, first, last_byte,
                            check.guest[first], check.mapped[first]);
                std::fflush(stdout);
            }
        }
        staging_pool.FreeDeferred(Vulkan::StagingBufferRef{
            .buffer = check.buffer,
            .offset = check.offset,
            .size = check.staging_size,
            .mapped = check.mapped,
            .type = VideoCore::MemoryType::HostCached,
        });
        share_checks_queued.pop_front();
    }
    // Pages due (200 ms after the fault) and clean: their GPU copy is queued now, with the guest
    // page as it is (write-protected while clean, so the GPU copy must match it).
    const auto now = std::chrono::steady_clock::now();
    std::vector<ShareCheck> due;
    {
        std::scoped_lock lock{share_checks_mutex};
        while (!share_checks.empty() && due.size() < 32 &&
               now - share_checks.front().since >= std::chrono::milliseconds{200}) {
            due.push_back(std::move(share_checks.front()));
            share_checks.pop_front();
        }
    }
    std::vector<ShareCheck> again;
    for (auto& check : due) {
        const VAddr page = check.page;
        if (now - check.since > std::chrono::seconds{3}) {
            ++stats[check.kind].never_clean;
            continue;
        }
        const u32 block = static_cast<u32>(page >> block_shift);
        if (memory_tracker->IsRegionCpuModified(page, 4096) || lazy_pages.contains(page) ||
            PendingReadbackTick(page, 4096, false) || !resident_ranges.Contains(block, block + 1)) {
            again.push_back(std::move(check));
            continue;
        }
        const auto* arena = GetArena(block, block);
        if (!arena) {
            continue;
        }
        const auto staging = staging_pool.Request(4096, VideoCore::MemoryType::HostCached, 0, true);
        const vk::BufferCopy copy = {
            .srcOffset = page - arena->cpu_addr,
            .dstOffset = staging.offset,
            .size = 4096,
        };
        runtime.CopyBuffer(arena, staging.buffer, std::span{&copy, 1});
        check.guest.resize(4096);
        if (!memory->TryReadBacking(page, check.guest.data(), 4096)) {
            memory->CopySparseMemory(page, check.guest.data(), 4096);
        }
        gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
            for (VAddr byte = lo; byte < hi; ++byte) {
                check.gpu_owned.set(byte - page);
            }
        });
        check.tick = scheduler.CurrentTick();
        check.buffer = staging.buffer;
        check.offset = staging.offset;
        check.staging_size = staging.size;
        check.mapped = staging.mapped;
        share_checks_queued.push_back(std::move(check));
    }
    if (!again.empty()) {
        std::scoped_lock lock{share_checks_mutex};
        for (auto& check : again) {
            share_checks.push_back(std::move(check));
        }
    }
    if (now - last >= std::chrono::seconds{5}) {
        last = now;
        std::string writers;
        for (const auto& [key, n] : bad_by_writer) {
            writers += fmt::format(" {} {}x{}", kinds[key.first], GpuWriterName(key.second), n);
        }
        std::printf("Share check (5 s): waited pages %u checked, %u with CPU bytes stale on the GPU "
                    "(%llu bytes), %u never clean; shared pages %u checked, %u stale (%llu bytes), %u "
                    "never clean, %u poison dwords left in the game's bytes;%s\n",
                    stats[0].checked, stats[0].bad_pages,
                    static_cast<unsigned long long>(stats[0].bad_bytes), stats[0].never_clean,
                    stats[1].checked, stats[1].bad_pages,
                    static_cast<unsigned long long>(stats[1].bad_bytes), stats[1].never_clean,
                    stats[1].poison_left, writers.c_str());
        std::fflush(stdout);
        stats = {};
        bad_by_writer.clear();
    }
}

void BufferCache::PollPeeks() {
    static u64 scanned = 0, bad = 0;
    static int printed = 0;
    static auto last = std::chrono::steady_clock::now();
    static std::map<std::pair<u64, u32>, float> max_abs; // (shader, slot) -> largest finite |x|
    while (!peeks.empty() && scheduler.IsFree(peeks.front().tick)) {
        Peek& peek = peeks.front();
        peek.buffer->Invalidate(peek.offset, peek.staging_size);
        const auto* values = reinterpret_cast<const float*>(peek.mapped);
        const u64 count = peek.size / 4;
        u64 nonfinite = 0, huge = 0;
        float largest = 0.0f;
        for (u64 i = 0; i < count; ++i) {
            const u32 bits = std::bit_cast<u32>(values[i]);
            if (((bits >> 23) & 0xff) == 0xff) {
                ++nonfinite;
            } else {
                const float a = std::abs(values[i]);
                // Plain integers (indices, counters) are tiny denormal-range floats: skip them.
                if ((bits & 0x7f800000u) != 0) {
                    largest = std::max(largest, a);
                    huge += a > 1e4f;
                }
            }
        }
        // Frame-to-frame jumps of the solvers' outputs: values that moved more than 2 units since
        // the last dispatch wrote the same range. A few of them (not most: that is a reset or a
        // camera cut) are flung vertices, which the 1e4 test above misses.
        u64 jumped = 0, compared = 0;
        float max_jump = 0.0f;
        if (peek.dispatch && peek.phase == 1) {
            static std::map<std::tuple<u64, u32, VAddr, u64>, std::vector<float>> previous;
            if (previous.size() > 4096) {
                previous.clear();
            }
            auto& old = previous[{peek.tag, peek.slot, peek.address, peek.size}];
            if (old.size() == count) {
                for (u64 i = 0; i < count; ++i) {
                    const u32 a = std::bit_cast<u32>(values[i]);
                    const u32 b = std::bit_cast<u32>(old[i]);
                    const auto normal = [](u32 bits) {
                        const u32 exponent = (bits >> 23) & 0xff;
                        return exponent != 0 && exponent != 0xff;
                    };
                    if (!normal(a) || !normal(b)) {
                        continue;
                    }
                    ++compared;
                    const float d = std::abs(values[i] - old[i]);
                    if (d > 2.0f) {
                        ++jumped;
                        max_jump = std::max(max_jump, d);
                    }
                }
            }
            old.assign(values, values + count);
        }
        ++scanned;
        // 1c957b01's constant block (slot 1, read before the dispatch): dwords 60-72 and 155-171
        // (68: the step time factor, 160: the step count), 30 s in, 12 dispatches.
        if (peek.tag == 0x1c957b01 && peek.slot == 1 && peek.phase == 0 && peek.ms > 30000 && peek.size >= 688) {
            static int const_lines = 0;
            if (const_lines < 12) {
                ++const_lines;
                const auto* words = reinterpret_cast<const u32*>(values);
                std::string text = fmt::format("Cloth const #{} t={:.1f}:", peek.dispatch, peek.ms);
                for (u32 i : {60u, 62u, 64u, 66u, 67u, 68u, 69u, 70u, 72u, 156u, 158u, 159u, 160u, 161u, 162u, 164u}) {
                    text += fmt::format(" d{}={:#x}/{:.4g}", i, words[i], std::bit_cast<float>(words[i]));
                }
                std::printf("%s\n", text.c_str());
                std::fflush(stdout);
            }
        }
        if (peek.dispatch) {
            struct SlotStat {
                u32 slot, phase;
                u64 size, nonfinite, huge, hash;
                float largest;
                VAddr address;
                u64 jumped, compared;
                float max_jump;
            };
            struct DispatchStat {
                u64 id, shader;
                double ms;
                std::vector<SlotStat> rows;
            };
            static DispatchStat current{};
            static std::deque<DispatchStat> history;
            static int dumps = 0;
            static u64 clean_streak = 0;
            u64 hash = 1469598103934665603ull;
            const auto* bytes = reinterpret_cast<const u8*>(values);
            for (u64 i = 0; i < peek.size; ++i) {
                hash = (hash ^ bytes[i]) * 1099511628211ull;
            }
            static u64 dispatches_1c = 0, cold_1c = 0, exploded_1c = 0;
            static double last_1c_ms = 0;
            static std::array<u64, 5> gaps{}; // <20, 20-40, 40-100, 100-500, >500 ms
            static auto cold_report = std::chrono::steady_clock::now();
            const auto finish = [&] {
                bool exploded = false, jumpy = false;
                for (const auto& row : current.rows) {
                    exploded |= row.phase == 1 && (row.nonfinite || row.huge);
                    jumpy |= row.phase == 1 && row.jumped && row.jumped * 4 < row.compared;
                }
                static u64 jumpy_1c = 0, jumpy_80 = 0;
                static int jump_lines = 0;
                static std::map<std::pair<u64, u32>, double> typical; // (shader, slot) -> average largest
                static u64 outliers_1c = 0, outliers_80 = 0;
                static int outlier_lines = 0;
                bool outlier = false;
                std::string outlier_text;
                for (const auto& row : current.rows) {
                    if (row.phase != 1 || row.size < 64 || row.nonfinite) {
                        continue;
                    }
                    auto [it, fresh] = typical.try_emplace({current.shader, row.slot}, row.largest);
                    const double avg = it->second;
                    if (!fresh && avg > 0.05 && avg < 100.0 && row.largest > 2.0 * avg + 0.5) {
                        outlier = true;
                        outlier_text += fmt::format(" s{}@{:x}: {:.3g} (average {:.3g});", row.slot,
                                                    row.address & 0xffffff, row.largest, avg);
                    } else {
                        it->second = avg * 0.95 + row.largest * 0.05;
                    }
                }
                (current.shader == 0x1c957b01 ? outliers_1c : outliers_80) += outlier;
                if (outlier && outlier_lines < 200) {
                    ++outlier_lines;
                    std::printf("Cloth outlier #%llu %08llx t=%.1f:%s\n", (unsigned long long)current.id,
                                (unsigned long long)current.shader, current.ms, outlier_text.c_str());
                }
                (current.shader == 0x1c957b01 ? jumpy_1c : jumpy_80) += jumpy;
                if (jumpy && jump_lines < 60) {
                    ++jump_lines;
                    std::string text = fmt::format("Cloth jump #{} {:08x} t={:.1f}:", current.id, current.shader,
                                                   current.ms);
                    for (const auto& row : current.rows) {
                        if (row.phase == 1 && row.jumped) {
                            text += fmt::format(" s{}@{:x}: {} of {} moved, max {:.3g};", row.slot,
                                                row.address & 0xffffff, row.jumped, row.compared, row.max_jump);
                        }
                    }
                    std::printf("%s\n", text.c_str());
                }
                // A run of consecutive dispatches, 30 s in: slot address (low 24 bits), max, hash.
                static int seq_lines = 0;
                if (current.shader == 0x1c957b01 && current.ms > 30000 && seq_lines < 40) {
                    ++seq_lines;
                    std::string text = fmt::format("Cloth seq #{} t={:.1f}{}:", current.id, current.ms,
                                                   exploded ? " EXPLODED" : "");
                    for (const auto& row : current.rows) {
                        if (row.slot == 0 || row.slot == 1 || row.slot == 3 || row.slot == 6 || row.slot == 17 ||
                            row.slot == 26 || row.slot == 29) {
                            text += fmt::format(" s{}{}@{:x}:{:.3g}:{:04x}", row.slot, row.phase ? "o" : "i",
                                                row.address & 0xffffff, row.largest, row.hash & 0xffff);
                        }
                    }
                    std::printf("%s\n", text.c_str());
                }
                // The 128-thread solver: dispatches and those with exploded outputs (with the
                // 1c957b01 report), and which output slots blew up.
                static u64 dispatches_80 = 0, exploded_80 = 0;
                static std::map<u32, u64> exploded_slots_80;
                if (current.shader == 0x809864d3) {
                    ++dispatches_80;
                    exploded_80 += exploded;
                    for (const auto& row : current.rows) {
                        if (row.phase == 1 && (row.nonfinite || row.huge)) {
                            ++exploded_slots_80[row.slot];
                        }
                    }
                }
                if (current.shader == 0x1c957b01) {
                    ++dispatches_1c;
                    exploded_1c += exploded;
                    // Cold start: the state slots read as all zero.
                    bool cold = true;
                    for (const auto& row : current.rows) {
                        if (row.phase == 0 && (row.slot == 0 || row.slot == 3 || row.slot == 17)) {
                            cold &= row.largest == 0.0f;
                        }
                    }
                    cold_1c += cold;
                    if (last_1c_ms > 0) {
                        const double gap = current.ms - last_1c_ms;
                        ++gaps[gap < 20 ? 0 : gap < 40 ? 1 : gap < 100 ? 2 : gap < 500 ? 3 : 4];
                    }
                    last_1c_ms = current.ms;
                    if (const auto now = std::chrono::steady_clock::now(); now - cold_report >= std::chrono::seconds{5}) {
                        cold_report = now;
                        std::printf("Cloth 1c957b01 (5 s): %llu dispatches, %llu cold starts, %llu exploded; gaps "
                                    "<20 ms %llu, 20-40 %llu, 40-100 %llu, 100-500 %llu, >500 %llu\n",
                                    (unsigned long long)dispatches_1c, (unsigned long long)cold_1c,
                                    (unsigned long long)exploded_1c, (unsigned long long)gaps[0],
                                    (unsigned long long)gaps[1], (unsigned long long)gaps[2],
                                    (unsigned long long)gaps[3], (unsigned long long)gaps[4]);
                        std::string slots;
                        for (const auto& [slot, n] : exploded_slots_80) {
                            slots += fmt::format(" s{}x{}", slot, n);
                        }
                        std::printf("Cloth 809864d3 (5 s): %llu dispatches, %llu exploded;%s\n",
                                    (unsigned long long)dispatches_80, (unsigned long long)exploded_80,
                                    slots.c_str());
                        std::printf("Cloth jumps (5 s): 1c957b01 %llu, 809864d3 %llu dispatches; outliers "
                                    "1c957b01 %llu, 809864d3 %llu\n",
                                    (unsigned long long)jumpy_1c, (unsigned long long)jumpy_80,
                                    (unsigned long long)outliers_1c, (unsigned long long)outliers_80);
                        jumpy_1c = jumpy_80 = outliers_1c = outliers_80 = 0;
                        std::fflush(stdout);
                        dispatches_1c = cold_1c = exploded_1c = 0;
                        dispatches_80 = exploded_80 = 0;
                        exploded_slots_80.clear();
                        gaps = {};
                    }
                }
                if (exploded && current.shader == 0x1c957b01 && clean_streak >= 20 && dumps < 6) {
                    ++dumps;
                    std::string text = fmt::format("Cloth history before exploded dispatch {}:", current.id);
                    history.push_back(current);
                    for (const auto& d : history) {
                        text += fmt::format("\n  #{} {:08x} t={:.1f}:", d.id, d.shader, d.ms);
                        for (const auto& row : d.rows) {
                            text += fmt::format(" [s{}{} {:x}+{} {:.3g} h{:04x}]", row.slot, row.phase ? "o" : "i",
                                                row.address & 0xffffff, row.size, row.largest, row.hash & 0xffff);
                        }
                    }
                    history.pop_back();
                    std::printf("%s\n", text.c_str());
                    std::fflush(stdout);
                }
                clean_streak = exploded ? 0 : clean_streak + 1;
                history.push_back(std::move(current));
                while (history.size() > 8) {
                    history.pop_front();
                }
                current = {};
            };
            if (peek.dispatch != current.id) {
                if (current.id) {
                    finish();
                }
                current.id = peek.dispatch;
                current.shader = peek.tag;
                current.ms = peek.ms;
            }
            current.rows.push_back({peek.slot, peek.phase, peek.size, nonfinite, huge, hash, largest, peek.address,
                                    jumped, compared, max_jump});
        }
        auto& m = max_abs[{peek.tag, peek.slot}];
        m = std::max(m, largest);
        if (nonfinite || huge) {
            ++bad;
            if (printed < 300) {
                ++printed;
                std::printf("Cloth peek: t=%.1f ms shader %08llx slot %u at %#llx size %llu: %llu non-finite, "
                            "%llu over 1e4, largest %g\n",
                            peek.ms, static_cast<unsigned long long>(peek.tag), peek.slot,
                            static_cast<unsigned long long>(peek.address),
                            static_cast<unsigned long long>(peek.size),
                            static_cast<unsigned long long>(nonfinite), static_cast<unsigned long long>(huge),
                            largest);
            }
        }
        staging_pool.FreeDeferred(Vulkan::StagingBufferRef{
            .buffer = peek.buffer,
            .offset = peek.offset,
            .size = peek.staging_size,
            .mapped = peek.mapped,
            .type = VideoCore::MemoryType::HostCached,
        });
        peeks.pop_front();
    }
    if (const auto now = std::chrono::steady_clock::now(); scanned && now - last >= std::chrono::seconds{5}) {
        last = now;
        std::string text;
        for (const auto& [key, m] : max_abs) {
            text += fmt::format(" {:08x}/{}:{:.3g}", key.first, key.second, m);
        }
        std::printf("Cloth peek (5 s): %llu buffers scanned, %llu with exploded values; largest |x| per slot:%s\n",
                    static_cast<unsigned long long>(scanned), static_cast<unsigned long long>(bad),
                    text.c_str());
        std::fflush(stdout);
        scanned = bad = 0;
        max_abs.clear();
    }
}

void BufferCache::SweepTracking() {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_tracking_sweep < std::chrono::seconds{2}) {
        return;
    }
    last_tracking_sweep = now;
    // Pages whose small GPU writes all finished before the last readback: GpuBytesInFlight
    // ignores such entries, so the page's record only cost memory and hash-table size.
    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    {
        std::unique_lock lock{gpu_ranges_mutex};
        std::erase_if(gpu_write_page_tick, [&](const auto& entry) {
            return semaphore->IsFree(entry.second) &&
                   !gpu_modified_ranges.Intersects(entry.first << 12, 4096);
        });
    }
    g_swept_pages += static_cast<u32>(std::erase_if(page_writes, [&](const auto& entry) {
        return std::ranges::all_of(entry.second, [&](const PageWrite& w) {
            return w.tick < last_readback_tick && semaphore->IsFree(w.tick);
        });
    }));
    if (HotWindowTtl().count()) {
        g_swept_windows += static_cast<u32>(std::erase_if(cpu_write_hot, [&](const auto& entry) {
            return now - entry.second >= HotWindowTtl();
        }));
    }
}

void BufferCache::CompleteReadbacks(bool wait, u64 wait_until) {
    // Pages of a readback are read-protected once its submission is out: a read fault waits
    // for submitted work only (never for the command buffer this thread is still recording).
    for (auto& readback : pending_readbacks) {
        if (readback.watching || readback.watched.empty() || readback.tick >= scheduler.CurrentTick()) {
            continue;
        }
        for (const auto& [address, size] : readback.watched) {
            page_manager->UpdatePageWatchers<true, true>(address, size);
        }
        readback.watching = true;
    }
    // GOW3_TEST_READBACK_HOLD=<ms>:<seconds> (tests): every <seconds>, polled readbacks do not
    // land for <ms>, as on a GPU that fell behind (the FrameBegin m_gfxEopTick crash reports
    // followed 45-55 ms stalls). Waiting callers still land them.
    static const auto hold = [] {
        const char* env = std::getenv("GOW3_TEST_READBACK_HOLD");
        unsigned ms = 0, seconds = 0;
        if (env && std::sscanf(env, "%u:%u", &ms, &seconds) == 2 && ms && seconds) {
            std::printf("Test: readbacks held %u ms every %u s\n", ms, seconds);
        }
        return std::pair<unsigned, unsigned>{ms, seconds};
    }();
    if (hold.first && !wait && !pending_readbacks.empty()) {
        static const auto start = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
                            .count();
        if (ms > 30000 && ms % (hold.second * 1000ull) < hold.first) {
            return;
        }
    }
    // GOW3_TEST_READBACK_LAG=<ms> (tests): polled readbacks land no sooner than <ms> after they
    // were queued, as on a GPU that runs that far behind.
    static const auto lag = [] {
        const char* env = std::getenv("GOW3_TEST_READBACK_LAG");
        const int ms = env ? std::atoi(env) : 0;
        if (ms > 0) {
            std::printf("Test: readbacks land %d ms after they are queued\n", ms);
        }
        return std::chrono::milliseconds(ms > 0 ? ms : 0);
    }();
    auto* semaphore = scheduler.GetWorkSemaphore();
    while (!pending_readbacks.empty()) {
        PendingReadback& readback = pending_readbacks.front();
        if (!wait && lag.count() && std::chrono::steady_clock::now() - readback.queued < lag) {
            return;
        }
        if (!semaphore->IsFree(readback.tick)) {
            if (wait && readback.tick <= wait_until) {
                scheduler.Wait(readback.tick);
            } else {
                // Querying the timeline semaphore is an ioctl: once per 16 polls.
                if (!wait && (++readback_polls & 15) != 0) {
                    return;
                }
                semaphore->Refresh();
                if (!semaphore->IsFree(readback.tick)) {
                    return;
                }
            }
        }
        NoteReadbackLanded(readback.tick);
        readback.buffer->Invalidate(readback.offset, readback.size);
        // Bytes downloaded meanwhile, or uploaded after a CPU write, are no longer GPU data.
        for (size_t i = 0; i < readback.pieces.size(); ++i) {
            const auto [address, size] = readback.pieces[i];
            const u8* data = readback.mapped + readback.offsets[i];
            if (!readback.before.empty()) {
                boost::container::small_vector<std::pair<VAddr, VAddr>, 8> runs;
                gpu_modified_ranges.ForEachInRange(address, size,
                                                   [&](VAddr lo, VAddr hi) { runs.emplace_back(lo, hi); });
                for (const auto& [lo, hi] : runs) {
                    LandGuarded(readback, i, lo, hi);
                }
                continue;
            }
            gpu_modified_ranges.ForEachInRange(address, size, [&](VAddr lo, VAddr hi) {
                WriteBackGpuData(lo, data + (lo - address), hi - lo, &readback.rewritten);
            });
        }
        for (const auto& [address, size] : readback.pieces) {
            readback.rewritten.ForEachNotInRange(address, size, [&](VAddr lo, size_t gap) {
                GpuRangesSubtract(lo, gap);
            });
        }
        {
            const auto now = std::chrono::steady_clock::now();
            if (landed_pieces.size() > 8192) {
                std::erase_if(landed_pieces, [&](const auto& entry) {
                    return now - entry.second.second >= RecurringWindow;
                });
            }
            for (const auto& [address, size] : readback.pieces) {
                landed_pieces[address] = {size, now};
            }
        }
        // Pages left without GPU data stop being GPU-modified (pieces are in address order).
        VAddr last_page = ~0ULL;
        for (const auto& [address, size] : readback.pieces) {
            for (VAddr page = Common::AlignDown(address, LAZY_PAGE_SIZE); page < address + size;
                 page += LAZY_PAGE_SIZE) {
                if (page == last_page) {
                    continue;
                }
                last_page = page;
                if (!gpu_modified_ranges.Intersects(page, LAZY_PAGE_SIZE)) {
                    lazy_pages.erase(page);
                    memory_tracker->UnmarkRegionAsGpuModified(page, LAZY_PAGE_SIZE);
                }
            }
        }
        if (readback.watching) {
            for (const auto& [address, size] : readback.watched) {
                page_manager->UpdatePageWatchers<false, true>(address, size);
            }
        }
        if (readback.open) {
            for (const auto& [address, size] : readback.pieces) {
                open_readback_ranges.Subtract(address, size);
            }
        }
        staging_pool.FreeDeferred(Vulkan::StagingBufferRef{
            .buffer = readback.buffer,
            .offset = readback.offset,
            .size = readback.size,
            .mapped = readback.mapped,
            .type = VideoCore::MemoryType::HostCached,
        });
        auto landed = std::move(readback.on_landed);
        RecycleSnapshot(readback.before);
        IndexReadbackWindows(readback, false);
        pending_readbacks.pop_front();
        for (auto& fn : landed) {
            --deferred_signals;
            fn();
        }
    }
}

bool BufferCache::DeferUntilLastReadback(std::function<void()> fn, VAddr label, u64 value, u32 bytes) {
    if (pending_readbacks.empty()) {
        return false;
    }
    ++deferred_signals;
    ++g_deferred_fences;
    {
        static std::mutex labels_mutex;
        static std::unordered_map<VAddr, u32> seen;
        std::scoped_lock lock{labels_mutex};
        if (seen.size() < 12 && seen[label]++ == 0) {
            std::printf("Deferred fence: label %#llx (tick %llu)\n", static_cast<unsigned long long>(label),
                        static_cast<unsigned long long>(pending_readbacks.back().tick));
            std::fflush(stdout);
        }
    }
    // (A wrapper is a heap allocation per fence: only with the fault statistics.)
    static const bool timed = std::getenv("GOW3_FAULT_STATS") != nullptr;
    if (timed) {
        fn = [fn = std::move(fn), start = std::chrono::steady_clock::now()] {
            const u64 us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            g_deferred_us += us;
            u64 max = g_deferred_max_us.load();
            while (us > max && !g_deferred_max_us.compare_exchange_weak(max, us)) {
            }
            fn();
        };
    }
    if (label) {
        const u32 label_bytes = bytes ? bytes : 4u;
        {
            // Read by the GPU command thread's waits (DeferredFenceTick) while this runs.
            std::scoped_lock lock{deferred_mutex};
            deferred_labels.insert(label);
            deferred_values[label] = {value, label_bytes, pending_readbacks.back().tick};
        }
        for (VAddr dword = label; dword < label + label_bytes; dword += 4) {
            deferred_label_filter[LabelSlot(dword)].fetch_add(1, std::memory_order_acq_rel);
        }
        pending_readbacks.back().on_landed.emplace_back([this, label, label_bytes, fn = std::move(fn)] {
            {
                std::scoped_lock lock{deferred_mutex};
                if (const auto it = deferred_labels.find(label); it != deferred_labels.end()) {
                    deferred_labels.erase(it);
                }
                if (!deferred_labels.contains(label)) {
                    deferred_values.erase(label);
                }
            }
            for (VAddr dword = label; dword < label + label_bytes; dword += 4) {
                deferred_label_filter[LabelSlot(dword)].fetch_sub(1, std::memory_order_acq_rel);
            }
            fn();
        });
    } else {
        pending_readbacks.back().on_landed.emplace_back(std::move(fn));
    }
    return true;
}

bool BufferCache::DeferredFenceTick(VAddr address, u64& tick, u32* value) const {
    std::scoped_lock lock{deferred_mutex};
    for (const auto& [label, deferred] : deferred_values) {
        if (address >= label && address + sizeof(u32) <= label + deferred.bytes) {
            tick = deferred.tick;
            if (value) {
                *value = static_cast<u32>(deferred.value >> ((address - label) * 8));
            }
            return true;
        }
    }
    return false;
}

void BufferCache::LandReadbacks(u64 tick) {
    CompleteReadbacks(true, tick);
}

bool BufferCache::HasUnsubmittedReadback() const {
    return !pending_readbacks.empty() && pending_readbacks.back().tick >= scheduler.CurrentTick();
}

void BufferCache::PollReadbacksNow() {
    if (pending_readbacks.empty()) {
        return;
    }
    scheduler.GetWorkSemaphore()->Refresh();
    readback_polls = 0;
    CompleteReadbacks(false);
}

// gow3 18 Oct 2026: a CPU write into a page a pending readback read-protects (the CPU's own data
// beside the readback's GPU bytes, or GPU bytes it may overwrite) left the page protected: the
// write fault marks it CPU-modified, but only landing the readback lifts the protection, so the
// store faulted again until the GPU thread happened to land it. With the GPU thread idle (the
// guest waits for the very job that writes) nothing did: a game on the Radeon 780M stopped
// flipping at level loads, one job worker spending ~20 s in the kernel on a single store. Watches
// start after submission, so the wait below is for work already sent to the GPU.
u64 BufferCache::WatchingTick(VAddr page) const {
    u64 tick = 0;
    for (const auto& readback : pending_readbacks) {
        if (!readback.watching) {
            continue;
        }
        for (const auto& [address, size] : readback.watched) {
            if (page >= address && page < address + size) {
                tick = std::max(tick, readback.tick);
            }
        }
    }
    return tick;
}

void BufferCache::LandReadbacksWatching(VAddr page) {
    const u64 tick = WatchingTick(page);
    if (tick) {
        CompleteReadbacks(true, tick);
        return;
    }
    // A read watch no pending readback owns would keep the page inaccessible for good.
    static std::atomic<u32> logged{0};
    if (page_manager->IsReadWatched(page) && logged.fetch_add(1, std::memory_order_relaxed) < 4) {
        std::printf("Readback watch: page %#llx is read-protected, but no pending buffer "
                    "readback watches it\n",
                    static_cast<unsigned long long>(page));
        std::fflush(stdout);
    }
    ReleaseOrphanReadWatch(page);
}

// gow3 23 Oct 2026 (GOW3_READBACKS=2, Precise): a read watch no pending readback owns stayed on a
// page at the main menu; a guest store faulted on it 1.3 million times (a black screen). It is
// lifted, so the access goes through.
void BufferCache::ReleaseOrphanReadWatch(VAddr page) {
    for (int i = 0; i < 64 && page_manager->IsReadWatched(page); ++i) {
        page_manager->UpdatePageWatchers<false, true>(page, 4096);
    }
}

void BufferCache::UnwatchPage(VAddr page) {
    for (auto& readback : pending_readbacks) {
        for (size_t i = 0; i < readback.watched.size(); ++i) {
            const auto [address, size] = readback.watched[i];
            if (page < address || page >= address + size) {
                continue;
            }
            if (readback.watching) {
                page_manager->UpdatePageWatchers<false, true>(page, 4096);
            }
            // Split the range around the page.
            const VAddr tail = page + 4096;
            const u64 tail_size = address + size - tail;
            if (page > address) {
                readback.watched[i].second = page - address;
                if (tail_size) {
                    readback.watched.emplace(readback.watched.begin() + i + 1, tail, tail_size);
                }
            } else if (tail_size) {
                readback.watched[i] = {tail, tail_size};
            } else {
                readback.watched.erase(readback.watched.begin() + i);
            }
            break;
        }
    }
}

u64 BufferCache::PendingReadbackTick(VAddr address, u64 size, bool newest_only) const {
    u64 tick = 0;
    const VAddr end = address + size;
    for (const auto& readback : pending_readbacks) {
        for (const auto& [piece, piece_size] : readback.pieces) {
            const VAddr lo = std::max(piece, address);
            const VAddr hi = std::min(piece + piece_size, end);
            if (lo >= hi) {
                continue;
            }
            if (!newest_only || (gpu_modified_ranges.Intersects(lo, hi - lo) &&
                                 !readback.rewritten.Contains(lo, hi - lo))) {
                tick = std::max(tick, readback.tick);
                break;
            }
        }
    }
    return tick;
}

u64 BufferCache::RangeTick() const {
    return scheduler.CurrentTick();
}

void BufferCache::NoteGpuWriteWindows(VAddr address, u64 size) {
    const u64 tick = scheduler.CurrentTick();
    if (size > 64_KB) {
        for (u64 window = address >> 20; window <= (address + size - 1) >> 20; ++window) {
            gpu_write_window_tick[window] = tick;
        }
        return;
    }
    for (u64 page = address >> 12; page <= (address + size - 1) >> 12; ++page) {
        gpu_write_page_tick[page] = tick;
    }
}

u64 BufferCache::FetchableWindowTick(VAddr page, bool count_refusals) {
    const u32 counted = count_refusals ? 1 : 0;
    if (lazy_pages.contains(page)) {
        g_fetch_refused[0] += counted;
        return 0;
    }
    if (PendingReadbackTick(page, 4096, false)) {
        g_fetch_refused[1] += counted;
        return 0;
    }
    u64 tick = 0;
    {
        std::shared_lock lock{gpu_ranges_mutex};
        const auto window = gpu_write_window_tick.find(page >> 20);
        const auto small = gpu_write_page_tick.find(page >> 12);
        if (window != gpu_write_window_tick.end()) {
            tick = window->second;
        }
        if (small != gpu_write_page_tick.end()) {
            tick = std::max(tick, small->second);
        }
        if (!tick) {
            g_fetch_refused[2] += counted;
            return 0;
        }
    }
    // Recorded in the open command buffer: only the full download gets it.
    if (!tick || tick >= scheduler.CurrentTick()) {
        g_fetch_refused[3] += counted;
        return 0;
    }
    return tick;
}

void BufferCache::FinishPageFetch(VAddr page) {
    GpuRangesSubtract(page, 4096);
    memory_tracker->UnmarkRegionAsGpuModified(page, 4096);
    page_writes.erase(page >> 12);
    ++g_direct_page_reads;
}

static bool UrgentCopiesEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_URGENT_COPIES");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

bool BufferCache::SideQueueFetch(std::span<const std::pair<VAddr, u64>> pieces, u64 wait_tick) {
    const vk::Queue queue = instance.GetCopyQueue();
    if (!UrgentCopiesEnabled() || !queue || pieces.empty() || !wait_tick) {
        return false;
    }
    const VAddr lo = pieces.front().first;
    const VAddr hi = pieces.back().first + pieces.back().second;
    const u32 first_block = lo >> block_shift;
    const u32 last_block = (hi - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    if (!arena) {
        return false;
    }
    // gow3: each piece resident (a batched window fetch can span blocks never made resident
    // between its pieces; those are not copied).
    for (const auto& [address, size] : pieces) {
        if (!resident_ranges.Contains(address >> block_shift,
                                      ((address + size - 1) >> block_shift) + 1)) {
            return false;
        }
    }
    u64 total = 0;
    for (const auto& [address, size] : pieces) {
        total += size;
    }
    const auto device = instance.GetDevice();
    if (!urgent_pool) {
        urgent_pool = Vulkan::Check<"urgent copy pool">(device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                     vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
        }));
        urgent_cmd = Vulkan::Check<"urgent copy buffer">(device.allocateCommandBuffers({
            .commandPool = *urgent_pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        }))[0];
        const vk::SemaphoreTypeCreateInfo timeline_ci = {
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        };
        urgent_timeline = Vulkan::Check<"urgent copy timeline">(
            device.createSemaphoreUnique({.pNext = &timeline_ci}));
    }
    const auto staging = staging_pool.Request(total, MemoryType::HostCached);
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    u64 offset = staging.offset;
    for (const auto& [address, size] : pieces) {
        copies.push_back({address - arena->cpu_addr, offset, size});
        offset += size;
    }
    std::ignore = urgent_cmd.reset();
    std::ignore = urgent_cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    urgent_cmd.copyBuffer(arena->Handle(), staging.buffer->Handle(), copies);
#pragma push_macro("MemoryBarrier")
#undef MemoryBarrier
    const vk::MemoryBarrier to_host = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eHostRead,
    };
#pragma pop_macro("MemoryBarrier")
    urgent_cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eHost, {}, to_host, {}, {});
    std::ignore = urgent_cmd.end();
    const vk::Semaphore timeline = scheduler.GetWorkSemaphore()->Handle();
    const vk::Semaphore done = *urgent_timeline;
    const u64 done_value = urgent_value + 1;
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eAllCommands;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = 1,
        .pWaitSemaphoreValues = &wait_tick,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &done_value,
    };
    const vk::SubmitInfo submit = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &timeline,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &urgent_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &done,
    };
    if (queue.submit(submit) != vk::Result::eSuccess) {
        return false;
    }
    urgent_value = done_value;
    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &done,
        .pValues = &done_value,
    };
    const auto wait_start = std::chrono::steady_clock::now();
    std::ignore = device.waitSemaphores(wait_info, std::numeric_limits<u64>::max());
    {
        const u32 us = static_cast<u32>(std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - wait_start)
                                            .count());
        g_side_fetches.fetch_add(1, std::memory_order_relaxed);
        g_side_fetch_us.fetch_add(us, std::memory_order_relaxed);
        for (u32 max = g_side_fetch_max_us.load(std::memory_order_relaxed);
             us > max && !g_side_fetch_max_us.compare_exchange_weak(max, us);) {
        }
    }
    staging.Invalidate();
    const u8* source = staging.mapped;
    for (const auto& [address, size] : pieces) {
        if (!memory->TryWriteBacking(std::bit_cast<u8*>(address), source, size)) {
            std::memcpy(std::bit_cast<void*>(address), source, size);
        }
        source += size;
    }
    return true;
}

bool BufferCache::CopyFinishedGpuPage(VAddr page, bool finished_only) {
    if (!UrgentCopiesEnabled() || !instance.GetCopyQueue()) {
        return false;
    }
    const u64 tick = FetchableWindowTick(page);
    if (!tick) {
        return false;
    }
    // gow3 21 Oct 2026 (GOW3_FETCH_NO_BLOCK=1): only GPU writes that have finished (the copy then
    // waits for no GPU work on this thread).
    auto* semaphore = scheduler.GetWorkSemaphore();
    if (finished_only) {
        if (!semaphore->IsFree(tick)) {
            semaphore->Refresh();
        }
        if (!semaphore->IsFree(tick)) {
            return false;
        }
    }
    // gow3 20 Oct 2026: the other pages of the fault window (GOW3_FAULT_WINDOW, 256 KiB) whose
    // GPU bytes are final come along in the same copy. When the game rewrites memory the GPU
    // wrote earlier (level streaming into reused memory:
    // ~9,000-14,000 such faults per 5 s, every page of whole 1 MiB windows), each 4 KiB page
    // was a copy on the second queue plus a wait (~0.25 ms): 35-70 ms frames. The fetched pages
    // stop being GPU data, so ExtendWriteFault right after lets the rest of the window be
    // written without faults. Safe for the same reason as the single page: a GPU-written page
    // that is not lazy and has no readback pending is write-protected, so the guest has not
    // written it since the GPU did; its GPU bytes are the newest. GOW3_FETCH_BATCH_KB=0: one
    // page per fault, as before.
    static const u64 batch_window = [] {
        const char* env = std::getenv("GOW3_FETCH_BATCH_KB");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        const u64 bytes = kib < 8 ? 0 : std::bit_ceil(std::min<u64>(kib, 1024)) * 1024;
        std::printf("GPU: write faults fetch %s\n",
                    bytes ? fmt::format("the finished GPU pages of their {} KiB window in one copy "
                                        "(GOW3_FETCH_BATCH_KB=0: one page)",
                                        bytes >> 10).c_str()
                          : "one page per fault (GOW3_FETCH_BATCH_KB=0)");
        return bytes;
    }();
    boost::container::small_vector<VAddr, 64> pages;
    pages.push_back(page);
    u64 wait_tick = tick;
    if (batch_window) {
        const VAddr window = Common::AlignDown(page, batch_window);
        for (VAddr other = window; other < window + batch_window; other += 4096) {
            if (other == page || !gpu_modified_ranges.Intersects(other, 4096)) {
                continue;
            }
            // Only pages the tracker holds as GPU data and not CPU-modified: write-protected
            // (a page is writable only with its CPU bit), so the guest cannot have newer bytes.
            if (!memory_tracker->IsRegionGpuModified(other, 4096) ||
                memory_tracker->IsRegionCpuModified(other, 4096)) {
                continue;
            }
            if (const u64 other_tick = FetchableWindowTick(other, false)) {
                if (finished_only && !semaphore->IsFree(other_tick)) {
                    continue; // (GOW3_FETCH_NO_BLOCK) its GPU write is still running
                }
                wait_tick = std::max(wait_tick, other_tick);
                pages.push_back(other);
            }
        }
        std::ranges::sort(pages);
    }
    const auto fetch = [&](std::span<const VAddr> list, u64 wait) {
        boost::container::small_vector<std::pair<VAddr, u64>, 64> pieces;
        for (const VAddr fetched : list) {
            gpu_modified_ranges.ForEachInRange(fetched, 4096, [&](VAddr lo, VAddr hi) {
                pieces.emplace_back(lo, hi - lo);
            });
        }
        if (!SideQueueFetch(pieces, wait)) {
            return false;
        }
        for (const VAddr fetched : list) {
            FinishPageFetch(fetched);
        }
        return true;
    };
    if (pages.size() > 1 && fetch(pages, wait_tick)) {
        return true;
    }
    const VAddr single[] = {page};
    return fetch(single, tick);
}

bool BufferCache::FetchOldGpuBytes(VAddr page) {
    if (!UrgentCopiesEnabled() || !instance.GetCopyQueue() || lazy_pages.contains(page)) {
        return false;
    }
    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    {
        // A large GPU write into the window still running could cover any byte.
        std::shared_lock lock{gpu_ranges_mutex};
        const auto window = gpu_write_window_tick.find(page >> 20);
        if (window != gpu_write_window_tick.end() && !semaphore->IsFree(window->second)) {
            return false;
        }
    }
    // The bytes still in flight (as GpuBytesInFlight counts them); the others are final.
    std::bitset<4096> covered;
    if (const auto it = page_writes.find(page >> 12); it != page_writes.end()) {
        for (const auto& write : it->second) {
            if (write.tick >= last_readback_tick || !semaphore->IsFree(write.tick)) {
                for (u32 byte = write.lo; byte < write.hi; ++byte) {
                    covered.set(byte);
                }
            }
        }
    }
    for (const auto& readback : pending_readbacks) {
        for (const auto& [address, size] : readback.pieces) {
            const VAddr lo = std::max(address, page);
            const VAddr hi = std::min(address + size, page + 4096);
            for (VAddr byte = lo; byte < hi; ++byte) {
                covered.set(byte - page);
            }
        }
    }
    boost::container::small_vector<std::pair<VAddr, u64>, 8> pieces;
    gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
        for (VAddr byte = lo; byte < hi; ++byte) {
            if (covered.test(byte - page)) {
                continue;
            }
            if (!pieces.empty() && pieces.back().first + pieces.back().second == byte) {
                ++pieces.back().second;
            } else {
                pieces.emplace_back(byte, 1);
            }
        }
    });
    const u64 done = semaphore->KnownGpuTick();
    if (pieces.empty() || !SideQueueFetch(pieces, done)) {
        return false;
    }
    for (const auto& [address, size] : pieces) {
        GpuRangesSubtract(address, size);
    }
    ++g_old_byte_fetches;
    return true;
}

bool BufferCache::ReadFinishedGpuPage(VAddr page) {
    if (!arena_host_mapped) {
        return false;
    }
    const u64 tick = FetchableWindowTick(page);
    if (!tick) {
        return false;
    }
    auto* semaphore = scheduler.GetWorkSemaphore();
    if (!semaphore->IsFree(tick)) {
        semaphore->Refresh();
        if (!semaphore->IsFree(tick)) {
            return false;
        }
    }
    // The page's GPU bytes and where the arena keeps them; all mapped, or the old path.
    struct Piece {
        VAddr address;
        u64 size;
        const u8* source;
    };
    boost::container::small_vector<Piece, 8> pieces;
    bool mapped = true;
    gpu_modified_ranges.ForEachInRange(page, 4096, [&](VAddr lo, VAddr hi) {
        for (VAddr address = lo; address < hi && mapped;) {
            const auto it = resident_ranges.Find(address >> block_shift);
            if (it == resident_ranges.end() || !it->mapped) {
                mapped = false;
                break;
            }
            const VAddr backing_start = it->start << block_shift;
            const u64 size = std::min<VAddr>(hi, it->end << block_shift) - address;
            pieces.push_back({address, size, it->mapped + it->offset + (address - backing_start)});
            address += size;
        }
    });
    if (!mapped || pieces.empty()) {
        return false;
    }
    std::array<u8, 4096> bytes;
    for (const auto& piece : pieces) {
        // One sequential read of the (write-combined) mapping, then the guest copy.
        std::memcpy(bytes.data(), piece.source, piece.size);
        if (!memory->TryWriteBacking(std::bit_cast<u8*>(piece.address), bytes.data(), piece.size)) {
            std::memcpy(std::bit_cast<void*>(piece.address), bytes.data(), piece.size);
        }
    }
    FinishPageFetch(page);
    return true;
}

void BufferCache::EraseLazyPages(VAddr address, u64 size) {
    if (lazy_pages.empty()) {
        return;
    }
    // gow3 24 Oct 2026: a poisoned page keeps its record while it tracks bytes outside the
    // downloaded range: they still hold the poison (the resource table walk downloads a few
    // dozen bytes of a page, Rasterizer::SyncSrtTables; erasing the whole record let a later
    // share_page take that poison for the guest's copy and keep it in the game's bytes).
    // Within the range, a tracked byte still GPU data (not downloaded: its tracker page was not
    // GPU-modified) stays tracked; one that is not but still holds the poison gets its guest
    // byte from before the poison back. No byte is left untracked with the poison in it.
    const VAddr end = address + size;
    for (auto it = lazy_pages.lower_bound(Common::AlignDown(address, LAZY_PAGE_SIZE));
         it != lazy_pages.end() && it->first < end;) {
        LazyPage& record = *it->second;
        if (record.poisoned) {
            const VAddr page = it->first;
            const VAddr lo = std::max(page, address);
            const VAddr hi = std::min(page + LAZY_PAGE_SIZE, end);
            std::bitset<LAZY_PAGE_SIZE> gpu_owned;
            gpu_modified_ranges.ForEachInRange(lo, hi - lo, [&](VAddr a, VAddr b) {
                for (VAddr byte = a; byte < b; ++byte) {
                    gpu_owned.set(byte - page);
                }
            });
            std::array<u8, LAZY_PAGE_SIZE> contents;
            const bool backed = memory->TryReadBacking(page, contents.data(), LAZY_PAGE_SIZE);
            bool restored = false;
            for (VAddr byte = lo; byte < hi; ++byte) {
                const size_t i = byte - page;
                if (!record.tracked.test(i) || gpu_owned.test(i)) {
                    continue;
                }
                if (backed && record.bytes[i] == PoisonByte(i) && contents[i] == PoisonByte(i)) {
                    contents[i] = record.original[i];
                    restored = true;
                }
                record.tracked.reset(i);
            }
            if (restored) {
                memory->TryWriteBacking(std::bit_cast<u8*>(lo), contents.data() + (lo - page), hi - lo);
                g_poison_restored.fetch_add(1, std::memory_order_relaxed);
            }
            NotePoisonEvent(page, fmt::format("erase{}{}", restored ? "-restored" : "",
                                              record.tracked.any() ? "-kept" : ""));
            if (restored) {
                ReportPoisonRestore("download", page);
            }
            if (record.tracked.any()) {
                ++it;
                continue;
            }
        }
        it = lazy_pages.erase(it);
    }
}

namespace {
// gow3: GOW3_BUFFER_STATS=1 — how buffer bindings reach the GPU, by guest region (256 MiB),
// printed every 5 s: small read-only copies into the stream buffer, arena bindings, and the
// bytes those re-upload after CPU writes. Input for engine-level short paths.
struct BufferStats {
    struct Region {
        u64 stream_count{}, stream_bytes{}, arena_count{}, arena_bytes{}, upload_bytes{};
    };
    std::map<u64, Region> regions;
    /// Hot window: frame of the last binding per 64 KiB block, and the distribution of the
    /// frames between uses (1, 2, 3, 4, 5-8, 9-16, 17+): the ring's reuse distance.
    std::unordered_map<u64, u64> last_use;
    std::array<u64, 7> reuse{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
    u64 calls = 0;
};
/// 256 MiB regions; 1 MiB inside the hot 0x104xxxxxxx window (the engine's frame data).
BufferStats& Stats();
void NoteHotUse(VAddr address, u32 size) {
    if ((address >> 28) != 0x104) {
        return;
    }
    auto& st = Stats();
    const u64 frame = Gow3Stats::gpu_frames.load(std::memory_order_relaxed);
    for (u64 block = address >> 16; block <= (address + size - 1) >> 16; ++block) {
        auto [it, inserted] = st.last_use.try_emplace(block, frame);
        if (!inserted && it->second != frame) {
            const u64 d = frame - it->second;
            ++st.reuse[d <= 4 ? d - 1 : d <= 8 ? 4 : d <= 16 ? 5 : 6];
            it->second = frame;
        }
    }
}
u64 RegionKey(VAddr address) {
    return (address >> 28) == 0x104 ? (address >> 20) | (1ull << 40) : address >> 28;
}
bool BufferStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GOW3_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
BufferStats& Stats() {
    static BufferStats stats;
    return stats;
}
void PrintBufferStats() {
    auto& st = Stats();
    if ((++st.calls & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - st.window).count();
    if (seconds < 5.0) {
        return;
    }
    std::printf("Buffer stats (%.1f s, per second): region, stream copies/bytes, arena bindings, "
                "re-uploaded bytes\n", seconds);
    std::printf("  hot window reuse distance in frames (1,2,3,4,5-8,9-16,17+): %llu %llu %llu %llu "
                "%llu %llu %llu\n",
                (unsigned long long)st.reuse[0], (unsigned long long)st.reuse[1],
                (unsigned long long)st.reuse[2], (unsigned long long)st.reuse[3],
                (unsigned long long)st.reuse[4], (unsigned long long)st.reuse[5],
                (unsigned long long)st.reuse[6]);
    st.reuse = {};
    for (const auto& [region, r] : st.regions) {
        const u64 base = (region >> 40) ? (region & ((1ull << 40) - 1)) << 20 : region << 28;
        if ((region >> 40) && r.stream_bytes + r.upload_bytes < 1e6 * seconds) {
            continue; // quiet MiB of the hot window
        }
        std::printf("  %#012llx: %8.0f copies %8.2f MB, %8.0f arena %8.2f MB bound, %8.2f MB uploads\n",
                    static_cast<unsigned long long>(base), r.stream_count / seconds,
                    r.stream_bytes / seconds / 1e6, r.arena_count / seconds,
                    r.arena_bytes / seconds / 1e6,
                    r.upload_bytes / seconds / 1e6);
    }
    st.regions.clear();
    st.window = now;
}
} // namespace

// gow3 17 Oct 2026: the crowd skinning shader (cs 8f067540, ~1,500 dispatches a frame)
// reads each character's mesh arrays (5 of 1-8 KB, never written after loading) as small read-only
// buffers: the stream path copied them from guest memory on every bind, ~43 MB a frame at 4K and
// the GPU command thread's main cost there. A range bound in 3 submissions moves to the arena,
// uploaded once and kept by CPU write tracking; once the CPU rewrites it (per-frame constants
// at a reused address) it goes back to the stream path for good. GOW3_STATIC_BUFFERS=0: off;
// GOW3_STATIC_MIN_BYTES (default 1024): smaller ranges always stream.
int BufferCache::UseArenaForStatic(VAddr device_addr, u32 size) {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_STATIC_BUFFERS");
        return !(env && env[0] == '0');
    }();
    static const u64 min_bytes = [] {
        const char* env = std::getenv("GOW3_STATIC_MIN_BYTES");
        return env ? std::strtoull(env, nullptr, 10) : 1024ull;
    }();
    if (!enabled || size < min_bytes) {
        return 0;
    }
    // Direct-mapped: a promoted range used in the last 256 submissions keeps its slot (others
    // stream); otherwise the newcomer takes it and starts over.
    const u64 key = device_addr ^ (u64(size) << 44);
    auto& range = static_ranges[(key * 0x9E3779B97F4A7C15ull) >> (64 - StaticRangeBits)];
    const u64 tick = scheduler.CurrentTick();
    if (range.key != key) {
        if (range.uploaded && !range.demoted && tick - range.last_tick < 256) {
            return 0;
        }
        range = {.key = key};
    }
    if (range.demoted) {
        return 0;
    }
    if (range.ticks >= 3) {
        range.last_tick = tick;
    } else {
        if (range.last_tick != tick) {
            range.last_tick = tick;
            ++range.ticks;
        }
        if (range.ticks < 3) {
            return 0;
        }
        if (Core::MemoryManager::KeptWritable(device_addr, size)) {
            range.demoted = true;
            return 0;
        }
        ++static_promoted;
    }
    // Only where guest memory changes are always seen by page tracking: no runtime write through
    // the backing view (fence labels, readback landings, image downloads) and no GPU write (lazy
    // and shared pages let the CPU write next to GPU bytes untracked) ever reached its 64 KiB
    // windows. Otherwise the arena copy could go stale where the stream path copied afresh
    // (17 Oct: the integrated GPU run stopped submitting frames with it).
    const VAddr page_lo = Common::AlignDown(device_addr, 4096);
    const VAddr page_hi = Common::AlignUp(device_addr + size, 4096);
    if (Core::MemoryManager::BackingWritten(device_addr, size) ||
        MayBeGpuWritten(page_lo, page_hi - page_lo)) {
        range.demoted = true;
        ++static_demoted;
        return 0;
    }
    const bool clean = range.uploaded;
    if (range.uploaded && memory_tracker->IsRegionCpuModified(device_addr, size)) {
        range.demoted = true;
        ++static_demoted;
        return 0;
    }
    range.uploaded = true;
    g_obtained_static = true;
    ++static_binds;
    static_bytes += size;
    return clean ? 2 : 1;
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferObtain);
    g_obtained_static = false;
    const bool stats = BufferStatsEnabled();
    if (stats) {
        PrintBufferStats();
        NoteHotUse(device_addr, size);
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    const int static_range = !is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)
                                 ? UseArenaForStatic(device_addr, size)
                                 : -1;
    if (static_range == 2) {
        // Uploaded before and clean since (no CPU write): resident, nothing to synchronize.
        const u64 first_block = device_addr >> block_shift;
        const auto* arena = GetArena(first_block, (device_addr + size - 1) >> block_shift);
        return {arena, arena->Offset(device_addr)};
    }
    if (static_range == 0) {
        if (stats) {
            auto& region = Stats().regions[RegionKey(device_addr)];
            ++region.stream_count;
            region.stream_bytes += size;
        }
        // gow3 17 Oct 2026: the same range bound again in the same submission, before the guest
        // has seen any GPU write (after which it may rewrite the memory), can reuse the copy: the
        // guest wrote it before submitting. At 4K: only 12% of the copies, and the
        // lookups cost more than they saved (GOW3_STREAM_REUSE=1: on).
        static const bool reuse = [] {
            const char* env = std::getenv("GOW3_STREAM_REUSE");
            return env && env[0] == '1';
        }();
        const u64 key = device_addr ^ (u64(size) << 44);
        const u64 tick = scheduler.CurrentTick();
        const u64 epoch = scheduler.GuestWriteEpoch();
        if (reuse) {
            if (const auto it = stream_copies.find(key);
                it != stream_copies.end() && it->second.tick == tick && it->second.epoch == epoch &&
                it->second.wraps == stream_buffer.Wraps()) {
                ++stream_reused;
                stream_reused_bytes += size;
                return {&stream_buffer, it->second.offset};
            }
            if (stream_copies.size() >= 16384) {
                stream_copies.clear();
            }
        }
        ++stream_copied;
        stream_copied_bytes += size;
        static const bool stream_stats = [] {
            const char* env = std::getenv("GOW3_STREAM_STATS");
            return env && env[0] == '1';
        }();
        if (stream_stats) {
            struct Entry {
                u64 count = 0, bytes = 0, used = 0;
            };
            static std::unordered_map<u64, Entry> entries;
            static auto last = std::chrono::steady_clock::now();
            auto& entry = entries[g_stream_stats_key];
            ++entry.count;
            entry.bytes += size;
            entry.used += g_stream_stats_used ? std::min<u64>(g_stream_stats_used, size) : size;
            if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
                last = now;
                std::vector<std::pair<u64, u64>> order;
                for (const auto& [key, e] : entries) {
                    order.emplace_back(e.bytes, key);
                }
                std::ranges::sort(order, std::greater{});
                std::string text;
                for (size_t i = 0; i < std::min<size_t>(order.size(), 20); ++i) {
                    const auto& e = entries[order[i].second];
                    text += fmt::format(" {:08x}/s{}: {} x {:.0f} B = {:.1f} MB (used {:.1f});",
                                        order[i].second >> 8, order[i].second & 0xff, e.count,
                                        double(e.bytes) / e.count, e.bytes / 1e6, e.used / 1e6);
                }
                std::printf("Stream copies by binding (5 s):%s\n", text.c_str());
                std::fflush(stdout);
                entries.clear();
            }
        }
        // gow3: the clock is read once per submission (or every 4096 copies), not per copy: up to
        // ~250k copies/s in heavy scenes, and these statistics are on for every player.
        static u64 stream_stats_tick = ~0ull;
        if (Gow3Stats::enabled && (tick != stream_stats_tick || (stream_copied & 4095) == 0)) {
            stream_stats_tick = tick;
            static auto last = std::chrono::steady_clock::now();
            if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
                last = now;
                std::printf("Stream copies (5 s): %llu copied (%.1f MB), %llu reused (%.1f MB); static "
                            "ranges: %llu binds from the arena (%.1f MB), %llu promoted, %llu demoted, "
                            "%zu known\n",
                            (unsigned long long)stream_copied, stream_copied_bytes / 1e6,
                            (unsigned long long)stream_reused, stream_reused_bytes / 1e6,
                            (unsigned long long)static_binds, static_bytes / 1e6,
                            (unsigned long long)static_promoted, (unsigned long long)static_demoted,
                            std::size_t(std::ranges::count_if(static_ranges, [](const StaticRange& r) {
                                return r.uploaded && !r.demoted;
                            })));
                std::fflush(stdout);
                stream_copied = stream_copied_bytes = stream_reused = stream_reused_bytes = 0;
                static_binds = static_bytes = static_promoted = static_demoted = 0;
            }
        }
        // gow3: the guest data is copied on a copy thread, started now; submission and
        // guest-visible fences wait for it (Scheduler::WaitHostCopies).
        if (!stream_buffer.mapped_data.empty() &&
            !Gow3Toggle::Disabled(Gow3Toggle::DeferredStreamCopies)) {
            if (const auto offset = stream_buffer.Reserve(size, instance.UniformMinAlignment())) {
                if (reuse) {
                    stream_copies[key] = {tick, epoch, stream_buffer.Wraps(), *offset};
                }
                SmallGuestCopy({
                    .run = &RunGuestCopy,
                    .context = this,
                    .source = device_addr,
                    .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset),
                    .size = size,
                    .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
                });
                return {&stream_buffer, *offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        if (reuse) {
            stream_copies[key] = {tick, epoch, stream_buffer.Wraps(), offset};
        }
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    const u64 uploaded_before = Gow3Stats::buffer_upload_bytes.load(std::memory_order_relaxed);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (stats) {
        auto& region = Stats().regions[RegionKey(device_addr)];
        ++region.arena_count;
        region.arena_bytes += size;
        region.upload_bytes +=
            Gow3Stats::buffer_upload_bytes.load(std::memory_order_relaxed) - uploaded_before;
    }
    if (is_written) {
        // Small writes only (counters, constant blocks: at most two pages): the CPU writes next
        // to these; larger GPU data in a page the CPU writes is downloaded as before.
        if (size <= 4096) {
            const u64 tick = scheduler.CurrentTick();
            const VAddr end = device_addr + size;
            for (VAddr page = Common::AlignDown(device_addr, 4096); page < end; page += 4096) {
                auto& writes = page_writes[page >> 12];
                const auto lo = static_cast<u16>(std::max(device_addr, page) - page);
                const auto hi = static_cast<u16>(std::min(end, page + 4096) - page);
                // The same range written again (every frame): its newest tick is what counts.
                // Appending instead grew each page's list by one entry per frame, and the prune
                // below scanned it all on every binding (about 1 MB of memory per
                // second, the frame rate falling from 42 to 31 FPS over a few minutes).
                const auto same = std::find_if(writes.begin(), writes.end(), [&](const PageWrite& w) {
                    return w.lo == lo && w.hi == hi;
                });
                if (same != writes.end()) {
                    same->tick = std::max(same->tick, tick);
                    continue;
                }
                if (writes.size() >= 16) {
                    // Entries GpuBytesInFlight no longer counts: finished, older than the last
                    // readback. If the page is still full, the oldest go (its CPU writes may
                    // then download as before).
                    auto* semaphore = scheduler.GetWorkSemaphore();
                    writes.erase(std::remove_if(writes.begin(), writes.end(),
                                                [&](const PageWrite& w) {
                                                    return w.tick < last_readback_tick &&
                                                           semaphore->IsFree(w.tick);
                                                }),
                                 writes.end());
                    if (writes.size() >= 16) {
                        std::sort(writes.begin(), writes.end(),
                                  [](const PageWrite& a, const PageWrite& b) { return a.tick < b.tick; });
                        writes.erase(writes.begin(), writes.begin() + (writes.size() - 8));
                    }
                }
                writes.push_back({lo, hi, tick});
            }
        }
        GpuRangesAdd(device_addr, size);
        if (TouchesPendingReadback(device_addr, size)) {
            for (auto& readback : pending_readbacks) {
                readback.NoteRewrite(device_addr, size);
            }
        }
        ResolvePoisonedPages(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    texture_cache.ResolveReadbacks(device_addr, size, true, ReadbackSource::BufferForImage);
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    if (!Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads)) {
        // gow3: texture data is copied on the copy threads (streaming: 100+ MB per frame);
        // the upload reads the staging only after submission, which waits for the copies.
        constexpr u64 Chunk = 1_MB;
        for (u64 offset = 0; offset < staging.size; offset += Chunk) {
            const Gow3Copy::Item item{
                .run = &RunGuestCopy,
                .context = this,
                .source = device_addr + offset,
                .destination = reinterpret_cast<u64>(staging.mapped + offset),
                .size = std::min(Chunk, staging.size - offset),
                .extra = reinterpret_cast<u64>(staging.buffer),
            };
            if (staging.size < Chunk) {
                SmallGuestCopy(item);
            } else {
                Gow3Copy::Async([item] { item.run(item); });
            }
        }
        return {staging.buffer, staging.offset};
    }
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager->ProcessFaultBuffer();
}

void BufferCache::SynchronizeDmaBuffers() {
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (arena_page_bits - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

void BufferCache::SynchronizeDmaRange(VAddr address, u64 size) {
    const u64 first = address >> block_shift;
    const u64 last = (address + size + (1ULL << block_shift) - 1) >> block_shift;
    if (last > first) {
        TouchArenaWindows(first, last - 1); // in use (ReclaimArenaMemory)
    }
    resident_ranges.ForEachInRange(first, last, [&](const Backing& backing) {
        const u64 start = std::max(first, backing.start);
        const u64 end = std::min(last, backing.end);
        if (start >= end) {
            return;
        }
        // One arena page at a time: SynchronizeMemory copies relative to a single arena.
        const u64 blocks_per_page = 1ULL << (arena_page_bits - block_shift);
        for (u64 block = start; block < end;) {
            const u64 page = block >> (arena_page_bits - block_shift);
            const u64 stop = std::min(end, (page + 1) * blocks_per_page);
            if (const auto* arena = address_space[page]) {
                SynchronizeMemory(arena, block << block_shift,
                                  static_cast<u32>((stop - block) << block_shift), false, false);
            }
            block = stop;
        }
    });
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    TouchArenaWindows(first_block, last_block);
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << arena_page_bits, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << arena_page_bits);
    const u64 first_size = first_arena ? first_arena->size_bytes : arena_page_size;
    const u64 last_size = last_arena ? last_arena->size_bytes : arena_page_size;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = backing.offset + ((start - backing.start) << block_shift),
        });
    });

    u64 base_page = first_addr >> arena_page_bits;
    for (u32 page = 0; page < (first_size >> arena_page_bits); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> arena_page_bits); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    Gow3Stats::Timer timer{Gow3Stats::t_resident};

    // gow3 20 Oct 2026 (GitHub #95): memory given back by ReclaimArenaMemory is bound again
    // first; new memory only for what it does not cover. Unbinds of these ranges still pending
    // (released and used again before they were submitted) are dropped.
    struct Piece {
        u64 start, end;
        vk::DeviceMemory memory;
        u64 offset;
        u8* mapped;
    };
    boost::container::small_vector<Piece, 8> pieces;
    boost::container::small_vector<std::pair<u64, u64>, 8> fresh;
    u64 fresh_blocks = 0;
    for (const auto& range : bind_ranges) {
        if (has_pending_unbinds) {
            TrimPendingUnbinds(arena, range.start, range.end);
        }
        u64 start = range.start;
        while (start < range.end && !free_runs.empty()) {
            auto& run = free_runs.back();
            const u64 count = std::min(run.blocks, range.end - start);
            pieces.push_back({start, start + count, run.memory, run.offset, run.mapped});
            run.offset += count << block_shift;
            run.blocks -= count;
            if (!run.blocks) {
                free_runs.pop_back();
            }
            start += count;
        }
        if (start < range.end) {
            fresh.emplace_back(start, range.end);
            fresh_blocks += range.end - start;
        }
    }
    if (fresh_blocks) {
        const u64 fresh_bytes = fresh_blocks << block_shift;
        vk::DeviceMemory device_memory{};
        u8* mapped = nullptr;
        u64 memory_offset{};
        if (!residency_pool ||
            !residency_pool->Take(fresh_bytes, device_memory, mapped, memory_offset)) {
            const vk::MemoryAllocateInfo alloc_info = {
                .allocationSize = fresh_bytes,
                .memoryTypeIndex = arena_memory_type_index,
            };
            device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));
            Gow3Stats::arena_memory_bytes += fresh_bytes;
            if (arena_host_mapped) {
                auto [map_result, pointer] =
                    instance.GetDevice().mapMemory(device_memory, 0, VK_WHOLE_SIZE);
                if (map_result == vk::Result::eSuccess) {
                    mapped = static_cast<u8*>(pointer);
                }
            }
        }
        for (const auto& [start, end] : fresh) {
            pieces.push_back({start, end, device_memory, memory_offset, mapped});
            memory_offset += (end - start) << block_shift;
        }
    }
    resident_in_use_blocks += resident_blocks;
    Gow3Stats::arena_in_use_bytes = resident_in_use_blocks << block_shift;

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& piece : pieces) {
        Backing backing;
        backing.start = piece.start;
        backing.end = piece.end;
        backing.memory = piece.memory;
        backing.offset = piece.offset;
        backing.mapped = piece.mapped;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (piece.start << block_shift) - arena->cpu_addr,
            .size = (piece.end - piece.start) << block_shift,
            .memory = piece.memory,
            .memoryOffset = piece.offset,
        });

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

void BufferCache::TrimPendingUnbinds(const Buffer* arena, u64 first_block, u64 end_block) {
    const u64 lo = (first_block << block_shift) - arena->cpu_addr;
    const u64 hi = (end_block << block_shift) - arena->cpu_addr;
    for (auto& pending : pending_binds) {
        if (pending.arena != arena) {
            continue;
        }
        decltype(pending.binds) kept;
        bool changed = false;
        for (const auto& bind : pending.binds) {
            if (bind.memory || bind.resourceOffset >= hi || bind.resourceOffset + bind.size <= lo) {
                kept.push_back(bind);
                continue;
            }
            changed = true;
            if (bind.resourceOffset < lo) {
                kept.push_back({.resourceOffset = bind.resourceOffset, .size = lo - bind.resourceOffset});
            }
            if (bind.resourceOffset + bind.size > hi) {
                kept.push_back({.resourceOffset = hi, .size = bind.resourceOffset + bind.size - hi});
            }
        }
        if (changed) {
            pending.binds = std::move(kept);
        }
    }
}

u64 BufferCache::ReleaseArenaBlocks(u64 first_block, u64 end_block) {
    boost::container::small_vector<Backing, 8> pieces;
    resident_ranges.ForEachInRange(first_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(first_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        if (start < end) {
            Backing piece = backing;
            piece.start = start;
            piece.end = end;
            piece.offset = backing.offset + ((start - backing.start) << block_shift);
            pieces.push_back(piece);
        }
    });
    if (pieces.empty()) {
        return 0;
    }
    resident_ranges.Remove(first_block, end_block, [this](const Backing& backing, u64 a, u64 b) {
        Backing part = backing;
        part.start = a;
        part.end = b;
        part.offset = backing.offset + ((a - backing.start) << block_shift);
        return part;
    });
    u64 blocks = 0;
    for (const auto& piece : pieces) {
        blocks += piece.end - piece.start;
    }
    // The page table entries of the blocks become 0, as for memory never made resident (shaders
    // reading guest memory directly then see no data instead of another range's).
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    std::memset(staging.mapped, 0, blocks * sizeof(vk::DeviceAddress));
    u64 offset = staging.offset;
    for (const auto& piece : pieces) {
        const VAddr address = piece.start << block_shift;
        const u64 size = (piece.end - piece.start) << block_shift;
        const Buffer* arena = address_space[address >> arena_page_bits];
        ASSERT_MSG(arena && address >= arena->cpu_addr &&
                       address + size <= arena->cpu_addr + arena->size_bytes,
                   "Released arena range outside its arena");
        BindsForArena(arena)->binds.push_back({
            .resourceOffset = address - arena->cpu_addr,
            .size = size,
        });
        quarantined_runs.push_back({piece.memory, piece.offset, piece.mapped, piece.end - piece.start});
        // The guest copy is the only one now: the next use uploads it (and a static range whose
        // fast path skipped uploads takes the stream path, see UseArenaForStatic).
        memory_tracker->MarkRegionAsCpuModified(address, size);
        const u64 copy_size = (piece.end - piece.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, piece.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }
    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    has_pending_unbinds = true;
    resident_in_use_blocks -= blocks;
    Gow3Stats::arena_in_use_bytes = resident_in_use_blocks << block_shift;
    Gow3Stats::arena_released_bytes += blocks << block_shift;
    return blocks;
}

void BufferCache::ReclaimArenaMemory() {
    // gow3 20 Oct 2026 (GitHub #95): the arenas only grew. Every range the GPU ever used stayed
    // bound to video memory: a laptop, 16 min, one level change: 320 ->
    // 1,920 MiB, never less, while images were collected (1,401 -> 502 MiB). Through a long
    // session that fills cards with 4-8 GB, and the driver then moves memory to system RAM
    // (frame rate falling) or runs out. Now, once the arenas hold more than their limit, 1 MiB
    // windows of guest memory no arena use touched for GOW3_ARENA_RECLAIM_AGE_S (default 60) are
    // given back, oldest first, until the arenas hold 3/4 of it: only windows without GPU-written
    // data, pending readbacks or lazy pages (the guest copy is complete there), whose last use the
    // GPU has finished. Their memory is bound again before new memory is allocated, so the arenas
    // stay near the limit. GOW3_ARENA_LIMIT_MB (default a quarter of video memory, at least
    // 768; 384 in the low video memory mode, where windows go after 20 s instead of 60);
    // GOW3_ARENA_RECLAIM=0: off.
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_ARENA_RECLAIM");
        return !(env && env[0] == '0');
    }();
    if (!enabled) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next_reclaim_epoch) {
        return;
    }
    next_reclaim_epoch = now + std::chrono::seconds(2);
    reclaim_epoch = static_cast<u16>(reclaim_epoch + 1);
    if (!reclaim_epoch) {
        reclaim_epoch = 1;
    }
    epoch_start_tick[reclaim_epoch % ReclaimEpochRing] = scheduler.CurrentTick();

    static const u64 limit = [this] {
        u64 mib = 0;
        if (const char* env = std::getenv("GOW3_ARENA_LIMIT_MB"); env && env[0]) {
            mib = std::strtoull(env, nullptr, 10);
        } else {
            const auto& props = instance.GetMemoryProperties();
            const u32 heap = props.memoryTypes[arena_memory_type_index].heapIndex;
            mib = std::max<u64>(Vulkan::LowVram::Enabled() ? 384 : 768,
                                (props.memoryHeaps[heap].size >> 20) / 4);
            // Arenas in system memory (integrated GPUs) share it with the game: at most
            // 2 GiB, an eighth of the heap on smaller systems.
            if (!(props.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) {
                mib = std::max<u64>(768, std::min<u64>(2048, (props.memoryHeaps[heap].size >> 20) / 8));
            }
        }
        std::printf("GPU: buffer arenas give back memory unused for a while above %llu MiB "
                    "(GOW3_ARENA_LIMIT_MB; GOW3_ARENA_RECLAIM=0: off)\n",
                    static_cast<unsigned long long>(mib));
        std::fflush(stdout);
        return std::max<u64>(mib, 64) << 20;
    }();
    static const u16 age_epochs = [] {
        const char* env = std::getenv("GOW3_ARENA_RECLAIM_AGE_S");
        const int seconds = env && env[0] ? std::atoi(env) : Vulkan::LowVram::Enabled() ? 20 : 60;
        return static_cast<u16>(std::clamp(seconds, 6, 2 * (int(ReclaimEpochRing) - 4)) / 2);
    }();
    if ((resident_in_use_blocks << block_shift) <= limit) {
        return;
    }
    const u64 target = limit / 4 * 3;

    // Windows with resident memory, the longest unused first.
    std::vector<std::pair<u16, u64>> candidates;
    u64 previous = ~u64{0};
    for (const auto& backing : resident_ranges) {
        const u64 first = (backing.start << block_shift) >> ReclaimWindowBits;
        const u64 last = ((backing.end << block_shift) - 1) >> ReclaimWindowBits;
        for (u64 w = first; w <= last; ++w) {
            if (w == previous) {
                continue;
            }
            previous = w;
            const u16 epoch = window_epoch[w].load(std::memory_order_relaxed);
            const u16 age = static_cast<u16>(reclaim_epoch - epoch);
            if (epoch && age < age_epochs) {
                continue;
            }
            candidates.emplace_back(epoch ? age : u16(0xFFFF), w);
        }
    }
    std::ranges::sort(candidates, std::greater{});

    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    u64 released = 0;
    u32 windows = 0, busy = 0;
    for (const auto& [age, w] : candidates) {
        if ((resident_in_use_blocks << block_shift) <= target) {
            break;
        }
        const VAddr lo = w << ReclaimWindowBits;
        const u64 size = u64(1) << ReclaimWindowBits;
        // The GPU has finished everything submitted up to the end of the window's last epoch.
        if (const u16 epoch = window_epoch[w].load(std::memory_order_relaxed);
            epoch && age < ReclaimEpochRing - 2 &&
            !semaphore->IsFree(epoch_start_tick[static_cast<u16>(epoch + 1) % ReclaimEpochRing])) {
            ++busy;
            continue;
        }
        bool gpu_data;
        {
            std::shared_lock lock{gpu_ranges_mutex};
            gpu_data = gpu_modified_ranges.Intersects(lo, size);
        }
        if (gpu_data || memory_tracker->IsRegionGpuModified(lo, size) ||
            PendingReadbackTick(lo, size, false)) {
            ++busy;
            continue;
        }
        if (const auto lazy = lazy_pages.lower_bound(lo); lazy != lazy_pages.end() && lazy->first < lo + size) {
            ++busy;
            continue;
        }
        if (const u64 blocks = ReleaseArenaBlocks(lo >> block_shift, (lo + size) >> block_shift)) {
            released += blocks;
            ++windows;
        }
    }
    if (windows || busy) {
        std::printf("GPU: buffer arenas gave back %llu MiB (%u windows unused for %u s or more; %u "
                    "kept: GPU data or in use); %llu MiB in use, limit %llu MiB\n",
                    static_cast<unsigned long long>((released << block_shift) >> 20), windows,
                    u32(age_epochs) * 2, busy,
                    static_cast<unsigned long long>((resident_in_use_blocks << block_shift) >> 20),
                    static_cast<unsigned long long>(limit >> 20));
        std::fflush(stdout);
    }
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const Buffer* src_buffer{};
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 addr, u64 size) {
            ForEachLazyUploadPiece(addr, size, [&](VAddr piece, u64 piece_size) {
                copies.emplace_back(total_size_bytes, piece, piece_size);
                total_size_bytes += piece_size;
                if (!is_written && RecentWritesEnabled() &&
                    gpu_modified_ranges.Intersects(piece, piece_size)) {
                    u64 gpu_bytes = 0;
                    gpu_modified_ranges.ForEachInRange(piece, piece_size,
                                                       [&](VAddr lo, VAddr hi) { gpu_bytes += hi - lo; });
                    NoteUploadOverGpu(piece, piece_size, gpu_bytes,
                                      NewestGpuWriteInto(piece, piece + piece_size).writer);
                }
            });
        },
        [&] { src_buffer = UploadCopies(arena, copies, total_size_bytes); });

    if (src_buffer) {
        const ProfileCopy mark{scheduler, 0xB0F7'0001ull, "buffer upload (CPU-written guest memory)"};
        runtime.CopyBuffer(src_buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

const Buffer* BufferCache::UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                                        size_t total_size_bytes) {
    if (copies.empty()) {
        return nullptr;
    }
    // Resolve before launching host copies: a copy-thread fault could deadlock submission.
    for (const auto& copy : copies) {
        texture_cache.ResolveReadbacks(copy.dstOffset, copy.size, true, ReadbackSource::BufferUpload);
    }
    Gow3Stats::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
    // gow3: the guest memory is copied into staging on the copy threads, started now in
    // groups of about 1 MiB; submission and guest-visible fences wait for them
    // (Scheduler::WaitHostCopies).
    if (!Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads) && total_size_bytes < 1_MB) {
        // Small uploads join the calling thread's batch.
        for (auto& copy : copies) {
            SmallGuestCopy({
                .run = &RunGuestCopy,
                .context = this,
                .source = copy.dstOffset,
                .destination = reinterpret_cast<u64>(staging.mapped + copy.srcOffset),
                .size = copy.size,
                .extra = reinterpret_cast<u64>(staging.buffer),
            });
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        return staging.buffer;
    }
    if (!Gow3Toggle::Disabled(Gow3Toggle::DeferredUploads)) {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            Gow3Copy::Async([group, memory = memory, staging] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
                staging.Flush();
            });
        };
        for (auto& copy : copies) {
            group->push_back({copy.dstOffset, staging.mapped + copy.srcOffset, copy.size});
            group_bytes += copy.size;
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
            if (group_bytes >= 1_MB) {
                launch();
                group = std::make_shared<Group>();
                group_bytes = 0;
            }
        }
        if (!group->empty()) {
            launch();
        }
        return staging.buffer;
    }
    const auto copy = [&](std::size_t i) {
        memory->CopySparseMemory(copies[i].dstOffset, staging.mapped + copies[i].srcOffset,
                                 copies[i].size);
    };
    if (total_size_bytes >= 2_MB && copies.size() > 1) {
        Gow3Copy::ParallelFor(copies.size(), copy);
    } else {
        for (std::size_t i = 0; i < copies.size(); ++i) {
            copy(i);
        }
    }
    for (auto& copy : copies) {
        copy.srcOffset += staging.offset;
        copy.dstOffset -= arena->cpu_addr;
    }
    staging.Flush();
    return staging.buffer;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    // gow3: most texel buffers alias no image; remember misses until images change.
    const u64 generation = texture_cache.RegistryGeneration();
    auto& miss = image_miss_cache[((device_addr >> 6) ^ size * 0x9E3779B1u) % image_miss_cache.size()];
    if (miss.address == device_addr && miss.size == size && miss.generation == generation &&
        !Gow3Toggle::Disabled(Gow3Toggle::TextureBindingMemo)) {
        return false;
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        miss = {device_addr, size, generation};
        return false;
    }
    // gow3: the lookups above only read what the texture binding helper leaves alone; the
    // copy below changes image state, so the helper finishes first.
    runtime.BeforeImageAccess();
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetSparseQueue().bindSparse(sparse_info);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        Vulkan::DumpGpuCheckpoints(instance);
        Vulkan::DumpDeviceFault(instance);
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
    // gow3 20 Oct 2026: memory whose unbinds are now submitted may be bound elsewhere.
    if (has_pending_unbinds) {
        has_pending_unbinds = false;
        free_runs.insert(free_runs.end(), quarantined_runs.begin(), quarantined_runs.end());
        quarantined_runs.clear();
    }
}

} // namespace VideoCore

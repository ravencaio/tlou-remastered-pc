#pragma once
// Keyboard and mouse button prompts (gow3_prompts.cpp).

#include <memory>
#include <vector>

#include "common/types.h"

namespace Vulkan {
class Instance;
class Scheduler;
class Runtime;
}

namespace Core { class MemoryManager; }
namespace AmdGpu { struct Image; }

namespace Gow3Prompts {

// The Last of Us Remastered's HUD sprite shader (pixel shader hash): world prompts, menu tabs and
// other HUD pictures, colour and alpha in two 127x127 textures. Its vertices are 64 bytes.
constexpr u64 PromptShaderHash = 0x0dfdb503;
constexpr u32 PromptVertexStride = 64;

// A draw of that shader: its index buffer at the draw's first index, vertex buffer and viewport height.
struct PromptGeometry {
    u32 num_indices = 0;
    bool is_index16 = true;
    VAddr ib_addr = 0;
    VAddr vb_addr = 0;
    float viewport_h = 1080.0f;
};

// Whether icons other than the PlayStation ones show (otherwise the draws are left alone).
bool FloatingPromptsActive();
// The button (GOW3_PROMPT_*) a draw of the HUD sprite shader shows as a prompt, or -1.
int IdentifyFloatingPrompt(const AmdGpu::Image& s0, const AmdGpu::Image& s1, Core::MemoryManager* memory,
                           const PromptGeometry& geom);

// Texels to upload instead of an image's guest data: linear, per mip its offset, row length and
// height in texels (BC: whole blocks).
struct Texels {
    std::vector<u8> data;
    struct Mip {
        u32 offset, row, height;
    };
    std::vector<Mip> mips;
};

// Texture cache upload (TextureCache::RefreshImage) of a 512x512 or 256x512 RGBA8 image or a BC1 one (64-512):
// when its first mip is one of the games' button icon atlases or button pictures and other icons
// than the PlayStation ones show, the texels to upload instead; nullptr otherwise.
std::shared_ptr<const Texels> Replacement(VAddr address, u32 guest_size, u32 mip0_size, u32 width, u32 height,
                                          u32 levels, bool bc1, u32 tile_mode);

// Binds the button's redrawn image in place of the draw's two textures; false: the game's stay.
bool BindFloatingPrompt(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                        Vulkan::Runtime& runtime, void* p_img0_info, void* p_img1_info,
                        int button);
void ClearFloatingPrompts();

} // namespace Gow3Prompts

// Pad side (runtime_pad.c): the labels per PS4 button (GOW3_PROMPT_* in gow3gpu.h) of set 0 (keys
// and mouse buttons) or 1 (other controllers), and the device whose icons show.
extern "C" void gow3gpu_prompts_set_labels(int set, const char* const* labels, int count);
extern "C" void gow3gpu_prompts_set_device(int device);

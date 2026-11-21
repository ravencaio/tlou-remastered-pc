// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 20 Nov 2026: Intel XeSS Super Resolution (XeSS-SR, Vulkan) next to FSR and DLSS, for native
// anti-aliasing and the upscaling presets. Adapted from SpartanPort's XeSS anti-aliasing
// (FelipePrado0/SpartanPort #1276, vk_xess.cpp), which runs it at the game's size only. It takes
// what DLSS takes: the jittered colour, the depth and the motion vectors at the render size
// (render pixels, previous minus current position, without the jitter), and writes the output
// size into the upscaler's output image.
//
// libxess.dll (Intel's binary, redistributable unmodified under the Intel Simplified Software
// License; tools/fetch_aa_files.sh xess) is loaded at run time from next to gow3-probe.exe, so the
// game runs without it. Windows only: Intel ships no other binaries. The headers in
// third_party/xess are Intel's (MIT). Its device extensions and features are enabled only when
// XeSS is the upscaler the game starts with (vk_instance.cpp): chosen later in the menu, FSR 3.1
// stands in until the next start.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class Xess {
public:
    Xess(const Instance& instance, Scheduler& scheduler);
    ~Xess();

    /// An image XeSS reads or writes; `view` covers `aspect` of mip 0, layer 0.
    struct Image {
        vk::Image image;
        vk::ImageView view;
        vk::Format format;
        vk::ImageAspectFlags aspect; ///< of the view
        u32 width, height;
    };
    struct Frame {
        vk::CommandBuffer cmdbuf;
        Image color, depth, motion; ///< layout General; the scene in their top-left corner
        Image output;               ///< output size, layout General, usage storage
        /// Optional responsive pixel mask (R8 at the render size, layout General; 1 = rely on
        /// this frame, not the history; view null = none), clipped to responsive_max.
        Image responsive;
        float responsive_max;
        u32 input_width, input_height; ///< the scene's size (the render size)
        std::array<float, 2> jitter;   ///< pixels, as the projection was shifted
        bool reset;
        bool hdr;           ///< linear HDR colour, else display-range (LDR) colour
        bool auto_exposure; ///< with hdr: XeSS's own exposure, else an exposure scale of 1
        const char* colour; ///< what the colour is, for the log
    };

    /// Records XeSS into `frame.cmdbuf`: the inputs go to the shader read-only layout and back
    /// to General, the output stays in General. False when XeSS cannot run (Problem() says why).
    /// A new size, quality or colour range reinitialises it after the submitted work is done.
    bool Record(const Frame& frame);

    /// Why the last Record failed, or null.
    [[nodiscard]] const char* Problem() const noexcept;
    /// XeSS will not run again in this session.
    [[nodiscard]] bool Fatal() const noexcept;

    /// XeSS is the upscaler the game starts with: the device is created for it.
    static bool SelectedAtStart();
    /// libxess.dll is next to the game and has the Vulkan entry points; `why` says what is
    /// missing otherwise.
    static bool LibraryAvailable(std::string* why = nullptr);
    /// Instance creation: the instance extensions XeSS asks for, those not enabled yet.
    static void AppendInstanceExtensions(std::vector<const char*>& enabled);
    /// The device extensions XeSS asks for (empty when the library is not there).
    static std::vector<const char*> DeviceExtensions(vk::Instance instance,
                                                     vk::PhysicalDevice physical_device);
    /// Patches the device features XeSS needs into the chain of the device being created:
    /// `chain` is its head (a VkPhysicalDeviceFeatures2) and comes back as the head to create the
    /// device with. False when the library is missing or gives no features; the chain is then
    /// left as it was.
    static bool PatchDeviceFeatures(vk::Instance instance, vk::PhysicalDevice physical_device,
                                    void*& chain);
    /// Device creation: whether the device has the extensions and features XeSS needs (`why`,
    /// kept, says what is missing). Without them Record fails instead of creating a context on a
    /// device that lacks them.
    static void SetDeviceReady(bool ready, const char* why = nullptr);
    [[nodiscard]] static bool DeviceReady();
    /// Why the device is not ready (empty when it is, or when XeSS was not selected at start).
    [[nodiscard]] static std::string DeviceProblem();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan

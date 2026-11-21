#!/usr/bin/env bash
# The files for NVIDIA DLAA and AMD FSR 4 Native AA (in-game menu -> Image -> Anti-aliasing), which
# are not part of this repository:
#   - NVIDIA DLSS SDK v310.9.16 (github.com/NVIDIA/DLSS, NVIDIA's license: LICENSE.txt): headers,
#     nvsdk_ngx_s.lib and nvngx_dlss.dll into downloads/dlss-sdk-<version>; with MSVC (Visual
#     Studio 2022 or its Build Tools) builds gow3_pc/gpu/dlss_bridge -> gow3_dlss.dll and puts
#     both DLLs in gow3_pc/out beside the runtime. About 63 MB.
#   - FSR 4 v07 INT8 native model (AMD's MIT-licensed FidelityFX SDK source, built by Q2RTX:
#     github.com/FireBurn/Q2RTX) into gow3_pc/fsr4_shaders, the 1080 and 2160 tiers. About 6 MB.
#   - Intel XeSS (in-game menu or launcher: XeSS): libxess.dll from github.com/intel/xess, Intel's
#     unmodified binary (redistributable under the Intel Simplified Software License, kept beside
#     it as libxess-LICENSE.txt with libxess-third-party-programs.txt), into gow3_pc/out. The pin
#     and checksum are SpartanPort's (#1276). About 78 MB.
# Usage: tools/fetch_aa_files.sh [dlss] [fsr4] [xess] (default dlss and fsr4). Files already
# there are kept.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
want=${*:-dlss fsr4}

fetch() {  # fetch URL FILE: downloads once, through a .part file
    [[ -s $2 ]] && return 0
    mkdir -p -- "$(dirname -- "$2")"
    curl -fsSL --retry 3 -o "$2.part" "$1"
    mv -- "$2.part" "$2"
}

if [[ $want == *dlss* ]]; then
    version=310.9.16
    sdk=downloads/dlss-sdk-$version
    base=https://raw.githubusercontent.com/NVIDIA/DLSS/v$version
    for header in nvsdk_ngx.h nvsdk_ngx_defs.h nvsdk_ngx_defs_dlssd.h nvsdk_ngx_defs_dlssg.h \
        nvsdk_ngx_defs_vk.h nvsdk_ngx_helpers.h nvsdk_ngx_helpers_cuda.h nvsdk_ngx_helpers_d3d.h \
        nvsdk_ngx_helpers_dlssd.h nvsdk_ngx_helpers_dlssd_cuda.h nvsdk_ngx_helpers_dlssd_d3d.h \
        nvsdk_ngx_helpers_dlssd_vk.h nvsdk_ngx_helpers_dlssg.h nvsdk_ngx_helpers_dlssg_d3d.h \
        nvsdk_ngx_helpers_dlssg_vk.h nvsdk_ngx_helpers_vk.h nvsdk_ngx_loader.h nvsdk_ngx_params.h \
        nvsdk_ngx_params_dlssd.h nvsdk_ngx_params_dlssg.h nvsdk_ngx_standalone_common.h \
        nvsdk_ngx_standalone_cuda.h nvsdk_ngx_vk.h; do
        fetch "$base/include/$header" "$sdk/include/$header"
    done
    for file in LICENSE.txt lib/Windows_x86_64/x64/nvsdk_ngx_s.lib lib/Windows_x86_64/rel/nvngx_dlss.dll; do
        fetch "$base/$file" "$sdk/$file"
    done
    # The bridge: MSVC through CMake's Visual Studio generator. Vulkan headers from the MSYS2
    # CLANG64 tree, copied alone (its C library headers must not reach MSVC).
    msys=${MSYS2_ROOT:-/c/msys64}
    msys=${msys%/}  # on GitHub's runner the MSYS2 root is "/": not "//clang64" (a network path)
    vulkan=downloads/vulkan-include
    mkdir -p "$vulkan"
    cp -r "$msys/clang64/include/vulkan" "$msys/clang64/include/vk_video" "$vulkan/"
    # Visual Studio (any edition with the C++ tools) through its own CMake; the generator follows
    # its version (17 = 2022, 18 = 2026: GitHub's windows-latest runners move between them).
    cmake_exe=$(command -v cmake.exe || true)
    generator="Visual Studio 17 2022"
    vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    if [[ -x $vswhere ]]; then
        vs=$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
             -property installationPath | tr -d '\r')
        major=$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
                -property installationVersion | tr -d '\r' | cut -d. -f1)
        [[ $major == 18 ]] && generator="Visual Studio 18 2026"
        if [[ -n $vs && -x "$(cygpath -u "$vs")/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ]]; then
            cmake_exe="$(cygpath -u "$vs")/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
        fi
    fi
    [[ -n $cmake_exe && -x $cmake_exe ]] || { echo "No MSVC (Visual Studio Build Tools) found: DLSS bridge not built" >&2; exit 1; }
    root=$(cygpath -w "$PWD")
    "$cmake_exe" -S "$root\\gow3_pc\\gpu\\dlss_bridge" -B "$root\\gow3_pc\\out\\dlss-bridge" \
        -G "$generator" -A x64 -DDLSS_SDK_ROOT="$root\\downloads\\dlss-sdk-$version" \
        -DVULKAN_INCLUDE="$root\\downloads\\vulkan-include" > /dev/null
    "$cmake_exe" --build "$root\\gow3_pc\\out\\dlss-bridge" --config Release > /dev/null
    cp gow3_pc/out/dlss-bridge/Release/gow3_dlss.dll "$sdk/lib/Windows_x86_64/rel/nvngx_dlss.dll" gow3_pc/out/
    echo "DLSS: gow3_pc/out/gow3_dlss.dll and nvngx_dlss.dll ($version)"
fi

if [[ $want == *xess* ]]; then
    commit=de0fb9c1c510661c571164e1418ceca8101dab69
    sha256=251659dd84a3e84de67c886a4186e01f3eca49b00641906fe38bb6b807e5d5b7
    base=https://raw.githubusercontent.com/intel/xess/$commit
    dest=downloads/xess-$commit
    fetch "$base/bin/libxess.dll" "$dest/libxess.dll"
    fetch "$base/LICENSE.txt" "$dest/LICENSE.txt"
    fetch "$base/third-party-programs.txt" "$dest/third-party-programs.txt"
    if [[ $(sha256sum "$dest/libxess.dll" | cut -d' ' -f1) != "$sha256" ]]; then
        rm -f -- "$dest/libxess.dll"
        echo "XeSS: libxess.dll does not match the pinned release" >&2
        exit 1
    fi
    mkdir -p gow3_pc/out
    cp "$dest/libxess.dll" gow3_pc/out/libxess.dll
    cp "$dest/LICENSE.txt" gow3_pc/out/libxess-LICENSE.txt
    cp "$dest/third-party-programs.txt" gow3_pc/out/libxess-third-party-programs.txt
    echo "XeSS: gow3_pc/out/libxess.dll and its license"
fi

if [[ $want == *fsr4* ]]; then
    commit=ae8d628fae208813172446d1e49ed94150b04658
    base="https://raw.githubusercontent.com/FireBurn/Q2RTX/$commit/baseq2/fsr4_shaders"
    dest=gow3_pc/fsr4_shaders
    model=fsr4_model_v07_i8_native
    files=(LICENSE-FSR4-v07.txt rcas.spv fsr4_rcas.spv spd_auto_exposure.spv "${model}_pre.spv" "${model}_post.spv")
    # Native AA, and the upscaling models a render resolution below the output picks by its
    # scale (1.5 Quality, 1.7 Balanced, 2 Performance, more Ultra Performance): ~5 MB each.
    for preset in native quality balanced performance ultraperf; do
        model=fsr4_model_v07_i8_$preset
        files+=("${model}_initializers.bin" "${model}_pre_weights.bin" "${model}_shader_manifest.json")
        for tier in 1080 2160; do
            files+=("${model}_${tier}_pre.spv" "${model}_${tier}_post.spv")
            for pass in $(seq 1 12); do
                files+=("${model}_${tier}_pass${pass}.spv")
            done
        done
    done
    for file in "${files[@]}"; do
        fetch "$base/$file" "$dest/$file"
    done
    echo "FSR 4: ${#files[@]} files in $dest"
fi

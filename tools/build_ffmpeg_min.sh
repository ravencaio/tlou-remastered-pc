#!/usr/bin/env bash
# Builds a small shared FFmpeg for the Linux package into a prefix (default /opt/ffmpeg-min):
# only what the port uses (H.264/HEVC/AAC/MP3 decoders, the MP4/MPEG-TS demuxers, swscale,
# swresample) and no external library, so its .so files run on any 64-bit Linux and ship in the
# package (tools/package_linux.sh) instead of the distribution's FFmpeg, whose soname changes with
# each major version (libavcodec.so.63 on Arch is not on a Steam Deck).
#   tools/build_ffmpeg_min.sh <version> [prefix]
# Then build with PKG_CONFIG_PATH=<prefix>/lib/pkgconfig. Needs a C compiler, make and nasm.
set -euo pipefail
version=${1:?FFmpeg version, e.g. 8.0}
prefix=${2:-/opt/ffmpeg-min}
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
curl -fL --retry 3 -o "$work/ffmpeg.tar.xz" "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz"
echo "ffmpeg-$version.tar.xz sha256 $(sha256sum "$work/ffmpeg.tar.xz" | cut -d' ' -f1)"
tar -xJf "$work/ffmpeg.tar.xz" -C "$work"
cd "$work/ffmpeg-$version"
./configure --prefix="$prefix" --enable-shared --disable-static --enable-pic \
    --disable-programs --disable-doc --disable-autodetect --disable-network \
    --disable-everything --disable-avdevice --disable-avfilter \
    --enable-decoder=h264,hevc,aac,aac_fixed,mp3,mp3float \
    --enable-parser=h264,hevc,aac,mpegaudio \
    --enable-demuxer=mov,mpegts,h264,hevc,aac,mp3 \
    --enable-protocol=file >/dev/null
make -j"$(nproc)" >/dev/null
make install >/dev/null
# LGPL 2.1: its text ships with the libraries (package_linux.sh copies it to licenses/ffmpeg).
mkdir -p "$prefix/share/licenses/ffmpeg"
cp COPYING.LGPLv2.1 LICENSE.md "$prefix/share/licenses/ffmpeg/"
echo "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz" > "$prefix/share/licenses/ffmpeg/SOURCE"
ls "$prefix/lib"/*.so.*

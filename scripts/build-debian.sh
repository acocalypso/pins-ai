#!/usr/bin/env bash
# ARM64 package containing the plugin, native runtime and verified model assets.
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo 'usage: build-debian.sh host-publish-directory core-package-version output-directory package-version' >&2
  exit 64
fi
repo=$(cd "$(dirname "$0")/.." && pwd)
host=$(realpath "$1"); core_version=$2; out=$(realpath -m "$3"); version=$4
test "$(dpkg --print-architecture)" = arm64
test -s "$host/NINA.Profile.dll"
mkdir -p "$out"
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
dotnet build "$repo/src/Pins.AI.Plugin/Pins.AI.Plugin.csproj" -c Release -p:HostDirectory="$host" -o "$work/plugin"
root="$work/debroot"; assets="$root/usr/share/pins-ai"; runtime="$root/usr/lib/pins-ai"
plugin="$root/home/pi/.local/share/NINA/Plugins/3.0.0/pins-ai"
mkdir -p "$plugin" "$assets/models" "$runtime" "$root/DEBIAN" "$root/usr/share/doc/pins-plugin-ai"
install -m644 "$work/plugin/Pins.AI.Plugin.dll" "$work/plugin/Pins.AI.Runtime.dll" "$plugin/"
cache_args=(); if [ -n "${PINS_AI_MODEL_CACHE:-}" ]; then cache_args=(--cache "$PINS_AI_MODEL_CACHE"); fi
python3 "$repo/scripts/fetch-bundled-model.py" --output "$assets/models" "${cache_args[@]}"
commit=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["llamaCommit"])' "$repo/models/bundled.json")
llama=${PINS_AI_LLAMA_SOURCE:-"$work/llama.cpp"}
if [ ! -d "$llama/.git" ]; then git clone https://github.com/ggml-org/llama.cpp.git "$llama"; git -C "$llama" checkout --detach "$commit"; fi
test "$(git -C "$llama" rev-parse HEAD)" = "$commit"
cmake -S "$llama" -B "$llama/build-package" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF
cmake --build "$llama/build-package" --target llama-server -j "${PINS_AI_BUILD_JOBS:-2}"
install -m755 "$llama/build-package/bin/llama-server" "$runtime/llama-server"
cc -O2 -Wall -Wextra -Werror "$repo/native/pins-ai-exec.c" -o "$runtime/pins-ai-exec"
readelf -h "$runtime/llama-server" | grep -q AArch64
install -m644 "$repo/LICENSE" "$repo/THIRD_PARTY_NOTICES.md" "$repo/models/bundled.json" "$root/usr/share/doc/pins-plugin-ai/"
install -m644 "$llama/LICENSE" "$root/usr/share/doc/pins-plugin-ai/llama-LICENSE"
cp -a "$llama/licenses" "$root/usr/share/doc/pins-plugin-ai/llama-licenses"
install -m644 "$repo/models/bundled/LICENSE.txt" "$root/usr/share/doc/pins-plugin-ai/Qwen3-LICENSE.txt"
install -m644 "$repo/packaging/default-config.json" "$assets/default-config.json"
install -m755 "$repo/packaging/postinst" "$root/DEBIAN/postinst"
size=$(du -sk "$root" | cut -f1)
cat > "$root/DEBIAN/control" <<EOF
Package: pins-plugin-ai
Version: $version
Section: science
Priority: optional
Architecture: arm64
Maintainer: PINS Team
Depends: pins (= $core_version), libc6, libstdc++6, libgcc-s1, libgomp1, python3, systemd
Installed-Size: $size
Description: Plugin-owned local Qwen3 MCP chat with diagnostic checkpoint 700
 Contains verified model and separate LoRA weights, llama.cpp, licenses and provenance.
EOF
dpkg-deb --root-owner-group --build "$root" "$out/pins-plugin-ai_${version}_arm64.deb"
(cd "$out" && sha256sum "pins-plugin-ai_${version}_arm64.deb" > "pins-plugin-ai_${version}_arm64.deb.sha256")

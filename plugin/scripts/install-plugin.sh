#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'usage: install-plugin.sh build-output-directory plugin-directory' >&2
  exit 64
fi
source_dir=$(realpath "$1")
plugin_dir=$(realpath -m "$2")
test -f "$source_dir/Pins.AI.Plugin.dll"
test -f "$source_dir/Pins.AI.Runtime.dll"
# Exact two managed assemblies, never duplicate PINS/framework dependencies.
mkdir -p "$plugin_dir"
if [ -f "$plugin_dir/Pins.AI.Plugin.dll" ]; then
  backup="${plugin_dir}.backup-$(date -u +%Y%m%dT%H%M%SZ)"
  cp -a "$plugin_dir" "$backup"
  echo "Previous plugin backed up to $backup"
fi
for name in Pins.AI.Plugin.dll Pins.AI.Runtime.dll; do
  # Atomic replacement preserves an already-loaded assembly's mapped inode.
  install -m 644 "$source_dir/$name" "$plugin_dir/.$name.new"
  mv -f "$plugin_dir/.$name.new" "$plugin_dir/$name"
done
echo "Installed in $plugin_dir. Restart PINS only when no session is active."

#!/bin/sh
set -eu

# Use the application downloader and pinned verification rules, once before running tests.
# Existing models are reused in XDG_CACHE_HOME, or ~/.cache when it is unset.
test_binary=${1:-target/release/visiongrep}
if [ ! -x "$test_binary" ]; then
    echo "Build visiongrep first with cargo build --release, or pass a built binary path." >&2
    exit 2
fi
model_setup_dir=$(mktemp -d)
trap 'rm -rf "$model_setup_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# One red RGB pixel; a fresh corpus ensures both vision and text artifacts are required.
printf '%s' 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC' | base64 -d > "$model_setup_dir/pixel.png"
"$test_binary" 'a red picture' "$model_setup_dir" \
    --no-cache --threshold -1 --json > /dev/null

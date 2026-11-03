#!/usr/bin/env bash
set -euo pipefail

: "${BUCKET:?BUCKET must name the benchmark artifact bucket}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

if ! python3 -m venv "$work_dir/venv"; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
    python3 -m venv "$work_dir/venv"
fi
"$work_dir/venv/bin/python" -m pip install \
    onnxruntime==1.27.0 onnx==1.20.1 numpy==2.4.6 Pillow==13.0.0 tokenizers==0.23.3

mkdir "$work_dir/coco"
curl --fail --location --silent --show-error \
    https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip -o "$work_dir/val2017.zip"
curl --fail --location --silent --show-error \
    https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip \
    -o "$work_dir/annotations_trainval2017.zip"
unzip -q "$work_dir/val2017.zip" -d "$work_dir/coco"
unzip -q "$work_dir/annotations_trainval2017.zip" 'annotations/captions_val2017.json' -d "$work_dir/coco"

vision_int8_sha256=f6ae78a288ea20d48c0c6190200aa62dff66f5ccc4c7c2bb7f2d7999b4236ffe
vision_int8="$work_dir/visual-int8.onnx"
aws s3 cp "s3://$BUCKET/artifacts/$vision_int8_sha256.onnx" "$vision_int8" --only-show-errors
printf '%s  %s\n' "$vision_int8_sha256" "$vision_int8" | sha256sum --check

objects="$HOME/.cache/visiongrep-bench/objects"
"$work_dir/venv/bin/python" benchmarks/model_contract/evaluate_int8.py \
    --coco "$work_dir/coco" \
    --visual-fp32 "$objects/3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c" \
    --textual-fp32 "$objects/ee267cd64f0f77362670ae0140476ed51ee8c5a761d41636e09997f2fdddcacc" \
    --visual-int8 "$vision_int8" \
    --tokenizer "$objects/72ed5c96db5729294468543e4bc75fce14ca63f58e37300290189ba1c1e52b85"

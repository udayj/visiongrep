# Third-party notices

## DataComp OpenCLIP model

VisionGrep downloads pinned ONNX format conversions of
[`laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K`](https://huggingface.co/laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K),
a ViT-B/32 model trained by Mehdi Cherti on DataComp-1B using OpenCLIP. The
upstream model is pinned at revision
`4afec35ffe57a943d569ff7ee888061830164da8`, and its model card declares the
model under the MIT license. The ONNX artifacts are pinned conversions from
the rclip model repository at revision
`17b9d07433aad73f70d338d8a1c7a4cef83887e0`. The OpenCLIP copyright and
permission notice is reproduced in
[`third_party/OPENCLIP_LICENSE.txt`](third_party/OPENCLIP_LICENSE.txt).

## OpenAI CLIP tokenizer

The OpenCLIP tokenizer is derived from OpenAI CLIP under the MIT license. The
OpenAI copyright and permission notice is reproduced in
[`third_party/OPENAI_CLIP_LICENSE.txt`](third_party/OPENAI_CLIP_LICENSE.txt).

## Pillow-compatible bicubic resizing

`src/pillow_resize.rs` implements the coefficient construction and fixed-point
rounding behavior of Pillow 12.3.0's `src/libImaging/Resample.c`. Pillow is
licensed under the MIT-CMU license, reproduced in
[`third_party/PILLOW_LICENSE.txt`](third_party/PILLOW_LICENSE.txt).

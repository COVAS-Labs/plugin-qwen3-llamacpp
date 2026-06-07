# COVAS:NEXT Plugin Qwen3 llama.cpp

Run `lucaelin/qwen3-0.6b-cn-gguf` locally with `llama-cpp-python`.

## Provider

- `qwen3-llamacpp`

## Defaults

- Context size: `16384`
- Max output tokens: `1024`
- GPU layers: `-1` (all)
- Device: `auto`

## Vulkan

The release workflow builds `llama-cpp-python` with `CMAKE_ARGS=-DGGML_VULKAN=on` and `FORCE_CMAKE=1`.

The plugin probes Vulkan devices with the Python `vulkan` binding during initialization and exposes detected devices in the device dropdown:

- Auto
- CPU only
- Vulkan auto
- Vulkan device N: detected device name

When a detected Vulkan device is selected, the plugin sets `GGML_VK_VISIBLE_DEVICES` to that device index before loading the model.

If the Python Vulkan probe is unavailable, the dropdown still shows `Vulkan auto`. You can then use the `Vulkan visible devices` setting to set `GGML_VK_VISIBLE_DEVICES`, for example `0`, `1`, or `0,1`. Leave it empty to let llama.cpp choose automatically.

## Packaging

Use `./pack.sh` or `./pack.ps1`. Release builds download `Qwen3-0.6B-grpo-ckpt700-q8_0.gguf` from Hugging Face into `model/`.

#!/bin/bash

if [ -d "dist" ]; then
    rm -rf dist
fi

mkdir dist

if [ -f "requirements.txt" ]; then
    CMAKE_ARGS="-DGGML_VULKAN=on" FORCE_CMAKE=1 pip install --target ./deps -r requirements.txt
fi

artifacts=(
    "cn-plugin-qwen3-llamacpp.py"
    "requirements.txt"
    "manifest.json" "__init__.py"
    "model"
)

if [ -d "deps" ]; then
    artifacts+=("deps")
fi

zip -r -9 "dist/cn-plugin-qwen3-llamacpp.zip" "${artifacts[@]}"

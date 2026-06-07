if (Test-Path "dist") {
    Remove-Item -Recurse -Force "dist"
}

New-Item "dist" -ItemType Directory

if (Test-Path "requirements.txt") {
    $env:CMAKE_ARGS = "-DGGML_VULKAN=on"
    if ($env:VULKAN_SDK) {
        $vulkanInclude = Join-Path $env:VULKAN_SDK "Include"
        $vulkanLibrary = Join-Path $env:VULKAN_SDK "Lib\vulkan-1.lib"
        $glslc = Join-Path $env:VULKAN_SDK "Bin\glslc.exe"
        $env:CMAKE_ARGS += " -DVulkan_INCLUDE_DIR=`"$vulkanInclude`" -DVulkan_LIBRARY=`"$vulkanLibrary`" -DVulkan_GLSLC_EXECUTABLE=`"$glslc`""
    }
    $env:FORCE_CMAKE = "1"
    pip install --target ./deps -r requirements.txt
}

$artifacts = "cn-plugin-qwen3-llamacpp.py", "requirements.txt", "manifest.json", "__init__.py", "model"

if (Test-Path "deps") {
    $artifacts += "deps"
}

$compress = @{
LiteralPath = $artifacts
CompressionLevel = "Fastest"
DestinationPath = "dist\cn-plugin-qwen3-llamacpp.zip"
}
Compress-Archive @compress

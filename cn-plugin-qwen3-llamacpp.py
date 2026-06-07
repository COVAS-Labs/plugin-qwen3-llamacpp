"""
Qwen3 llama.cpp plugin for COVAS:NEXT.
"""

from __future__ import annotations

import os
import html
import json
import re
from time import time
from typing import Any, List, Optional, override

try:
    from lib.Logger import ModelUsageStats, log
except ImportError:
    from lib.Logger import log

    class ModelUsageStats:  # type: ignore[no-redef]
        def __init__(
            self,
            input_tokens: int | None = None,
            output_tokens: int | None = None,
            total_tokens: int | None = None,
            cached_tokens: int | None = None,
            reasoning_tokens: int | None = None,
            provider: str | None = None,
            model_name: str | None = None,
            response_ms: float | None = None,
            output_chars: int | None = None,
        ):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.total_tokens = total_tokens
            self.cached_tokens = cached_tokens
            self.reasoning_tokens = reasoning_tokens
            self.provider = provider
            self.model_name = model_name
            self.response_ms = response_ms
            self.output_chars = output_chars

from lib.Models import LLMError, LLMModel
from lib.PluginBase import PluginBase, PluginManifest
from lib.PluginSettingDefinitions import (
    ModelProviderDefinition,
    NumericalSetting,
    ParagraphSetting,
    PluginSettings,
    SelectOption,
    SelectSetting,
    SettingsGrid,
    TextSetting,
    ToggleSetting,
)


MODEL_FILE = "Qwen3-0.6B-grpo-ckpt700-q8_0.gguf"
THINK_PREFILL = "<think>\n\n</think>\n\n"


def _strip_thinking(text: str) -> str:
    candidate = text
    if candidate.startswith(THINK_PREFILL):
        candidate = candidate[len(THINK_PREFILL) :]
    candidate = re.sub(r"<think>.*?</think>\s*", "", candidate, flags=re.DOTALL)
    candidate = re.sub(r"^</think>\s*", "", candidate)
    candidate = re.sub(r"^<think>.*$", "", candidate, flags=re.DOTALL)
    return candidate.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _decode_vk_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.split("\x00", 1)[0]
    try:
        return bytes(value).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    except Exception:
        return str(value)


def _detect_vulkan_devices() -> list[tuple[int, str]]:
    try:
        import vulkan as vk

        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="COVAS Qwen3 llama.cpp",
            applicationVersion=1,
            pEngineName="COVAS:NEXT",
            engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )
        instance = vk.vkCreateInstance(create_info, None)
        try:
            physical_devices = vk.vkEnumeratePhysicalDevices(instance)
            devices: list[tuple[int, str]] = []
            for index, physical_device in enumerate(physical_devices):
                properties = vk.vkGetPhysicalDeviceProperties(physical_device)
                devices.append((index, _decode_vk_string(properties.deviceName)))
            return devices
        finally:
            vk.vkDestroyInstance(instance, None)
    except Exception as exc:
        log("debug", f"Vulkan device probe unavailable: {exc}")
        return []


def _device_select_options() -> list[SelectOption]:
    options: list[SelectOption] = [
        SelectOption(key="auto", label="Auto", value="auto", disabled=False),
        SelectOption(key="cpu", label="CPU only", value="cpu", disabled=False),
        SelectOption(key="vulkan", label="Vulkan auto", value="vulkan", disabled=False),
    ]

    devices = _detect_vulkan_devices()
    options.extend(
        SelectOption(
            key=f"vulkan_{index}",
            label=f"Vulkan device {index}: {name}",
            value=f"vulkan:{index}",
            disabled=False,
        )
        for index, name in devices
    )
    return options


def _int_setting(settings: dict[str, Any], key: str, default: int) -> int:
    value = settings.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_setting(settings: dict[str, Any], key: str, default: float) -> float:
    value = settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_setting(settings: dict[str, Any], key: str, default: bool) -> bool:
    value = settings.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


class Qwen3LlamaCppModel(LLMModel):
    def __init__(self, model_dir: str, settings: dict[str, Any]):
        super().__init__("qwen3-0.6b-cn-gguf", provider_name="qwen3-llamacpp")
        self.model_dir = model_dir
        self.settings = settings
        self._llm = None

        self.n_ctx = _int_setting(settings, "n_ctx", 16384)
        self.max_tokens = _int_setting(settings, "max_tokens", 1024)
        self.temperature = _float_setting(settings, "temperature", 0.3)
        self.top_p = _float_setting(settings, "top_p", 0.95)
        self.repeat_penalty = _float_setting(settings, "repeat_penalty", 1.05)
        self.n_threads = _int_setting(settings, "n_threads", max(1, (os.cpu_count() or 4) // 2))
        self.n_gpu_layers = _int_setting(settings, "n_gpu_layers", -1)
        self.main_gpu = _int_setting(settings, "main_gpu", 0)
        self.device = str(settings.get("device", "auto"))
        self.vulkan_visible_devices = str(settings.get("vulkan_visible_devices", "")).strip()
        self.verbose = _bool_setting(settings, "verbose", False)

    def _model_path(self) -> str:
        path = os.path.join(self.model_dir, MODEL_FILE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    def _configure_device(self) -> int:
        if self.device == "cpu":
            os.environ.pop("GGML_VK_VISIBLE_DEVICES", None)
            return 0

        if self.device.startswith("vulkan:"):
            os.environ["GGML_VK_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
            return self.n_gpu_layers

        if self.device == "vulkan" and self.vulkan_visible_devices:
            os.environ["GGML_VK_VISIBLE_DEVICES"] = self.vulkan_visible_devices
            return self.n_gpu_layers

        os.environ.pop("GGML_VK_VISIBLE_DEVICES", None)
        return self.n_gpu_layers

    def _get_model(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama

                n_gpu_layers = self._configure_device()
                log(
                    "info",
                    f"Loading Qwen3 llama.cpp model ctx={self.n_ctx} threads={self.n_threads} gpu_layers={n_gpu_layers} device={self.device}",
                )
                self._llm = Llama(
                    model_path=self._model_path(),
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=n_gpu_layers,
                    main_gpu=self.main_gpu,
                    verbose=self.verbose,
                )
            except Exception as exc:
                raise LLMError(f"Failed to initialize Qwen3 llama.cpp model: {exc}", exc)

        return self._llm

    def _usage(self, started_at: float, response: dict[str, Any], output_text: str | None) -> ModelUsageStats:
        usage = response.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        return ModelUsageStats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            provider=self.provider_name,
            model_name=self.model_name,
            response_ms=(time() - started_at) * 1000,
            output_chars=len(output_text) if output_text is not None else None,
        )

    def _tool_call(self, name: str, arguments: Any, index: int, call_id: str | None = None) -> Any:
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if isinstance(arguments, dict) else {}, ensure_ascii=False)
        try:
            from openai.types.chat import ChatCompletionMessageFunctionToolCall
        except Exception:
            return {
                "id": call_id or f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }

        return ChatCompletionMessageFunctionToolCall.model_validate(
            {
                "id": call_id or f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )

    def _convert_tool_calls(self, raw_tool_calls: Any) -> list[Any] | None:
        if not raw_tool_calls:
            return None

        converted = []
        for index, tool_call in enumerate(raw_tool_calls):
            if not isinstance(tool_call, dict):
                converted.append(tool_call)
                continue

            function = tool_call.get("function") or {}
            converted.append(
                self._tool_call(function.get("name", ""), function.get("arguments") or {}, index, tool_call.get("id"))
            )
        return converted or None

    def _parse_tool_calls_from_text(self, text: str | None) -> list[Any] | None:
        if not text:
            return None

        candidate = _strip_thinking(text)
        calls: list[Any] = []

        for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", candidate, re.DOTALL):
            payload_text = _extract_json_object(match.group(1))
            if not payload_text:
                continue
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            name = payload.get("name") or payload.get("function", {}).get("name")
            arguments = payload.get("arguments")
            if arguments is None:
                arguments = payload.get("function", {}).get("arguments", {})
            if isinstance(name, str) and name:
                calls.append(self._tool_call(name, arguments, len(calls)))

        for match in re.finditer(
            r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
            candidate,
            re.DOTALL,
        ):
            name = match.group(1).strip()
            body = match.group(2)
            arguments: dict[str, Any] = {}
            for param_match in re.finditer(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                param_name = param_match.group(1).strip()
                param_value_raw = html.unescape(param_match.group(2).strip())
                try:
                    param_value = json.loads(param_value_raw)
                except json.JSONDecodeError:
                    param_value = param_value_raw
                if param_name:
                    arguments[param_name] = param_value
            if name:
                calls.append(self._tool_call(name, arguments, len(calls)))

        if calls:
            return calls

        payload_text = _extract_json_object(candidate)
        if not payload_text:
            return None
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return None

        raw_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
        if isinstance(raw_calls, list):
            return self._convert_tool_calls(raw_calls)

        if isinstance(payload, dict):
            name = payload.get("name") or payload.get("function", {}).get("name")
            arguments = payload.get("arguments")
            if arguments is None:
                arguments = payload.get("function", {}).get("arguments", {})
            if isinstance(name, str) and name:
                return [self._tool_call(name, arguments, 0)]

        return None

    def _messages_with_think_prefill(self, messages: List[dict]) -> list[dict[str, Any]]:
        prepared = [dict(message) for message in messages]
        if prepared and prepared[-1].get("role") == "assistant":
            prepared[-1]["content"] = f"{prepared[-1].get('content') or ''}{THINK_PREFILL}"
        else:
            prepared.append({"role": "assistant", "content": THINK_PREFILL})
        return prepared

    def generate(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[Any] = None,
    ) -> tuple[str | None, List[Any] | None, ModelUsageStats]:
        started_at = time()
        llm = self._get_model()

        params: dict[str, Any] = {
            "messages": self._messages_with_think_prefill(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
        }
        if tools:
            params["tools"] = tools
        if tool_choice:
            params["tool_choice"] = tool_choice

        try:
            response = llm.create_chat_completion(**params)
        except Exception as exc:
            raise LLMError(f"Qwen3 llama.cpp generation failed: {exc}", exc)

        choices = response.get("choices") or []
        if not choices:
            return None, None, self._usage(started_at, response, None)

        message = choices[0].get("message") or {}
        response_text = message.get("content") or None
        response_actions = self._convert_tool_calls(message.get("tool_calls"))

        if response_actions is None:
            response_actions = self._parse_tool_calls_from_text(response_text)
            if response_actions is not None:
                response_text = None

        if response_text is not None:
            response_text = _strip_thinking(response_text) or None

        usage = self._usage(started_at, response, response_text)

        if response_text is None and response_actions is None:
            return None, None, usage
        return response_text, response_actions, usage


class Qwen3LlamaCppPlugin(PluginBase):
    def __init__(self, plugin_manifest: PluginManifest):
        super().__init__(plugin_manifest)

        device_options = _device_select_options()

        provider_settings: list[SettingsGrid] = [
            SettingsGrid(
                key="runtime",
                label="Runtime",
                fields=[
                    SelectSetting(
                        key="device",
                        label="Device",
                        type="select",
                        readonly=False,
                        placeholder=None,
                        default_value="auto",
                        select_options=device_options,
                        multi_select=False,
                    ),
                    TextSetting(
                        key="vulkan_visible_devices",
                        label="Vulkan visible devices",
                        type="text",
                        readonly=False,
                        placeholder="0 or 1 or 0,1",
                        default_value="",
                        max_length=64,
                        min_length=None,
                        hidden=False,
                    ),
                    NumericalSetting(
                        key="n_ctx",
                        label="Context size",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=16384,
                        min_value=1024,
                        max_value=40960,
                        step=1024,
                    ),
                    NumericalSetting(
                        key="n_gpu_layers",
                        label="GPU layers (-1 = all)",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=-1,
                        min_value=-1,
                        max_value=200,
                        step=1,
                    ),
                    NumericalSetting(
                        key="main_gpu",
                        label="Main GPU index",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=0,
                        min_value=0,
                        max_value=16,
                        step=1,
                    ),
                    NumericalSetting(
                        key="n_threads",
                        label="CPU threads",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=max(1, (os.cpu_count() or 4) // 2),
                        min_value=1,
                        max_value=128,
                        step=1,
                    ),
                ],
            ),
            SettingsGrid(
                key="generation",
                label="Generation",
                fields=[
                    NumericalSetting(
                        key="max_tokens",
                        label="Max output tokens",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=1024,
                        min_value=1,
                        max_value=8192,
                        step=1,
                    ),
                    NumericalSetting(
                        key="temperature",
                        label="Temperature",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=0.3,
                        min_value=0,
                        max_value=2,
                        step=0.05,
                    ),
                    NumericalSetting(
                        key="top_p",
                        label="Top P",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=0.95,
                        min_value=0,
                        max_value=1,
                        step=0.01,
                    ),
                    NumericalSetting(
                        key="repeat_penalty",
                        label="Repeat penalty",
                        type="number",
                        readonly=False,
                        placeholder=None,
                        default_value=1.05,
                        min_value=0.5,
                        max_value=2,
                        step=0.01,
                    ),
                    ToggleSetting(
                        key="verbose",
                        label="Verbose llama.cpp logs",
                        type="toggle",
                        readonly=False,
                        placeholder=None,
                        default_value=False,
                    ),
                ],
            ),
        ]

        self.settings_config = PluginSettings(
            key="Qwen3 llama.cpp",
            label="Qwen3 llama.cpp",
            icon="memory",
            grids=[
                SettingsGrid(
                    key="general",
                    label="General",
                    fields=[
                        ParagraphSetting(
                            key="info_text",
                            label=None,
                            type="paragraph",
                            readonly=False,
                            placeholder=None,
                            content="Runs lucaelin/qwen3-0.6b-cn-gguf locally with llama.cpp. The release packages llama-cpp-python with Vulkan support.",
                        )
                    ],
                )
            ],
        )
        self.model_providers = [
            ModelProviderDefinition(
                kind="llm",
                id="qwen3-llamacpp",
                label="Qwen3 0.6B CN llama.cpp",
                settings_config=provider_settings,
            )
        ]

    @override
    def create_model(self, provider_id: str, settings: dict[str, Any]) -> LLMModel:
        if provider_id == "qwen3-llamacpp":
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            return Qwen3LlamaCppModel(model_dir=os.path.join(plugin_dir, "model"), settings=settings)
        raise ValueError(f"Unknown Qwen3 llama.cpp provider: {provider_id}")

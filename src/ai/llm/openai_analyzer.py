"""OpenAI analyzer adapter with optional tool execution.

This module avoids hard dependencies on external SDKs by using dynamic imports
and narrows exception handling to keep linting clean while preserving safety.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import uuid
from contextlib import suppress
from types import SimpleNamespace
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Set

from ai.llm.interface import AnalyzerAdapter, AnalyzerError
from config.env import get_settings, is_tracing_enabled

# Optional observability
try:
    from observability.tracing import tracer  # type: ignore
except ImportError:
    def _noop_tracer() -> Any:  # type: ignore[override]
        return None

    tracer = _noop_tracer  # type: ignore

try:
    from observability.metrics import ANALYSIS_TOKEN_USAGE  # type: ignore
except ImportError:
    ANALYSIS_TOKEN_USAGE = None  # type: ignore


# Tools registry is imported dynamically at runtime to avoid hard dependency during offline/CI.


_LOGGER = logging.getLogger("sd_onboarding")

# Sensitive keys to redact from tool results echoed back to the model
SENSITIVE_KEYS: Set[str] = {
    "secret",
    "secrets",
    "api_key",
    "token",
    "access_token",
    "password",
    "key",
}


def _build_messages(context: str, question: str, comparison_mode: bool) -> List[dict]:
    """
    Build a conservative, structured system+user prompt for chat.completions.
    No secrets or dynamic code is injected. Deterministic message ordering.
    """
    system_preamble = (
        "You are a senior service operations analyst. "
        "Analyze incident tickets to propose onboarding improvements for the Service Desk. "
        "Respond in structured Markdown with sections: Summary, Distributions, "
        "Insights, Recommendations, Onboarding Key-Values. Avoid sensitive data. "
        "Be concise and actionable."
    )
    if comparison_mode:
        system_preamble += " Compare datasets when possible and call out differences clearly."

    return [
        {"role": "system", "content": system_preamble},
        {
            "role": "user",
            "content": (
                f"Question: {question.strip()}\n\n"
                "Context:\n"
                f"{context.strip()}"
            ),
        },
    ]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not str(raw or "").strip():
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _build_tool_specs_from_registry(registry: Any) -> List[Dict[str, Any]]:
    """
    Build OpenAI function tool specs from registry input schemas.
    Uses Pydantic model_json_schema() and enforces Draft 2020-12 compatible shapes.
    """
    specs: List[Dict[str, Any]] = []
    # Introspect registry tools map; strictly whitelisted
    tools_map = getattr(registry, "_tools", {})  # internal map; stable in our implementation
    for name, spec in tools_map.items():
        # Ensure schema is built; suppress rebuild failures
        with suppress(AttributeError, TypeError, ValueError):
            spec.input_model.model_rebuild()
        try:
            schema = spec.input_model.model_json_schema()
        except (AttributeError, TypeError, ValueError) as e:
            _LOGGER.warning("tool.spec.schema.error: name=%s error=%s", name, str(e))
            continue

        # Normalize minimal requirements for OpenAI tools JSON schema
        if "type" not in schema:
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        specs.append(
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": str(spec.description or ""),
                    "parameters": schema,
                },
            }
        )
    return specs


def _sanitize_for_model(data: Any) -> Any:
    """
    Remove potential secrets/PII keys before echoing tool result to the model context.
    """
    if isinstance(data, dict):
        out: Dict[str, Any] = {}
        for k, v in data.items():
            k_norm = str(k).lower().strip()
            if all(s not in k_norm for s in SENSITIVE_KEYS):
                out[k] = _sanitize_for_model(v)
        return out
    if isinstance(data, list):
        return [_sanitize_for_model(v) for v in data]
    return data


def _append_tool_result_message(
    history: List[Dict[str, Any]],
    tool_call_id: str,
    result: Dict[str, Any],
) -> None:
    """
    Append a 'tool' role message with sanitized JSON content.
    """
    history.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(
                _sanitize_for_model(result), ensure_ascii=False
            ),
        }
    )


def _build_tool_context_from_claims_or_params(
    *,
    request_id: str,
    args: Dict[str, Any],
    token_budget: Optional[int],
    step_limit: Optional[int],
    claims: Optional[Dict[str, Any]] = None,
) -> Any:
    # Import registry helpers dynamically to avoid import-time failures in offline environments
    build_ctx = None
    tool_context_cls = None
    with suppress(ImportError):
        mod = importlib.import_module("ai.llm.tools.registry")
        build_ctx = getattr(mod, "build_tool_context_from_claims", None)
        tool_context_cls = getattr(mod, "ToolContext", None)

    dataset_id: Optional[int] = None
    try:
        if "dataset_id" in args and args["dataset_id"] is not None:
            dataset_id = int(args["dataset_id"])
    except (TypeError, ValueError):
        dataset_id = None

    if callable(build_ctx) and isinstance(claims, dict):
        with suppress(Exception):
            return build_ctx(
                claims,
                request_id=request_id,
                token_budget=token_budget,
                step_limit=step_limit,
                dataset_id=dataset_id,
            )

    if tool_context_cls is not None:
        with suppress(Exception):
            return tool_context_cls(
                subject="anonymous",
                roles=set(),
                request_id=request_id,
                token_budget=token_budget,
                step_limit=step_limit,
                dataset_id=dataset_id,
            )

    # Final fallback: lightweight object with required attributes
    return SimpleNamespace(
        subject="anonymous",
        roles=set(),
        request_id=request_id,
        token_budget=token_budget,
        step_limit=step_limit,
        dataset_id=dataset_id,
    )


class OpenAIAnalyzer(AnalyzerAdapter):
    """
    OpenAI/Azure OpenAI analyzer adapter.

    - Uses environment to auto-detect provider:
      * OPENAI_API_KEY -> OpenAI
      * AZURE_OPENAI_API_KEY/AZURE_OPENAI_ENDPOINT (or APP_AZURE_OPENAI_*) -> Azure OpenAI
    - No logs of secrets. Raises AnalyzerError for configuration and provider errors.
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        enable_tools: Optional[bool] = None,
        token_budget: Optional[int] = None,
        step_limit: Optional[int] = None,
    ) -> None:
        # Lazy import openai client to avoid static import errors when dependency is not installed.
        if importlib.util.find_spec("openai") is None:
            raise AnalyzerError(
                "OpenAI Python SDK is not installed. Install 'openai' to use the OpenAI analyzer."
            )
        mod = importlib.import_module("openai")
        openai_cls = getattr(mod, "OpenAI")
        azure_openai_cls = getattr(mod, "AzureOpenAI")

        self._openai_cls = openai_cls  # type: ignore[attr-defined]
        self._azure_openai_cls = azure_openai_cls  # type: ignore[attr-defined]

        # Load settings early (.env via Pydantic);
        # provide fallbacks via legacy env keys where needed
        self._settings = get_settings()
        self._azure_endpoint: Optional[str] = None

        # Detect provider intent (prefer Azure when any Azure config provided)
        self._provider = self._detect_provider()
        self._client = self._create_client()

        # Tools/agent loop configuration
        self._registry: Optional[Any] = registry
        # tools path only when registry present AND enable_tools explicitly True
        self._tools_enabled: bool = self._registry is not None and enable_tools is True
        # Budgets
        self._step_limit: int = int(step_limit or _env_int("TOOL_MAX_STEPS", 8))
        # token budget: prefer explicit param, else env TOOL_TOKEN_BUDGET,
        # else None (no explicit cap beyond model limits)
        env_tb = os.getenv("TOOL_TOKEN_BUDGET")
        self._token_budget: Optional[int] = (
            token_budget
            if token_budget is not None
            else (int(env_tb) if env_tb else None)
        )

        # Observability flags
        with suppress(Exception):
            self._tracing_enabled = bool(is_tracing_enabled())
        if not hasattr(self, "_tracing_enabled"):
            self._tracing_enabled = False

        # Initialization log (single line; no secrets)
        with suppress(Exception):
            model_or_deployment = self._resolve_model()
            endpoint_host = ""
            if self._provider == "azure":
                endpoint_host = urlparse(self._azure_endpoint or "").netloc
            _LOGGER.info(
                "analyzer.init backend=%s endpoint_host=%s model=%s",
                self._provider,
                endpoint_host,
                model_or_deployment,
            )

    def _detect_provider(self) -> str:
        """
        Determine provider intent using Settings first (APP_*), then legacy env fallbacks.
        Prefer Azure when any Azure config is present; otherwise OpenAI when API key is present.
        """
        s = self._settings
        # Resolve Azure intent
        azure_endpoint = (
            (str(s.azure_openai_endpoint) if getattr(s, "azure_openai_endpoint", None) else None)
            or os.environ.get("APP_AZURE_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
        )
        azure_key = (
            getattr(s, "azure_openai_key", None)
            or os.environ.get("APP_AZURE_OPENAI_KEY")
            or os.environ.get("APP_AZURE_OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_API_KEY")
            or os.environ.get("AZURE_OPENAI_KEY")
        )
        openai_key = (
            getattr(s, "openai_api_key", None)
            or os.environ.get("APP_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        if azure_endpoint or azure_key:
            return "azure"
        if openai_key:
            return "openai"
        raise AnalyzerError(
            "No OpenAI or Azure OpenAI credentials found in environment or settings"
        )

    def _create_client(self):
        """
        Instantiate SDK client with safe defaults and no secret logging.
        """
        try:
            s = self._settings
            if self._provider == "openai":
                api_key = (
                    getattr(s, "openai_api_key", None)
                    or os.environ.get("APP_OPENAI_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                )
                if not api_key:
                    raise AnalyzerError("OPENAI_API_KEY is not configured")
                # Optional org/project (legacy env support)
                org = os.environ.get("OPENAI_ORG")
                project = os.environ.get("OPENAI_PROJECT")
                # Use slightly lower timeout and retry defaults than SDK default 10m/2
                return self._openai_cls(
                    api_key=api_key,
                    organization=org,
                    project=project,
                    timeout=60.0,
                    max_retries=2,
                )
            else:
                # Azure intent (prefer Settings)
                endpoint = (
                    (
                        str(s.azure_openai_endpoint)
                        if getattr(s, "azure_openai_endpoint", None)
                        else None
                    )
                    or os.environ.get("APP_AZURE_OPENAI_ENDPOINT")
                    or os.environ.get("AZURE_OPENAI_ENDPOINT")
                )
                api_key = (
                    getattr(s, "azure_openai_key", None)
                    or os.environ.get("APP_AZURE_OPENAI_KEY")
                    or os.environ.get("APP_AZURE_OPENAI_API_KEY")
                    or os.environ.get("AZURE_OPENAI_API_KEY")
                    or os.environ.get("AZURE_OPENAI_KEY")
                )
                api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-08-01-preview"

                if endpoint and api_key:
                    self._azure_endpoint = endpoint
                    return self._azure_openai_cls(
                        azure_endpoint=endpoint,
                        api_key=api_key,
                        api_version=api_version,
                        timeout=60.0,
                        max_retries=2,
                    )

                if fallback_openai_key := (
                    getattr(s, "openai_api_key", None)
                    or os.environ.get("APP_OPENAI_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                ):
                    _LOGGER.warning("config.azure.incomplete: falling back to OpenAI backend")
                    return self._openai_cls(
                        api_key=fallback_openai_key,
                        organization=os.environ.get("OPENAI_ORG"),
                        project=os.environ.get("OPENAI_PROJECT"),
                        timeout=60.0,
                        max_retries=2,
                    )

                raise AnalyzerError(
                    "Azure OpenAI credentials are incomplete (endpoint/key) "
                    "and no OpenAI key available"
                )
        except (ValueError, TypeError, AttributeError) as e:
            raise AnalyzerError(f"Failed to initialize OpenAI client: {e}") from e

    def _resolve_model(self) -> str:
        """
        Resolve model/deployment using Settings first (APP_*), with sane defaults:
          - OpenAI: APP_OPENAI_MODEL (default 'gpt-5')
          - Azure:  APP_AZURE_OPENAI_DEPLOYMENT (default 'gpt-5')
        """
        s = self._settings
        if self._provider == "openai":
            return (
                getattr(s, "openai_model", None)
                or os.environ.get("APP_OPENAI_MODEL")
                or os.environ.get("OPENAI_MODEL")
                or "gpt-5"
            )
        # Azure uses 'model' param as deployment name for chat.completions in new SDKs
        return (
            getattr(s, "azure_openai_deployment", None)
            or os.environ.get("APP_AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or "gpt-5"
        )

    def _run_model_call(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Perform a single chat.completions call with optional tools.
        Creates a tracing child span when enabled.
        """
        tr: Any = None
        if self._tracing_enabled:
            with suppress(Exception):
                tr = tracer()

        # Use context manager only when tracing is available
        if tr:
            with tr.start_as_current_span("analyzer.model_call") as span:
                # type: ignore[attr-defined]
                span.set_attribute("tool_count", len(tools or []))
                # type: ignore[attr-defined]
                span.set_attribute("step_limit", self._step_limit)
                # type: ignore[attr-defined]
                span.set_attribute("token_budget", int(self._token_budget or 0))
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "top_p": 0.9,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                    **kwargs
                )
                # Metrics: token usage if available
                with suppress(Exception):
                    usage = getattr(resp, "usage", None)
                    total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                    if ANALYSIS_TOKEN_USAGE is not None and total_tokens > 0:
                        ANALYSIS_TOKEN_USAGE.inc(total_tokens)  # pragma: no cover
                return resp
        else:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.9,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
                **kwargs
            )
            # Metrics: token usage if available
            with suppress(Exception):
                usage = getattr(resp, "usage", None)
                total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                if ANALYSIS_TOKEN_USAGE is not None and total_tokens > 0:
                    ANALYSIS_TOKEN_USAGE.inc(total_tokens)  # pragma: no cover
            return resp

    def analyze(
        self,
        context: str,
        question: str,
        prompt_version: str,
        comparison_mode: bool = False,
    ) -> str:
        if not isinstance(context, str) or not context.strip():
            raise AnalyzerError("Context is required")
        if not isinstance(question, str) or not question.strip():
            raise AnalyzerError("Question is required")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise AnalyzerError("prompt_version is required")

        messages = _build_messages(
            context=context,
            question=question,
            comparison_mode=comparison_mode,
        )
        model = self._resolve_model()

        # Backward-compatible path: no tools
        if not self._tools_enabled or self._registry is None:
            # Use chat.completions API for broad compatibility
            resp = self._run_model_call(model=model, messages=messages)
            content = (resp.choices[0].message.content or "").strip()  # type: ignore[index]
            if not content:
                raise AnalyzerError("Analyzer returned empty content")
            return content

        # Tool-enabled agent loop
        run_request_id = str(uuid.uuid4())
        executed_ids: Set[str] = set()
        total_steps = 0
        tools = _build_tool_specs_from_registry(self._registry)
        cum_tokens = 0

        # Trace the full run
        run_span_ctx: Any = None
        tr: Any = None
        if self._tracing_enabled:
            with suppress(Exception):
                tr = tracer()
                if tr:
                    run_span_ctx = tr.start_as_current_span("analyzer.run")

        try:
            # If tracing is active, enter context manager for the full run
            if run_span_ctx:
                # type: ignore[attr-defined]
                with run_span_ctx as span:  # type: ignore[attr-defined]
                    # type: ignore[attr-defined]
                    span.set_attribute("step_limit", self._step_limit)
                    # type: ignore[attr-defined]
                    span.set_attribute("token_budget", int(self._token_budget or 0))
                    while True:
                        # Step/budget enforcement (step limit hard cap)
                        if total_steps >= self._step_limit:
                            # Abort with controlled message
                            err_payload = {
                                "error": {
                                    "category": "budget_exceeded",
                                    "message": "Step limit exceeded",
                                    "details": {"step_limit": int(self._step_limit)},
                                }
                            }
                            # Surface error to caller as final assistant content
                            return (
                                "Tool execution aborted.\n\n"
                                f"{json.dumps(err_payload, ensure_ascii=False)}"
                            )

                        # Token budget pre-check before another model call
                        if self._token_budget is not None and cum_tokens >= self._token_budget:
                            err_payload = {
                                "error": {
                                    "category": "budget_exceeded",
                                    "message": "Token budget exhausted before next step",
                                    "details": {
                                        "token_budget": int(self._token_budget or 0),
                                        "used": int(cum_tokens),
                                    },
                                }
                            }
                            return (
                                "Tool execution aborted.\n\n"
                                f"{json.dumps(err_payload, ensure_ascii=False)}"
                            )

                        resp = self._run_model_call(model=model, messages=messages, tools=tools)

                        # Track token usage and enforce budget after call
                        step_tokens = 0
                        with suppress(Exception):
                            usage = getattr(resp, "usage", None)
                            step_tokens = (
                                int(getattr(usage, "total_tokens", 0) or 0)
                                if usage
                                else 0
                            )
                            cum_tokens += step_tokens
                        
                        if self._token_budget is not None and cum_tokens > self._token_budget:
                            err_payload = {
                                "error": {
                                    "category": "budget_exceeded",
                                    "message": "Token budget exceeded",
                                    "details": {
                                        "token_budget": int(self._token_budget or 0),
                                        "used": int(cum_tokens),
                                        "last_step_tokens": int(step_tokens),
                                    },
                                }
                            }
                            return (
                                "Tool execution aborted.\n\n"
                                f"{json.dumps(err_payload, ensure_ascii=False)}"
                            )
                        choice = resp.choices[0]  # type: ignore[index]
                        msg = choice.message

                        # If the assistant produced tool calls, dispatch them via the registry
                        tool_calls = getattr(msg, "tool_calls", None)
                        if tool_calls:
                            # Observability: log decision
                            _LOGGER.info("analyzer.tool_calls: count=%s", len(tool_calls))
                            # Dispatch each call
                            for call in tool_calls:
                                call_id = str(getattr(call, "id", "") or "")
                                fn = getattr(call, "function", None)
                                name = str(getattr(fn, "name", "") or "")
                                args_raw = str(getattr(fn, "arguments", "") or "").strip()

                                if not name:
                                    # Unknown/malformed call name
                                    result = {
                                        "error": {
                                            "category": "validation_error",
                                            "message": "Missing tool name in call",
                                            "details": {"tool_call_id": call_id},
                                        }
                                    }
                                    _append_tool_result_message(
                                        messages,
                                        call_id or f"missing-{uuid.uuid4()}",
                                        result,
                                    )
                                    continue

                                if call_id in executed_ids:
                                    # Idempotency guard: do not re-execute
                                    _LOGGER.debug(
                                        "analyzer.skip_duplicate_tool_call: id=%s name=%s",
                                        call_id,
                                        name,
                                    )
                                    continue

                                # Parse args JSON robustly
                                try:
                                    args = json.loads(args_raw) if args_raw else {}
                                    if not isinstance(args, dict):
                                        raise ValueError("Tool arguments must be a JSON object")
                                except (json.JSONDecodeError, ValueError) as e:
                                    result = {
                                        "error": {
                                            "category": "validation_error",
                                            "message": "Invalid tool arguments JSON",
                                            "details": {"tool_name": name, "hint": str(e)},
                                        }
                                    }
                                    _append_tool_result_message(
                                        messages,
                                        call_id or f"badjson-{uuid.uuid4()}",
                                        result,
                                    )
                                    executed_ids.add(call_id)
                                    total_steps += 1
                                    continue

                                # Strict whitelist: only registered tools
                                spec = self._registry.get_spec(name)
                                if spec is None:
                                    result = {
                                        "error": {
                                            "category": "tool_unavailable",
                                            "message": "Requested tool is not available",
                                            "details": {"tool_name": name},
                                        }
                                    }
                                    _append_tool_result_message(
                                        messages,
                                        call_id or f"unknown-{uuid.uuid4()}",
                                        result,
                                    )
                                    executed_ids.add(call_id)
                                    total_steps += 1
                                    continue

                                # Build ToolContext (anonymous if no claims available)
                                ctx = _build_tool_context_from_claims_or_params(
                                    request_id=run_request_id,
                                    args=args,
                                    token_budget=self._token_budget,
                                    step_limit=self._step_limit,
                                    claims=None,
                                )

                                # Execute via registry with validation/RBAC/audit/metrics
                                result = self._registry.execute(name=name, args=args, context=ctx)
                                _append_tool_result_message(
                                    messages,
                                    call_id or f"exec-{uuid.uuid4()}",
                                    result,
                                )
                                executed_ids.add(call_id)
                                total_steps += 1

                                # If registry surfaced an error, stop immediately per safety policy
                                if isinstance(result, dict) and "error" in result:
                                    # Bubble error in final content
                                    return (
                                        "Tool execution failed.\n\n"
                                        f"{json.dumps(
                                            _sanitize_for_model(result), ensure_ascii=False
                                        )}"
                                    )

                            # Continue so the model can consume tool results
                            continue

                        # No tool calls: treat assistant content as final completion
                        content = (msg.content or "").strip()
                        if not content:
                            raise AnalyzerError("Analyzer returned empty content")
                        return content
            else:
                while True:
                    # Step/budget enforcement (step limit hard cap)
                    if total_steps >= self._step_limit:
                        # Abort with controlled message
                        err_payload = {
                            "error": {
                                "category": "budget_exceeded",
                                "message": "Step limit exceeded",
                                "details": {"step_limit": int(self._step_limit)},
                            }
                        }
                        # Surface error to caller as final assistant content
                        return (
                            "Tool execution aborted.\n\n"
                            f"{json.dumps(err_payload, ensure_ascii=False)}"
                        )

                    # Token budget pre-check before another model call
                    if self._token_budget is not None and cum_tokens >= self._token_budget:
                        err_payload = {
                            "error": {
                                "category": "budget_exceeded",
                                "message": "Token budget exhausted before next step",
                                "details": {
                                    "token_budget": int(self._token_budget or 0),
                                    "used": int(cum_tokens),
                                },
                            }
                        }
                        return (
                            "Tool execution aborted.\n\n"
                            f"{json.dumps(err_payload, ensure_ascii=False)}"
                        )

                    resp = self._run_model_call(model=model, messages=messages, tools=tools)

                    # Track token usage and enforce budget after call
                    step_tokens = 0
                    with suppress(Exception):
                        usage = getattr(resp, "usage", None)
                        step_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                        cum_tokens += step_tokens
                    
                    if self._token_budget is not None and cum_tokens > self._token_budget:
                        err_payload = {
                            "error": {
                                "category": "budget_exceeded",
                                "message": "Token budget exceeded",
                                "details": {
                                    "token_budget": int(self._token_budget or 0),
                                    "used": int(cum_tokens),
                                    "last_step_tokens": int(step_tokens),
                                },
                            }
                        }
                        return (
                            "Tool execution aborted.\n\n"
                            f"{json.dumps(err_payload, ensure_ascii=False)}"
                        )
                    choice = resp.choices[0]  # type: ignore[index]
                    msg = choice.message

                    # If the assistant produced tool calls, dispatch them via the registry
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        # Observability: log decision
                        _LOGGER.info("analyzer.tool_calls: count=%s", len(tool_calls))
                        # Dispatch each call
                        for call in tool_calls:
                            call_id = str(getattr(call, "id", "") or "")
                            fn = getattr(call, "function", None)
                            name = str(getattr(fn, "name", "") or "")
                            args_raw = str(getattr(fn, "arguments", "") or "").strip()

                            if not name:
                                # Unknown/malformed call name
                                result = {
                                    "error": {
                                        "category": "validation_error",
                                        "message": "Missing tool name in call",
                                        "details": {"tool_call_id": call_id},
                                    }
                                }
                                _append_tool_result_message(
                                    messages,
                                    call_id or f"missing-{uuid.uuid4()}",
                                    result,
                                )
                                continue

                            if call_id in executed_ids:
                                # Idempotency guard: do not re-execute
                                _LOGGER.debug(
                                    "analyzer.skip_duplicate_tool_call: id=%s name=%s",
                                    call_id,
                                    name,
                                )
                                continue

                            # Parse args JSON robustly
                            try:
                                args = json.loads(args_raw) if args_raw else {}
                                if not isinstance(args, dict):
                                    raise ValueError("Tool arguments must be a JSON object")
                            except (json.JSONDecodeError, ValueError) as e:
                                result = {
                                    "error": {
                                        "category": "validation_error",
                                        "message": "Invalid tool arguments JSON",
                                        "details": {"tool_name": name, "hint": str(e)},
                                    }
                                }
                                _append_tool_result_message(
                                    messages,
                                    call_id or f"badjson-{uuid.uuid4()}",
                                    result,
                                )
                                executed_ids.add(call_id)
                                total_steps += 1
                                continue

                            # Strict whitelist: only registered tools
                            spec = self._registry.get_spec(name)
                            if spec is None:
                                result = {
                                    "error": {
                                        "category": "tool_unavailable",
                                        "message": "Requested tool is not available",
                                        "details": {"tool_name": name},
                                    }
                                }
                                _append_tool_result_message(
                                    messages,
                                    call_id or f"unknown-{uuid.uuid4()}",
                                    result,
                                )
                                executed_ids.add(call_id)
                                total_steps += 1
                                continue

                            # Build ToolContext (anonymous if no claims available)
                            ctx = _build_tool_context_from_claims_or_params(
                                request_id=run_request_id,
                                args=args,
                                token_budget=self._token_budget,
                                step_limit=self._step_limit,
                                claims=None,
                            )

                            # Execute via registry with validation/RBAC/audit/metrics
                            result = self._registry.execute(name=name, args=args, context=ctx)
                            _append_tool_result_message(
                                messages,
                                call_id or f"exec-{uuid.uuid4()}",
                                result,
                            )
                            executed_ids.add(call_id)
                            total_steps += 1

                            # If registry surfaced an error, stop immediately per safety policy
                            if isinstance(result, dict) and "error" in result:
                                # Bubble error in final content
                                return (
                                    "Tool execution failed.\n\n"
                                    f"{json.dumps(_sanitize_for_model(result), ensure_ascii=False)}"
                                )

                        # Continue for the model to consume tool results and produce final content
                        continue

                    # No tool calls: treat assistant content as final completion
                    content = (msg.content or "").strip()
                    if not content:
                        raise AnalyzerError("Analyzer returned empty content")
                    return content
                # Step/budget enforcement (step limit hard cap)
                if total_steps >= self._step_limit:
                    # Abort with controlled message
                    err_payload = {
                        "error": {
                            "category": "budget_exceeded",
                            "message": "Step limit exceeded",
                            "details": {"step_limit": int(self._step_limit)},
                        }
                    }
                    # Surface error to caller as final assistant content
                    return (
                        "Tool execution aborted.\n\n"
                        f"{json.dumps(err_payload, ensure_ascii=False)}"
                    )

                # Token budget pre-check before another model call
                if self._token_budget is not None and cum_tokens >= self._token_budget:
                    err_payload = {
                        "error": {
                            "category": "budget_exceeded",
                            "message": "Token budget exhausted before next step",
                            "details": {
                                "token_budget": int(self._token_budget or 0),
                                "used": int(cum_tokens),
                            },
                        }
                    }
                    return (
                        "Tool execution aborted.\n\n"
                        f"{json.dumps(err_payload, ensure_ascii=False)}"
                    )

                resp = self._run_model_call(model=model, messages=messages, tools=tools)

                # Track token usage and enforce budget after call
                with suppress(Exception):
                    usage = getattr(resp, "usage", None)
                    step_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
                    cum_tokens += step_tokens
                if 'step_tokens' not in locals():
                    step_tokens = 0
                if self._token_budget is not None and cum_tokens > self._token_budget:
                    err_payload = {
                        "error": {
                            "category": "budget_exceeded",
                            "message": "Token budget exceeded",
                            "details": {
                                "token_budget": int(self._token_budget or 0),
                                "used": int(cum_tokens),
                                "last_step_tokens": int(step_tokens),
                            },
                        }
                    }
                    return (
                        "Tool execution aborted.\n\n"
                        f"{json.dumps(err_payload, ensure_ascii=False)}"
                    )
                choice = resp.choices[0]  # type: ignore[index]
                msg = choice.message

                # If the assistant produced tool calls, dispatch them via the registry
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    # Observability: log decision
                    _LOGGER.info("analyzer.tool_calls: count=%s", len(tool_calls))
                    # Dispatch each call
                    for call in tool_calls:
                        call_id = str(getattr(call, "id", "") or "")
                        fn = getattr(call, "function", None)
                        name = str(getattr(fn, "name", "") or "")
                        args_raw = str(getattr(fn, "arguments", "") or "").strip()

                        if not name:
                            # Unknown/malformed call name
                            result = {
                                "error": {
                                    "category": "validation_error",
                                    "message": "Missing tool name in call",
                                    "details": {"tool_call_id": call_id},
                                }
                            }
                            _append_tool_result_message(
                                messages,
                                call_id or f"missing-{uuid.uuid4()}",
                                result,
                            )
                            continue

                        if call_id in executed_ids:
                            # Idempotency guard: do not re-execute
                            _LOGGER.debug(
                                "analyzer.skip_duplicate_tool_call: id=%s name=%s",
                                call_id,
                                name,
                            )
                            continue

                        # Parse args JSON robustly
                        try:
                            args = json.loads(args_raw) if args_raw else {}
                            if not isinstance(args, dict):
                                raise ValueError("Tool arguments must be a JSON object")
                        except (json.JSONDecodeError, ValueError) as e:
                            result = {
                                "error": {
                                    "category": "validation_error",
                                    "message": "Invalid tool arguments JSON",
                                    "details": {"tool_name": name, "hint": str(e)},
                                }
                            }
                            _append_tool_result_message(
                                messages,
                                call_id or f"badjson-{uuid.uuid4()}",
                                result,
                            )
                            executed_ids.add(call_id)
                            total_steps += 1
                            continue

                        # Strict whitelist: only registered tools
                        spec = self._registry.get_spec(name)
                        if spec is None:
                            result = {
                                "error": {
                                    "category": "tool_unavailable",
                                    "message": "Requested tool is not available",
                                    "details": {"tool_name": name},
                                }
                            }
                            _append_tool_result_message(
                                messages,
                                call_id or f"unknown-{uuid.uuid4()}",
                                result,
                            )
                            executed_ids.add(call_id)
                            total_steps += 1
                            continue

                        # Build ToolContext (anonymous if no claims available)
                        ctx = _build_tool_context_from_claims_or_params(
                            request_id=run_request_id,
                            args=args,
                            token_budget=self._token_budget,
                            step_limit=self._step_limit,
                            claims=None,
                        )

                        # Execute via registry with validation/RBAC/audit/metrics
                        result = self._registry.execute(name=name, args=args, context=ctx)
                        _append_tool_result_message(
                            messages,
                            call_id or f"exec-{uuid.uuid4()}",
                            result,
                        )
                        executed_ids.add(call_id)
                        total_steps += 1

                        # If registry surfaced an error, stop immediately per safety policy
                        if isinstance(result, dict) and "error" in result:
                            # Bubble error in final content
                            return (
                                "Tool execution failed.\n\n"
                                f"{json.dumps(_sanitize_for_model(result), ensure_ascii=False)}"
                            )

                    # Tool results appended; model will consume them next loop

                # No tool calls: treat assistant content as final completion
                content = (msg.content or "").strip()
                if not content:
                    raise AnalyzerError("Analyzer returned empty content")
                return content
        except AnalyzerError:
            raise
        except (ValueError, TypeError, RuntimeError) as e:
            # Map common provider/HTTP errors to AnalyzerError without leaking secrets
            raise AnalyzerError(f"Analyzer provider error: {e}") from e
        finally:
            pass

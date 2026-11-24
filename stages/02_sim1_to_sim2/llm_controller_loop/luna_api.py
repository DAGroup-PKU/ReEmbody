"""Existing local GPT-6 Luna route, with unmodified Responses provenance.

Importing this module does not load credentials or make network calls.
`request_strategy(..., execute=True)` is the only billable entrypoint.
Use a fresh output directory per actual request. Unknown outcomes are never
automatically retried; raw arguments and raw provider output are immutable.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

MODEL = os.environ.get("REEMBODY_LLM_MODEL", "gpt-5")
PROVIDER_MODELS = {MODEL, "openai/openai/gpt-6-luna", "openai/gpt-6-luna", "gpt-6-luna"}
IMPLEMENTATION = Path(__file__).resolve().parents[1] / "rssr_augment"
EXPECTED_ENDPOINT = os.environ.get("REEMBODY_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError("Nonfinite JSON value")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def validate(value, schema, path="arguments"):
    """Validate the explicitly supported JSON-schema subset without dependencies."""
    supported = {"type", "description", "title", "enum", "const", "properties",
                 "required", "additionalProperties", "items", "minItems", "maxItems",
                 "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                 "minLength", "maxLength", "anyOf"}
    if set(schema) - supported:
        raise ValueError("Unsupported schema keywords at " + path)
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                break
            except ValueError:
                pass
        else:
            raise ValueError("No anyOf alternative matches " + path)
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {"null": value is None, "boolean": type(value) is bool,
               "integer": type(value) is int, "number": type(value) in (int, float),
               "string": isinstance(value, str), "array": isinstance(value, list),
               "object": isinstance(value, dict)}
    if types and not any(matches.get(t, False) for t in types):
        raise ValueError("Incorrect type at " + path)
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("Incorrect enum at " + path)
    if "const" in schema and value != schema["const"]:
        raise ValueError("Incorrect constant at " + path)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not set(schema.get("required", [])) <= set(value):
            raise ValueError("Missing required fields at " + path)
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("Unexpected fields at " + path)
        for key in set(value) & set(properties):
            validate(value[key], properties[key], path + "." + key)
    if isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", math.inf):
            raise ValueError("Array length out of range at " + path)
        if "items" in schema:
            for index, item in enumerate(value):
                validate(item, schema["items"], path + "[" + str(index) + "]")
    if isinstance(value, str):
        if not schema.get("minLength", 0) <= len(value) <= schema.get("maxLength", math.inf):
            raise ValueError("String length out of range at " + path)
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite number at " + path)
        for key, okay in (("minimum", lambda x: value >= x), ("maximum", lambda x: value <= x),
                          ("exclusiveMinimum", lambda x: value > x), ("exclusiveMaximum", lambda x: value < x)):
            if key in schema and not okay(schema[key]):
                raise ValueError("Number out of range at " + path)


def check_schema(schema, path="arguments"):
    supported = {"type", "description", "title", "enum", "const", "properties",
                 "required", "additionalProperties", "items", "minItems", "maxItems",
                 "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                 "minLength", "maxLength", "anyOf"}
    if not isinstance(schema, dict) or set(schema) - supported:
        raise ValueError("Unsupported schema at " + path)
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if not isinstance(types, list) or set(types) - {"null", "boolean", "integer", "number", "string", "array", "object"}:
        raise ValueError("Unsupported schema type at " + path)
    for index, option in enumerate(schema.get("anyOf", [])):
        check_schema(option, path + ".anyOf[" + str(index) + "]")
    if "object" in types:
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False or set(schema.get("required", [])) != set(properties):
            raise ValueError("Strict object schema needs all properties required and additionalProperties=false at " + path)
        for key, subschema in properties.items():
            check_schema(subschema, path + "." + key)
    if "array" in types:
        if "items" not in schema:
            raise ValueError("Array schema requires items at " + path)
        check_schema(schema["items"], path + ".items")


def load_profile(profile_source="env"):
    """Resolve LLM API configuration from standard environment variables without exposing secrets."""
    api_key = os.environ.get("REEMBODY_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("REEMBODY_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or None
    model = os.environ.get("REEMBODY_LLM_MODEL", MODEL)
    if not api_key:
        raise ValueError(
            "Missing LLM API key. Set REEMBODY_LLM_API_KEY or OPENAI_API_KEY in your environment."
        )
    profile = {"model": model, "base_url": base_url, "llm_api": "responses"}
    metadata = dict(
        profile_source=profile_source,
        profile_name="env",
        selected_profile_sha256=digest(profile),
        model=model,
        llm_api="responses",
        endpoint=base_url or "https://api.openai.com/v1",
    )
    return profile, metadata


def make_client(profile_source="env", *, return_metadata=False):
    """Construct an OpenAI Responses-compatible client from environment variables."""
    from openai import OpenAI

    profile, metadata = load_profile(profile_source)
    api_key = os.environ.get("REEMBODY_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    kwargs = {"api_key": api_key, "timeout": 300, "max_retries": 0}
    if profile.get("base_url"):
        kwargs["base_url"] = profile["base_url"]
    client = OpenAI(**kwargs)
    return (client, metadata) if return_metadata else client


def build_kwargs(prompt, evidence, strategy_schema, *, effort="medium", max_output_tokens=16000):
    if effort not in ("low", "medium", "high") or not 1 <= max_output_tokens <= 32768:
        raise ValueError("Invalid bounded request options")
    check_schema(strategy_schema)
    return dict(model=MODEL,
        input=[dict(role="developer", content=prompt),
               dict(role="user", content=canonical(evidence))],
        tools=[dict(type="function", name="propose_trajectory", strict=True,
                    description="Propose the complete next trajectory refinement strategy from supplied evidence.",
                    parameters=strategy_schema)],
        tool_choice=dict(type="function", name="propose_trajectory"), parallel_tool_calls=False,
        reasoning=dict(effort=effort), max_output_tokens=max_output_tokens, store=False, stream=False)


def verify_response(output_dir):
    """Verify saved request/raw response/arguments and return exact parsed strategy."""
    folder = Path(output_dir)
    request = loads((folder / "request.json").read_text())
    response = loads((folder / "response.json").read_text())
    raw_bytes = (folder / "raw_provider_response.json").read_bytes()
    raw = loads(raw_bytes)
    if digest(request) != response["request_sha256"]:
        raise ValueError("Request provenance changed")
    if hashlib.sha256(raw_bytes).hexdigest() != response["raw_provider_response_sha256"]:
        raise ValueError("Raw response provenance changed")
    if raw.get("model") not in PROVIDER_MODELS or raw.get("status") != "completed":
        raise ValueError("Unexpected model or incomplete provider response")
    if response.get("provider_model") != raw.get("model") or response.get("provider_response_id") != raw.get("id") or response.get("provider_status") != raw.get("status"):
        raise ValueError("Provider response metadata binding changed")
    calls = [(i, x) for i, x in enumerate(raw.get("output", [])) if x.get("type") == "function_call"]
    if len(calls) != 1 or calls[0][1].get("name") != "propose_trajectory":
        raise ValueError("Expected exactly one bound strategy function call")
    index, call = calls[0]
    if call.get("arguments") != response["content"] or response["raw_arguments_json_pointer"] != f"/output/{index}/arguments":
        raise ValueError("Strategy argument binding changed")
    strategy = loads(call["arguments"])
    validate(strategy, request["actual_sdk_kwargs"]["tools"][0]["parameters"])
    return strategy


def request_strategy(prompt, evidence, strategy_schema, output_dir, *, execute=False,
                     effort="medium", max_output_tokens=16000, profile_source="franka-latest"):
    """Explicit opt-in API request; refuses to resubmit any existing request folder."""
    if not execute:
        raise ValueError("Billable entrypoint requires explicit execute=True")
    kwargs = build_kwargs(prompt, evidence, strategy_schema, effort=effort, max_output_tokens=max_output_tokens)
    client, profile_metadata = make_client(profile_source, return_metadata=True)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    request = dict(schema="rssr.trajectory.luna.request.v1", profile="gpt6luna", model=MODEL,
                   profile_provenance=profile_metadata,
                   created_utc=datetime.now(timezone.utc).isoformat(), timeout_s=300,
                   transport_retries=0, actual_sdk_kwargs=kwargs,
                   endpoint_sha256=hashlib.sha256(EXPECTED_ENDPOINT.encode()).hexdigest())
    _write(folder / "request.json", request)
    _write(folder / "submission.json", dict(request_sha256=digest(request),
                                            actual_sdk_method="client.responses.create"))
    began = time.monotonic()
    try:
        sdk_response = client.responses.create(**kwargs)
    except Exception as error:
        status = getattr(error, "status_code", None)
        _write(folder / "error.json", dict(error_type=type(error).__name__, status_code=status,
            outcome="known_http_rejection" if isinstance(status, int) else "unknown_transport_outcome",
            automatic_retry=False))
        raise RuntimeError("Luna request failed; sanitized provenance retained; never resubmit this directory") from None
    raw = sdk_response.model_dump(mode="json", exclude_none=False)
    _write(folder / "raw_provider_response.json", raw)
    calls = [(i, x) for i, x in enumerate(raw.get("output", [])) if x.get("type") == "function_call"]
    index, call = calls[0] if len(calls) == 1 else (None, {})
    response = dict(schema="rssr.trajectory.luna.response.v1", request_sha256=digest(request),
                    provider_api="responses", provider_model=raw.get("model"),
                    provider_response_id=raw.get("id"), provider_status=raw.get("status"),
                    provider_request_id=getattr(sdk_response, "_request_id", None),
                    received_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-began,
                    usage=raw.get("usage"), content=call.get("arguments"),
                    raw_arguments_json_pointer=f"/output/{index}/arguments" if index is not None else None,
                    raw_provider_response_sha256=hashlib.sha256((folder / "raw_provider_response.json").read_bytes()).hexdigest())
    _write(folder / "response.json", response)
    strategy = verify_response(folder)
    _write(folder / "strategy.json", strategy)
    return strategy

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from dotenv import load_dotenv

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - optional dependency
    tabulate = None

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)

try:
    from deepeval.metrics import ToolUseMetric
except ImportError:  # pragma: no cover - gracefully degrade when unavailable
    ToolUseMetric = None  # type: ignore[assignment]

try:
    from deepeval import evaluate as deepeval_evaluate
except ImportError:  # pragma: no cover - optional dependency
    deepeval_evaluate = None
from deepeval.test_case import ConversationalTestCase, LLMTestCase
from deepeval.test_case.conversational_test_case import Turn
from deepeval.test_case.llm_test_case import ToolCall

SKIP_TEXT_KEYS = {"toolUseId", "status", "role", "finish_reason", "id", "name"}
CaseBundle = Tuple[str, LLMTestCase, Dict[str, object], ConversationalTestCase]


def _iter_table_objects(log_path: Path) -> Iterable[Dict]:
    decoder = json.JSONDecoder()
    text = log_path.read_text(encoding="utf-8")
    length = len(text)
    idx = 0
    while idx < length:
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            payload, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(payload, dict) and "resource" in payload and "timeUnixNano" in payload:
            yield payload
            idx = end
        else:
            idx = brace + 1


def iter_log_objects(log_path: Path) -> Iterable[Dict]:
    text = log_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        yield from _iter_table_objects(log_path)
        return

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
    elif isinstance(payload, dict):
        yield payload


def extract_texts(node) -> List[str]:
    seen: Set[str] = set()
    texts: List[str] = []

    def _walk(value, key_hint: str | None = None) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or key_hint in SKIP_TEXT_KEYS:
                return
            if stripped.startswith(("{", "[")):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
                else:
                    _walk(parsed, key_hint)
                    return
            if stripped not in seen:
                seen.add(stripped)
                texts.append(stripped)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in SKIP_TEXT_KEYS:
                    continue
                _walk(child, key)
        elif isinstance(value, list):
            for item in value:
                _walk(item, key_hint)

    _walk(node)
    return texts


def extract_tool_names(node) -> Set[str]:
    names: Set[str] = set()

    def _walk(value) -> None:
        if isinstance(value, dict):
            if "toolUse" in value and isinstance(value["toolUse"], dict):
                maybe = value["toolUse"].get("name")
                if isinstance(maybe, str) and maybe:
                    names.add(maybe)
            if "function" in value and isinstance(value["function"], dict):
                maybe = value["function"].get("name")
                if isinstance(maybe, str) and maybe:
                    names.add(maybe)
            if "name" in value and any(k in value for k in {"arguments", "input_parameters"}):
                maybe = value.get("name")
                if isinstance(maybe, str) and maybe:
                    names.add(maybe)
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(node)
    return names


def _ingest_strands_tracer_event(details: Dict[str, object], body: Dict) -> None:
    messages = body.get("input", {}).get("messages", [])
    for message in messages:
        role = message.get("role")
        texts = extract_texts(message)
        if role == "user" and texts:
            if not details["user"]:
                details["user"] = texts[0]
        elif role == "tool":
            for ctx in texts:
                if ctx not in details["contexts"]:
                    details["contexts"].append(ctx)
        details["tool_names"].update(extract_tool_names(message))

    outputs = body.get("output", {}).get("messages", [])
    for message in outputs:
        role = message.get("role")
        texts = extract_texts(message)
        if role == "assistant" and texts:
            details["answer"] = texts[-1]
        elif role == "tool":
            for ctx in texts:
                if ctx not in details["contexts"]:
                    details["contexts"].append(ctx)
        details["tool_names"].update(extract_tool_names(message))


def collect_trace_data(log_path: Path) -> Dict[str, Dict[str, object]]:
    traces: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "user": None,
            "answer": None,
            "contexts": [],
            "tool_names": set(),
            "source_files": {log_path.name},
        }
    )

    for entry in iter_log_objects(log_path):
        trace_id = entry.get("traceId") or entry.get("attributes", {}).get("otelTraceID")
        if not trace_id or trace_id in {"0", ""}:
            continue
        event_name = entry.get("attributes", {}).get("event.name")
        body = entry.get("body")
        details = traces[trace_id]
        details["source_files"].add(log_path.name)
        if event_name == "gen_ai.user.message":
            texts = extract_texts(body)
            if texts and not details["user"]:
                details["user"] = texts[0]
        elif event_name == "gen_ai.choice":
            texts = extract_texts(body)
            if texts:
                details["answer"] = texts[-1]
        elif event_name == "gen_ai.assistant.message":
            texts = extract_texts(body)
            if texts:
                details["answer"] = texts[-1]
        elif event_name == "gen_ai.tool.message":
            for ctx in extract_texts(body):
                if ctx not in details["contexts"]:
                    details["contexts"].append(ctx)
        elif event_name == "strands.telemetry.tracer" and isinstance(body, dict):
            _ingest_strands_tracer_event(details, body)
        if body:
            details["tool_names"].update(extract_tool_names(body))

    return traces


def build_cases(log_path: Path) -> List[CaseBundle]:
    traces = collect_trace_data(log_path)
    bundles: List[CaseBundle] = []
    for trace_id, info in traces.items():
        if not info["user"] or not info["answer"]:
            continue
        retrieval_context = info["contexts"] or []
        tool_calls = [ToolCall(name=name) for name in sorted(info["tool_names"])]
        case = LLMTestCase(
            input=info["user"],
            actual_output=info["answer"],
            retrieval_context=retrieval_context,
            tools_called=tool_calls,
            additional_metadata={
                "trace_id": trace_id,
                "source_files": sorted(info["source_files"]),
            },
        )
        # Tool correctness metrics require an expected tool plan; default to observed tools.
        case.expected_tools = tool_calls  # type: ignore[attr-defined]

        turns = [
            Turn(role="user", content=info["user"], retrieval_context=retrieval_context or None),
            Turn(
                role="assistant",
                content=info["answer"],
                tools_called=tool_calls or None,
                retrieval_context=retrieval_context or None,
            ),
        ]
        conversational_case = ConversationalTestCase(
            turns=turns,
            context=retrieval_context or None,
            additional_metadata={
                "trace_id": trace_id,
                "source_files": sorted(info["source_files"]),
            },
        )

        bundles.append((trace_id, case, info, conversational_case))
    return bundles


def evaluate_rag(cases: List[CaseBundle]) -> None:
    eval_model_name = os.getenv("DEEPEVAL_MODEL_NAME", "gpt-4o-mini")
    metrics = [
        ("Faithfulness", FaithfulnessMetric(model=eval_model_name)),
        ("Answer Relevance", AnswerRelevancyMetric(model=eval_model_name)),
        ("Context Relevance", ContextualRelevancyMetric(model=eval_model_name)),
    ]
    rows = []
    for trace_id, case, info, _ in cases:
        row = {
            "trace": trace_id[:8],
            "tools": ", ".join(sorted(info["tool_names"])) or "None",
            "contexts": len(info["contexts"]),
        }
        for label, metric in metrics:
            try:
                score = metric.measure(case, _show_indicator=False)
            except Exception as exc:  # pragma: no cover - network / LLM issues
                score = float("nan")
                print(f"[WARN] {label} failed for trace {trace_id}: {exc}")
            row[label] = score
        rows.append(row)
    if tabulate:
        print(tabulate(rows, headers="keys", floatfmt=".3f"))
    else:
        for row in rows:
            print(row)


def evaluate_tool_usage(cases: List[CaseBundle]) -> None:
    tool_use_metric = None
    if ToolUseMetric:
        try:
            threshold = float(os.getenv("TOOL_USE_THRESHOLD", "0.5"))
        except ValueError:
            threshold = 0.5
        available_tools_env = os.getenv("TOOL_USE_AVAILABLE_TOOLS")
        if available_tools_env:
            available_tools = [name.strip() for name in available_tools_env.split(",") if name.strip()]
        else:
            available_tools = [
                "mitre_attack_search",
                "virustotal_lookup",
                "nvd_vulnerability_search",
            ]
        tool_use_metric = ToolUseMetric(
            available_tools=available_tools,
            threshold=threshold,
        )
    else:
        print("[INFO] ToolUseMetric unavailable; install deepeval with agentic extras")

    rows = []
    tool_frequency: Dict[str, int] = defaultdict(int)
    conversational_cases = []
    for trace_id, case, info, conv_case in cases:
        names = sorted(info["tool_names"])
        for name in names:
            tool_frequency[name] += 1
        row = {
            "trace": trace_id[:8],
            "tool_count": len(names),
            "tools": ", ".join(names) or "None",
        }
        if tool_use_metric:
            conversational_cases.append(conv_case)
            try:
                row["Tool Use"] = tool_use_metric.measure(conv_case, _show_indicator=False)
            except Exception as exc:  # pragma: no cover - LLM/network failures
                row["Tool Use"] = float("nan")
                print(f"[WARN] Tool Use metric failed for trace {trace_id}: {exc}")
        rows.append(row)

    if tabulate:
        print(tabulate(rows, headers="keys", floatfmt=".3f"))
    else:
        for row in rows:
            print(row)

    if tool_frequency:
        print("\nAggregated tool usage:")
        for name, count in sorted(tool_frequency.items(), key=lambda item: (-item[1], item[0])):
            print(f"- {name}: {count} trace(s)")
    else:
        print("\nNo tool usage captured in the provided trace data.")

    if tool_use_metric and deepeval_evaluate and conversational_cases:
        try:
            deepeval_evaluate(
                test_cases=conversational_cases,
                metrics=[tool_use_metric],
                show_indicator=False,
            )
        except TypeError:
            # Older deepeval versions may not support keyword args.
            try:
                deepeval_evaluate(conversational_cases, [tool_use_metric])
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Unable to run deepeval.evaluate: {exc}")
        except Exception as exc:  # pragma: no cover - network/LLM issues
            print(f"[WARN] ToolUseMetric evaluation failed: {exc}")
    elif tool_use_metric and not conversational_cases:
        print("[INFO] No conversational test cases available for ToolUseMetric evaluation.")
    elif tool_use_metric:
        print("[INFO] deepeval.evaluate not available; skipped evaluator run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Strands agent traces using DeepEval metrics."
    )
    parser.add_argument(
        "log_path",
        help="Path to the OTEL trace log file (JSON or CloudWatch table export).",
    )
    parser.add_argument(
        "mode",
        choices=["rag", "tool"],
        help="Select 'rag' for RAG quality metrics or 'tool' for tool usage summaries.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=False)
    args = parse_args()
    log_path = Path(args.log_path).expanduser()
    if not log_path.exists():
        raise FileNotFoundError(f"Trace file not found: {log_path}")
    cases = build_cases(log_path)
    if not cases:
        print("No completed traces found to evaluate.")
        return
    if args.mode == "rag":
        evaluate_rag(cases)
    else:
        evaluate_tool_usage(cases)


if __name__ == "__main__":
    main()

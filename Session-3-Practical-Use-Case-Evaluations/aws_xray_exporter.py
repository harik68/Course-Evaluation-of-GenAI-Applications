import json
import logging
from typing import Any, Dict, Sequence, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult, SpanExporter
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

_NANOS_IN_SECOND = 1_000_000_000
_ALLOWED_ANNOTATION_TYPES = (bool, int, float)
_MAX_DOCS_PER_CALL = 50


def _trace_id_to_xray(trace_id: int, start_time_ns: int) -> str:
    """Convert a 16-byte OpenTelemetry trace ID into the AWS X-Ray format."""
    trace_hex = f"{trace_id:032x}"
    epoch_seconds = max(0, int(start_time_ns / _NANOS_IN_SECOND))
    return f"1-{epoch_seconds:08x}-{trace_hex[8:]}"


def _sanitize_name(name: str) -> str:
    sanitized = [ch if ch.isalnum() or ch in "._-:/" else "-" for ch in name[:200]]
    cleaned = "".join(sanitized).strip("-")
    return cleaned or "span"


def _split_attributes(span: ReadableSpan) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    annotations: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

    combined: Dict[str, Any] = {}
    if span.resource is not None:
        combined.update(span.resource.attributes)
    if span.attributes:
        combined.update(span.attributes)

    for key, value in combined.items():
        key_str = str(key)[:250]
        if isinstance(value, _ALLOWED_ANNOTATION_TYPES):
            annotations[key_str] = value
        elif isinstance(value, str):
            metadata[key_str] = value[:1024]
        else:
            metadata[key_str] = str(value)[:1024]

    return annotations, metadata


def _extract_exceptions(span: ReadableSpan) -> Dict[str, Any]:
    exceptions = []
    for event in span.events:
        if event.name != "exception":
            continue
        attrs = event.attributes or {}
        exceptions.append(
            {
                "type": str(attrs.get("exception.type", "Exception"))[:128],
                "message": str(attrs.get("exception.message", ""))[:512],
                "stack": attrs.get("exception.stacktrace"),
            }
        )
    if not exceptions:
        return {}
    return {"exceptions": exceptions}


class AwsXRaySpanExporter(SpanExporter):
    """Minimal AWS X-Ray exporter that calls PutTraceSegments directly."""

    def __init__(
        self,
        region: str,
        service_name: str,
        client=None,
        max_documents_per_call: int = _MAX_DOCS_PER_CALL,
    ) -> None:
        self._region = region
        self._service_name = service_name
        self._client = client or boto3.client("xray", region_name=region)
        self._max_docs = max(1, min(max_documents_per_call, _MAX_DOCS_PER_CALL))

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if not spans:
            return SpanExportResult.SUCCESS

        documents = []
        for span in spans:
            try:
                segment_doc = json.dumps(self._span_to_segment(span))
                documents.append(segment_doc)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.error("Failed to serialize span '%s': %s", span.name, exc, exc_info=True)

        for idx in range(0, len(documents), self._max_docs):
            batch = documents[idx : idx + self._max_docs]
            try:
                self._client.put_trace_segments(TraceSegmentDocuments=batch)
            except (BotoCoreError, ClientError) as exc:
                logger.error("AWS X-Ray export failed: %s", exc, exc_info=True)
                return SpanExportResult.FAILURE

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # pragma: no cover - noop
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover - noop
        return True

    def _span_to_segment(self, span: ReadableSpan) -> Dict[str, Any]:
        ctx = span.context
        start_time_ns = span.start_time or 0
        end_time_ns = span.end_time or start_time_ns
        start_time = start_time_ns / _NANOS_IN_SECOND
        end_time = max(start_time, end_time_ns / _NANOS_IN_SECOND)

        is_root = span.parent is None
        segment_name = span.name if is_root else span.name or self._service_name

        segment: Dict[str, Any] = {
            "name": _sanitize_name(segment_name),
            "id": f"{ctx.span_id:016x}",
            "trace_id": _trace_id_to_xray(ctx.trace_id, start_time_ns),
            "start_time": start_time,
            "end_time": end_time,
            "aws": {"region": self._region},
        }

        if not is_root and span.parent is not None:
            segment["type"] = "subsegment"
            segment["parent_id"] = f"{span.parent.span_id:016x}"

        annotations, metadata = _split_attributes(span)
        if annotations:
            segment["annotations"] = annotations
        if metadata:
            segment["metadata"] = {"default": metadata}

        resource_attrs = span.resource.attributes if span.resource is not None else {}
        service_block: Dict[str, Any] = {"name": resource_attrs.get("service.name", self._service_name)}
        if version := resource_attrs.get("service.version"):
            service_block["version"] = version
        if service_block:
            segment["service"] = service_block
        if namespace := resource_attrs.get("service.namespace"):
            segment["namespace"] = namespace

        if span.status.status_code is StatusCode.ERROR:
            segment["fault"] = True
            if span.status.description:
                segment.setdefault("cause", {}).setdefault(
                    "exceptions",
                    [],
                ).append({"type": "Error", "message": span.status.description[:512]})
        elif span.status.status_code is StatusCode.UNSET:
            if span.status.description:
                segment.setdefault("annotations", {})["otel.status_description"] = span.status.description[:256]

        exception_block = _extract_exceptions(span)
        if exception_block:
            existing = segment.setdefault("cause", {})
            existing.setdefault("exceptions", []).extend(exception_block["exceptions"])

        return segment

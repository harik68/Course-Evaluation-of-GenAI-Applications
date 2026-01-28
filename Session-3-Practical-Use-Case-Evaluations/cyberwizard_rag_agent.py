import os
import logging
import argparse
import json
import atexit
from datetime import datetime
from io import StringIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus
from uuid import uuid4

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from opentelemetry import baggage, context, trace
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.extension.aws.trace.aws_xray_id_generator import AwsXRayIdGenerator
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
import numpy as np
from pypdf import PdfReader

from aws_xray_exporter import AwsXRaySpanExporter

def parse_arguments():
    """Parse the CLI arguments that control session metadata and ingestion."""
    parser = argparse.ArgumentParser(description='Strands Cybersecurity Agent with Session Tracking')
    parser.add_argument('--session-id', 
                       type=str, 
                       required=True,
                       help='Session ID to associate with this agent run')
    parser.add_argument('--question',
                       type=str,
                       required=False,
                       help='Cybersecurity investigation question for the agent to analyze')
    parser.add_argument('--ingest-pdfs',
                       action='store_true',
                       help='When set, ingest PDFs from the local folder before running the agent')
    parser.add_argument('--pdf-folder',
                       type=str,
                       default='pdf_files',
                       help='Folder containing PDFs to ingest into the vector store')
    return parser.parse_args()

def set_session_context(session_id):
    """Set the session ID in OpenTelemetry baggage for trace correlation"""
    ctx = baggage.set_baggage("session.id", session_id)
    token = context.attach(ctx)
    logging.info(f"Session ID '{session_id}' attached to telemetry context")
    return token

###########################
#### Agent Code below: ####
###########################

from strands import Agent, tool
from strands.models import BedrockModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Strands logging
logging.getLogger("strands").setLevel(logging.INFO)


S3_BUCKET_ENV = "AGENT_LOG_BUCKET"
S3_PREFIX_ENV = "AGENT_LOG_PREFIX"
S3_BUFFER_BYTES_ENV = "AGENT_LOG_MAX_BUFFER"
_s3_log_handler = None


class S3LogHandler(logging.Handler):
    """Buffer logs locally and periodically upload them to S3."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "cloudwatch-export",
        session_id: Optional[str] = None,
        client=None,
        max_bytes: int = 262_144,
    ) -> None:
        super().__init__()
        self.bucket_name = bucket_name
        self.session_id = session_id or "unknown-session"
        self._prefix = prefix.strip("/") or "cloudwatch-export"
        self._client = client or boto3.client("s3")
        self._buffer = StringIO()
        self._lock = Lock()
        self._run_id = uuid4().hex
        self._sequence = 0
        self._max_bytes = max(4096, int(max_bytes))

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Refresh the session identifier used in generated log keys."""
        if session_id:
            self.session_id = session_id

    def emit(self, record: logging.LogRecord) -> None:
        """Format a log record, buffer it, and flush if the buffer is full."""
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - fallback path
            message = record.getMessage()

        payload = None
        with self._lock:
            self._buffer.write(message + "\n")
            if self._buffer.tell() >= self._max_bytes:
                payload = self._drain_buffer_locked()

        if payload:
            self._upload_payload(payload)

    def flush(self) -> None:
        """Force a buffered upload regardless of the current buffer size."""
        payload = None
        with self._lock:
            payload = self._drain_buffer_locked()
        if payload:
            self._upload_payload(payload)

    def close(self) -> None:
        """Flush remaining bytes and release resources."""
        try:
            self.flush()
        finally:
            try:
                self._buffer.close()
            except Exception:  # pragma: no cover - defensive
                pass
            super().close()

    def _drain_buffer_locked(self) -> Optional[Tuple[str, bytes]]:
        """Return the current buffer contents as bytes and reset the buffer."""
        contents = self._buffer.getvalue()
        if not contents.strip():
            return None
        self._buffer.close()
        self._buffer = StringIO()
        key = self._build_key()
        return key, contents.encode("utf-8")

    def _build_key(self) -> str:
        """Generate a deterministic S3 object key for the current session."""
        self._sequence += 1
        now = datetime.utcnow()
        date_path = now.strftime("%Y/%m/%d")
        filename = f"{self._sanitize(self.session_id)}-{now.strftime('%H%M%S')}-{self._run_id}-{self._sequence:04d}.log"
        return f"{self._prefix}/{date_path}/{filename}"

    def _sanitize(self, value: str) -> str:
        """Replace unsafe characters in path components with hyphens."""
        safe = [ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in value[:80]]
        return "".join(safe) or "session"

    def _upload_payload(self, payload: Tuple[str, bytes]) -> None:
        """Send the buffered payload to S3 and log the outcome locally."""
        key, data = payload
        try:
            self._client.put_object(Bucket=self.bucket_name, Key=key, Body=data)
            print(f"[S3LogHandler] Uploaded logs to s3://{self.bucket_name}/{key}")
        except Exception as exc:  # pragma: no cover - network errors
            print(f"[S3LogHandler] Failed to upload logs to s3://{self.bucket_name}/{key}: {exc}")


def configure_s3_log_archival(session_id: Optional[str]) -> Optional[S3LogHandler]:
    """Ensure the shared S3 log handler exists and is bound to the session."""
    global _s3_log_handler
    bucket = os.getenv(S3_BUCKET_ENV)
    if not bucket:
        return None

    max_buffer = os.getenv(S3_BUFFER_BYTES_ENV, "262144")
    try:
        max_buffer_bytes = int(max_buffer)
    except ValueError:
        logger.warning(
            "Invalid %s value '%s'. Using default buffer size.",
            S3_BUFFER_BYTES_ENV,
            max_buffer,
        )
        max_buffer_bytes = 262_144

    if _s3_log_handler is None:
        prefix = os.getenv(S3_PREFIX_ENV, "cloudwatch-export")
        handler = S3LogHandler(
            bucket_name=bucket,
            prefix=prefix,
            session_id=session_id,
            max_bytes=max_buffer_bytes,
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logging.getLogger().addHandler(handler)
        _s3_log_handler = handler
        atexit.register(handler.close)
        logger.info("S3 log archival enabled (bucket=%s, prefix=%s)", bucket, prefix)
    else:
        _s3_log_handler.set_session_id(session_id)

    return _s3_log_handler


def _parse_sample_ratio(value: str) -> float:
    """Safely coerce the OTEL sampling ratio into the valid [0, 1] range."""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid OTEL_TRACES_SAMPLER_ARG '%s'. Falling back to 1.0", value)
        return 1.0
    return min(1.0, max(0.0, ratio))


def configure_aws_tracing() -> None:
    """Configure an in-process AWS X-Ray exporter (Option 2)."""
    if os.getenv("STRANDS_DISABLE_CUSTOM_XRAY") in {"1", "true", "True"}:
        logger.info("Custom AWS X-Ray exporter disabled via STRANDS_DISABLE_CUSTOM_XRAY")
        return

    region = (
        os.getenv("AWS_TRACES_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )
    service_name = os.getenv("OTEL_SERVICE_NAME", "strands-cyber-rag")
    resource_attrs = {SERVICE_NAME: service_name}
    if service_version := os.getenv("SERVICE_VERSION"):
        resource_attrs["service.version"] = service_version
    if service_namespace := os.getenv("SERVICE_NAMESPACE"):
        resource_attrs["service.namespace"] = service_namespace

    sample_ratio = _parse_sample_ratio(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))

    tracer_provider = TracerProvider(
        resource=Resource.create(resource_attrs),
        id_generator=AwsXRayIdGenerator(),
        sampler=ParentBased(TraceIdRatioBased(sample_ratio)),
    )
    exporter = AwsXRaySpanExporter(region=region, service_name=service_name)
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
    logger.info(
        "AWS X-Ray tracing configured (region=%s, service=%s, sample_ratio=%s)",
        region,
        service_name,
        sample_ratio,
    )

    requests_instrumentor = RequestsInstrumentor()
    try:
        already_requests = requests_instrumentor.is_instrumented()  # type: ignore[attr-defined]
    except AttributeError:
        already_requests = False
    if not already_requests:
        try:
            requests_instrumentor.instrument()
        except Exception as exc:  # pragma: no cover - guard against re-entry issues
            logger.warning("Requests instrumentation failed: %s", exc)

    botocore_instrumentor = BotocoreInstrumentor()
    try:
        already_botocore = botocore_instrumentor.is_instrumented()  # type: ignore[attr-defined]
    except AttributeError:
        already_botocore = False
    if not already_botocore:
        try:
            botocore_instrumentor.instrument()
        except Exception as exc:  # pragma: no cover
            logger.warning("Botocore instrumentation failed: %s", exc)


configure_aws_tracing()

DEFAULT_VECTOR_DIR = "vector_store"


def _resolve_vector_store_dir(override: Optional[str] = None) -> Path:
    """Resolve the folder that stores embeddings and metadata."""
    if override:
        return Path(override)

    for env_key in ("VECTOR_STORE_DIR", "MITRE_VECTOR_DB_PATH"):
        configured = os.getenv(env_key)
        if configured:
            return Path(configured)

    return Path(DEFAULT_VECTOR_DIR)


def _vector_store_paths(store_dir: Path) -> Tuple[Path, Path]:
    """Return the embeddings.npy and documents.jsonl paths, ensuring the folder exists."""
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / "embeddings.npy", store_dir / "documents.jsonl"


def _extend_vector_store(store_dir: Path, embeddings: List[List[float]], records: List[dict]) -> int:
    """Append new embeddings and records to the local vector store."""
    if not embeddings:
        return 0

    embeddings_path, documents_path = _vector_store_paths(store_dir)
    new_array = np.array(embeddings, dtype=np.float32)

    if embeddings_path.exists():
        existing = np.load(embeddings_path)
        existing = existing.astype(np.float32, copy=False)
        new_array = np.vstack([existing, new_array])

    np.save(embeddings_path, new_array)

    with open(documents_path, "a", encoding="utf-8") as doc_file:
        for record in records:
            doc_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def _load_vector_store(store_dir: Path) -> Tuple[Optional[np.ndarray], List[dict]]:
    """Load embeddings and associated metadata from disk."""
    embeddings_path, documents_path = _vector_store_paths(store_dir)

    if not embeddings_path.exists() or not documents_path.exists():
        return None, []

    embeddings = np.load(embeddings_path)
    embeddings = embeddings.astype(np.float32, copy=False)

    with open(documents_path, "r", encoding="utf-8") as doc_file:
        records = [json.loads(line) for line in doc_file if line.strip()]

    if len(records) != len(embeddings):
        logger.warning(
            "Vector store record count mismatch (records=%s, embeddings=%s). Truncating to smallest.",
            len(records),
            len(embeddings),
        )
        min_len = min(len(records), len(embeddings))
        embeddings = embeddings[:min_len]
        records = records[:min_len]

    return embeddings, records


def _cosine_similarity_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a matrix of embeddings and a single vector."""
    if matrix.size == 0:
        return np.array([])

    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        return np.zeros(matrix.shape[0])

    normalized_vector = vector / vector_norm
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms[matrix_norms == 0] = 1e-10

    scores = matrix @ normalized_vector
    return scores / matrix_norms


def _chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """Yield overlapping text windows for downstream embedding."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return
    text_len = len(cleaned)
    start = 0
    while start < text_len:
        end = min(text_len, start + chunk_size)
        yield cleaned[start:end]
        if end == text_len:
            break
        start = max(0, end - overlap)


def _embed_texts_with_bedrock(texts: List[str]) -> List[List[float]]:
    """Call an AWS Bedrock embedding model and return dense vectors."""
    if not texts:
        return []

    model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")
    client = boto3.client("bedrock-runtime", region_name=region)

    embeddings: List[List[float]] = []
    for text in texts:
        payload = json.dumps({"inputText": text})
        try:
            response = client.invoke_model(
                body=payload,
                modelId=model_id,
                accept="application/json",
                contentType="application/json",
            )
        except (BotoCoreError, ClientError) as exc:
            logger.error("Embedding request failed: %s", exc)
            raise RuntimeError(f"Embedding request failed: {exc}") from exc

        body = response.get("body")
        data = json.loads(body.read()) if hasattr(body, "read") else json.loads(body)  # type: ignore[arg-type]
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError("Embedding response missing 'embedding' field")
        embeddings.append(embedding)

    return embeddings


def ingest_pdf_folder(
    folder_path: str = "pdf_files",
    store_dir: Optional[str] = None,
    batch_size: int = 16,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> str:
    """Ingest PDFs into the local numpy-based vector store."""
    source_dir = Path(folder_path)
    if not source_dir.exists():
        return f"PDF folder '{folder_path}' not found. Skipping ingestion."

    pdf_files = sorted(path for path in source_dir.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_files:
        return f"No PDF files discovered in '{folder_path}'."

    vector_dir = _resolve_vector_store_dir(store_dir)

    pending_records: List[dict] = []
    staged_records: List[dict] = []
    staged_embeddings: List[List[float]] = []

    def flush_batch():
        nonlocal pending_records
        if not pending_records:
            return
        batch = pending_records
        pending_records = []
        texts = [record["text"] for record in batch]
        embeddings = _embed_texts_with_bedrock(texts)
        staged_records.extend(batch)
        staged_embeddings.extend(embeddings)

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # pragma: no cover - depends on local files
            logger.error("Failed to read %s: %s", pdf_path, exc)
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk_index, chunk in enumerate(_chunk_text(text, chunk_size, chunk_overlap), start=1):
                snippet = chunk.strip()
                if not snippet:
                    continue
                record_id = f"pdf::{pdf_path.stem}::p{page_number}::c{chunk_index}::{uuid4().hex}"
                metadata = {
                    "source": "pdf",
                    "file": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_index,
                }
                pending_records.append({"id": record_id, "text": snippet, "metadata": metadata})
                if len(pending_records) >= batch_size:
                    flush_batch()

    flush_batch()
    added_chunks = _extend_vector_store(vector_dir, staged_embeddings, staged_records)
    store_label = str(vector_dir)
    return (
        f"Ingested {added_chunks} PDF chunks from {len(pdf_files)} files into vector store "
        f"'{store_label}'."
    )


def _format_json(metadata):
    """Return metadata as a compact JSON string suitable for logging."""
    try:
        return json.dumps(metadata, ensure_ascii=True)
    except (TypeError, ValueError):
        return str(metadata)


@tool
def mitre_attack_search(query: str) -> str:
    """Search the local numpy-based MITRE ATT&CK vector store for techniques and mitigations."""
    store_dir = _resolve_vector_store_dir()
    top_k = int(os.getenv("MITRE_TOP_K", "3"))

    embeddings, records = _load_vector_store(store_dir)
    if embeddings is None or not records:
        return (
            f"Vector store '{store_dir}' is empty. Ingest MITRE or PDF data before searching."
        )

    query_embedding = _embed_texts_with_bedrock([query])[0]
    query_vector = np.array(query_embedding, dtype=np.float32)

    scores = _cosine_similarity_matrix(embeddings, query_vector)
    if scores.size == 0:
        return f"No MITRE ATT&CK matches for '{query}'."

    ranked_indices = np.argsort(scores)[::-1][:max(1, top_k)]

    formatted = []
    for rank, idx in enumerate(ranked_indices, start=1):
        record = records[idx]
        meta = record.get("metadata", {})
        snippet_text = (record.get("text") or "").strip().replace("\n", " ")
        technique = meta.get("technique_id") or meta.get("technique") or "Unknown Technique"
        tactic = meta.get("tactic") or meta.get("phase_name")
        reference = meta.get("url") or meta.get("reference") or meta.get("file")
        similarity = f"{scores[idx]:.4f}"

        formatted.append(
            f"{rank}. Technique: {technique}"
            f"{' | Tactic: ' + tactic if tactic else ''}\n"
            f"   Cosine similarity: {similarity}\n"
            f"   Insight: {snippet_text[:600]}\n"
            f"   Reference: {reference or _format_json(meta)}"
        )

    return "\n".join(formatted)


@tool
def virustotal_lookup(query: str) -> str:
    """Query VirusTotal for malware, hashes, domains, or URLs related to the input."""
    api_key = os.getenv("VIRUSTOTAL_API_KEY")
    if not api_key:
        return "VirusTotal API key not configured. Set VIRUSTOTAL_API_KEY to enable this tool."

    url = f"https://www.virustotal.com/api/v3/search?query={quote_plus(query)}"
    headers = {"x-apikey": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network call
        logger.error("VirusTotal lookup error: %s", exc)
        return f"VirusTotal lookup error: {exc}"

    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        return f"No VirusTotal matches found for '{query}'."

    formatted = []
    for idx, row in enumerate(rows[:5]):
        attributes = row.get("attributes", {})
        stats = attributes.get("last_analysis_stats", {})
        total = sum(stats.values()) or 1
        detection_ratio = f"{stats.get('malicious', 0)}/{total}"
        threat_label = attributes.get("popular_threat_classification", {}).get("suggested_threat_label")
        item_type = row.get("type", "unknown")
        vt_id = row.get("id", "unknown")
        gui_url = f"https://www.virustotal.com/gui/{item_type}/{vt_id}"

        formatted.append(
            f"{idx + 1}. Type: {item_type} | Detection ratio: {detection_ratio}\n"
            f"   Threat label: {threat_label or 'Not classified'}\n"
            f"   VT link: {gui_url}\n"
            f"   Query context: {query}"
        )

    return "\n".join(formatted)


def _extract_cvss(metrics: dict) -> Tuple[str, str, str]:
    """Pick the most specific CVSS tuple (severity, score, vector) from NVD metrics."""
    metric_priority = ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
    for metric_name in metric_priority:
        metric_entries = metrics.get(metric_name)
        if metric_entries:
            entry = metric_entries[0]
            data = entry.get("cvssData", {})
            severity = entry.get("baseSeverity") or data.get("baseSeverity") or "UNKNOWN"
            score = data.get("baseScore") or entry.get("baseScore") or "N/A"
            vector = data.get("vectorString") or "N/A"
            return severity, str(score), vector
    return "UNKNOWN", "N/A", "N/A"


@tool
def nvd_vulnerability_search(query: str) -> str:
    """Search the National Vulnerability Database for CVEs related to the query."""
    params = {
        "keywordSearch": query,
        "resultsPerPage": os.getenv("NVD_RESULTS_PER_PAGE", "5"),
    }
    headers = {}
    api_key = os.getenv("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key

    try:
        response = requests.get(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network call
        logger.error("NVD lookup error: %s", exc)
        return f"NVD lookup error: {exc}"

    payload = response.json()
    vulns = payload.get("vulnerabilities", [])
    if not vulns:
        return f"No NVD entries found for '{query}'."

    formatted = []
    for idx, item in enumerate(vulns[:5]):
        cve = item.get("cve", {})
        cve_id = cve.get("id", "Unknown CVE")
        descriptions = cve.get("descriptions", [])
        description = next((d.get("value") for d in descriptions if d.get("lang") == "en"), "No English description available.")
        metrics = cve.get("metrics", {})
        severity, score, vector = _extract_cvss(metrics)
        published = cve.get("published")

        formatted.append(
            f"{idx + 1}. {cve_id}\n"
            f"   Severity: {severity} (score {score}, vector {vector})\n"
            f"   Published: {published or 'Unknown'}\n"
            f"   Summary: {description}"
        )

    return "\n".join(formatted)


def get_bedrock_model():
    """Instantiate the configured Bedrock text model for conversational responses."""
    model_id = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
    region = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

    try:
        bedrock_model = BedrockModel(
            model_id=model_id,
            region_name=region,
            temperature=0.7,
            max_tokens=1024
        )
        logger.info(f"Successfully initialized Bedrock model: {model_id} in region: {region}")
        return bedrock_model
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock model: {str(e)}")
        logger.error("Please ensure you have proper AWS credentials configured and access to the Bedrock model")
        raise


def main():
    """Entry point that ingests data, configures telemetry, and runs the agent."""
    # Parse command line arguments
    args = parse_arguments()
    s3_handler = configure_s3_log_archival(args.session_id)

    # Set session context for telemetry
    context_token = set_session_context(args.session_id)

    try:
        # Optionally ingest PDFs before initializing the conversational model
        ingest_summary = None
        if args.ingest_pdfs:
            try:
                ingest_summary = ingest_pdf_folder(folder_path=args.pdf_folder)
                logger.info(ingest_summary)
            except Exception as exc:  # pragma: no cover - depends on local data
                logger.error("PDF ingestion failed: %s", exc)
                ingest_summary = f"PDF ingestion failed: {exc}"

        # Initialize Bedrock model
        bedrock_model = get_bedrock_model()

        # Create cybersecurity agent
        cybersecurity_agent = Agent(
            model=bedrock_model,
            system_prompt="""You are a cybersecurity copilot that triages analyst questions, selects the most relevant 
            data source, and synthesizes a concise incident response report. Use the MITRE ATT&CK tool for questions 
            about adversary behavior or attack techniques, the VirusTotal tool for malware, hashes, or domains, and the 
            NVD tool for software vulnerabilities or CVE lookups. Always explain which source you used and cite key 
            evidence.""",
            tools=[mitre_attack_search, virustotal_lookup, nvd_vulnerability_search],
            trace_attributes={
                "user.id": "user@domain.com",
                "tags": ["Strands", "Cybersecurity"],
            }
        )

        # Execute the cybersecurity investigation
        query = args.question or """Investigate credential dumping detected via LSASS memory access on a Windows 
        domain controller. Identify the relevant MITRE ATT&CK techniques plus containment and remediation guidance."""

        result = cybersecurity_agent(query)
        if ingest_summary:
            print("PDF ingestion:", ingest_summary)
        print("Result:", result)

    finally:
        # Detach context when done
        context.detach(context_token)
        logger.info(f"Session context for '{args.session_id}' detached")
        if s3_handler:
            s3_handler.close()


if __name__ == "__main__":
    main()

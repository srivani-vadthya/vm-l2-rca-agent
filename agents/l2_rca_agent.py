import time
import uuid

from models.request import IncidentRequest
from utils.logger import log
from services.context_builder import ContextBuilder
from services.diagnosis_service import DiagnosisService
from services.report_service import ReportService


class L2RCAAgent:

    def __init__(self):
        self.context_builder = ContextBuilder()
        self.diagnosis_service = DiagnosisService()
        self.report_service = ReportService()

    def analyze(self, incident: IncidentRequest):

        correlation_id = str(uuid.uuid4())
        ids = {"ticket_id": incident.ticket_id, "correlation_id": correlation_id}
        pipeline_start = time.perf_counter()

        # ── Stage 1: Incident received ────────────────────────────────────────
        log("Incident received",
            stage="INCIDENT_RECEIVED",
            extra={
                "application": incident.application,
                "technology": incident.technology,
                "priority": incident.priority,
                "classification": incident.classification
            },
            **ids)

        # ── Stage 2 & 3: Context building / created ───────────────────────────
        log("Context building", stage="CONTEXT_BUILDING", **ids)
        t0 = time.perf_counter()

        context = self.context_builder.build_context(incident, correlation_id=correlation_id)

        log("Context successfully created",
            stage="CONTEXT_CREATED",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            **ids)

        # ── Stage 4: Calling Groq LLM ─────────────────────────────────────────
        log("Calling Groq LLM", stage="LLM_CALL", **ids)
        t0 = time.perf_counter()

        diagnosis = self.diagnosis_service.diagnose(
            context,
            ticket_id=incident.ticket_id,
            correlation_id=correlation_id
        )

        diagnosis_elapsed = (time.perf_counter() - t0) * 1000

        # ── Stage 8-10: Parsed diagnosis, problem domain, confidence ──────────
        log("Parsed diagnosis",
            stage="DIAGNOSIS_PARSED",
            elapsed_ms=diagnosis_elapsed,
            **ids)

        log(f"Problem domain: {diagnosis.problem_domain}",
            stage="PROBLEM_DOMAIN",
            extra={"problem_domain": diagnosis.problem_domain},
            **ids)

        log(f"Confidence score: {diagnosis.confidence}",
            stage="CONFIDENCE_SCORE",
            extra={"confidence": diagnosis.confidence},
            **ids)

        # ── Stage 11 & 12: Recommended agent + RCA report generated ──────────
        t0 = time.perf_counter()

        report = self.report_service.generate_report(
            ticket_id=incident.ticket_id,
            application=incident.application,
            technology=incident.technology,
            diagnosis=diagnosis
        )

        log(f"Recommended agent: {report.recommended_agent}",
            stage="RECOMMENDED_AGENT",
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            extra={"recommended_agent": report.recommended_agent},
            **ids)

        log("RCA report generated",
            stage="RCA_REPORT_GENERATED",
            extra={"status": report.status},
            **ids)

        # ── Stage 13: Preparing request for downstream agent ─────────────────
        log("Preparing request for downstream agent",
            stage="PREPARING_DOWNSTREAM_REQUEST",
            extra={"target_agent": report.recommended_agent},
            **ids)

        payload = report.model_dump() if hasattr(report, "model_dump") else report.dict()
        payload_size = len(str(payload))

        # ── Stage 14-17: HTTP POST to downstream agent ────────────────────────
        log(f"Sending HTTP POST request to {report.recommended_agent}",
            stage="SENDING_HTTP_POST",
            extra={"agent": report.recommended_agent},
            **ids)

        log(f"Target URL: {report.recommended_agent}",
            stage="TARGET_URL",
            extra={"agent": report.recommended_agent},
            **ids)

        log(f"Payload size: {payload_size} bytes",
            stage="PAYLOAD_SIZE",
            extra={"payload_bytes": payload_size},
            **ids)

        log("Waiting for response from downstream agent",
            stage="WAITING_FOR_RESPONSE",
            **ids)

        t0 = time.perf_counter()
        execution = self.agent_executor.execute(report)
        execution_elapsed = (time.perf_counter() - t0) * 1000

        # ── Stage 18-20: Response received ───────────────────────────────────
        http_status = getattr(execution, "status_code", 200)

        log("Response received from downstream agent",
            stage="RESPONSE_RECEIVED",
            elapsed_ms=execution_elapsed,
            **ids)

        log(f"HTTP status: {http_status}",
            stage="HTTP_STATUS",
            extra={"http_status": http_status},
            **ids)

        log("Execution summary returned from downstream agent",
            stage="EXECUTION_SUMMARY_RECEIVED",
            extra={"agent": report.recommended_agent},
            **ids)

        # ── Stage 21: Returning final response ───────────────────────────────
        total_elapsed = (time.perf_counter() - pipeline_start) * 1000

        log("Returning final response",
            stage="RETURNING_FINAL_RESPONSE",
            elapsed_ms=total_elapsed,
            **ids)

        # ── Orchestration summary ─────────────────────────────────────────────
        log(
            f"ORCHESTRATION SUMMARY | "
            f"L2 RCA Agent analyzed incident [{incident.ticket_id}] | "
            f"Problem domain: {diagnosis.problem_domain} | "
            f"Selected agent: {report.recommended_agent} | "
            f"Confidence: {diagnosis.confidence} | "
            f"Delegated execution to {report.recommended_agent} | "
            f"Execution response received | "
            f"Final response returned to caller | "
            f"Total pipeline duration: {round(total_elapsed, 3)} ms",
            stage="ORCHESTRATION_SUMMARY",
            elapsed_ms=total_elapsed,
            extra={
                "ticket_id": incident.ticket_id,
                "application": incident.application,
                "technology": incident.technology,
                "problem_domain": diagnosis.problem_domain,
                "confidence": diagnosis.confidence,
                "recommended_agent": report.recommended_agent,
                "rca_status": report.status,
                "pipeline_duration_ms": round(total_elapsed, 3)
            },
            **ids
        )

        return {
            "rca": report,
            "execution": execution
        }

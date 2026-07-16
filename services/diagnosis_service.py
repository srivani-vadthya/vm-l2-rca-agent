import time

from llm.groq_client import GroqClient
from utils.output_parser import OutputParser
from utils.logger import log
from models.diagnosis import Diagnosis
from utils.exceptions import DiagnosisException


class DiagnosisService:

    def __init__(self):
        self.groq = GroqClient()

    def diagnose(self, context: dict, ticket_id: str = "", correlation_id: str = "") -> Diagnosis:

        ids = {"ticket_id": ticket_id, "correlation_id": correlation_id}

        try:
            log("Calling Groq LLM", stage="LLM_CALL", **ids)
            t0 = time.perf_counter()

            response = self.groq.diagnose(context, ticket_id=ticket_id, correlation_id=correlation_id)

            llm_elapsed = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            diagnosis_json = OutputParser.parse(response)
            diagnosis = Diagnosis(**diagnosis_json)
            parse_elapsed = (time.perf_counter() - t1) * 1000

            log("Parsed diagnosis",
                stage="DIAGNOSIS_PARSED",
                elapsed_ms=parse_elapsed,
                **ids)

            log(f"Problem domain: {diagnosis.problem_domain}",
                stage="PROBLEM_DOMAIN",
                extra={"problem_domain": diagnosis.problem_domain},
                **ids)

            log(f"Confidence score: {diagnosis.confidence}",
                stage="CONFIDENCE_SCORE",
                extra={"confidence": diagnosis.confidence},
                **ids)

            return diagnosis

        except Exception as ex:
            raise DiagnosisException(f"Failed to generate diagnosis: {str(ex)}")

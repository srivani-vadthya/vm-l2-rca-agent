from pydantic import BaseModel
from typing import List


class Diagnosis(BaseModel):

    problem_domain: str

    recommended_agent: str

    confidence: float

    reason: str

    recommended_checks: List[str]
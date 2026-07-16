from fastapi import APIRouter, HTTPException

from agents.l2_rca_agent import L2RCAAgent
from models.api_response import APIResponse
from models.request import IncidentRequest
from utils.exceptions import DiagnosisException

router = APIRouter()

agent = L2RCAAgent()


@router.post("/analyze", response_model=APIResponse)
def analyze_incident(request: IncidentRequest):

    try:
        report = agent.analyze(request)

        return APIResponse(
            success=True,
            message="Diagnosis completed successfully.",
            data=report
        )

    except DiagnosisException as ex:
        raise HTTPException(
            status_code=500,
            detail=str(ex)
        )
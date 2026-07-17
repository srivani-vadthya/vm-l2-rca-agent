import requests

from config.settings import settings


class AgentExecutor:

    def execute(self, diagnosis):

        agent = diagnosis.recommended_agent

        if agent == "db_fix_agent":

            response = requests.post(

                settings.DB_FIX_AGENT_URL +

                "/api/v1/execute",

                json={

                    "ticket_id": diagnosis.ticket_id,

                    "application": diagnosis.application,

                    "technology": diagnosis.technology,

                    "problem_domain": diagnosis.problem_domain,

                    "recommended_agent": diagnosis.recommended_agent,

                    "confidence": diagnosis.confidence,

                    "reason": diagnosis.reason,

                    "recommended_checks": diagnosis.recommended_checks,

                    "status": diagnosis.status

                },

                timeout=120

            )

            return response.json() if response.text else {"status_code": response.status_code}

        raise Exception("Unknown Agent")
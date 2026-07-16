import json
import time

from groq import Groq

from config.settings import settings
from llm.prompts import SYSTEM_PROMPT
from utils.logger import log


class GroqClient:

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def diagnose(self, context: dict, ticket_id: str = "", correlation_id: str = "") -> str:

        ids = {"ticket_id": ticket_id, "correlation_id": correlation_id}

        prompt = json.dumps(context, indent=2)

        log("Prompt generated",
            stage="PROMPT_GENERATED",
            extra={"prompt_chars": len(prompt), "model": "llama-3.3-70b-versatile"},
            **ids)

        log("Waiting for LLM response", stage="LLM_WAITING", **ids)
        t0 = time.perf_counter()

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]
        )

        elapsed = (time.perf_counter() - t0) * 1000
        content = response.choices[0].message.content

        log("LLM response received",
            stage="LLM_RESPONSE_RECEIVED",
            elapsed_ms=elapsed,
            extra={"response_chars": len(content)},
            **ids)

        return content

import json
import re


class JSONParser:
    """Utility for robust JSON extraction."""

    @staticmethod
    def extract_json(
        content: str,
    ) -> dict:

        cleaned = re.sub(
            r"```json|```",
            "",
            content,
        )

        cleaned = re.sub(
            r"<think>.*?</think>",
            "",
            cleaned,
            flags=re.DOTALL,
        )

        cleaned = cleaned.strip()

        return json.loads(
            cleaned,
        )
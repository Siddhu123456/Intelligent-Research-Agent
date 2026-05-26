import json
import re


class JSONParser:
    """Utility for robust JSON extraction."""

    @staticmethod
    def clean_response(
        content: str,
    ) -> str:
        """Clean raw LLM response."""

        # Coerce non-string content (models, dicts, lists) to string/json
        if not isinstance(content, str):
            try:
                # Pydantic model v2
                if hasattr(content, "model_dump"):
                    content = content.model_dump()

                # Pydantic model v1 or other objects
                if hasattr(content, "dict") and not isinstance(content, dict):
                    content = content.dict()

                if isinstance(content, (dict, list)):
                    content = json.dumps(content)
                else:
                    content = str(content)
            except Exception:
                try:
                    content = str(content)
                except Exception:
                    content = ""

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

        return cleaned

    @staticmethod
    def extract_json(
        content: str,
    ) -> dict:
        """Extract JSON object from LLM response."""

        cleaned = (
            JSONParser
            .clean_response(
                content,
            )
        )

        try:

            return json.loads(
                cleaned,
            )

        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                cleaned,
                flags=re.DOTALL,
            )

            if not match:

                raise ValueError(
                    "No valid JSON object found.",
                )

            return json.loads(
                match.group(),
            )

    @staticmethod
    def safe_extract(
        content: str,
        fallback: dict,
    ) -> dict:
        """Safely extract JSON with fallback."""

        try:

            return (
                JSONParser
                .extract_json(
                    content,
                )
            )

        except Exception:

            return fallback